# p3o.py — 离散提示参数的 Policy Gradient 优化器
#
# P3O（Prompt Parameter Policy Optimization）是一个 PyTorch 风格的优化器，
# 专门用于优化 Template 中的可学习 Variable（离散选项的 logit 分布）。
#
# 核心思路（REINFORCE + 熵正则化）：
#   1. Behavior.sample() 时，对每个可学习 Variable 按当前 softmax 分布采样选项索引
#   2. LLM 调用后得到各分组的预测值（group_outputs）
#   3. P3O.step() 计算 reward（预测值与目标值的函数），再用 REINFORCE loss 更新 logit
#      Loss = -advantage * log_prob(sampled_idx) - entropy_coef * entropy
#   4. 经过多步迭代，高 reward 选项的概率逐渐上升（prompt 自动进化）
#
# 与标准 SGD/Adam 的区别：
#   - 参数是离散选项的 logit（不可对 LLM 输出求梯度），因此用策略梯度绕过
#   - auto_update_from_archetype=True 时，step() 内部自动从 archetype 读取采样结果
#
# 典型用法：
#   opt = P3O(archetype=arch, lr=0.05)
#   arch.sample()       # 内部填充 last_group_keys / last_group_outputs / last_slot_choices
#   opt.step()          # 计算 reward → REINFORCE loss → 反向传播 → SGD 更新 logit
#   opt.zero_grad()
#
# 进阶用法（手动控制时机）：
#   opt = P3O(archetype=arch, auto_update_from_archetype=False)
#   arch.sample()
#   opt.update_from_archetype()
#   opt.step()
#   opt.zero_grad()

from typing import List, Optional, Callable, Any
import torch
import torch.nn as nn


class P3O:
    """离散提示参数的 PyTorch 风格策略梯度优化器。

    通过 REINFORCE 算法优化 Template 中 Variable 的 logit 分布，
    使高 reward 的选项（Prompt 表述方式）被 LLM 更频繁地采样到。
    """

    def __init__(
        self,
        *,
        archetype: Any,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        reward_fn: Optional[Callable[[float, float], float]] = None,
        auto_update_from_archetype: bool = True,
        fix_choices_after_step: bool = False,
        reducer: str = "mean",
        verbose: bool = False,
        # 解耦的 reward / target 提供器（用户自定义奖励逻辑）
        rewards_provider: Optional[Callable[[List[str], List[float], Any], List[float]]] = None,
        targets_provider: Optional[Callable[[List[str], Any], List[float]]] = None,
        # PSPGO 超参数（EMA 基线 + 熵正则化）
        entropy_coef: float = 0.01,    # 熵正则化系数，防止策略过早收敛到单一选项
        lambda_param: float = 0.5,     # 保留字段，暂未使用
        rho: float = 0.9,              # y_bar（运行均值）的 EMA 衰减系数
        beta: float = 0.9,             # baseline（基线）的 EMA 衰减系数
    ):
        """初始化 P3O 优化器。

        Args:
            archetype:                    与之绑定的 Archetype 实例，必须提供
            lr:                           学习率（SGD 步长）
            momentum:                     动量系数（预留，当前未实现）
            weight_decay:                 L2 正则化系数
            reward_fn:                    自定义 fitness 函数 F(pred, target) -> reward；
                                          若为 None，默认使用 F = 1 - (y-t)^2
            auto_update_from_archetype:   True 时，step() 自动调用 update_from_archetype()
            fix_choices_after_step:       True 时，每次 step 后将 template 中各 Variable
                                          固定为当前概率最高的选项（贪心固定）
            reducer:                      多组 reward 的聚合方式（"mean" 等，暂未扩展）
            verbose:                      打印详细调试信息
            rewards_provider:             直接提供 reward 列表（优先于 targets_provider + reward_fn）
            targets_provider:             提供目标值列表（与 reward_fn 配合计算 reward）
            entropy_coef:                 熵正则项系数
            rho:                          y_bar EMA 衰减系数
            beta:                         baseline EMA 衰减系数
        """
        self.archetype = archetype
        # 从 archetype 中自动提取可学习参数列表（Variable logit tensors）
        self.param_groups = [
            {
                'params': list(getattr(self.archetype, 'parameters', lambda: [])()),
                'lr': lr,
                'momentum': momentum,
                'weight_decay': weight_decay,
            }
        ]
        self.state = {}
        # reward_fn 为 None 时使用默认 fitness：F(y, t) = 1 - (y-t)^2
        self.reward_fn = reward_fn
        self.auto_update_from_archetype = auto_update_from_archetype
        self.fix_choices_after_step = fix_choices_after_step
        self.reducer = reducer
        self.verbose = bool(verbose)
        self.rewards_provider = rewards_provider
        self.targets_provider = targets_provider
        # PSPGO 运行状态（EMA 基线）
        self.entropy_coef = float(entropy_coef)
        self.lambda_param = float(lambda_param)
        self.rho = float(rho)
        self.beta = float(beta)
        self._baseline: float = 0.0   # reward 的指数移动基线，用于计算 advantage
        self._y_bar: Optional[float] = None  # reward 的运行均值

    # ------------------------------------------------------------------
    # Reward 计算辅助
    # ------------------------------------------------------------------

    def compute_group_targets(self, group_keys: list[str]) -> list[float]:
        """通过 targets_provider 计算各分组的目标值。

        若未提供 targets_provider，抛出异常并提示改用 rewards_provider。
        """
        if self.targets_provider is None:
            raise ValueError("P3O: no targets_provider provided; use rewards_provider instead")
        targets = self.targets_provider(group_keys, self.archetype)
        return [float(v) for v in targets]

    def reinforce_step(self, group_preds: list[float], group_keys: list[str]) -> None:
        """计算分组 reward 并对参数梯度进行 REINFORCE 缩放（当前简化实现）。

        注意：此方法当前将所有梯度置零（mul_(0.0)），是尚未完全接入真实
        梯度流的占位实现。实际梯度更新在 update_from_archetype() 中完成。
        """
        if self.rewards_provider is not None:
            rewards = self.rewards_provider(group_keys, [float(p) for p in group_preds], self.archetype)
        else:
            targets = self.compute_group_targets(group_keys)
            if self.reward_fn is None:
                rewards = [1.0 - (float(y) - float(t)) ** 2 for y, t in zip(group_preds, targets)]
            else:
                rewards = [float(self.reward_fn(float(y), float(t))) for y, t in zip(group_preds, targets)]
        avg_reward = sum(rewards) / max(len(rewards), 1)
        if self.verbose:
            print(f"P3O: avg_reward={avg_reward:.4f} over {len(rewards)} groups")
        # 占位：当前不通过此路径更新参数；实际更新走 update_from_archetype()
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.mul_(0.0)

    # ------------------------------------------------------------------
    # 核心更新逻辑
    # ------------------------------------------------------------------

    def update_from_archetype(self) -> None:
        """从 archetype._behavior 读取上次采样结果，计算 REINFORCE loss 并反向传播。

        读取的状态字段（由 Behavior.sample() 填充）：
          last_group_keys    — 各分组标识符列表
          last_group_outputs — 各分组 LLM 输出的均值（预测值）
          last_slot_choices  — 各可学习 Variable 本次采样到的选项索引

        更新逻辑：
          1. 计算各分组 reward
          2. EMA 更新 baseline，计算 advantage = R - baseline
          3. 对每个可学习 Variable，计算：
             loss = -advantage * log_prob(sampled_idx) - entropy_coef * H
          4. loss.backward() → 梯度累积到 logit tensors
          （由后续 step() 中的 SGD 更新消费这些梯度）
        """
        beh = getattr(self.archetype, "_behavior", None)
        if beh is None:
            print("P3O: no behavior bound; call arch.broadcast(...); arch.sample() first")
            return
        keys = getattr(beh, "last_group_keys", None)
        preds = getattr(beh, "last_group_outputs", None)
        slot_choices = getattr(beh, "last_slot_choices", None)
        if not keys or not preds:
            print("P3O: no group info available; call arch.sample() after broadcast")
            return

        # Step 1：计算每组 reward
        if self.rewards_provider is not None:
            rewards_list = self.rewards_provider([str(k) for k in keys], [float(p) for p in preds], self.archetype)
        else:
            targets = self.compute_group_targets([str(k) for k in keys])
            if self.reward_fn is None:
                rewards_list = [1.0 - (float(p) - float(t)) ** 2 for p, t in zip(preds, targets)]
            else:
                rewards_list = [float(self.reward_fn(float(p), float(t))) for p, t in zip(preds, targets)]

        # Step 2：EMA 更新 baseline，计算 advantage
        R = sum(rewards_list) / max(1, len(rewards_list))
        if self._y_bar is None:
            self._y_bar = R
        # y_bar：reward 的慢速运行均值（用于未来可能的归一化）
        self._y_bar = self.rho * self._y_bar + (1.0 - self.rho) * R
        # baseline：reward 的快速 EMA，advantage = R - baseline 衡量本次表现超出基线多少
        self._baseline = self.beta * self._baseline + (1.0 - self.beta) * R
        advantage = R - self._baseline
        if self.verbose:
            print(f"P3O: reward={R:.4f}, adv={advantage:.4f} over {len(rewards_list)} groups")
            if isinstance(slot_choices, dict) and slot_choices:
                print(f"P3O: selected indices = {slot_choices}")

        if not slot_choices:
            if self.verbose:
                print("P3O: no slot choices present; ensure template has learnable Variables")
            return

        # Step 3：构建 REINFORCE loss 并反向传播
        loss = None
        template = getattr(self.archetype, "_prompt", None)
        if template is None:
            return

        fields = list(slot_choices.keys())
        num_groups = max(1, len(rewards_list))
        for field_name in fields:
            # 从 template 的 _variables 字典取出对应的 Variable 对象
            var = getattr(template, "_variables", {}).get(field_name)
            if var is None or not getattr(var, 'learnable', False):
                # 备选路径：通过 create_slots() 动态查找
                try:
                    var = template.create_slots().get(field_name)
                except Exception:
                    var = None
            if var is None or not getattr(var, 'learnable', False):
                continue

            # 取出该 Variable 当前 logit，计算 softmax 概率分布
            idx = slot_choices[field_name]
            logits = var.get_parameter(template)
            if logits is None:
                continue
            probs = torch.softmax(logits, dim=0)
            eps = 1e-8  # 数值稳定性：防止 log(0)
            # 采样选项的对数概率
            logp = torch.log(probs[idx].clamp_min(eps))
            # 策略熵（鼓励探索，防止过早收敛）
            entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
            # REINFORCE loss：advantage > 0 时降低 loss（增大采样概率）
            field_loss = -(advantage * logp) - (self.entropy_coef * entropy)
            loss = field_loss if loss is None else (loss + field_loss)

        if loss is not None:
            # 将梯度累积到 logit tensors，由 step() 中的 SGD 消费
            loss.backward()

    # ------------------------------------------------------------------
    # 标准优化器接口
    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        """清零所有可学习参数的梯度（每次 step 后调用）。"""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.zero_()

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """执行一次参数更新。

        流程：
          1. （可选）执行 closure 计算 loss
          2. 若 auto_update_from_archetype=True，自动调用 update_from_archetype()
             触发 REINFORCE loss 反向传播，填充 logit 梯度
          3. 对所有参数执行 SGD 更新：param -= lr * (grad + weight_decay * param)
          4. 若 fix_choices_after_step=True，将各 Variable 固定为贪心最优选项

        Args:
            closure: 可选的 loss 重计算函数（兼容 PyTorch 优化器接口）

        Returns:
            closure 返回的 loss 值，或 None
        """
        loss = None
        if closure is not None:
            loss = closure()

        # 自动从 archetype 拉取上次采样结果并计算梯度
        if self.auto_update_from_archetype and self.archetype is not None:
            try:
                self.update_from_archetype()
            except Exception as e:
                if self.verbose:
                    try:
                        import traceback as _tb
                        print(f"P3O: update_from_archetype failed: {e}")
                        _tb.print_exc()
                    finally:
                        raise
                else:
                    raise

        # SGD 更新：param -= lr * grad（含 weight decay）
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)
                param.data.add_(grad, alpha=-lr)

        # 可选：step 后将 template 中各 Variable 固定为当前概率最高的选项
        # 适用于训练完毕后的推理阶段，使 prompt 完全确定化
        if self.fix_choices_after_step and self.archetype is not None:
            try:
                template = getattr(self.archetype, "_prompt", None)
                if template is not None and hasattr(template, "create_slots"):
                    slots = template.create_slots()
                    best = {}
                    for name, var in slots.items():
                        if getattr(var, 'learnable', False):
                            best[name] = var.get_best_index(template)
                    if best:
                        template.set_optimized_slots(best)
            except Exception:
                pass  # 固定选项失败不应阻断优化流程

        return loss

    # ------------------------------------------------------------------
    # 状态序列化（兼容 PyTorch checkpoint 风格）
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """返回优化器状态字典，用于保存 checkpoint。"""
        return {
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """从 checkpoint 恢复优化器状态。"""
        self.state = state_dict['state']
        self.param_groups = state_dict['param_groups']
