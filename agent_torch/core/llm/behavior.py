# behavior.py — LLM 驱动的 Agent 决策核心
#
# 职责：
#   1. 将整个人口（population）按特征分组，每组生成一条 Prompt
#   2. 并发调用一个或多个 Archetype（LLM 实例）获取数值化决策
#   3. 将各组输出广播回个体维度，返回 shape=(population_size, 1) 的行为张量
#
# 两种工作流：
#   A. Template 流（推荐）：使用 Template 对象分组，支持可学习 Variable（P3O 优化）
#   B. 基础 PromptManager 流：使用字符串模板 + 人口特征组合枚举分组
#
# 关键依赖：
#   Archetype      — LLM 调用单元，持有对话记忆（core/llm/archetype.py）
#   PromptManager  — 字符串模板渲染与分组（core/llm/prompt_manager.py）
#   LoadPopulation — 加载人口数据（core/dataloader.py）
#   P3O            — 从 last_slot_choices/last_group_outputs 读取梯度信号（optim/p3o.py）

from agent_torch.core.llm.prompt_manager import PromptManager
from agent_torch.core.dataloader import LoadPopulation
import torch


class Behavior:
    """LLM 行为采样器：将人口分组、调用 LLM、聚合输出为行为张量。

    Args:
        archetype:             Archetype 实例列表（支持多 archetype 集成）
        region:                人口区域标识（当 population=None 时用于加载）
        template:              Template 对象（None 则走基础 PromptManager 流）
        population:            预加载的人口对象，或可传入 region 字符串
        optimization_interval: P3O 优化间隔步数（暂未在此类内部使用，由调用方控制）
    """

    def __init__(self, archetype, region, template=None, population=None, optimization_interval: int = 3):
        self.archetype = archetype

        if population is None:
            # 未传入人口对象时，根据 region 从磁盘/数据库加载
            self.population = LoadPopulation(region)
        else:
            if hasattr(population, 'population_size'):
                # 已是预加载的人口对象，直接使用
                self.population = population
            else:
                # 传入的是区域标识符，转交给 LoadPopulation 加载
                self.population = LoadPopulation(population)

        self.template = template
        self._memory_initialized = False

        # P3O 优化所需的中间状态，由 sample() 在每次调用后填充：
        #   last_slot_choices  — 本次采样时各可学习 Variable 选择的选项索引
        #   last_slot_logp_sum — 所有可学习 slot 的 log 概率之和（用于 REINFORCE loss）
        #   last_slot_entropy_sum — 所有 slot 的熵之和（用于熵正则化）
        self.last_slot_choices = None

        if self.template is None:
            # 基础 PromptManager 流：用最后一个 archetype 的 user_prompt 初始化
            self.prompt_manager = PromptManager(self.archetype[-1].user_prompt, self.population)
            # 每个 archetype 的记忆槽数量 = distinct_groups（人口特征组合数）
            for arch in self.archetype:
                arch.initialize_memory(num_agents=self.prompt_manager.distinct_groups)
        else:
            # Template 流：先做一次分组预计算以确定组数，再初始化记忆
            prompts, _, _ = self.template.get_grouped_prompts(self.population, kwargs={})
            for arch in self.archetype:
                arch.initialize_memory(num_agents=len(prompts))

    # ------------------------------------------------------------------
    # 可供子类覆写的钩子（子类定制前后处理逻辑时使用）
    # ------------------------------------------------------------------

    def pre_sample_hook(self, kwargs):
        """采样前钩子，子类可覆写以注入自定义前处理逻辑。"""
        pass

    def post_sample_hook(self, sampled_behavior, kwargs):
        """采样后钩子，子类可覆写以注入自定义后处理逻辑（如平滑、截断等）。"""
        pass

    # ------------------------------------------------------------------
    # 主采样入口
    # ------------------------------------------------------------------

    def sample(self, kwargs=None):
        """驱动一次完整的 LLM 决策采样，返回 (population_size, 1) 行为张量。

        流程（Template 流）：
          1. 对可学习 Variable 采样选项索引，记录 logp/entropy 供 P3O 使用
          2. 按 Template 分组生成 prompt_list
          3. 并发调用所有 archetype，收集 float 输出
          4. 多 archetype 输出取均值，广播到个体维度

        流程（基础 PromptManager 流）：
          1. 枚举人口特征组合生成 prompt_list
          2. 并发调用所有 archetype
          3. 用 mask 将每组输出叠加到对应个体
        """
        verbose = bool(kwargs.get("verbose", False)) if kwargs else False
        if verbose:
            print("Behavior: Decision")
        self.pre_sample_hook(kwargs)

        device = kwargs["device"]
        # 初始化全零行为张量，shape=(population_size, 1)
        sampled_behavior = torch.zeros(self.population.population_size, 1, device=device)

        # ============================================================
        # 分支 A：Template 流
        # ============================================================
        if self.template is not None:
            # Step 1：对所有可学习 Variable 采样一次展示选项
            slots = self.template.create_slots()
            sampled_choices = {}   # 字段名 -> 采样到的选项索引
            logp_sum = None
            entropy_sum = None
            for name, var in slots.items():
                if getattr(var, 'learnable', False):
                    idx, logp, entropy = var.sample_index(self.template)
                    sampled_choices[name] = int(idx)
                    try:
                        # 累加各字段的 logp 和 entropy，用于 P3O loss 计算
                        logp_sum = logp if logp_sum is None else (logp_sum + logp)
                        entropy_sum = entropy if entropy_sum is None else (entropy_sum + entropy)
                    except Exception:
                        pass

            if sampled_choices:
                # 将采样结果写入 template，后续 get_grouped_prompts 会使用这些固定选项
                self.template.set_optimized_slots(sampled_choices)
                # 保存供 P3O optimizer 在 step() 中读取
                self.last_slot_choices = sampled_choices
                self.last_slot_logp_sum = logp_sum
                self.last_slot_entropy_sum = entropy_sum

            # Step 2：按 Template 分组生成 prompt 列表
            prompt_list, group_keys, group_indices = self.template.get_grouped_prompts(self.population, kwargs or {})
            # 保存供 P3O optimizer 诊断 / reward 计算
            self.last_prompt_list = prompt_list
            self.last_group_indices = group_indices

            if verbose:
                print(f"\n=== Population Broadcast LLM Calls ===")
                print(f"Number of unique prompts: {len(prompt_list)}")
                print(f"Number of archetypes: {self.archetype[-1].n_arch}")
                for i, prompt in enumerate(prompt_list):
                    print(f"\nPrompt {i+1}:\n{prompt}")

            self.last_group_keys = group_keys

            # Step 3：并发调用所有 archetype，每个 archetype 返回 len(prompt_list) 个输出
            agent_outputs = []
            for n_arch in range(self.archetype[-1].n_arch):
                outputs = self.archetype[n_arch](prompt_list, last_k=12)
                agent_outputs.append(outputs)

            # Step 4：将各 archetype 输出叠加到对应分组个体上
            group_values_accum = [0.0 for _ in range(len(prompt_list))]
            for arch_outputs in agent_outputs:
                for en, output_value in enumerate(arch_outputs):
                    try:
                        # 兼容字符串输出和字典输出（{"text": "0.7"}）两种格式
                        text_value = output_value["text"] if isinstance(output_value, dict) and "text" in output_value else output_value
                        value_for_group = float(text_value)
                        if torch.isnan(torch.tensor(value_for_group, device=device)):
                            value_for_group = 0.0
                    except Exception:
                        value_for_group = 0.0
                    group_values_accum[en] += value_for_group
                    # 将该组的数值广播给组内所有 agent（group_indices[en] 为个体下标列表）
                    idx = torch.tensor(group_indices[en], dtype=torch.long, device=device)
                    sampled_behavior[idx, 0] = sampled_behavior[idx, 0] + value_for_group

            # 多 archetype 取均值
            n = len(agent_outputs) if agent_outputs else 1
            sampled_behavior = sampled_behavior / max(n, 1)
            # 保存各组均值输出，供 P3O 计算 reward 时使用
            self.last_group_outputs = [v / max(n, 1) for v in group_values_accum]

            # 打印采样摘要（无论 verbose 设置如何，始终输出一行元信息）
            try:
                mean_val = float(sampled_behavior.mean().item())
            except Exception:
                mean_val = float('nan')
            print(f"Population sample complete: outputs shape={tuple(sampled_behavior.shape)}, mean={mean_val:.4f}")

            if verbose:
                print(f"=== End Population LLM Calls ===\n")

            # 将本次调用历史写入磁盘（供后续审计或调试）
            self.archetype[-1].export_memory_to_file(file_dir=kwargs["current_memory_dir"], last_k=len(prompt_list))
            self.post_sample_hook(sampled_behavior, kwargs)
            return sampled_behavior

        # ============================================================
        # 分支 B：基础 PromptManager 流
        # ============================================================
        # 枚举人口特征组合，为每种组合生成一条 prompt
        prompt_list = self.prompt_manager.get_prompt_list(kwargs=kwargs)
        self.last_prompt_list = prompt_list

        if verbose:
            print(f"\n=== Population Broadcast LLM Calls (base) ===")
            print(f"Number of prompts: {len(prompt_list)}")
            print(f"Number of archetypes: {self.archetype[-1].n_arch}")
            for i, prompt in enumerate(prompt_list):
                print(f"\nPrompt {i+1}:\n{prompt}")

        # 为每个分组构建 boolean mask，用于将组输出叠加到对应个体
        masks = self.get_masks_for_each_group(self.prompt_manager.dict_variables_with_values, kwargs)

        # 并发调用所有 archetype
        agent_outputs = []
        for n_arch in range(self.archetype[-1].n_arch):
            agent_outputs.append(self.archetype[n_arch](prompt_list, last_k=12))

        # 用 mask 将各组输出叠加，多 archetype 取均值
        sampled_behavior = self.get_sampled_behavior(sampled_behavior, masks, agent_outputs)

        try:
            mean_val = float(sampled_behavior.mean().item())
        except Exception:
            mean_val = float('nan')
        print(f"Population sample complete: outputs shape={tuple(sampled_behavior.shape)}, mean={mean_val:.4f}")

        if verbose:
            print(f"=== End Population LLM Calls ===\n")

        self.archetype[-1].export_memory_to_file(file_dir=kwargs["current_memory_dir"], last_k=len(prompt_list))
        self.post_sample_hook(sampled_behavior, kwargs)
        return sampled_behavior

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def get_sampled_behavior(self, sampled_behavior, masks, agent_outputs):
        """将多个 archetype 的组级输出通过 mask 叠加到个体维度，并取均值。

        Args:
            sampled_behavior: 初始全零行为张量 (population_size, 1)
            masks:            每个分组对应的 float mask 列表
            agent_outputs:    多个 archetype 各自的输出列表（外层=archetype，内层=group）

        Returns:
            shape=(population_size, 1) 的行为张量，各 archetype 输出已取均值
        """
        for agent_output in agent_outputs:
            for en, output_value in enumerate(agent_output):
                try:
                    # 兼容字符串和字典两种输出格式
                    if isinstance(output_value, dict) and "text" in output_value:
                        text_value = output_value["text"]
                    else:
                        text_value = output_value
                    value_for_group = float(text_value)
                    if torch.isnan(torch.tensor(value_for_group)):
                        value_for_group = 0.0
                except Exception:
                    value_for_group = 0.0
                # mask 为 (population_size, 1)，仅属于该分组的 agent 位置为 1
                sampled_behavior_for_group = masks[en] * value_for_group
                sampled_behavior = torch.add(sampled_behavior, sampled_behavior_for_group)

        n = len(agent_outputs)
        average_sampled_behavior = sampled_behavior / n
        return average_sampled_behavior

    def get_masks_for_each_group(self, variables, kwargs=None):
        """为 PromptManager 枚举出的每个特征组合构建 float mask。

        逻辑：遍历 prompt_manager 中所有组合（如 age=30, income=high），
        对每个组合在人口中找出满足条件的个体，返回对应的 float mask 列表。

        特殊处理：age==0 视为无效数据，直接返回全零 mask（跳过该分组）。

        Returns:
            masks: 长度等于组合数的列表，每项 shape=(population_size, 1)
        """
        masks = []
        for en, target_values in enumerate(
            self.prompt_manager.combinations_of_prompt_variables_with_index
        ):
            device = kwargs['device']
            mask = torch.tensor([True] * self.population.population_size, device=device)
            for key, value in target_values.items():
                if key in variables:
                    if key == "age" and value == 0:
                        # age=0 为脏数据标志，整组置零
                        mask = torch.zeros_like(mask)
                    else:
                        comparison = variables[key] == value
                        # 兼容 CuPy 数组：统一转换为 PyTorch tensor
                        if not isinstance(comparison, torch.Tensor):
                            comparison = torch.tensor(comparison, device=device)
                        mask = torch.logical_and(mask, comparison)
            mask = mask.unsqueeze(1)     # (population_size,) -> (population_size, 1)
            float_mask = mask.float()
            masks.append(float_mask)

        return masks
