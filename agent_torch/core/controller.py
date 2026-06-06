"""
AgentTorch 控制器模块
====================
Controller 是仿真主循环的"调度员"，不含任何业务逻辑，
只负责在正确的时机调用正确的 substep 函数。

它的三个核心方法对应 substep 三阶段：
  observe()  → 调用 SubstepObservation.forward()
  act()      → 调用 SubstepAction.forward()
  progress() → 调用 SubstepTransition.forward()，并把结果写回 state

数据流：
  config["substeps"]["0"]["observation"]["residents"]["get_features"]
       └─ 对应 observation_function["0"]["residents"]["get_features"]（nn.Module 实例）
       └─ Controller 按 substep 编号和 agent_type 查找并调用
"""

import asyncio
import torch.nn as nn
import re
from agent_torch.core.helpers import get_by_path, set_by_path, copy_module
from agent_torch.core.utils import is_async_method


class Controller(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.returns = []
        # 缓存每个 substep 的函数 key 列表，避免热路径中重复解析 config
        self._obs_keys_cache = {}
        self._policy_keys_cache = {}

    def observe(self, state, observation_function, agent_type):
        """第一阶段：为 agent_type 执行观察，返回 observation dict。

        流程：
          1. 从 state["current_substep"] 确定当前是第几个子步骤
          2. 从 config 查出该 substep 下该 agent_type 注册了哪些观察函数
          3. 逐一调用，合并结果
        """
        substep = state["current_substep"]
        obs_block = self.config["substeps"].get(substep, {}).get("observation")
        if not isinstance(obs_block, dict):
            return None

        # 缓存 key 列表，避免每步都重新解析 config
        if substep not in self._obs_keys_cache:
            self._obs_keys_cache[substep] = {}
        if agent_type not in self._obs_keys_cache[substep]:
            agent_map = obs_block.get(agent_type) or {}
            if not isinstance(agent_map, dict):
                self._obs_keys_cache[substep][agent_type] = []
            else:
                self._obs_keys_cache[substep][agent_type] = list(agent_map.keys())

        keys = self._obs_keys_cache[substep][agent_type]
        if not keys:
            return None

        result = {}
        funcs = observation_function[substep][agent_type]
        for obs_key in keys:
            # 调用 SubstepObservation.forward(state)
            result.update(funcs[obs_key](state))
        return result

    def act(self, state, observation, policy_function, agent_type):
        """第二阶段：为 agent_type 执行策略，返回 action dict。

        与 observe() 结构相同，区别是调用 forward(state, observation)。
        """
        substep = state["current_substep"]
        pol_block = self.config["substeps"].get(substep, {}).get("policy")
        if not isinstance(pol_block, dict):
            return None

        if substep not in self._policy_keys_cache:
            self._policy_keys_cache[substep] = {}
        if agent_type not in self._policy_keys_cache[substep]:
            agent_map = pol_block.get(agent_type) or {}
            if not isinstance(agent_map, dict):
                self._policy_keys_cache[substep][agent_type] = []
            else:
                self._policy_keys_cache[substep][agent_type] = list(agent_map.keys())

        keys = self._policy_keys_cache[substep][agent_type]
        if not keys:
            return None

        result = {}
        funcs = policy_function[substep][agent_type]
        for pol_key in keys:
            # 调用 SubstepAction.forward(state, observation)
            result.update(funcs[pol_key](state, observation))
        return result

    def progress(self, state, action, transition_function):
        """第三阶段：执行状态转移，返回新的 state。

        关键细节：
          1. 先 copy_module(state) 生成副本，保持原 state 不变（支持梯度回溯）
          2. 将 current_substep 递增（环形，到头回 0）
          3. 调用所有 transition 函数，并把返回值按 config 路径写回 next_state
        """
        # 复制一份 state（深拷贝张量），保证梯度图不被破坏
        next_state = copy_module(state)
        del state  # 主动释放，避免多份 state 共存占用内存

        substep = next_state["current_substep"]
        # substep 用字符串存储（nn.ModuleDict 的 key 必须是字符串）
        next_substep = (int(substep) + 1) % self.config["simulation_metadata"][
            "num_substeps_per_step"
        ]
        next_state["current_substep"] = str(next_substep)

        for trans_func in self.config["substeps"][substep]["transition"].keys():
            # 调用 SubstepTransition.forward(state=next_state, action=action)
            updated_vals = transition_function[substep][trans_func](
                state=next_state, action=action
            )
            # 把返回值写回 state 中对应的路径（路径在 YAML config 中定义）
            # 例如 "environment/vitality" → next_state["environment"]["vitality"] = value
            for var_name, value in updated_vals.items():
                source_path = self.config["substeps"][substep]["transition"][
                    trans_func
                ]["input_variables"][var_name]
                set_by_path(next_state, re.split("/", source_path), value)

        return next_state

    def progress_inplace(self, state, action, transition_function):
        """原地更新 state（不复制），适用于确认无梯度回溯需求的场景，节省内存。"""
        substep = state["current_substep"]
        next_substep = (int(substep) + 1) % self.config["simulation_metadata"][
            "num_substeps_per_step"
        ]
        state["current_substep"] = str(next_substep)

        for trans_func in self.config["substeps"][substep]["transition"].keys():
            updated_vals = transition_function[substep][trans_func](
                state=state, action=action
            )
            for var_name, value in updated_vals.items():
                source_path = self.config["substeps"][substep]["transition"][
                    trans_func
                ]["input_variables"][var_name]
                set_by_path(state, re.split("/", source_path), value)

        return state

    def learn_after_episode(self, episode_traj, initializer, optimizer):
        """示例：单集结束后的 RL 学习步骤（仅用于消费者市场示例模型）。"""
        optimizer.zero_grad()
        ret_episode_all = sum(
            [i[0]["agents"]["consumers"]["Q_exp"] for i in episode_traj["states"]]
        )
        ret_episode_0 = ret_episode_all[0]
        ret_episode = ret_episode_all.sum()
        self.returns.append(ret_episode)
        loss = -1e6 * ret_episode
        loss.backward()
        F_t_param = initializer.policy_function["0"]["consumers"][
            "purchase_product"
        ].learnable_args["F_t_params"]
        print(
            f"return is {ret_episode}, return for agent 0 is {ret_episode_0} and the F_t_param for agent 0 is {F_t_param[0]}"
        )
        optimizer.step()
