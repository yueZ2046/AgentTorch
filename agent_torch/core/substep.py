"""
AgentTorch 子步骤基类模块
========================
这是整个框架最核心的抽象层。每个仿真"步骤"由若干"子步骤"(substep)组成，
每个子步骤严格遵循 观察(Observe) → 行动(Act) → 转移(Transition) 三阶段流水线。

你写的所有业务逻辑，都是通过继承这里的基类来实现的：
  - SubstepObservation  ：智能体"看"世界（从 state 中提取信息）
  - SubstepAction       ：智能体"决策"（根据观测生成动作）
  - SubstepTransition   ：世界"更新"（用动作修改 state）

调用链（由 Runner 驱动）：
  Runner.step()
    └─ Controller.observe()  → SubstepObservation.forward(state)
    └─ Controller.act()      → SubstepAction.forward(state, observation)
    └─ Controller.progress() → SubstepTransition.forward(state, action)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from abc import ABC, abstractmethod

from agent_torch.core.helpers.general import *

# 方便用户直接从 substep 导入 vmap（向量化装饰器）
from agent_torch.core.helpers.vmap import vmap, sample_grid


class SubstepObservation(nn.Module, ABC):
    """子步骤第一阶段：观察。

    智能体从当前 state 中"看"到它需要的信息，打包成 observation dict 返回。
    子类必须实现 forward(self, state) → dict。

    典型用途：
        - 读取邻居的疾病状态（流行病模型）
        - 读取当前街坊的特征向量（城市活力模型）
    """

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        # input_variables：从 YAML config 中解析的输入变量路径字典，
        # 形如 {"block_features": "environment/block_features"}
        self.input_variables = input_variables
        self.output_variables = output_variables

        # 将参数分为"可学习"和"固定"两类
        self.learnable_args, self.fixed_args = (
            arguments["learnable"],
            arguments["fixed"],
        )
        # 可学习参数注册为 nn.ParameterDict，使 PyTorch 能跟踪梯度
        if self.learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

        # calibration 模式：为每个可学习参数额外挂一个 calibrate_<name> 属性，
        # 供外部校准器（如 P3O）直接读写
        if self.config["simulation_metadata"]["calibration"] == True:
            for key, value in self.learnable_args.items():
                tensor_name = f"calibrate_{key}"
                setattr(self, tensor_name, torch.tensor(value, requires_grad=True))

        self.args = {**self.fixed_args, **self.learnable_args}
        self.custom_observation_network = None

    @abstractmethod
    def forward(self, state):
        """子类必须实现。输入当前仿真状态，输出观测字典。"""
        pass


class SubstepAction(nn.Module, ABC):
    """子步骤第二阶段：行动（策略/决策）。

    智能体根据观测决定下一步的行动。返回的 action dict 会传给 Transition。
    子类必须实现 forward(self, state, observation) → dict。

    典型用途：
        - 根据感染概率决定是否接触（流行病模型）
        - 根据街坊特征预测留家概率和吸引力（城市活力模型，即 MovePolicy）
    """

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables

        self.learnable_args, self.fixed_args = (
            arguments["learnable"],
            arguments["fixed"],
        )
        if self.learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

        if self.config["simulation_metadata"]["calibration"] == True:
            for key, value in self.learnable_args.items():
                tensor_name = f"calibrate_{key}"
                setattr(self, tensor_name, torch.tensor(value, requires_grad=True))

        self.args = {**self.fixed_args, **self.learnable_args}
        self.custom_action_network = None

    @abstractmethod
    def forward(self, state, observation):
        """子类必须实现。输入状态+观测，输出动作字典。"""
        pass


class SubstepTransition(nn.Module, ABC):
    """子步骤第三阶段：状态转移。

    用智能体的动作更新仿真 state，生成下一时刻的世界状态。
    子类必须实现 forward(self, state, action) → dict。

    返回的字典的 key 必须与 YAML config 中 transition.input_variables 对应，
    框架会自动将返回值写回 state 的对应路径。

    典型用途：
        - 根据接触动作更新疾病状态（流行病模型）
        - 根据移动动作更新街坊活力值（城市活力模型，即 AggregateVitality）
    """

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables

        self.learnable_args, self.fixed_args = (
            arguments["learnable"],
            arguments["fixed"],
        )

        if self.learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

        if self.config["simulation_metadata"]["calibration"] == True:
            for key, value in self.learnable_args.items():
                tensor_name = f"calibrate_{key}"
                setattr(self, tensor_name, torch.tensor(value, requires_grad=True))

        self.args = {**self.fixed_args, **self.learnable_args}
        self.custom_transition_network = None

    @abstractmethod
    def forward(self, state, action):
        """子类必须实现。输入状态+动作，返回更新后的变量字典。"""
        pass


class SubstepTransitionMessagePassing(MessagePassing, ABC):
    """基于图消息传递的状态转移（适用于网络/图结构仿真）。

    继承自 PyTorch Geometric 的 MessagePassing，aggr="add" 表示邻居消息求和聚合。
    适合需要沿图结构传播信息的场景，如疾病在社交网络上的扩散。
    """

    def __init__(self, config, input_variables, output_variables, arguments):
        super(SubstepTransitionMessagePassing, self).__init__(aggr="add")
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables

        self.learnable_args, self.fixed_args = (
            arguments["learnable"],
            arguments["fixed"],
        )
        if self.learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

        if self.config["simulation_metadata"]["calibration"] == True:
            for key, value in self.learnable_args.items():
                tensor_name = f"calibrate_{key}"
                setattr(self, tensor_name, torch.tensor(value, requires_grad=True))

    @abstractmethod
    def forward(self, state, action):
        pass
