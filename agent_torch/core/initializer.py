"""
AgentTorch 初始化器模块
=======================
Initializer 负责两件事：
  1. 从 YAML config（或 Python 字典）构建初始 state（环境/智能体/对象/网络四个域）
  2. 从 Registry 查找并实例化所有 substep 的 Observation/Policy/Transition 模块

完成后 Runner 就可以直接使用：
  - self.state                          — 仿真初始状态字典
  - self.observation_function[substep]  — 各 substep 的观察模块
  - self.policy_function[substep]       — 各 substep 的策略模块
  - self.transition_function[substep]   — 各 substep 的转移模块

state 的顶层结构（由 config["state"] 决定）：
  state = {
      "current_step":     0,
      "current_substep":  "0",          # 字符串，因为 nn.ModuleDict 的 key 必须是字符串
      "environment":  { "vitality": Tensor, ... },
      "agents":       { "residents": { "home_block": Tensor, ... } },
      "objects":      { ... },
      "network":      { ... },
      "parameters":   nn.ParameterDict  # 所有可学习参数
  }

设备处理策略：
  - config["simulation_metadata"]["device"] 支持 "auto" / "cpu" / "cuda"
  - "auto" 在有 GPU 时自动解析为 "cuda"，否则为 "cpu"
  - CUDA 模式下使用多流异步传输加速初始化（见 _to_device_streamed）
"""

import torch
import torch.nn as nn
import os

from agent_torch.core.helpers.general import *


class Initializer(nn.Module):
    """从 config + registry 构建初始仿真状态和所有 substep 模块。

    继承 nn.Module 是为了让 learnable_parameters 能被 PyTorch 的参数追踪机制管理，
    使优化器可以通过 runner.parameters() 找到所有可训练参数。
    """

    def __init__(self, config, registry):
        """构造函数：解析设备配置，创建 CUDA 流池，初始化所有容器。

        Args:
            config:   仿真配置字典（来自 YAML 或 build_config()）
            registry: Registry 实例，持有所有已注册的类和函数

        副作用：
            - 将 config["simulation_metadata"]["device"] 从 "auto" 解析并写回为
              "cuda" 或 "cpu"，供下游代码直接读取
            - CUDA 模式下创建 4 条（默认）异步传输流
        """
        super().__init__()
        self.config = config
        self.registry = registry

        # 解析设备：支持 "auto"（有 GPU 用 GPU）/ "cpu" / "cuda"
        cfg_dev = str(self.config["simulation_metadata"].get("device", "auto")).lower()
        self.device = torch.device(
            "cuda" if (cfg_dev == "auto" and torch.cuda.is_available())
            else (cfg_dev if cfg_dev != "auto" else "cpu")
        )
        self.is_cuda = (self.device.type == 'cuda')

        # 把解析后的实际设备名写回 config，下游无需再判断 "auto"
        self.config["simulation_metadata"]["device"] = self.device.type

        # CUDA 模式：创建多条异步流，用于初始化时并行搬运张量（见 _next_stream）
        if self.is_cuda:
            try:
                self._num_streams = int(os.getenv('AGENT_TORCH_INIT_NUM_STREAMS', '4'))
            except Exception:
                self._num_streams = 4
            self._streams = [
                torch.cuda.Stream(device=self.device)
                for _ in range(max(1, self._num_streams))
            ]
            self._stream_rr = 0   # round-robin 轮询指针，每次取流后 +1
        else:
            self._streams = []
            self._stream_rr = 0

        # 状态容器（由 simulator() 填充）
        self.state = {}
        self.environment, self.agents, self.objects, self.networks = {}, {}, {}, {}

        # 可学习 / 固定参数注册表（由 _initialize_property 分类填充）
        self.fixed_parameters, self.learnable_parameters = {}, {}

        # substep 模块容器（由 substeps() 填充）
        # 使用 nn.ModuleDict 而非普通 dict，确保 PyTorch 能追踪其中的参数
        (
            self.observation_function,
            self.policy_function,
            self.transition_function,
            self.reward_function,
        ) = (nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict())

    # ─────────────────────── 设备传输工具 ────────────────────────────────────

    def _next_stream(self):
        """轮询返回下一条 CUDA 异步流（仅 CUDA 模式有效）。

        使用 round-robin 策略在 N 条流之间循环，让多个张量传输任务
        分散到不同流上并行执行，而不是在同一条流上排队。
        CPU 模式下返回当前默认流（调用方不会真正使用）。
        """
        if not self.is_cuda or not self._streams:
            return torch.cuda.current_stream(self.device) if self.is_cuda else None
        s = self._streams[self._stream_rr]
        self._stream_rr = (self._stream_rr + 1) % len(self._streams)
        return s

    def _to_device_streamed(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """把 CPU 张量异步搬到目标设备，CUDA 模式下使用锁页内存 + 多流并行。

        三层优化叠加：
          1. pin_memory()：把张量锁定在物理内存，GPU DMA 可直接读取，无需 CPU 中转
          2. non_blocking=True：发出传输指令后立即返回，传输在 GPU 流上后台进行
          3. _next_stream()：不同张量分配到不同流，传输真正并行而非顺序排队

        注意：non_blocking=True 必须配合 pin_memory() 才有效；
              若张量内存不连续（非 C-contiguous），先调 contiguous() 整理。
        """
        if not self.is_cuda:
            return cpu_tensor.to(self.device)   # CPU 模式：同步传输，无需额外处理

        stream = self._next_stream()

        # 保证内存连续，否则 pin_memory 可能失败
        if not cpu_tensor.is_contiguous():
            cpu_tensor = cpu_tensor.contiguous()

        # 锁页内存：让 GPU DMA 绕过 CPU，直接从物理内存读取数据
        if hasattr(cpu_tensor, 'is_pinned') and not cpu_tensor.is_pinned():
            try:
                cpu_tensor = cpu_tensor.pin_memory()
            except Exception:
                pass   # 锁页失败则退化为普通传输，不中断初始化

        with torch.cuda.stream(stream):
            gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)  # 异步传输
        return gpu_tensor

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """统一的设备传输入口：CUDA 走异步多流路径，CPU 走同步路径。

        代码中所有需要搬运张量的地方都应调用此方法，而不是直接调用 .to(device)，
        以保证 CUDA 模式下始终享受多流并行优化。
        """
        return self._to_device_streamed(tensor) if self.is_cuda else tensor.to(self.device)

    # ─────────────────────── 属性初始化工具 ──────────────────────────────────

    def _initialize_from_default(self, src_val, shape):
        """从标量、列表或数值直接构造初始张量，不调用任何外部生成器。

        处理三种输入类型：
          - str：原样返回（用于文件路径等字符串属性）
          - list：转为 Tensor，若 shape 与列表形状不符，则广播填充到目标形状
          - 数值（int/float）：用该值填充目标 shape 的全量张量

        构造完毕后调用 _to_device 搬到目标设备。
        """
        processed_shape = shape
        if type(src_val) == str:
            return src_val   # 字符串属性（如文件路径）不做任何处理

        if type(src_val) == list:
            src_tensor = torch.tensor(src_val)

            # 如果指定了 shape 且与列表形状不同，需要广播
            if processed_shape and list(src_tensor.shape) != (
                [processed_shape] if isinstance(processed_shape, int) else processed_shape
            ):
                init_value = torch.zeros(
                    size=[processed_shape] if isinstance(processed_shape, int) else processed_shape
                )
                if (
                    len(src_tensor.shape) == 1
                    and len([processed_shape] if isinstance(processed_shape, int) else processed_shape) >= 2
                ):
                    if src_tensor.shape[0] == processed_shape[-1]:
                        # 源列表长度匹配最后一维，沿其他维广播
                        init_value = src_tensor.unsqueeze(0).expand(processed_shape)
                    else:
                        # 无法对齐，用第一个值填满
                        init_value.fill_(src_tensor[0].item())
                else:
                    init_value.fill_(src_tensor.flatten()[0].item())
            else:
                init_value = src_tensor   # 形状已匹配，直接使用
        else:
            # 数值类型：生成全量张量并填充该值
            init_value = src_val * torch.ones(size=processed_shape)

        init_value = self._to_device(init_value)
        return init_value

    def _initialize_from_generator(self, initializer_object, initialize_shape, name_root):
        """调用 Registry 中注册的初始化函数生成属性张量。

        适用于无法用常量表达的属性，例如"从 CSV 文件读取坐标"、
        "用随机分布生成初始能量"等。

        流程：
          1. 解析生成器函数名（initializer_object["generator"]）
          2. 解析并初始化生成器所需的各个参数（可能递归调用 _initialize_from_default）
          3. 调用 registry.initialization_helpers[function](shape, params) 得到张量
          4. 把结果搬到目标设备

        Args:
            initializer_object: config 中描述生成器的字典，包含 generator / arguments 字段
            initialize_shape:   目标张量形状
            name_root:          参数的命名前缀，用于唯一标识可学习参数
        """
        function = initializer_object["generator"]

        params = {}
        for argument in initializer_object["arguments"].keys():
            arg_object = initializer_object["arguments"][argument]

            arg_name = f"{name_root}_{argument}"
            arg_learnable, arg_shape = arg_object["learnable"], arg_object["shape"]
            arg_init_func = arg_object["initialization_function"]

            if arg_init_func is None:
                arg_value = self._initialize_from_default(arg_object["value"], arg_shape)
            else:
                raise NotImplementedError("Dynamic argument initialization is not supported yet.")

            params[argument] = arg_value

            # 按是否可学习分类注册
            if arg_learnable:
                self.learnable_parameters[arg_name] = arg_value
            else:
                self.fixed_parameters[arg_name] = arg_value

        # 调用 Registry 里注册的生成器函数
        init_value = self.registry.initialization_helpers[function](initialize_shape, params)
        init_value = self._to_device(init_value)
        return init_value

    def _initialize_property(self, property_object, property_key):
        """初始化单个状态属性，根据 config 决定走默认值路径还是生成器路径。

        读取 config 中某个属性的完整描述（name/shape/dtype/learnable/
        initialization_function/value），选择对应的初始化方式，
        并把结果按 learnable 标志分入 learnable_parameters 或 fixed_parameters。

        Returns:
            (property_value, property_is_learnable)
        """
        property_name = property_object["name"]
        property_shape, property_dtype = (
            property_object["shape"],
            property_object["dtype"],
        )
        property_is_learnable = property_object["learnable"]
        property_initializer = property_object["initialization_function"]

        if property_initializer is None:
            # 用常量/列表直接初始化
            property_value = self._initialize_from_default(
                property_object["value"], property_shape
            )
        else:
            # 调用 Registry 注册的生成器函数初始化
            property_value = self._initialize_from_generator(
                property_initializer, property_shape, property_key
            )

        if property_is_learnable:
            self.learnable_parameters[property_key] = property_value
        else:
            self.fixed_parameters[property_key] = property_value

        return property_value, property_is_learnable

    # ─────────────────────── 四域初始化 ──────────────────────────────────────

    def init_environment(self, key="environment"):
        """初始化环境域：遍历 config["state"]["environment"] 下的所有属性并构建张量。

        环境域存储全局状态（不属于任何具体 Agent），例如：
          - 街坊特征矩阵 block_features
          - 活力预测值 predicted_vitality
          - 空间图 edge_index
        结果写入 self.environment，最终由 initialize() 挂到 state["environment"]。
        """
        if self.config["state"][key] is None:
            return

        for prop in self.config["state"][key].keys():
            property_object = self.config["state"][key][prop]
            property_value, property_is_learnable = self._initialize_property(
                property_object, property_key=f"{key}_{prop}"
            )
            self.environment[prop] = property_value

    def init_agents(self, key="agents"):
        """初始化智能体域：遍历所有 Agent 类型及其属性并构建张量。

        Agent 按类型分组（如 "residents"、"predator"），每种类型下有多个属性
        （如 home_block、demo_group、weight）。
        结果写入 self.agents，最终挂到 state["agents"]。

        跳过 "metadata" key（它不是 Agent 类型，而是类型级别的描述信息）。
        """
        if self.config["state"][key] is None:
            return

        for instance_type in self.config["state"][key].keys():
            if instance_type == "metadata":
                continue

            self.agents[instance_type] = {}
            instance_properties = self.config["state"][key][instance_type]["properties"]
            if instance_properties is None:
                continue

            for prop in instance_properties.keys():
                property_object = instance_properties[prop]
                property_value, property_is_learnable = self._initialize_property(
                    property_object, property_key=f"{key}_{instance_type}_{prop}"
                )
                self.agents[instance_type][prop] = property_value

    def init_objects(self, key="objects"):
        """初始化对象域：与 init_agents 结构完全相同，但针对静态对象（Object）。

        Object 与 Agent 的区别：Object 不主动决策，只作为环境中的被动实体存在，
        例如"草地"（捕食者-猎物模型）或"建筑物"。
        结果写入 self.objects，最终挂到 state["objects"]。
        """
        if self.config["state"][key] is None:
            return

        for instance_type in self.config["state"][key].keys():
            if instance_type == "metadata":
                continue

            self.objects[instance_type] = {}
            instance_properties = self.config["state"][key][instance_type]["properties"]
            if instance_properties is None:
                continue

            for prop in instance_properties.keys():
                property_object = instance_properties[prop]
                property_value, property_is_learnable = self._initialize_property(
                    property_object, property_key=f"{key}_{instance_type}_{prop}"
                )
                self.objects[instance_type][prop] = property_value

    def init_network(self, key="network"):
        """初始化网络域：从 Registry 调用图构建函数，生成边列表和邻接矩阵。

        网络域描述 Agent / Object 之间的交互结构，例如：
          - 社交接触网络（流行病模型）
          - 空间邻接图（城市活力模型）

        每个网络由 type 字段指定构建函数名，在 Registry 的 network_helpers 中查找。
        构建结果（graph 对象 + adjacency_matrix）存入 self.networks，
        最终挂到 state["network"]。

        adjacency_matrix 格式：(edge_list, attr_list) 二元组，均搬到目标设备。
        """
        if self.config["state"][key] is None:
            return

        for interaction_type in self.config["state"][key].keys():
            self.networks[interaction_type] = {}

            if self.config["state"][key][interaction_type] is None:
                continue

            for contact_network in self.config["state"][key][interaction_type].keys():
                self.networks[interaction_type][contact_network] = {}

                network_type = self.config["state"][key][interaction_type][
                    contact_network
                ]["type"]
                params = self.config["state"][key][interaction_type][contact_network][
                    "arguments"
                ]

                # 调用 Registry 中注册的图构建函数
                graph, adjacency_matrix = self.registry.network_helpers[network_type](params)

                if len(adjacency_matrix) == 2:
                    edge_list, attr_list = adjacency_matrix
                    edge_list = self._to_device(edge_list)
                    attr_list = self._to_device(attr_list)
                    adjacency_matrix = (edge_list, attr_list)

                self.networks[interaction_type][contact_network]["graph"] = graph
                self.networks[interaction_type][contact_network][
                    "adjacency_matrix"
                ] = adjacency_matrix

    def simulator(self):
        """依次调用四域初始化函数，完成仿真状态的数据层构建。

        调用顺序：environment → agents → objects → network
        调用完成后，self.environment / agents / objects / networks 均已填充，
        且所有可学习参数已收集进 self.learnable_parameters。

        最后把 learnable_parameters 包装成 nn.ParameterDict，
        让 PyTorch 将其纳入参数追踪（optimizer.step() 才能更新这些参数）。
        """
        self.init_environment()
        self.init_agents(key="agents")
        self.init_objects(key="objects")
        self.init_network()

        # nn.ParameterDict 让优化器能通过 runner.parameters() 找到所有可学习参数
        self.parameters_dict = nn.ParameterDict(self.learnable_parameters)

    # ─────────────────────── Substep 模块实例化 ──────────────────────────────

    def _parse_function(self, function_object, name_root):
        """解析单个 substep 函数的配置，返回实例化该函数所需的三元组。

        从 config 中读取 generator（类名）/ input_variables / output_variables /
        arguments，把 arguments 按 learnable 标志分为两组并初始化为张量。

        Returns:
            (input_variables, output_variables, arguments)
            其中 arguments = {"learnable": {...}, "fixed": {...}}

        这个方法是 substeps() 的核心工具，被 observation / policy / transition
        三种函数的实例化循环共同调用。
        """
        generator = function_object["generator"]
        input_variables = function_object["input_variables"]
        output_variables = function_object["output_variables"]

        arguments = function_object["arguments"]
        learnable_args, fixed_args = {}, {}

        if arguments is not None:
            for argument in arguments:
                arg_name = f"{name_root}_{argument}"
                arg_object = arguments[argument]
                arg_function = arg_object["initialization_function"]
                arg_learnable = arg_object["learnable"]
                arg_shape = arg_object["shape"]

                if arg_function is None:
                    arg_value = self._initialize_from_default(
                        arg_object["value"], arg_shape
                    )
                else:
                    arg_value = self._initialize_from_generator(
                        arg_function, arg_shape, name_root=arg_name
                    )

                if arg_learnable:
                    self.learnable_parameters[arg_name] = arg_value
                    learnable_args[argument] = arg_value
                else:
                    self.fixed_parameters[arg_name] = arg_value
                    fixed_args[argument] = arg_value

        arguments = {"learnable": learnable_args, "fixed": fixed_args}
        return input_variables, output_variables, arguments

    def substeps(self):
        """从 Registry 查找并实例化所有 substep 的 Observation / Policy / Transition 模块。

        遍历 config["substeps"] 下的每个 substep（以字符串编号为 key，如 "0"、"1"），
        对每种 active_agent 类型：
          1. 解析 observation 配置 → 实例化 SubstepObservation 子类
          2. 解析 policy 配置     → 实例化 SubstepAction 子类
          3. 解析 transition 配置 → 实例化 SubstepTransition 子类

        实例化时通过 Registry 按字符串名查找类（如 "MovePolicy" → MovePolicy 类），
        再用 _parse_function 解析出的三元组调用类构造函数。

        结果存入三个 nn.ModuleDict（observation_function / policy_function /
        transition_function），层级结构为 [substep][agent_type][func_name]。
        """
        for substep in self.config["substeps"].keys():
            active_agents = self.config["substeps"][substep]["active_agents"]

            (
                self.observation_function[substep],
                self.policy_function[substep],
                self.transition_function[substep],
            ) = (nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict())

            for agent_type in active_agents:
                # ── Observation 模块 ────────────────────────────────────────
                agent_observations = self.config["substeps"][substep]["observation"][
                    agent_type
                ]
                self.observation_function[substep][agent_type] = nn.ModuleDict()
                if agent_observations is not None:
                    for obs_func in agent_observations:
                        input_variables, output_variables, arguments = (
                            self._parse_function(
                                agent_observations[obs_func],
                                name_root=f"{agent_type}_observation_{obs_func}",
                            )
                        )
                        # registry.observation_helpers["GetFeatures"] → GetFeatures 类
                        self.observation_function[substep][agent_type][
                            obs_func
                        ] = self.registry.observation_helpers[obs_func](
                            self.config, input_variables, output_variables, arguments,
                        )

                # ── Policy 模块 ─────────────────────────────────────────────
                agent_policies = self.config["substeps"][substep]["policy"][agent_type]
                self.policy_function[substep][agent_type] = nn.ModuleDict()

                if agent_policies is not None:
                    for policy_func in agent_policies:
                        input_variables, output_variables, arguments = (
                            self._parse_function(
                                agent_policies[policy_func],
                                name_root=f"{agent_type}_policy_{policy_func}",
                            )
                        )
                        # registry.policy_helpers["MovePolicy"] → MovePolicy 类
                        self.policy_function[substep][agent_type][
                            policy_func
                        ] = self.registry.policy_helpers[policy_func](
                            self.config, input_variables, output_variables, arguments,
                        )

            # ── Transition 模块（不按 agent_type 分组，直接挂在 substep 下）───
            substep_transitions = self.config["substeps"][substep]["transition"]
            self.transition_function[substep] = nn.ModuleDict()

            for transition_func in substep_transitions:
                input_variables, output_variables, arguments = self._parse_function(
                    substep_transitions[transition_func],
                    name_root=f"_transition_{transition_func}",
                )
                # registry.transition_helpers["AggregateVitality"] → AggregateVitality 类
                self.transition_function[substep][transition_func] = (
                    self.registry.transition_helpers[transition_func](
                        self.config, input_variables, output_variables, arguments
                    )
                )

    # ─────────────────────── 对外接口 ────────────────────────────────────────

    def initialize(self):
        """完整初始化入口：构建 state 数据层 + 实例化所有 substep 模块。

        调用顺序：
          1. 设置 current_step=0 / current_substep="0"
          2. simulator() → 构建 environment / agents / objects / network
          3. substeps()  → 实例化所有 Observation / Policy / Transition 模块
          4. 把四域数据和参数字典挂进 state

        Runner.init() 内部调用此方法一次，之后仿真循环就可以开始运行。
        """
        self.state["current_step"] = 0
        self.state["current_substep"] = "0"  # 字符串，nn.ModuleDict 的 key 必须是字符串

        self.simulator()
        self.substeps()

        self.state["environment"] = self.environment
        self.state["network"] = self.networks
        self.state["agents"] = self.agents
        self.state["objects"] = self.objects
        self.state["parameters"] = self.parameters_dict

    def reset_state(self):
        """仅重置 state 张量，不重建 substep 模块（训练循环专用）。

        与 initialize() 的关键区别：
          - initialize()：同时重建 substep 模块 → substep 中的可学习参数变成新对象，
                          优化器失去对它们的追踪，梯度断裂
          - reset_state()：只重新执行 simulator()，保留已实例化的 substep 模块 →
                          可学习参数对象不变，优化器继续正常追踪

        典型用法：训练循环中每个 epoch 开始时调用，让仿真从初始状态重新跑，
        同时保持参数梯度链路完整。
        """
        self.state["current_step"] = 0
        self.state["current_substep"] = "0"

        # 清空数据容器，重新从 config 初始化（不触碰 substep 模块）
        self.environment, self.agents, self.objects, self.networks = {}, {}, {}, {}
        self.fixed_parameters, self.learnable_parameters = {}, {}

        self.simulator()   # 重建数据层

        # 更新 state 中的引用（simulator 生成了新的 dict，需要重新挂载）
        self.state["environment"] = self.environment
        self.state["network"] = self.networks
        self.state["agents"] = self.agents
        self.state["objects"] = self.objects
        self.state["parameters"] = self.parameters_dict

    def forward(self):
        """nn.Module 标准接口：直接委托给 initialize()，方便函数式调用。"""
        self.initialize()

    # ─────────────────────── 序列化支持 ──────────────────────────────────────

    def __getstate__(self):
        """pickle 序列化时调用：把 state 字典安全地导出为可序列化格式。

        nn.ParameterDict 不能直接 pickle，需要提前调用 state_dict() 转为普通字典。
        用于模型保存（torch.save）或多进程数据传递。
        """
        state_dict = self.state.copy()
        params = getattr(self, 'parameters_dict', None)
        if isinstance(params, nn.ParameterDict):
            state_dict["parameters"] = params.state_dict()
        return state_dict

    def __setstate__(self, state):
        """pickle 反序列化时调用：从导出格式恢复 state 和 parameters_dict。

        与 __getstate__ 对称：把普通字典恢复为 nn.ParameterDict，
        并重新挂回 state["parameters"]。
        """
        self.parameters_dict = nn.ParameterDict(state.get("parameters", {}))
        self.state = state
        self.state["parameters"] = self.parameters_dict
