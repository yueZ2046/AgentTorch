# executor.py — 仿真执行器
#
# 职责：将 Runner 封装成一个高层"执行器"，对外提供 init/execute 两步接口。
# 使用者无需直接操作 Runner，只需通过 Executor 完成：
#   1. 加载数据 / 人口（DataLoader / PopLoader）
#   2. 初始化仿真状态
#   3. 按 episode 循环推进仿真并收集结果
#
# 关键依赖：
#   Runner   — 仿真主循环（core/runner.py）
#   DataLoader — 从 PopLoader 构造 config（core/dataloader.py）
#   dask.dataframe — 惰性读取大规模 state trajectory

import importlib
import sys
from tqdm import trange
import dask.dataframe as dd
from agent_torch.core.dataloader import DataLoader
from agent_torch.core.runner import Runner


class BaseExecutor:
    """执行器基类：持有 model 引用，提供从 model 构造 Runner 的公共方法。"""

    def __init__(self, model):
        # model 是一个 Python 模块对象，该模块内部需暴露全局 registry
        self.model = model

    def _get_runner(self, config):
        """根据 config 和 model 内置的 registry 实例化 Runner。

        注意：要求 model 模块在顶层暴露 `registry` 变量，
        否则 importlib.import_module 后无法找到注册表。
        """
        module_name = self.model.__name__
        # 动态导入 model 模块（已在 sys.modules 中时直接返回缓存）
        module = importlib.import_module(module_name)

        registry = module.registry
        print("Registry: ", registry)
        runner = Runner(config, registry)
        return runner


class Executor(BaseExecutor):
    """主执行器：将 DataLoader、Runner、episode 循环整合为三步式 API。

    典型用法：
        executor = Executor(model=my_model, pop_loader=loader)
        executor.init()
        executor.execute(key="infected")
        values = executor.get_simulation_values("infected")
    """

    def __init__(self, model, data_loader=None, pop_loader=None) -> None:
        super().__init__(model)
        self.model = model

        if pop_loader:
            # 优先使用 PopLoader：从人口数据自动生成 DataLoader
            self.pop_loader = pop_loader
            self.data_loader = DataLoader(model, self.pop_loader)
        else:
            # 直接使用外部传入的 DataLoader（已预先构建好 config）
            self.data_loader = data_loader

        # 从 DataLoader 读取 YAML config，再据此构建 Runner
        self.config = self.data_loader.get_config()
        self.runner = self._get_runner(self.config)

    def init(self):
        """重新加载 config 并初始化 Runner 的内部状态。

        必须在 execute() 之前调用。每次重新 init 会重置仿真状态。
        """
        self.config = self.data_loader.get_config()
        self.runner = self._get_runner(self.config)
        self.runner.init()
        # 以下为可微分训练预留接口（当前未启用）：
        # self.learnable_params = [
        #     param for param in self.runner.parameters() if param.requires_grad
        # ]
        # self.opt = opt(self.learnable_params)

    def execute(self, key=None):
        """按照 config 中配置的 episode/step 数驱动仿真完整运行。

        Args:
            key: 若指定，执行完毕后自动调用 get_simulation_values(key) 缓存结果。
        """
        num_episodes = self.config["simulation_metadata"]["num_episodes"]
        num_steps_per_episode = self.config["simulation_metadata"][
            "num_steps_per_episode"
        ]

        for episode in trange(num_episodes):
            # 每个 episode 开始前重置 agent 状态（不重建 Runner）
            # self.opt.zero_grad()  # 可微分训练时在此清零梯度
            self.runner.reset()
            # 推进 num_steps_per_episode 步，内部由 Controller 编排各 Substep
            self.runner.step(num_steps_per_episode)

        if key is not None:
            self.simulation_values = self.runner.get_simulation_values(key)

    def get_simulation_values(self, key, key_type="environment"):
        """从 state trajectory 末尾帧取出指定变量的张量值。

        Args:
            key:      state 字典中的变量名（如 "infected"、"position"）
            key_type: 变量所在的顶层分区，默认 "environment"；
                      agent 属性通常在 "agents" 下

        Returns:
            对应变量的张量（或数值）

        注意：trajectory 若为 dask DataFrame（大规模仿真时），
        会先调用 .compute() 触发实际计算，可能较慢。
        """
        # dask 惰性 DataFrame 在此处强制求值
        if isinstance(self.runner.state_trajectory, dd.DataFrame):
            self.runner.state_trajectory = self.runner.state_trajectory.compute()

        # state_trajectory[-1][-1] 取最后一个 episode 的最后一步 state
        self.simulation_values = self.runner.state_trajectory[-1][-1][key_type][key]
        return self.simulation_values
