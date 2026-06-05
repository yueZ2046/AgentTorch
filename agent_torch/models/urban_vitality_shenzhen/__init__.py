"""深圳街坊级城市活力仿真模型。

核心思路：
    不直接用神经网络预测活力值，而是用可微分的 Agent 仿真：
    居民（按人口群体分组的 Agent）决定"留家 vs 外出"，
    外出人口按街坊吸引力路由，最终叠加得到各街坊各时段的活力值。
    整个仿真过程对参数可微，可以用 PyTorch 自动求导训练。

Agent 的构成：
    Agent 数量 = 街坊数 × 人口群体数（默认 4 个群体：青少年/青年/中年/老年）
    每个 Agent 代表某街坊某人口群体，weight 属性是该群体的人口数量。

仿真每步执行两个 Substep（定义在 substeps/ 目录）：
    1. MovePolicy（policy）     → 输出留家概率、街坊吸引力、规模修正
    2. AggregateVitality（transition）→ 把居民散射到城市各街坊，输出预测活力
"""

from agent_torch.core import Registry, Runner

from .data import ShenzhenVitalityDataset, build_config, load_shenzhen_vitality_data
from .substeps import AggregateVitality, MovePolicy


def get_registry() -> Registry:
    """创建并返回注册表，把字符串名称映射到对应的 Python 类。

    AgentTorch 的 Runner 读取 YAML/dict 配置时，通过 generator 字段的字符串
    在注册表里查找对应的类并实例化，所以每个自定义 Substep 都必须在这里注册。

    key="policy"     → 决策函数（Agent 行动）
    key="transition" → 状态转移函数（环境更新）
    """
    registry = Registry()
    registry.register(MovePolicy,        "move_policy",        key="policy")
    registry.register(AggregateVitality, "aggregate_vitality", key="transition")
    return registry


def create_runner(
    data_dir="data_shenzhen",
    hidden_dim: int = 64,
    validation_fraction: float = 0.2,
    seed: int = 42,
    device: str = "auto",
    split_strategy: str = "random",
    holdout_district=None,
):
    """加载深圳数据，构建配置，返回已初始化的 AgentTorch Runner。

    执行流程：
        1. load_shenzhen_vitality_data → 读 CSV/SHP，处理特征，划分训练/验证集
        2. build_config               → 把数据集打包成 AgentTorch 需要的配置字典
        3. Runner(config, registry)   → 根据配置实例化仿真环境和 Substep
        4. runner.init()              → 初始化所有张量（包括模型参数）
        5. runner.to(device)          → 把所有参数搬到 GPU 或 CPU
    """
    dataset = load_shenzhen_vitality_data(
        data_dir=data_dir,
        validation_fraction=validation_fraction,
        seed=seed,
        split_strategy=split_strategy,
        holdout_district=holdout_district,
    )
    config = build_config(dataset, hidden_dim=hidden_dim, device=device)
    runner = Runner(config, get_registry())
    runner.init()
    runner.to(runner.initializer.device)
    return runner, dataset


__all__ = [
    "ShenzhenVitalityDataset",
    "build_config",
    "create_runner",
    "get_registry",
    "load_shenzhen_vitality_data",
]
