"""深圳城市活力模型的两个 Substep（仿真子步骤）。

MovePolicy       — 居民移动决策（policy 类型）
AggregateVitality — 活力聚合计算（transition 类型）
"""

from .aggregate import AggregateVitality
from .move import MovePolicy

__all__ = ["AggregateVitality", "MovePolicy"]
