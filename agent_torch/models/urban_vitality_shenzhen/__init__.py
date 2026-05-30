"""Shenzhen block-level urban vitality simulation model.

Residents are agent groups indexed by (block, demographic cohort).
Urban vitality emerges from their movement decisions rather than
being predicted directly.
"""

from agent_torch.core import Registry, Runner

from .data import ShenzhenVitalityDataset, build_config, load_shenzhen_vitality_data
from .substeps import AggregateVitality, MovePolicy


def get_registry() -> Registry:
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
    """Load Shenzhen data and return an initialized AgentTorch runner."""
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
