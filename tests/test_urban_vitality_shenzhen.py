import pandas as pd
import torch

from agent_torch.models.urban_vitality_shenzhen import create_runner
from agent_torch.models.urban_vitality_shenzhen.data import (
    DEMO_GROUPS,
    N_DEMO_GROUPS,
    PORTRAIT_FEATURES,
    TARGET_NAMES,
    load_shenzhen_vitality_data,
)


def _write_fixture_data_with_spatial(path):
    """Write fixture data plus a minimal GeoDataFrame-like shapefile substitute.

    We can't write a real .shp without geopandas, so this just tests the
    no-shapefile fallback path — has_spatial stays False, spatial_proj is None.
    """
    _write_fixture_data(path)


def _write_fixture_data(path, *, with_portrait=False):
    features = pd.DataFrame(
        {
            "Block_ID": [1, 1, 2, 3],
            "Shape_Area": [100.0, 120.0, 200.0, 300.0],
            "FAR": [1.0, 3.0, 2.0, 4.0],
            "RoadLevel": ["[1]", "[1]", "[2]", "[3]"],
            # demographic groups required for agent weight creation
            "青少年与儿童": [100.0, 100.0, 200.0, 50.0],
            "青年":         [300.0, 300.0, 500.0, 150.0],
            "中年":         [400.0, 400.0, 600.0, 200.0],
            "老年":         [200.0, 200.0, 300.0, 100.0],
        }
    )
    target_rows = []
    for block_id, base in [(1, 10.0), (2, 20.0), (3, 30.0)]:
        row = {"Block_ID": block_id}
        row.update({name: base + hour for hour, name in enumerate(TARGET_NAMES)})
        target_rows.append(row)
    features.to_csv(path / "街坊_数据连接.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(target_rows).to_csv(path / "街坊_LBS统计.csv", index=False, encoding="utf-8-sig")

    if with_portrait:
        portrait_rows = []
        for block_id, factor in [(1, 1.0), (2, 2.0), (3, 3.0)]:
            row = {"Block_ID": block_id}
            row.update({col: factor * 100.0 for col in PORTRAIT_FEATURES})
            portrait_rows.append(row)
        pd.DataFrame(portrait_rows).to_csv(
            path / "街坊_人口画像.csv", index=False, encoding="utf-8-sig"
        )


def test_loader_builds_dataset_with_demo_weights(tmp_path):
    _write_fixture_data(tmp_path)
    dataset = load_shenzhen_vitality_data(tmp_path, validation_fraction=0.0)

    assert dataset.num_blocks == 3
    assert dataset.vitality.shape == (3, 48)
    assert dataset.demo_weights.shape == (3, N_DEMO_GROUPS)
    assert torch.isfinite(dataset.features).all()
    # duplicate Block_ID=1 rows collapsed; weight from first group > 0
    assert (dataset.demo_weights >= 0).all()


def test_portrait_features_are_merged_when_file_present(tmp_path):
    _write_fixture_data(tmp_path)
    baseline_features = load_shenzhen_vitality_data(tmp_path, validation_fraction=0.0).num_features

    _write_fixture_data(tmp_path, with_portrait=True)
    dataset = load_shenzhen_vitality_data(tmp_path, validation_fraction=0.0)

    assert dataset.num_features == baseline_features + len(PORTRAIT_FEATURES)
    assert torch.isfinite(dataset.features).all()
    assert all(name in dataset.feature_names for name in PORTRAIT_FEATURES)


def test_runner_produces_emergent_vitality(tmp_path):
    _write_fixture_data(tmp_path)
    runner, dataset = create_runner(tmp_path, hidden_dim=8, validation_fraction=0.0)

    runner.step(1)
    state = runner.state

    predicted = state["environment"]["predicted_vitality"]
    assert predicted.shape == (3, 48)
    assert torch.isfinite(predicted).all()
    # vitality should be non-negative (comes from population weights)
    assert (predicted >= 0).all()


def test_gradients_flow_to_policy_parameters(tmp_path):
    _write_fixture_data(tmp_path)
    runner, dataset = create_runner(tmp_path, hidden_dim=8, validation_fraction=0.0)

    runner.reset_state()
    runner.step(1)
    predicted = runner.state["environment"]["predicted_vitality_scaled"]
    observed  = runner.state["environment"]["observed_vitality_scaled"]
    loss = (predicted - observed).square().mean()
    loss.backward()

    move_policy = runner.initializer.policy_function["0"]["residents"]["move_policy"]
    assert move_policy.home_logits.grad is not None
    assert move_policy.attract_net[0].weight.grad is not None
    assert move_policy.scale_net.weight.grad is not None


def test_spatial_network_disabled_without_shapefile(tmp_path):
    """Without a shapefile, has_spatial=False and spatial_proj is None."""
    _write_fixture_data(tmp_path)
    dataset = load_shenzhen_vitality_data(tmp_path, validation_fraction=0.0)
    assert not dataset.has_spatial
    assert dataset.edge_index is None

    runner, dataset = create_runner(tmp_path, hidden_dim=8, validation_fraction=0.0)
    policy = runner.initializer.policy_function["0"]["residents"]["move_policy"]
    assert policy.spatial_attn is None

    runner.step(1)
    predicted = runner.state["environment"]["predicted_vitality"]
    assert predicted.shape == (3, 48)
    assert torch.isfinite(predicted).all()


def test_temporal_prior_encodes_day_night_pattern(tmp_path):
    """home_logits should start with higher p_home at night than at midday."""
    _write_fixture_data(tmp_path)
    runner, _ = create_runner(tmp_path, hidden_dim=8, validation_fraction=0.0)
    policy = runner.initializer.policy_function["0"]["residents"]["move_policy"]
    p = torch.sigmoid(policy.home_logits.detach())
    # All demo groups: p_home at 2am (slot 2) > p_home at 1pm (slot 13)
    assert (p[:, 2] > p[:, 13]).all(), "Prior should have higher p_home at night than midday"
