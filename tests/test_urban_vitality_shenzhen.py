import pandas as pd
import torch

from agent_torch.models.urban_vitality_shenzhen import create_runner
from agent_torch.models.urban_vitality_shenzhen.scenario import (
    ODPredictor,
    RenewalScheme,
    ScenarioPlan,
    SiteSpec,
    _apply_scheme_features,
    _resolve_site,
    _scheme_to_normalized,
    run_scenario_plan,
)
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


def _write_fixture_data(path, *, with_portrait=False, with_od=False):
    data = {
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
    if with_od:
        data["od_mock"] = [5.0, 6.0, 20.0, 40.0]
    features = pd.DataFrame(data)
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



def test_scenario_resolves_block_site_spec():
    site = SiteSpec(blocks=[{"block_id": 1, "coverage": 0.25}, {"block_id": 2}])
    assert _resolve_site(site, data_dir="unused") == {1: 0.25, 2: 1.0}


def test_scheme_normalization_and_coverage_application(tmp_path, capsys):
    _write_fixture_data(tmp_path)
    dataset = load_shenzhen_vitality_data(tmp_path, validation_fraction=0.0)
    scheme = RenewalScheme(
        name="test",
        building={"FAR": 6.0, "missing_feature": 1.0},
    )

    norm = _scheme_to_normalized(
        scheme, dataset.feature_names, dataset.feature_mean, dataset.feature_scale
    )
    out = capsys.readouterr().out
    assert "missing_feature" in out
    far_idx = dataset.feature_names.index("FAR")
    assert set(norm) == {far_idx}

    modified = _apply_scheme_features(dataset.features, norm, [0], [0.5])
    expected = dataset.features[0, far_idx] * 0.5 + torch.tensor(norm[far_idx]) * 0.5
    assert torch.isclose(modified[0, far_idx], expected)
    assert torch.isclose(modified[1, far_idx], dataset.features[1, far_idx])


def test_od_feedback_updates_only_target_blocks(tmp_path):
    _write_fixture_data(tmp_path, with_od=True)
    dataset = load_shenzhen_vitality_data(tmp_path, validation_fraction=0.0)
    od = ODPredictor().fit(dataset)
    far_idx = dataset.feature_names.index("FAR")
    od_idx = dataset.feature_names.index("od_mock")

    modified = dataset.features.clone()
    modified[0, far_idx] = modified[0, far_idx] + 5.0
    result = od.predict_od_for_targets(dataset.features, modified, [0])

    assert not torch.isclose(result[0, od_idx], dataset.features[0, od_idx])
    assert torch.allclose(result[1:, od_idx], dataset.features[1:, od_idx])


def test_run_scenario_plan_exports_expected_csvs(tmp_path):
    _write_fixture_data(tmp_path)
    runner, dataset = create_runner(tmp_path, hidden_dim=8, validation_fraction=0.0)
    plan = ScenarioPlan(
        plan_name="fixture plan",
        site=SiteSpec(blocks=[{"block_id": 1, "coverage": 1.0}]),
        schemes=[RenewalScheme(name="scheme_a", building={"FAR": 5.0})],
    )

    result = run_scenario_plan(runner, dataset, plan, data_dir=tmp_path, od_feedback=True)
    assert result.baseline.shape == (dataset.num_blocks, 48)
    assert result.schemes["scheme_a"].shape == (dataset.num_blocks, 48)

    out_dir = tmp_path / "scenario_out"
    result.to_csv(out_dir)
    assert (out_dir / "baseline_vitality.csv").exists()
    delta = pd.read_csv(out_dir / "delta_scheme_a.csv")
    assert list(delta.columns[:2]) == ["block_id", "t0"]
    assert len(delta) == dataset.num_blocks
