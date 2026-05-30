"""Phase 4: Urban renewal intervention scenario comparison.

Usage (Python API):
    from agent_torch.models.urban_vitality_shenzhen.scenario import (
        RenewalScheme, ScenarioPlan, run_scenario_plan
    )

    plan = ScenarioPlan.from_json("my_plan.json")
    result = run_scenario_plan(runner, dataset, plan, data_dir="data_shenzhen")
    result.print_report()
    result.to_csv("outputs/scenario_result.csv")

JSON format:
    {
      "plan_name": "福田某地块更新比选",
      "site": {
        "shp_path": "path/to/site.shp"          // SHP boundary of renewal site
        // OR: "blocks": [{"block_id": 957, "coverage": 0.7}]
      },
      "schemes": [
        {
          "name": "方案A：高密度商业综合体",
          "description": "商业开发，容积率4.5",
          "building": {"FAR": 4.5, "TOT_Area": 60000},
          "poi": {"poi_restaurant": 80, "poi_shopping": 60, "poi_company": 40}
        },
        {
          "name": "方案B：文化创意园",
          "description": "低密度文化功能",
          "building": {"FAR": 2.0, "TOT_Area": 25000},
          "poi": {"poi_education": 15, "poi_sports": 10, "poi_restaurant": 20}
        }
      ]
    }

Feature value reference (city-wide P25 / P50 / P75):
    Building:  FAR 0.6 / 1.5 / 2.9    TOT_Area 87k / 282k / 658k (㎡)
    POI:       poi_restaurant 1 / 11 / 39    poi_shopping 2 / 17 / 59
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


# Features that use log1p transform before normalisation.
_LOG1P_PREFIXES = ("poi_", "od_")


def _is_log1p(feature_name: str) -> bool:
    return any(feature_name.startswith(p) for p in _LOG1P_PREFIXES)


# ─────────────────────────── data classes ────────────────────────────────────

@dataclass
class RenewalScheme:
    """One candidate renewal scheme for a site.

    building: dict of building feature name → raw value (e.g. FAR=4.5, TOT_Area=60000)
    poi:      dict of POI feature name → raw count  (e.g. poi_restaurant=80)
    """
    name: str
    description: str = ""
    building: Dict[str, float] = field(default_factory=dict)
    poi:      Dict[str, float] = field(default_factory=dict)

    def feature_overrides(self) -> Dict[str, float]:
        """Merge building and poi dicts into one feature-name → raw-value dict."""
        merged = {}
        merged.update(self.building)
        merged.update(self.poi)
        return merged


@dataclass
class SiteSpec:
    """Spatial definition of the renewal site.

    Either shp_path (SHP / GeoJSON file) or blocks (list of dicts with
    block_id and optional coverage) must be provided.
    """
    shp_path:  Optional[str]                     = None
    blocks:    Optional[List[Dict]]               = None  # [{"block_id": 957, "coverage": 0.7}, ...]


@dataclass
class ScenarioPlan:
    plan_name: str
    site:      SiteSpec
    schemes:   List[RenewalScheme]

    @classmethod
    def from_json(cls, path: str | Path) -> "ScenarioPlan":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        site_d = d["site"]
        site = SiteSpec(
            shp_path=site_d.get("shp_path"),
            blocks=site_d.get("blocks"),
        )
        schemes = []
        for s in d["schemes"]:
            schemes.append(RenewalScheme(
                name=s["name"],
                description=s.get("description", ""),
                building=s.get("building", {}),
                poi=s.get("poi", {}),
            ))
        return cls(plan_name=d["plan_name"], site=site, schemes=schemes)

    def to_json(self, path: str | Path) -> None:
        d = {
            "plan_name": self.plan_name,
            "site": {
                **({"shp_path": self.site.shp_path} if self.site.shp_path else {}),
                **({"blocks": self.site.blocks} if self.site.blocks else {}),
            },
            "schemes": [
                {
                    "name": s.name,
                    "description": s.description,
                    "building": s.building,
                    "poi": s.poi,
                }
                for s in self.schemes
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)


@dataclass
class ScenarioResult:
    plan_name:        str
    block_ids:        np.ndarray          # (N_blocks,)
    block_districts:  Optional[np.ndarray]
    target_coverage:  Dict[int, float]    # block_id → coverage fraction
    baseline:         np.ndarray          # (N_blocks, 48) raw LBS vitality
    schemes:          Dict[str, np.ndarray]  # scheme_name → (N_blocks, 48)
    scheme_descs:     Dict[str, str]
    feature_names:    List[str]

    # ── helpers ──

    def _target_mask(self) -> np.ndarray:
        """Boolean mask for target blocks."""
        return np.isin(self.block_ids, list(self.target_coverage.keys()))

    def delta(self, scheme_name: str) -> np.ndarray:
        """Vitality change vs baseline: (N_blocks, 48)."""
        return self.schemes[scheme_name] - self.baseline

    def _vitality_summary(self, vitality: np.ndarray, mask: np.ndarray) -> Dict:
        v = vitality[mask]
        return {
            "weekday_mean": float(v[:, :24].mean()),
            "weekend_mean": float(v[:, 24:].mean()),
            "weekday_peak": float(v[:, :24].max(axis=1).mean()),
            "peak_hour_wd": int(v[:, :24].mean(axis=0).argmax()),
        }

    def summary_table(self) -> pd.DataFrame:
        """Per-scheme comparison of target-block vitality."""
        mask = self._target_mask()
        rows = []
        base = self._vitality_summary(self.baseline, mask)
        rows.append({
            "方案": "现状基准",
            "描述": "",
            "工作日均值活力": round(base["weekday_mean"]),
            "周末均值活力": round(base["weekend_mean"]),
            "工作日峰值活力": round(base["weekday_peak"]),
            "工作日峰值时段": f"{base['peak_hour_wd']:02d}:00",
            "活力变化(工作日)": "—",
            "活力变化率": "—",
        })
        for name, v in self.schemes.items():
            s = self._vitality_summary(v, mask)
            delta_wd = s["weekday_mean"] - base["weekday_mean"]
            rate = delta_wd / (base["weekday_mean"] + 1e-6) * 100
            rows.append({
                "方案": name,
                "描述": self.scheme_descs.get(name, ""),
                "工作日均值活力": round(s["weekday_mean"]),
                "周末均值活力": round(s["weekend_mean"]),
                "工作日峰值活力": round(s["weekday_peak"]),
                "工作日峰值时段": f"{s['peak_hour_wd']:02d}:00",
                "活力变化(工作日)": f"{delta_wd:+.0f}",
                "活力变化率": f"{rate:+.1f}%",
            })
        return pd.DataFrame(rows)

    def temporal_delta_table(self) -> pd.DataFrame:
        """Hourly vitality delta per scheme for target blocks."""
        mask = self._target_mask()
        hours_wd = [f"工作日{h:02d}时" for h in range(24)]
        hours_we = [f"周末{h:02d}时" for h in range(24)]
        rows = {"时段": hours_wd + hours_we}
        for name, v in self.schemes.items():
            d = (v - self.baseline)[mask].mean(axis=0)
            rows[name] = d.round(1).tolist()
        return pd.DataFrame(rows)

    def spillover_table(self, dataset, top_n: int = 10) -> pd.DataFrame:
        """Non-target blocks with largest vitality delta (spillover / displacement)."""
        target_ids = set(self.target_coverage.keys())
        nontarget = ~np.isin(self.block_ids, list(target_ids))

        rows = []
        for name, v in self.schemes.items():
            d = (v - self.baseline)[nontarget].mean(axis=1)
            block_ids_nt = self.block_ids[nontarget]
            top_idx = np.argsort(np.abs(d))[::-1][:top_n]
            for i in top_idx:
                row = {
                    "方案": name,
                    "block_id": int(block_ids_nt[i]),
                    "工作日活力变化": round(float(d[i]), 1),
                    "方向": "溢出增益" if d[i] > 0 else "替代损失",
                }
                if self.block_districts is not None:
                    row["district"] = self.block_districts[nontarget][i]
                rows.append(row)
        return pd.DataFrame(rows)

    def print_report(self, dataset=None) -> None:
        target_ids = list(self.target_coverage.keys())
        print(f"\n{'='*60}")
        print(f"  {self.plan_name}")
        print(f"{'='*60}")
        print(f"涉及街坊: {target_ids}")
        for bid, cov in self.target_coverage.items():
            print(f"  Block_{bid}  覆盖率 {cov*100:.0f}%")

        print(f"\n── 方案对比（目标街坊均值）──")
        print(self.summary_table().to_string(index=False))

        print(f"\n── 工作日分时段活力变化 ──")
        td = self.temporal_delta_table()
        # Print weekday only for brevity
        wd = td[td["时段"].str.startswith("工作日")].copy()
        print(wd.to_string(index=False))

        if dataset is not None:
            print(f"\n── 空间溢出效应（周边街坊，前10）──")
            sp = self.spillover_table(dataset, top_n=10)
            print(sp.to_string(index=False))

        print(f"\n⚠️  结果为模型反事实估计，非因果预测。模型准确率约 68–72%（中高活力街坊）。")

    def to_csv(self, output_dir: str | Path) -> None:
        """Export baseline and all scheme delta matrices as CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_df = pd.DataFrame(self.baseline, columns=[f"t{i}" for i in range(48)])
        base_df.insert(0, "block_id", self.block_ids)
        base_df.to_csv(output_dir / "baseline_vitality.csv", index=False)

        for name, v in self.schemes.items():
            delta = v - self.baseline
            df = pd.DataFrame(delta, columns=[f"t{i}" for i in range(48)])
            df.insert(0, "block_id", self.block_ids)
            safe = name.replace("/", "_").replace(" ", "_")
            df.to_csv(output_dir / f"delta_{safe}.csv", index=False, encoding="utf-8-sig")

        print(f"[scenario] 结果已导出到 {output_dir}/")


# ─────────────────────────── site resolution ─────────────────────────────────

def _resolve_site_from_shp(
    shp_path: str, data_dir: Path
) -> Dict[int, float]:
    """Compute coverage fraction for each overlapping block from a site SHP/GeoJSON."""
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas required for SHP-based site specification.")

    block_shp = data_dir / "街坊范围shp/深圳_街坊_Pro.shp"
    if not block_shp.exists():
        raise FileNotFoundError(f"Block shapefile not found: {block_shp}")

    blocks = gpd.read_file(block_shp)
    blocks["Block_ID"] = blocks["Block_ID"].astype(float).round().astype(int)

    site_gdf = gpd.read_file(shp_path)
    if site_gdf.crs != blocks.crs:
        site_gdf = site_gdf.to_crs(blocks.crs)

    site_union = site_gdf.union_all() if hasattr(site_gdf, "union_all") else site_gdf.unary_union

    coverage: Dict[int, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        intersected = blocks[blocks.intersects(site_union)].copy()

    for _, row in intersected.iterrows():
        inter_area = row.geometry.intersection(site_union).area
        block_area = row.geometry.area
        frac = inter_area / block_area if block_area > 0 else 0.0
        if frac > 0.01:   # ignore trivial slivers < 1%
            coverage[int(row["Block_ID"])] = round(min(frac, 1.0), 4)

    if not coverage:
        raise ValueError("Site geometry does not overlap with any block.")

    return coverage


def _resolve_site(site: SiteSpec, data_dir: Path) -> Dict[int, float]:
    """Return {block_id: coverage} from SiteSpec."""
    if site.shp_path:
        return _resolve_site_from_shp(site.shp_path, data_dir)
    if site.blocks:
        return {
            int(b["block_id"]): float(b.get("coverage", 1.0))
            for b in site.blocks
        }
    raise ValueError("SiteSpec must have either shp_path or blocks.")


# ─────────────────────────── feature application ─────────────────────────────

def _scheme_to_normalized(
    scheme: RenewalScheme,
    feature_names: List[str],
    feature_mean:  torch.Tensor,
    feature_scale: torch.Tensor,
) -> Dict[int, float]:
    """Convert scheme raw values to normalised feature space.

    Returns {feature_index: normalised_value}.
    Unknown feature names are skipped with a warning.
    """
    result: Dict[int, float] = {}
    overrides = scheme.feature_overrides()
    for feat_name, raw_val in overrides.items():
        if feat_name not in feature_names:
            print(f"  [scenario] 警告：特征 '{feat_name}' 不存在于数据集，已跳过。")
            continue
        idx = feature_names.index(feat_name)
        mean_  = float(feature_mean[idx].item())
        scale_ = float(feature_scale[idx].item())
        if _is_log1p(feat_name):
            log_val = float(np.log1p(max(raw_val, 0.0)))
            norm_val = (log_val - mean_) / (scale_ + 1e-8)
        else:
            norm_val = (raw_val - mean_) / (scale_ + 1e-8)
        result[idx] = norm_val
    return result


def _apply_scheme_features(
    features:     torch.Tensor,           # (N_blocks, n_feat)  — current normalised
    scheme_norm:  Dict[int, float],       # feature_idx → new normalised value
    block_indices: List[int],             # which blocks to modify
    coverage:     List[float],            # coverage fraction per block
) -> torch.Tensor:
    """Return modified feature matrix with scheme applied at target blocks."""
    modified = features.clone()
    for b_idx, cov in zip(block_indices, coverage):
        for feat_idx, new_val in scheme_norm.items():
            orig = float(modified[b_idx, feat_idx].item())
            modified[b_idx, feat_idx] = orig * (1.0 - cov) + new_val * cov
    return modified


# ─────────────────────────── OD predictor ────────────────────────────────────

class ODPredictor:
    """Two-stage feedback model: static features → predicted OD → vitality.

    Trains a Ridge regression from static features (building + POI, no OD)
    to OD features (arrival/departure time series).  When a scheme changes
    POI or building attributes, the predictor estimates the new OD patterns
    that would result — capturing the feedback loop:

        add restaurants → more noon arrivals → higher noon vitality

    The predictor is trained on cross-sectional variation across 3023 blocks
    (blocks with more restaurants do have more noon arrivals, r≈0.67).
    This is an approximation: cross-sectional ≠ causal, but it is far more
    informative than freezing OD at the current state.

    R² of static features → OD: ~0.65 (mean across 96 OD columns).
    """

    def __init__(self):
        self._model = None
        self._static_indices: List[int] = []
        self._od_indices: List[int] = []

    def fit(self, dataset) -> "ODPredictor":
        from sklearn.linear_model import Ridge

        fnames = dataset.feature_names
        self._static_indices = [i for i, f in enumerate(fnames) if not f.startswith("od_")]
        self._od_indices     = [i for i, f in enumerate(fnames) if f.startswith("od_")]
        if not self._od_indices:
            raise ValueError("No OD features found; cannot fit OD feedback model.")

        X = dataset.features[:, self._static_indices].numpy()
        Y = dataset.features[:, self._od_indices].numpy()

        self._model = Ridge(alpha=1.0).fit(X, Y)
        r2 = self._model.score(X, Y)
        print(f"[scenario] OD预测器训练完成  R²={r2:.3f}  "
              f"({len(self._static_indices)} 静态特征 → {len(self._od_indices)} OD列)")
        return self

    def predict_od(self, features: torch.Tensor) -> torch.Tensor:
        """Predict OD columns from (possibly scheme-modified) feature matrix."""
        X = features[:, self._static_indices].numpy()
        od_pred = np.asarray(self._model.predict(X))
        if od_pred.ndim == 1:
            od_pred = od_pred[:, None]
        result = features.clone()
        result[:, self._od_indices] = torch.from_numpy(od_pred.astype("float32"))
        return result

    def predict_od_for_targets(
        self,
        baseline_features: torch.Tensor,
        modified_features: torch.Tensor,
        block_indices: List[int],
    ) -> torch.Tensor:
        """Update OD columns only for target blocks.

        This keeps scenario deltas on the same baseline OD footing and avoids
        contaminating non-target spillover with cross-sectional OD model error.
        """
        od_all = self.predict_od(modified_features)
        result = modified_features.clone()
        if block_indices:
            od_cols = torch.tensor(self._od_indices, dtype=torch.long)
            idx = torch.tensor(block_indices, dtype=torch.long)
            result[:, od_cols] = baseline_features[:, od_cols]
            result[idx[:, None], od_cols] = od_all[idx[:, None], od_cols]
        return result


def build_od_predictor(dataset) -> ODPredictor:
    """Train and return an ODPredictor from the full dataset."""
    return ODPredictor().fit(dataset)


# ─────────────────────────── main runner ─────────────────────────────────────

def _run_inference(runner, dev: torch.device, feat_override: Optional[torch.Tensor] = None) -> np.ndarray:
    """Reset state (optionally inject features), run one step, return raw vitality."""
    runner.reset_state()
    if feat_override is not None:
        runner.state["environment"]["block_features"] = feat_override.to(dev)
    with torch.no_grad():
        runner.step(1)
    return runner.state["environment"]["predicted_vitality"].detach().cpu().numpy()


def run_scenario_plan(
    runner,
    dataset,
    plan: ScenarioPlan,
    data_dir: str | Path = "data_shenzhen",
    od_feedback: bool = True,
) -> ScenarioResult:
    """Run baseline and all schemes; return ScenarioResult for comparison.

    Args:
        od_feedback: If True (default), train an OD predictor and update OD
            features when a scheme changes POI/building attributes.  This
            captures the two-stage feedback loop:
                scheme (POI change) → predicted new OD → vitality prediction
            If False, OD features are frozen at their current values.

    The runner must already be trained (runner.init() called, parameters fitted).
    """
    data_dir = Path(data_dir)
    dev = runner.initializer.device

    # ── OD predictor (two-stage feedback) ──
    od_pred: Optional[ODPredictor] = None
    if od_feedback:
        try:
            od_pred = build_od_predictor(dataset)
        except ValueError as exc:
            print(f"[scenario] OD反馈已禁用：{exc}")

    # ── resolve site → {block_id: coverage} ──
    print(f"[scenario] 解析场地范围...")
    coverage_map = _resolve_site(plan.site, data_dir)
    print(f"[scenario] 覆盖街坊: {list(coverage_map.keys())}  覆盖率: {list(coverage_map.values())}")

    # Map block_id → dataset row index
    block_id_to_idx: Dict[int, int] = {
        int(bid): i for i, bid in enumerate(dataset.block_ids.numpy())
    }
    block_indices: List[int] = []
    coverage_vals: List[float] = []
    for bid, cov in coverage_map.items():
        if bid not in block_id_to_idx:
            print(f"  [scenario] 警告：Block_{bid} 不在数据集中，已跳过。")
            continue
        block_indices.append(block_id_to_idx[bid])
        coverage_vals.append(cov)

    if not block_indices:
        raise ValueError("场地内没有匹配的数据集街坊，请检查 block_id 或 SHP 范围。")

    base_features = dataset.features  # (N_blocks, n_feat) — normalised

    # ── baseline ──
    print("[scenario] 运行基准预测...")
    baseline_vitality = _run_inference(runner, dev)

    # ── each scheme ──
    scheme_vitalities: Dict[str, np.ndarray] = {}
    scheme_descs: Dict[str, str] = {}
    for scheme in plan.schemes:
        print(f"[scenario] 运行方案：{scheme.name}...")
        scheme_norm = _scheme_to_normalized(
            scheme,
            dataset.feature_names,
            dataset.feature_mean,
            dataset.feature_scale,
        )
        modified_features = _apply_scheme_features(
            base_features, scheme_norm, block_indices, coverage_vals
        )
        # Two-stage feedback: re-predict OD from modified static features
        if od_pred is not None:
            modified_features = od_pred.predict_od_for_targets(
                base_features, modified_features, block_indices
            )

        vitality = _run_inference(runner, dev, feat_override=modified_features)
        scheme_vitalities[scheme.name] = vitality
        scheme_descs[scheme.name] = scheme.description

    # Restore clean state
    runner.reset_state()

    return ScenarioResult(
        plan_name=plan.plan_name,
        block_ids=dataset.block_ids.numpy(),
        block_districts=dataset.districts,
        target_coverage=coverage_map,
        baseline=baseline_vitality,
        schemes=scheme_vitalities,
        scheme_descs=scheme_descs,
        feature_names=dataset.feature_names,
    )


# ─────────────────────────── utility ─────────────────────────────────────────

def list_scheme_features(dataset) -> None:
    """Print available building and POI features with their city-wide value ranges."""
    fnames = dataset.feature_names
    f_mean = dataset.feature_mean.numpy()
    f_scale = dataset.feature_scale.numpy()
    feats_raw = dataset.features.numpy()

    print("── 可在方案中指定的建筑特征 (building) ──")
    building = [f for f in fnames if not _is_log1p(f) and f not in
                {'CZZL20_24','CZZL25_29','CZZL30_34','CZZL35_39','CZZL40_44',
                 'CZZL45_49','CZZL50_54','CZZL55_59','XB1CZRK','XB2CZRK',
                 '就业人','CZRKMD','FHJCZRKZSL','JZSJD1RKSL','JZSJD5RKSL'}]
    for feat in building:
        idx = fnames.index(feat)
        raw = feats_raw[:, idx] * f_scale[idx] + f_mean[idx]
        p25, p50, p75 = np.percentile(raw, [25, 50, 75])
        print(f"  {feat:<20}  P25={p25:>10.1f}  P50={p50:>10.1f}  P75={p75:>10.1f}")

    print("\n── 可在方案中指定的 POI 特征 (poi)，单位：个 ──")
    poi = [f for f in fnames if f.startswith("poi_")]
    for feat in poi:
        idx = fnames.index(feat)
        log_raw = feats_raw[:, idx] * f_scale[idx] + f_mean[idx]
        raw = np.expm1(log_raw)
        p25, p50, p75 = np.percentile(raw, [25, 50, 75])
        print(f"  {feat:<22}  P25={p25:>6.0f}  P50={p50:>6.0f}  P75={p75:>6.0f}")
