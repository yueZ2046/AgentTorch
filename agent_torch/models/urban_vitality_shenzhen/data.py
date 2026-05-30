"""Data preparation and runtime configuration for Shenzhen urban vitality."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import warnings
import pandas as pd
import torch


TARGET_NAMES = [
    *(f"WD_C_{hour:02d}" for hour in range(24)),
    *(f"WE_C_{hour:02d}" for hour in range(24)),
]
NON_NUMERIC_FEATURES = {"Block_ID", "H_List", "FL_List", "RoadLevel"}

# Demographic groups — map each resident to one of these four cohorts.
# The columns must be present in 街坊_数据连接.csv.
DEMO_GROUPS = ["青少年与儿童", "青年", "中年", "老年"]
N_DEMO_GROUPS = len(DEMO_GROUPS)

PORTRAIT_FILE = "街坊_人口画像.csv"
SHP_FILE      = "街坊范围shp/深圳_街坊_Pro.shp"

POI_DIR = "POI2026/POI_SHP"
# Ordered by coverage / predictive relevance; poi_road excluded (99% zero, 189 total POIs).
POI_CATEGORIES = {
    "poi_medical":    "深圳市-医疗保健服务.shp",   # 26369 POIs — dense, good signal
    "poi_scenic":     "深圳市-风景名胜.shp",        # 5875 POIs — leisure destinations
    "poi_auto_sales": "深圳市-汽车销售.shp",        # 4297 POIs — commercial proxy
    "poi_motorcycle": "深圳市-摩托车服务.shp",      # 764 POIs — local mobility proxy
}

# 2026 POI update: CSV files include wgslng/wgslat (WGS84) coordinates.
# Categories aligned with Shenzhen vitality study activity dimensions.
POI_DIR_2026 = "POI2026/POI_CSV"
POI_CATEGORIES_2026 = {
    "poi_restaurant": "深圳市-餐饮服务.csv",         # 108 953 — dining, highest vitality driver
    "poi_shopping":   "深圳市-购物服务.csv",          # 162 881 — retail, daytime attractor
    "poi_life_svc":   "深圳市-生活服务.csv",          # 101 544 — everyday services
    "poi_transport":  "深圳市-交通设施服务.csv",      # 53 020 — transit hubs, accessibility
    "poi_company":    "深圳市-公司企业.csv",           # 130 706 — employment, weekday flow
    "poi_sports":     "深圳市-体育休闲服务.csv",      # 24 409 — leisure, weekend flow
    "poi_hotel":      "深圳市-住宿服务.csv",           # 21 244 — accommodation, overnight activity
    "poi_education":  "深圳市-科教文化服务.csv",      # 24 709 — cultural and educational
}

# Curated columns from 街坊_人口画像.csv ordered by predictive value.
# CZZL = 常住人口 by 5-year age cohort; other cols add employment,
# gender split, population density, mobility and household structure.
PORTRAIT_FEATURES = [
    "CZZL20_24", "CZZL25_29", "CZZL30_34", "CZZL35_39",  # peak working age
    "CZZL40_44", "CZZL45_49", "CZZL50_54", "CZZL55_59",  # mid-late career
    "XB1CZRK",                                             # male resident count
    "XB2CZRK",                                             # female resident count
    "就业人",                                               # employed population
    "CZRKMD",                                               # resident population density
    "FHJCZRKZSL",                                           # non-registered residents (floating)
    "JZSJD1RKSL",                                           # residents < 1 year (recent arrivals)
    "JZSJD5RKSL",                                           # residents > 10 years (long-term stable)
]


@dataclass
class ShenzhenVitalityDataset:
    """Tensors and metadata for agent-based block-level vitality simulation."""

    block_ids: torch.Tensor
    features: torch.Tensor
    vitality: torch.Tensor
    vitality_scaled: torch.Tensor
    demo_weights: torch.Tensor      # (N_blocks, N_DEMO_GROUPS) raw population counts
    train_mask: torch.Tensor
    validation_mask: torch.Tensor
    feature_names: List[str]
    target_names: List[str]
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    edge_index: Optional[torch.Tensor] = field(default=None)           # (2, E) Neighbor80 proximity
    edge_index_mobility: Optional[torch.Tensor] = field(default=None)  # (2, E) k-NN commute graph
    districts: Optional[np.ndarray] = field(default=None)              # (N_blocks,) str — admin district per block

    @property
    def num_blocks(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def has_spatial(self) -> bool:
        return self.edge_index is not None

    @property
    def has_mobility(self) -> bool:
        return self.edge_index_mobility is not None


def _canonical_block_id(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="raise").round().astype("int64")


def _training_masks(num_blocks: int, validation_fraction: float, seed: int):
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")

    validation_size = int(round(num_blocks * validation_fraction))
    if validation_fraction > 0 and validation_size == 0 and num_blocks > 1:
        validation_size = 1
    validation_size = min(validation_size, max(0, num_blocks - 1))

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_blocks, generator=generator)
    validation_mask = torch.zeros(num_blocks, dtype=torch.bool)
    validation_mask[order[:validation_size]] = True
    return ~validation_mask, validation_mask


def _load_block_districts(
    data_dir: Path, block_id_order: np.ndarray
) -> Optional[np.ndarray]:
    """Assign each block a Shenzhen administrative district label via spatial join.

    Uses block centroids joined to the OD departure grid (which carries district
    and street labels).  Returns a string array aligned with block_id_order,
    or None when geopandas / the OD shapefile is unavailable.
    """
    od_path  = data_dir / "LBS原始数据/工作日出发人口_网格.shp"
    shp_path = data_dir / SHP_FILE
    if not od_path.exists() or not shp_path.exists():
        return None
    try:
        import geopandas as gpd
    except ImportError:
        return None

    gdf = gpd.read_file(shp_path)
    gdf["Block_ID"] = _canonical_block_id(gdf["Block_ID"])
    od = gpd.read_file(od_path)[["district", "geometry"]]

    centroids = gdf[["Block_ID", "geometry"]].copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids = centroids.to_crs(od.crs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        joined = gpd.sjoin(centroids, od[["district", "geometry"]],
                           how="left", predicate="within")

    # Drop duplicate matches (rare boundary cases) and keep first
    joined = joined.drop_duplicates(subset=["Block_ID"])

    # Fallback for centroids outside all OD cells (~0.6%)
    missing_mask = joined["district"].isna()
    if missing_mask.any():
        missing_gdf = centroids[missing_mask.values].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nearest = gpd.sjoin_nearest(missing_gdf, od[["district", "geometry"]],
                                        how="left")
        nearest = nearest.drop_duplicates(subset=["Block_ID"])
        joined.loc[missing_mask, "district"] = nearest["district"].values

    id_to_district = dict(
        zip(joined["Block_ID"].astype("int64"), joined["district"].fillna("unknown"))
    )
    return np.array([id_to_district.get(int(bid), "unknown") for bid in block_id_order])


def _district_masks(
    districts: np.ndarray, holdout: str
):
    """Return (train_mask, val_mask) where validation = blocks in `holdout` district."""
    available = sorted(set(districts))
    if holdout not in available:
        raise ValueError(
            f"District '{holdout}' not found. Available: {available}"
        )
    val_mask = torch.tensor(districts == holdout, dtype=torch.bool)
    if val_mask.all():
        raise ValueError(f"All blocks belong to district '{holdout}'; cannot train.")
    return ~val_mask, val_mask


def _build_knn_mobility_edges(
    data_dir: Path, block_id_order: np.ndarray, k: int = 30
) -> Optional[torch.Tensor]:
    """Build a k-nearest-centroid edge index for mobility routing.

    Unlike Neighbor80 (80 m proximity), this graph connects each block to its
    k spatially closest blocks (~2 km at k=30), matching typical intra-city
    commute distances in Shenzhen.  Used by AggregateVitality for local softmax.
    Returns None if geopandas / scipy are unavailable.
    """
    shp_path = data_dir / SHP_FILE
    if not shp_path.exists():
        return None
    try:
        import geopandas as gpd
        from scipy.spatial import cKDTree
    except ImportError:
        return None

    gdf = gpd.read_file(shp_path)
    gdf["Block_ID"] = _canonical_block_id(gdf["Block_ID"])
    id_to_idx = {int(bid): idx for idx, bid in enumerate(block_id_order)}

    # Build centroid array ordered by dataset row index
    n = len(block_id_order)
    centroids = np.zeros((n, 2), dtype="float64")
    for _, row in gdf.iterrows():
        bid = int(row.Block_ID)
        if bid in id_to_idx:
            c = row.geometry.centroid
            centroids[id_to_idx[bid]] = [c.x, c.y]

    tree = cKDTree(centroids)
    _, nbr_indices = tree.query(centroids, k=k + 1)   # k+1: first result is self

    src_list, dst_list = [], []
    for i, neighbors in enumerate(nbr_indices):
        for j in neighbors[1:]:    # skip self
            src_list.append(i)
            dst_list.append(int(j))

    return torch.tensor([src_list, dst_list], dtype=torch.long)


def _build_spatial_edges(
    data_dir: Path, block_id_order: np.ndarray
) -> Optional[torch.Tensor]:
    """Parse pre-computed Neighbor80 adjacency from the block shapefile.

    Returns a (2, E) int64 edge_index (both directions, no self-loops),
    or None if geopandas is unavailable or the shapefile doesn't exist.
    block_id_order gives Block_ID values in the same row order as the dataset.
    """
    shp_path = data_dir / SHP_FILE
    if not shp_path.exists():
        return None
    try:
        import geopandas as gpd
    except ImportError:
        return None

    gdf = gpd.read_file(shp_path)
    gdf["Block_ID"] = _canonical_block_id(gdf["Block_ID"])

    id_to_idx = {int(bid): idx for idx, bid in enumerate(block_id_order)}

    src_list, dst_list = [], []
    for _, row in gdf.iterrows():
        src_bid = int(row.Block_ID)
        if src_bid not in id_to_idx:
            continue
        src_idx = id_to_idx[src_bid]
        nbr_str = row.get("Neighbor80", "")
        if pd.isna(nbr_str):
            continue
        for token in str(nbr_str).split(","):
            token = token.strip()
            if not token.isdigit():
                continue
            dst_bid = int(token)
            if dst_bid == src_bid or dst_bid not in id_to_idx:
                continue
            dst_idx = id_to_idx[dst_bid]
            src_list.append(src_idx)
            dst_list.append(dst_idx)

    if not src_list:
        return None

    # Ensure both directions and deduplicate
    both_src = torch.tensor(src_list + dst_list, dtype=torch.long)
    both_dst = torch.tensor(dst_list + src_list, dtype=torch.long)
    edge_index = torch.stack([both_src, both_dst], dim=0)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def _count_poi_per_block_csv(data_dir: Path) -> Optional[pd.DataFrame]:
    """Count POIs per block from 2026 CSV files using wgslng/wgslat coordinates.

    CSV files carry WGS84 (EPSG:4326) lon/lat; blocks are in EPSG:3857.
    Points are reprojected before spatial join.
    Returns a DataFrame [Block_ID, poi_restaurant, ...] or None on failure.
    """
    poi_dir = data_dir / POI_DIR_2026
    shp_path = data_dir / SHP_FILE
    if not poi_dir.exists() or not shp_path.exists():
        return None
    try:
        import geopandas as gpd
    except ImportError:
        return None

    blocks = gpd.read_file(shp_path)
    blocks["Block_ID"] = _canonical_block_id(blocks["Block_ID"])
    all_block_ids = blocks["Block_ID"].unique()

    col_frames = {}
    for col_name, filename in POI_CATEGORIES_2026.items():
        fp = poi_dir / filename
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp, encoding="utf-8-sig", usecols=["wgslng", "wgslat"])
            df = df.dropna(subset=["wgslng", "wgslat"])
            poi = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["wgslng"], df["wgslat"]),
                crs="EPSG:4326",
            ).to_crs(blocks.crs)
            joined = gpd.sjoin(
                blocks[["Block_ID", "geometry"]],
                poi[["geometry"]],
                how="left",
                predicate="contains",
            )
            matched = joined.dropna(subset=["index_right"])
            counts = matched.groupby("Block_ID").size()
            counts = counts.reindex(all_block_ids, fill_value=0).rename(col_name)
            col_frames[col_name] = counts
        except Exception:
            continue

    if not col_frames:
        return None
    result = pd.DataFrame(col_frames).reset_index()
    result.rename(columns={"index": "Block_ID"}, inplace=True)
    return result


def _count_poi_per_block(data_dir: Path) -> Optional[pd.DataFrame]:
    """Count POIs per block for each category in POI_CATEGORIES.

    Returns a DataFrame with columns [Block_ID, poi_medical, poi_scenic, ...],
    or None if geopandas is unavailable or the POI directory doesn't exist.
    Blocks with no POIs of a given type receive a count of 0.
    """
    poi_dir = data_dir / POI_DIR
    shp_path = data_dir / SHP_FILE
    if not poi_dir.exists() or not shp_path.exists():
        return None
    try:
        import geopandas as gpd
    except ImportError:
        return None

    blocks = gpd.read_file(shp_path)
    blocks["Block_ID"] = _canonical_block_id(blocks["Block_ID"])
    all_block_ids = blocks["Block_ID"].unique()

    col_frames = {}
    for col_name, filename in POI_CATEGORIES.items():
        fp = poi_dir / filename
        if not fp.exists():
            continue
        try:
            poi = gpd.read_file(fp)
            if poi.crs is None:
                poi = poi.set_crs("EPSG:4326")
            poi = poi.to_crs(blocks.crs)
            joined = gpd.sjoin(
                blocks[["Block_ID", "geometry"]],
                poi[["geometry"]],
                how="left",
                predicate="contains",
            )
            matched = joined.dropna(subset=["index_right"])
            counts = matched.groupby("Block_ID").size()
            counts = counts.reindex(all_block_ids, fill_value=0).rename(col_name)
            col_frames[col_name] = counts
        except Exception:
            continue

    if not col_frames:
        return None

    result = pd.DataFrame(col_frames).reset_index()
    result.rename(columns={"index": "Block_ID"}, inplace=True)
    return result


_OD_FILES = {
    "arr_wd": "LBS原始数据/工作日到达人口_网格.shp",
    "dep_wd": "LBS原始数据/工作日出发人口_网格.shp",
    "arr_we": "LBS原始数据/周末到达人口_网格.shp",
    "dep_we": "LBS原始数据/周末出发人口_网格.shp",
}


def _aggregate_od_to_blocks(data_dir: Path) -> Optional[pd.DataFrame]:
    """Aggregate grid-level arrival/departure flows to block level.

    Spatial-joins ~90k LBS grid centroids to 3023 blocks and sums hourly
    flow counts per block.  Returns a DataFrame with columns:

        Block_ID,
        od_arr_wd_h00..h23  (weekday hourly arrivals,   24 cols)
        od_dep_wd_h00..h23  (weekday hourly departures, 24 cols)
        od_arr_we_h00..h23  (weekend  hourly arrivals,  24 cols)
        od_dep_we_h00..h23  (weekend  hourly departures,24 cols)

    Returns None if geopandas is unavailable or the OD files are missing.
    Blocks without any overlapping grids receive zeros.
    """
    shp_path = data_dir / SHP_FILE
    # Check at least one OD file exists before importing geopandas
    if not shp_path.exists() or not any(
        (data_dir / p).exists() for p in _OD_FILES.values()
    ):
        return None
    try:
        import geopandas as gpd
    except ImportError:
        return None

    blocks = gpd.read_file(shp_path)
    blocks["Block_ID"] = _canonical_block_id(blocks["Block_ID"])
    all_block_ids = sorted(blocks["Block_ID"].unique())

    # Build grid-to-block mapping once (reused for all four OD layers).
    # Use the first available OD file to establish the grid index.
    first_path = next(
        (data_dir / p for p in _OD_FILES.values() if (data_dir / p).exists()), None
    )
    if first_path is None:
        return None

    ref_gdf = gpd.read_file(first_path, columns=["grid_id", "geometry"])
    ref_gdf = ref_gdf.to_crs(blocks.crs)
    ref_gdf["centroid"] = ref_gdf.geometry.centroid
    centroids = ref_gdf[["grid_id"]].copy()
    centroids["geometry"] = ref_gdf["centroid"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mapping = gpd.sjoin(
            centroids.set_geometry("geometry"),
            blocks[["Block_ID", "geometry"]],
            how="inner",
            predicate="within",
        )[["grid_id", "Block_ID"]]

    grid_to_block = mapping.set_index("grid_id")["Block_ID"].to_dict()

    result = pd.DataFrame({"Block_ID": all_block_ids}).set_index("Block_ID")

    for tag, rel_path in _OD_FILES.items():
        fp = data_dir / rel_path
        if not fp.exists():
            continue
        gdf = gpd.read_file(fp)
        h_cols = [f"h{h:02d}" for h in range(24)]
        h_cols_present = [c for c in h_cols if c in gdf.columns]
        if not h_cols_present:
            continue

        gdf["Block_ID"] = gdf["grid_id"].map(grid_to_block)
        gdf_matched = gdf.dropna(subset=["Block_ID"])
        gdf_matched = gdf_matched.copy()
        gdf_matched["Block_ID"] = gdf_matched["Block_ID"].astype("int64")

        agg = gdf_matched.groupby("Block_ID")[h_cols_present].sum()
        for h_col in h_cols_present:
            out_col = f"od_{tag}_{h_col}"
            result[out_col] = agg[h_col].reindex(result.index, fill_value=0.0)

    result = result.reset_index()
    od_cols = [c for c in result.columns if c.startswith("od_")]
    if not od_cols:
        return None
    return result


def load_shenzhen_vitality_data(
    data_dir="data_shenzhen",
    validation_fraction: float = 0.2,
    seed: int = 42,
    split_strategy: str = "random",
    holdout_district: Optional[str] = None,
) -> ShenzhenVitalityDataset:
    """Read block attributes and LBS curves and prepare supervised tensors.

    Feature records duplicated by upstream spatial joins are collapsed by
    `Block_ID` using the mean of numeric attributes. The prediction target is
    the hourly population present in each block for weekdays and weekends.
    """
    data_dir = Path(data_dir)
    feature_path = data_dir / "街坊_数据连接.csv"
    target_path = data_dir / "街坊_LBS统计.csv"
    if not feature_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            "Expected 街坊_数据连接.csv and 街坊_LBS统计.csv under " f"{data_dir}."
        )

    feature_frame = pd.read_csv(feature_path, encoding="utf-8-sig")
    target_frame = pd.read_csv(target_path, encoding="utf-8-sig")
    feature_frame["Block_ID"] = _canonical_block_id(feature_frame["Block_ID"])
    target_frame["Block_ID"] = _canonical_block_id(target_frame["Block_ID"])

    portrait_path = data_dir / PORTRAIT_FILE
    if portrait_path.exists():
        portrait_frame = pd.read_csv(portrait_path, encoding="utf-8-sig")
        portrait_frame["Block_ID"] = _canonical_block_id(portrait_frame["Block_ID"])
        available = [c for c in PORTRAIT_FEATURES if c in portrait_frame.columns]
        if available:
            portrait_sub = portrait_frame[["Block_ID"] + available].copy()
            # portrait has one row per block; left-join preserves feature_frame duplicates
            feature_frame = feature_frame.merge(portrait_sub, on="Block_ID", how="left")

    missing_targets = [name for name in TARGET_NAMES if name not in target_frame.columns]
    if missing_targets:
        raise ValueError(f"Missing LBS target columns: {missing_targets}")

    numeric_candidates = feature_frame.drop(
        columns=[name for name in NON_NUMERIC_FEATURES if name in feature_frame.columns]
    ).apply(pd.to_numeric, errors="coerce")
    feature_names = [
        name
        for name in numeric_candidates.columns
        if name != "Block_ID" and not numeric_candidates[name].isna().all()
    ]
    numeric_features = feature_frame[["Block_ID"]].copy()
    for name in feature_names:
        numeric_features[name] = numeric_candidates[name]

    # Spatial joins can produce several source rows for the same city block.
    numeric_features = numeric_features.groupby("Block_ID", as_index=False).mean()

    # Optionally enrich with POI counts (requires geopandas + shapefile).
    poi_df = _count_poi_per_block(data_dir)
    if poi_df is not None:
        poi_cols = [c for c in poi_df.columns if c != "Block_ID"]
        numeric_features = numeric_features.merge(poi_df, on="Block_ID", how="left")
        for col in poi_cols:
            # POI counts are power-law distributed (many zeros, few extreme values).
            # log1p compresses the range (e.g. 160→5.1) so normalization is stable.
            numeric_features[col] = np.log1p(numeric_features[col].fillna(0.0))
        feature_names = feature_names + poi_cols

    # Enrich with 2026 POI categories from CSV (餐饮/购物/生活/交通/公司/体育/住宿/科教).
    poi_csv_df = _count_poi_per_block_csv(data_dir)
    if poi_csv_df is not None:
        poi_csv_cols = [c for c in poi_csv_df.columns if c != "Block_ID"]
        numeric_features = numeric_features.merge(poi_csv_df, on="Block_ID", how="left")
        for col in poi_csv_cols:
            numeric_features[col] = np.log1p(numeric_features[col].fillna(0.0))
        feature_names = feature_names + poi_csv_cols

    # Enrich with aggregated OD flow features (Phase 3).
    # Each block gets 24-hour arrival + departure profiles for weekday and weekend
    # (96 features total) derived from the raw LBS grid-level OD data.  These
    # directly encode each block's empirical temporal activity pattern, giving
    # scale_net a strong signal it cannot infer from static building/POI features.
    od_df = _aggregate_od_to_blocks(data_dir)
    if od_df is not None:
        od_cols = [c for c in od_df.columns if c != "Block_ID"]
        numeric_features = numeric_features.merge(od_df, on="Block_ID", how="left")
        for col in od_cols:
            numeric_features[col] = np.log1p(numeric_features[col].fillna(0.0))
        feature_names = feature_names + od_cols
        print(f"[data] OD features loaded: {len(od_cols)} cols "
              f"({od_df['Block_ID'].nunique()}/{len(numeric_features)} blocks matched)")

    target_frame = target_frame[["Block_ID", *TARGET_NAMES]].drop_duplicates("Block_ID")
    merged = numeric_features.merge(target_frame, on="Block_ID", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No matching Block_ID values found between features and LBS targets.")

    # Extract raw demographic population counts before normalization.
    # Fallback to zeros for any missing column (e.g., synthetic test fixtures).
    demo_array = np.stack(
        [
            merged[col].fillna(0.0).clip(lower=0.0).to_numpy(dtype="float32")
            if col in merged.columns
            else np.zeros(len(merged), dtype="float32")
            for col in DEMO_GROUPS
        ],
        axis=1,
    )  # (N_blocks, N_DEMO_GROUPS)

    feature_values = merged[feature_names].replace([np.inf, -np.inf], np.nan)
    feature_values = feature_values.fillna(feature_values.median()).fillna(0.0)
    target_values = merged[TARGET_NAMES].apply(pd.to_numeric, errors="coerce")
    if target_values.isna().any().any():
        raise ValueError("LBS target columns contain missing or non-numeric values.")

    block_id_order = merged["Block_ID"].to_numpy(dtype="int64", copy=True)
    districts = _load_block_districts(data_dir, block_id_order)

    if split_strategy == "district":
        if districts is None:
            raise ValueError(
                "District data unavailable (LBS原始数据/工作日出发人口_网格.shp not found). "
                "Use split_strategy='random'."
            )
        if holdout_district is None:
            raise ValueError("holdout_district must be specified when split_strategy='district'.")
        train_mask, validation_mask = _district_masks(districts, holdout_district)
    else:
        train_mask, validation_mask = _training_masks(len(merged), validation_fraction, seed)

    train_indices = train_mask.numpy()
    feature_array = feature_values.to_numpy(dtype="float32", copy=True)
    target_array = target_values.to_numpy(dtype="float32", copy=True)
    target_log_array = np.log1p(np.maximum(target_array, 0.0))

    feature_mean = feature_array[train_indices].mean(axis=0)
    feature_scale = feature_array[train_indices].std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    target_mean = target_log_array[train_indices].mean(axis=0)
    target_scale = target_log_array[train_indices].std(axis=0)
    target_scale[target_scale < 1e-6] = 1.0

    features = (feature_array - feature_mean) / feature_scale
    vitality_scaled = (target_log_array - target_mean) / target_scale

    edge_index          = _build_spatial_edges(data_dir, block_id_order)
    edge_index_mobility = _build_knn_mobility_edges(data_dir, block_id_order, k=30)

    return ShenzhenVitalityDataset(
        block_ids=torch.from_numpy(block_id_order),
        features=torch.from_numpy(features),
        vitality=torch.from_numpy(target_array),
        vitality_scaled=torch.from_numpy(vitality_scaled),
        demo_weights=torch.from_numpy(demo_array),
        train_mask=train_mask,
        validation_mask=validation_mask,
        feature_names=feature_names,
        target_names=list(TARGET_NAMES),
        feature_mean=torch.from_numpy(feature_mean.astype("float32")),
        feature_scale=torch.from_numpy(feature_scale.astype("float32")),
        target_mean=torch.from_numpy(target_mean.astype("float32")),
        target_scale=torch.from_numpy(target_scale.astype("float32")),
        edge_index=edge_index,
        edge_index_mobility=edge_index_mobility,
        districts=districts,
    )


def _property(name, value):
    value = value.float() if torch.is_tensor(value) and value.dtype != torch.bool else value
    return {
        "name": name,
        "dtype": "float",
        "shape": list(value.shape),
        "learnable": False,
        "initialization_function": None,
        "value": value,
    }


def _long_property(name, value: torch.Tensor):
    """Property entry for integer tensors (e.g., edge_index).

    The initializer converts via src_val * ones(...), which would change dtype.
    Storing as float32 is lossless for block IDs ≤ 3023 (< 2^24); substeps
    cast back to long on use.
    """
    return _property(name, value.float())


def build_config(
    dataset: ShenzhenVitalityDataset,
    hidden_dim: int = 64,
    device: str = "auto",
):
    """Build an in-memory AgentTorch config for the agent-based Shenzhen model.

    Agents are resident groups indexed by (block, demographic_cohort).
    Vitality emerges from their movement decisions rather than being
    predicted directly.
    """
    n_agents = dataset.num_blocks * N_DEMO_GROUPS

    # Resident agent tensors: (block × demo_group) ordering
    block_idx = torch.arange(dataset.num_blocks)
    demo_idx  = torch.arange(N_DEMO_GROUPS)
    home_block = block_idx.unsqueeze(1).expand(-1, N_DEMO_GROUPS).reshape(-1).float()
    demo_group = demo_idx.unsqueeze(0).expand(dataset.num_blocks, -1).reshape(-1).float()
    weight     = dataset.demo_weights.reshape(-1).float()

    zeros_vitality = torch.zeros(dataset.num_blocks, len(dataset.target_names))

    env_state = {
        "block_features":            _property("block_features",            dataset.features),
        "observed_vitality":         _property("observed_vitality",         dataset.vitality),
        "observed_vitality_scaled":  _property("observed_vitality_scaled",  dataset.vitality_scaled),
        "predicted_vitality":        _property("predicted_vitality",        zeros_vitality.clone()),
        "predicted_vitality_scaled": _property("predicted_vitality_scaled", zeros_vitality.clone()),
        "target_mean":               _property("target_mean",               dataset.target_mean),
        "target_scale":              _property("target_scale",              dataset.target_scale),
    }
    if dataset.has_spatial:
        env_state["edge_index"] = _long_property("edge_index", dataset.edge_index)
    if dataset.has_mobility:
        env_state["edge_index_mobility"] = _long_property(
            "edge_index_mobility", dataset.edge_index_mobility
        )

    move_policy_inputs = {"block_features": "environment/block_features"}
    if dataset.has_mobility:
        # Use the 2 km k-NN graph for attention-based spatial feature enrichment.
        # Wider context (vs Neighbor80 at 80 m) captures functional neighbourhood
        # patterns (commercial, residential clusters) that Neighbor80 misses.
        move_policy_inputs["edge_index"] = "environment/edge_index_mobility"
    elif dataset.has_spatial:
        move_policy_inputs["edge_index"] = "environment/edge_index"

    aggregate_inputs = {
        "home_block":                "agents/residents/home_block",
        "demo_group":                "agents/residents/demo_group",
        "weight":                    "agents/residents/weight",
        "target_mean":               "environment/target_mean",
        "target_scale":              "environment/target_scale",
        "predicted_vitality":        "environment/predicted_vitality",
        "predicted_vitality_scaled": "environment/predicted_vitality_scaled",
    }
    # Local softmax requires real OD origin-destination pairs to work correctly.
    # High-vitality attractors draw population city-wide, not just from 30 neighbours;
    # restricting routing to a 2km k-NN graph consistently hurts the top tier.

    return {
        "simulation_metadata": {
            "calibration": False,
            "device": device,
            "num_agents": n_agents,
            "num_blocks": dataset.num_blocks,
            "num_features": dataset.num_features,
            "has_spatial":   dataset.has_spatial,
            "has_mobility":  dataset.has_mobility,
            "num_demo_groups": N_DEMO_GROUPS,
            "num_targets": len(dataset.target_names),
            "hidden_dim": hidden_dim,
            "temporal_rank": 8,
            "num_episodes": 1,
            "num_steps_per_episode": 1,
            "num_substeps_per_step": 1,
        },
        "state": {
            "environment": env_state,
            "agents": {
                "residents": {
                    "number": n_agents,
                    "properties": {
                        "home_block": _property("home_block", home_block),
                        "demo_group": _property("demo_group", demo_group),
                        "weight":     _property("weight",     weight),
                    },
                }
            },
            "objects": None,
            "network": {},
        },
        "substeps": {
            "0": {
                "name": "Simulate resident movement",
                "description": (
                    "Each resident group decides home/away and distributes "
                    "across blocks; vitality is the emergent aggregate."
                ),
                "active_agents": ["residents"],
                "observation": {"residents": None},
                "policy": {
                    "residents": {
                        "move_policy": {
                            "generator": "MovePolicy",
                            "input_variables": move_policy_inputs,
                            "output_variables": ["p_home", "attract_logits", "block_log_scale"],
                            "arguments": None,
                        }
                    }
                },
                "transition": {
                    "aggregate_vitality": {
                        "generator": "AggregateVitality",
                        "input_variables": aggregate_inputs,
                        "output_variables": [
                            "predicted_vitality",
                            "predicted_vitality_scaled",
                        ],
                        "arguments": None,
                    }
                },
                "reward": None,
            }
        },
    }
