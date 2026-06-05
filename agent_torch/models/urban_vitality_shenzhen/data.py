"""深圳城市活力模型的数据准备与仿真配置模块。

本模块负责两件事：
    1. 数据加载与预处理（load_shenzhen_vitality_data）：
       从 CSV / SHP 文件中读取建筑特征、POI、OD 流量和 LBS 活力目标，
       做归一化、构建空间图，封装成 ShenzhenVitalityDataset。

    2. 仿真配置构建（build_config）：
       把数据集转化为 AgentTorch Runner 需要的字典格式，
       定义 Agent 属性、环境状态、以及 Substep 的输入输出关系。

数据文件依赖（位于 data_shenzhen/ 目录）：
    街坊_数据连接.csv   — 街坊建筑与用地特征（必须）
    街坊_LBS统计.csv    — 每小时 LBS 实时人口（预测目标，必须）
    街坊_人口画像.csv   — 人口统计特征（可选，缺失则不加入特征）
    POI2026/POI_CSV/    — 2026 年 POI 数据（可选，需 geopandas）
    LBS原始数据/*.shp   — OD 到达/出发流量网格（可选，需 geopandas）
    街坊范围shp/        — 街坊边界 shapefile（空间图构建用，可选）
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import warnings
import pandas as pd
import torch


# ────────────────────────── 全局常量 ──────────────────────────────────────────

# 预测目标列名：工作日 0–23 时 + 周末 0–23 时，共 48 列
# 对应 街坊_LBS统计.csv 中的 WD_C_00...WD_C_23, WE_C_00...WE_C_23
TARGET_NAMES = [
    *(f"WD_C_{hour:02d}" for hour in range(24)),   # 工作日各小时实时人口
    *(f"WE_C_{hour:02d}" for hour in range(24)),   # 周末各小时实时人口
]

# 这些列不能作为数值特征：Block_ID 是 ID 列，H_List/FL_List 是列表字符串，RoadLevel 是类别
NON_NUMERIC_FEATURES = {"Block_ID", "H_List", "FL_List", "RoadLevel"}

# 人口群体分组（来源：街坊_数据连接.csv 中已有的列）
# 年龄段定义：青少年与儿童=0-19岁，青年=20-34岁，中年=35-59岁，老年=60+岁
# 每个（街坊, 群体）对构成一个 Agent，weight = 该群体在该街坊的实际人口数
DEMO_GROUPS = ["青少年与儿童", "青年", "中年", "老年"]
N_DEMO_GROUPS = len(DEMO_GROUPS)   # 4

PORTRAIT_FILE = "街坊_人口画像.csv"
SHP_FILE      = "街坊范围shp/深圳_街坊_Pro.shp"   # 街坊边界，用于空间连接

# POI 数据目录（2026 年 CSV 格式，含经纬度列 wgslng/wgslat）
POI_DIR_2026 = "POI2026/POI_CSV"
# 全部 12 个 POI 类别均从 CSV 读取（统一坐标系转换流程）
# 括号内数字为全深圳 POI 总数，反映各类别数据密度
POI_CATEGORIES_2026 = {
    "poi_restaurant": "深圳市-餐饮服务.csv",         # 108 953 个 — 餐饮，活力最强驱动因素
    "poi_shopping":   "深圳市-购物服务.csv",          # 162 881 个 — 零售，白天吸引力
    "poi_life_svc":   "深圳市-生活服务.csv",          # 101 544 个 — 日常生活服务
    "poi_transport":  "深圳市-交通设施服务.csv",      #  53 020 个 — 交通枢纽，可达性代理
    "poi_company":    "深圳市-公司企业.csv",           # 130 706 个 — 就业岗位，工作日流量
    "poi_sports":     "深圳市-体育休闲服务.csv",      #  24 409 个 — 休闲设施，周末流量
    "poi_hotel":      "深圳市-住宿服务.csv",           #  21 244 个 — 住宿，过夜活动
    "poi_education":  "深圳市-科教文化服务.csv",      #  24 709 个 — 文化教育
    "poi_medical":    "深圳市-医疗保健服务.csv",      #  26 369 个 — 医疗设施，高密度信号
    "poi_scenic":     "深圳市-风景名胜.csv",           #   5 875 个 — 景点，休闲目的地
    "poi_auto_sales": "深圳市-汽车销售.csv",           #   4 297 个 — 商业区代理指标
    "poi_motorcycle": "深圳市-摩托车服务.csv",         #     764 个 — 本地出行代理
}

# 从街坊_人口画像.csv 中精选的人口统计特征，按预测价值排序
# CZZL = 常住人口（总量），按 5 岁年龄段分列；其余列见字段解释.xlsx
PORTRAIT_FEATURES = [
    "CZZL20_24", "CZZL25_29", "CZZL30_34", "CZZL35_39",  # 主力劳动年龄段（20-39 岁）
    "CZZL40_44", "CZZL45_49", "CZZL50_54", "CZZL55_59",  # 中晚期职业年龄（40-59 岁）
    "XB1CZRK",                                             # 男性常住人口数
    "XB2CZRK",                                             # 女性常住人口数
    "就业人",                                               # 就业人口数
    "CZRKMD",                                               # 常住人口密度（人/平方公里）
    "FHJCZRKZSL",                                           # 非户籍常住人口（外来人口）
    "JZSJD1RKSL",                                           # 居住不足 1 年的人口（新迁入）
    "JZSJD5RKSL",                                           # 居住 10 年以上的人口（长期稳定）
]


# ────────────────────────── 数据容器 ──────────────────────────────────────────

@dataclass
class ShenzhenVitalityDataset:
    """Agent 仿真所需的全部张量和元数据，打包在一个数据类里。

    各字段说明：
        block_ids:        (N_blocks,)          街坊编号（对应 CSV 中的 Block_ID 列）
        features:         (N_blocks, F)        归一化后的输入特征矩阵
        vitality:         (N_blocks, 48)       原始 LBS 实时人口数（预测目标，未归一化）
        vitality_scaled:  (N_blocks, 48)       log1p 后再标准化的活力值（训练时用）
        demo_weights:     (N_blocks, 4)        各人口群体的原始人口数，作为 Agent 的 weight
        train_mask:       (N_blocks,) bool     True = 训练集街坊
        validation_mask:  (N_blocks,) bool     True = 验证集街坊（与 train_mask 互补）
        feature_names:    长度 F 的字符串列表  与 features 各列一一对应
        target_names:     长度 48 的字符串列表 与 vitality 各列对应（WD_C_00...WE_C_23）
        feature_mean/scale: 仅用训练集计算的标准化统计量（防止数据泄露）
        target_mean/scale:  同上，用于 log1p(活力) 的标准化
        edge_index:           (2, E) 邻居图（Neighbor80，80m 以内相邻街坊）
        edge_index_mobility:  (2, E) 通勤图（k-NN，30 个最近街坊，约 2km 范围）
        districts:            (N_blocks,) 每个街坊所属的行政区名称（如"福田区"）
    """

    block_ids: torch.Tensor
    features: torch.Tensor
    vitality: torch.Tensor
    vitality_scaled: torch.Tensor
    demo_weights: torch.Tensor
    train_mask: torch.Tensor
    validation_mask: torch.Tensor
    feature_names: List[str]
    target_names: List[str]
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    edge_index: Optional[torch.Tensor] = field(default=None)
    edge_index_mobility: Optional[torch.Tensor] = field(default=None)
    districts: Optional[np.ndarray] = field(default=None)

    @property
    def num_blocks(self) -> int:
        """数据集中的街坊总数（features 的行数）。"""
        return int(self.features.shape[0])

    @property
    def num_features(self) -> int:
        """输入特征的维度 F（建筑 + POI + OD + 人口统计列数之和）。"""
        return int(self.features.shape[1])

    @property
    def has_spatial(self) -> bool:
        """是否构建了 Neighbor80 邻接图（需要 geopandas + shapefile）"""
        return self.edge_index is not None

    @property
    def has_mobility(self) -> bool:
        """是否构建了 k-NN 通勤图（需要 geopandas + scipy）"""
        return self.edge_index_mobility is not None


def _canonical_block_id(values: pd.Series) -> pd.Series:
    """把 Block_ID 列统一转换为 int64，避免不同来源（float/str）混用导致合并失败。"""
    return pd.to_numeric(values, errors="raise").round().astype("int64")


def _training_masks(num_blocks: int, validation_fraction: float, seed: int):
    """用固定随机种子生成训练集/验证集的布尔掩码（随机划分策略）。"""
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
    """通过空间连接，把每个街坊分配到深圳行政区（如"福田区"、"南山区"）。

    方法：
        1. 读取街坊 shapefile，计算每个街坊的几何中心（centroid）
        2. 将中心点与 OD 网格 shapefile 做空间连接（predicate="within"）
           OD 网格文件的 district 字段记录了每个网格所属的行政区
        3. 约 0.6% 的街坊中心点落在所有 OD 网格之外（边界角落），
           用 sjoin_nearest 找最近网格补填

    返回：
        与 block_id_order 长度相同的字符串数组（如 ["福田区", "南山区", ...]），
        当 geopandas 不可用或文件缺失时返回 None（不影响训练，只影响分区holdout）
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

    # 极少数情况下，一个街坊中心可能落在多个 OD 网格的边界上，产生重复匹配，取第一条
    joined = joined.drop_duplicates(subset=["Block_ID"])

    # 约 0.6% 的街坊中心点在所有 OD 网格之外（深圳边缘角落），用最近邻补填
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
    """按行政区划分训练集/验证集（空间留出验证法）。

    把指定行政区的所有街坊作为验证集，其余街坊作为训练集。
    这比随机划分更严格：模型必须对完全没见过的地区做预测，
    是检验模型空间泛化能力的标准方法。
    """
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
    """构建 k-近邻通勤图（edge_index_mobility），用于空间注意力的邻域上下文。

    与 Neighbor80（只连接 80m 以内的紧邻街坊）不同，
    k-NN 图连接每个街坊与其 k=30 个空间最近街坊（约 2km 范围），
    匹配深圳市内通勤距离的典型尺度，有助于捕捉功能区聚集效应。

    返回：
        (2, E) long 型张量，edge[0]=源节点索引，edge[1]=目标节点索引
        当 geopandas 或 scipy 不可用时返回 None（模型退化为无空间注意力）

    注意：
        k+1 是因为 cKDTree.query 的结果第一个总是节点自身（距离=0），需跳过。
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

    # 按数据集行索引顺序排列各街坊中心点坐标
    n = len(block_id_order)
    centroids = np.zeros((n, 2), dtype="float64")
    for _, row in gdf.iterrows():
        bid = int(row.Block_ID)
        if bid in id_to_idx:
            c = row.geometry.centroid
            centroids[id_to_idx[bid]] = [c.x, c.y]

    # cKDTree 是高效的 KD 树，查询每个街坊的 k+1 个最近邻（第 0 个是自身）
    tree = cKDTree(centroids)
    _, nbr_indices = tree.query(centroids, k=k + 1)

    src_list, dst_list = [], []
    for i, neighbors in enumerate(nbr_indices):
        for j in neighbors[1:]:    # 跳过第 0 个（自身）
            src_list.append(i)
            dst_list.append(int(j))

    return torch.tensor([src_list, dst_list], dtype=torch.long)


def _build_spatial_edges(
    data_dir: Path, block_id_order: np.ndarray
) -> Optional[torch.Tensor]:
    """从街坊 shapefile 的 Neighbor80 字段解析 80m 邻接图。

    Neighbor80 是 shapefile 中预计算的字段，存储了每个街坊 80m 以内所有邻居的
    Block_ID，以逗号分隔（如 "1023,1024,1031"）。

    返回：
        (2, E) long 型张量，双向边（A→B 和 B→A 都会包含），无自环，并去重。
        当 geopandas 不可用或 shapefile 不存在时返回 None。

    注意：
        torch.unique(dim=1) 对列去重，因为 shapefile 中可能同时记录了 A→B 和 B→A，
        如果不去重，同一条边会被计算两次，权重加倍。
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

    # 建立 Block_ID → 数据集行索引 的映射（因为行索引才是张量用的下标）
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
        # 解析逗号分隔的邻居 Block_ID 列表
        for token in str(nbr_str).split(","):
            token = token.strip()
            if not token.isdigit():
                continue
            dst_bid = int(token)
            if dst_bid == src_bid or dst_bid not in id_to_idx:
                continue    # 跳过自环和不在数据集里的邻居
            dst_idx = id_to_idx[dst_bid]
            src_list.append(src_idx)
            dst_list.append(dst_idx)

    if not src_list:
        return None

    # 拼接正向边和反向边，然后去重，保证图是无向的
    both_src = torch.tensor(src_list + dst_list, dtype=torch.long)
    both_dst = torch.tensor(dst_list + src_list, dtype=torch.long)
    edge_index = torch.stack([both_src, both_dst], dim=0)
    edge_index = torch.unique(edge_index, dim=1)   # 按列去重（每列是一条边）
    return edge_index


def _count_poi_per_block_csv(data_dir: Path) -> Optional[pd.DataFrame]:
    """统计每个街坊内各类 POI 的数量（从 2026 年 CSV 文件读取）。

    流程：
        1. 读取街坊 shapefile（EPSG:3857 坐标系）
        2. 对每个 POI 类别的 CSV 文件：
           a. 读取 wgslng/wgslat 列（WGS84 经纬度），构建点几何
           b. 坐标系转换：EPSG:4326 → 街坊的坐标系（EPSG:3857）
           c. 空间连接（sjoin predicate="contains"）：判断哪个 POI 落在哪个街坊内
           d. 按 Block_ID 分组计数，没有该类 POI 的街坊填 0
        3. 所有类别合并为一张宽表 [Block_ID, poi_restaurant, poi_shopping, ...]

    返回：
        DataFrame，或者当目录/文件不存在或 geopandas 不可用时返回 None。

    注意：
        单个类别的 CSV 处理失败（如文件格式异常）会被静默跳过（except Exception: continue），
        不影响其他类别的加载。
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



# OD 流量文件映射（四个方向的 LBS 网格级 OD 数据）
# 每个文件包含约 90k 个 500m×500m 网格，每格有 h00~h23 的逐小时流量
_OD_FILES = {
    "arr_wd": "LBS原始数据/工作日到达人口_网格.shp",   # 工作日到达流量
    "dep_wd": "LBS原始数据/工作日出发人口_网格.shp",   # 工作日出发流量
    "arr_we": "LBS原始数据/周末到达人口_网格.shp",     # 周末到达流量
    "dep_we": "LBS原始数据/周末出发人口_网格.shp",     # 周末出发流量
}


def _aggregate_od_to_blocks(data_dir: Path) -> Optional[pd.DataFrame]:
    """把网格级 OD 流量聚合到街坊级别，生成 96 列 OD 特征。

    输出列名格式（共 96 列）：
        od_arr_wd_h00 ~ od_arr_wd_h23  工作日逐小时到达人口（24列）
        od_dep_wd_h00 ~ od_dep_wd_h23  工作日逐小时出发人口（24列）
        od_arr_we_h00 ~ od_arr_we_h23  周末逐小时到达人口（24列）
        od_dep_we_h00 ~ od_dep_we_h23  周末逐小时出发人口（24列）

    这些特征直接编码了每个街坊的实测时序活动模式，
    给 scale_net 提供了强信号（它无法从静态建筑/POI 特征中推断出时间峰值）。

    处理流程：
        1. 读取街坊 shapefile 和第一个可用 OD 文件，建立"网格→街坊"映射
           （所有四个 OD 文件用相同的网格系统，映射只需建立一次）
        2. 对四个 OD 文件分别加载，通过映射关系把 grid_id 转为 Block_ID
        3. 按 Block_ID 对逐小时流量求和，无网格匹配的街坊填 0

    返回：
        DataFrame [Block_ID, od_arr_wd_h00, ...] 或 None（文件缺失/geopandas不可用）
    """
    shp_path = data_dir / SHP_FILE
    # 避免无谓导入：先检查文件是否存在
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

    # 用第一个可用的 OD 文件建立"网格中心点→街坊"映射，四个文件共用
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
        # 用网格中心点做空间连接：判断哪个中心点在哪个街坊内
        mapping = gpd.sjoin(
            centroids.set_geometry("geometry"),
            blocks[["Block_ID", "geometry"]],
            how="inner",         # inner join：只保留有匹配的网格（边缘网格可能落在市外）
            predicate="within",
        )[["grid_id", "Block_ID"]]

    grid_to_block = mapping.set_index("grid_id")["Block_ID"].to_dict()

    # 初始化结果表（索引为 Block_ID，后续逐列填入 OD 特征）
    result = pd.DataFrame({"Block_ID": all_block_ids}).set_index("Block_ID")

    for tag, rel_path in _OD_FILES.items():
        fp = data_dir / rel_path
        if not fp.exists():
            continue
        gdf = gpd.read_file(fp)
        h_cols = [f"h{h:02d}" for h in range(24)]
        h_cols_present = [c for c in h_cols if c in gdf.columns]  # 兼容列名缺失的情况
        if not h_cols_present:
            continue

        # 用映射关系把 grid_id 转为 Block_ID（不在任何街坊内的网格得到 NaN）
        gdf["Block_ID"] = gdf["grid_id"].map(grid_to_block)
        gdf_matched = gdf.dropna(subset=["Block_ID"]).copy()
        gdf_matched["Block_ID"] = gdf_matched["Block_ID"].astype("int64")

        # 同一街坊内的所有网格流量求和
        agg = gdf_matched.groupby("Block_ID")[h_cols_present].sum()
        for h_col in h_cols_present:
            out_col = f"od_{tag}_{h_col}"
            # reindex 确保所有街坊都有值（无匹配网格的街坊填 0）
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
    """读取深圳街坊数据，完成特征工程，返回可直接用于训练的数据集对象。

    处理流程（按顺序）：
        1. 读取基础 CSV（街坊_数据连接.csv + 街坊_LBS统计.csv）
        2. 合并人口画像（街坊_人口画像.csv），可选
        3. 提取数值特征列，排除 ID 列和字符串列
        4. 去重：GIS 空间连接可能产生一个街坊多行，取均值
        5. 加入 POI 计数特征（12 类 CSV POI），可选
        6. 加入 OD 流量特征（96 列逐小时到达/出发），可选
        7. 合并特征和目标，提取人口群体权重（demo_weights）
        8. 归一化：Z-score（特征）和 log1p+Z-score（目标），统计量只从训练集算
        9. 构建空间图（Neighbor80）和通勤图（k-NN），可选
        10. 划分训练/验证集（随机或按行政区）

    关键设计决策：
        - 目标值先取 log1p 再标准化（log1p 压缩幂律分布的极端值）
        - 特征归一化统计量只从训练集计算，验证集不参与（防止数据泄露）
        - 特征缺失值用该列的中位数填充（中位数对极端值比均值更鲁棒）
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

    # GIS 空间连接可能导致同一街坊出现多行（与多个网格匹配），对数值列取均值去重
    numeric_features = numeric_features.groupby("Block_ID", as_index=False).mean()

    # 加入 12 类 POI 计数特征（来源：POI2026/POI_CSV/*.csv）
    # POI 数量呈幂律分布（多数街坊很少，少数街坊极多），log1p 压缩使归一化稳定
    poi_csv_df = _count_poi_per_block_csv(data_dir)
    if poi_csv_df is not None:
        poi_csv_cols = [c for c in poi_csv_df.columns if c != "Block_ID"]
        numeric_features = numeric_features.merge(poi_csv_df, on="Block_ID", how="left")
        for col in poi_csv_cols:
            numeric_features[col] = np.log1p(numeric_features[col].fillna(0.0))
        feature_names = feature_names + poi_csv_cols

    # 加入 OD 流量特征（96 列：4 个方向 × 24 小时）
    # 这些特征直接编码了每个街坊的实测时序活动模式，
    # 给 scale_net 提供了无法从静态特征推断的时段峰值信号
    od_df = _aggregate_od_to_blocks(data_dir)
    if od_df is not None:
        od_cols = [c for c in od_df.columns if c != "Block_ID"]
        numeric_features = numeric_features.merge(od_df, on="Block_ID", how="left")
        for col in od_cols:
            numeric_features[col] = np.log1p(numeric_features[col].fillna(0.0))
        feature_names = feature_names + od_cols
        print(f"[数据] OD 特征已加载：{len(od_cols)} 列，"
              f"{od_df['Block_ID'].nunique()}/{len(numeric_features)} 个街坊有匹配")

    # 只保留 48 个目标列，按 Block_ID 去重（LBS 文件可能有重复行）
    target_frame = target_frame[["Block_ID", *TARGET_NAMES]].drop_duplicates("Block_ID")
    # inner join：只保留同时有特征和目标的街坊；validate 检查是否真正一对一
    merged = numeric_features.merge(target_frame, on="Block_ID", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("特征和 LBS 目标之间没有匹配的 Block_ID，请检查两份 CSV 的 Block_ID 格式。")

    # 在归一化之前提取人口群体的原始人口数（用作 Agent 的 weight）
    # clip(lower=0)：确保人口数非负（极少数数据异常时可能出现负值）
    # 如果某群体列不存在（如单元测试的合成数据），用全零代替
    demo_array = np.stack(
        [
            merged[col].fillna(0.0).clip(lower=0.0).to_numpy(dtype="float32")
            if col in merged.columns
            else np.zeros(len(merged), dtype="float32")
            for col in DEMO_GROUPS
        ],
        axis=1,
    )  # (N_blocks, 4)

    # 把无穷大替换为 NaN，再用每列中位数填充（中位数比均值对极端值更鲁棒）
    feature_values = merged[feature_names].replace([np.inf, -np.inf], np.nan)
    feature_values = feature_values.fillna(feature_values.median()).fillna(0.0)

    target_values = merged[TARGET_NAMES].apply(pd.to_numeric, errors="coerce")
    if target_values.isna().any().any():
        raise ValueError("LBS 目标列包含缺失值或非数值，请检查街坊_LBS统计.csv。")

    block_id_order = merged["Block_ID"].to_numpy(dtype="int64", copy=True)
    # 行政区信息：需要 geopandas + OD shapefile，用于分区留出验证
    districts = _load_block_districts(data_dir, block_id_order)

    # 划分训练/验证集
    if split_strategy == "district":
        if districts is None:
            raise ValueError(
                "行政区数据不可用（找不到 LBS原始数据/工作日出发人口_网格.shp）。"
                "请改用 split_strategy='random'。"
            )
        if holdout_district is None:
            raise ValueError("使用 split_strategy='district' 时必须指定 holdout_district。")
        train_mask, validation_mask = _district_masks(districts, holdout_district)
    else:
        train_mask, validation_mask = _training_masks(len(merged), validation_fraction, seed)

    train_indices = train_mask.numpy()
    feature_array = feature_values.to_numpy(dtype="float32", copy=True)
    target_array = target_values.to_numpy(dtype="float32", copy=True)

    # 目标值先取 log1p，再做 Z-score 标准化
    # maximum(..., 0) 保证参数非负（理论上 LBS 人口数 ≥ 0，但数据异常时可能有负数）
    target_log_array = np.log1p(np.maximum(target_array, 0.0))

    # 标准化统计量只从训练集计算（关键！验证集不能参与，否则产生数据泄露）
    feature_mean  = feature_array[train_indices].mean(axis=0)
    feature_scale = feature_array[train_indices].std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0    # 方差近零的列（常数列）不做缩放
    target_mean   = target_log_array[train_indices].mean(axis=0)
    target_scale  = target_log_array[train_indices].std(axis=0)
    target_scale[target_scale < 1e-6] = 1.0

    # 应用标准化（广播到全量数据，包括验证集）
    features = (feature_array - feature_mean) / feature_scale
    vitality_scaled = (target_log_array - target_mean) / target_scale

    # 构建空间图（可选，需要 geopandas）
    edge_index          = _build_spatial_edges(data_dir, block_id_order)      # Neighbor80 邻接图
    edge_index_mobility = _build_knn_mobility_edges(data_dir, block_id_order, k=30)  # k-NN 通勤图

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
    """把张量/值包装成 AgentTorch 状态属性格式。

    AgentTorch 的 Initializer 用这个格式读取初始状态值并构建状态字典。
    bool 张量不转 float（掩码类属性保持 bool），其余全部转 float32。
    learnable=False 表示这个属性不是可训练参数（由 Substep 更新而非优化器）。
    """
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
    """把整数类型的张量（如 edge_index）包装为状态属性格式。

    AgentTorch Initializer 初始化时会做 src_val * ones(...)，这个操作会把 long 转为 float。
    这里直接存为 float32（街坊索引 ≤ 3023，远小于 float32 的精度上限 2^24），
    Substep 使用时再转回 long（通过 .long()）。
    """
    return _property(name, value.float())


def build_config(
    dataset: ShenzhenVitalityDataset,
    hidden_dim: int = 64,
    device: str = "auto",
):
    """把 ShenzhenVitalityDataset 转化为 AgentTorch Runner 需要的配置字典。

    这个配置字典完整描述了仿真的初始状态和执行逻辑：
        - simulation_metadata：仿真超参数（Agent 数量、特征维度等）
        - state.environment：全局状态张量（特征矩阵、活力目标、空间图等）
        - state.agents.residents：每个 Agent 的属性（家街坊、人口群体、人口数）
        - substeps：定义仿真每步执行哪些 Policy 和 Transition，输入输出是哪些状态

    Agent 的构建逻辑：
        n_agents = N_blocks × 4（街坊数 × 人口群体数）
        Agent 排列方式：街坊0的4个群体，街坊1的4个群体，...（行优先展开）
        home_block[agent_i] = agent_i 所属街坊的行索引（0 到 N_blocks-1）
        demo_group[agent_i] = agent_i 所属群体的编号（0=青少年, 1=青年, 2=中年, 3=老年）
        weight[agent_i]     = 该街坊该群体的实际人口数
    """
    n_agents = dataset.num_blocks * N_DEMO_GROUPS

    # 构建 Agent 属性张量（行优先展开）
    block_idx = torch.arange(dataset.num_blocks)
    demo_idx  = torch.arange(N_DEMO_GROUPS)
    # 每个街坊重复 4 次（4 个群体），unsqueeze+expand+reshape 实现
    # 结果：[0,0,0,0, 1,1,1,1, 2,2,2,2, ..., 3022,3022,3022,3022]
    home_block = block_idx.unsqueeze(1).expand(-1, N_DEMO_GROUPS).reshape(-1).float()
    # 4 个群体编号循环 N_blocks 次
    # 结果：[0,1,2,3, 0,1,2,3, ..., 0,1,2,3]
    demo_group = demo_idx.unsqueeze(0).expand(dataset.num_blocks, -1).reshape(-1).float()
    # demo_weights 形状 (N_blocks, 4)，reshape(-1) 展开为 (N_agents,)
    weight     = dataset.demo_weights.reshape(-1).float()

    # 预测活力初始为全零（每轮训练通过 runner.reset_state() 重置到这里）
    zeros_vitality = torch.zeros(dataset.num_blocks, len(dataset.target_names))

    # 环境状态：所有街坊级别的全局张量
    env_state = {
        "block_features":            _property("block_features",            dataset.features),
        "observed_vitality":         _property("observed_vitality",         dataset.vitality),          # 训练目标（原始）
        "observed_vitality_scaled":  _property("observed_vitality_scaled",  dataset.vitality_scaled),  # 训练目标（标准化）
        "predicted_vitality":        _property("predicted_vitality",        zeros_vitality.clone()),   # 仿真输出（每步覆写）
        "predicted_vitality_scaled": _property("predicted_vitality_scaled", zeros_vitality.clone()),   # 仿真输出（标准化，用于损失）
        "target_mean":               _property("target_mean",               dataset.target_mean),       # 反标准化用
        "target_scale":              _property("target_scale",              dataset.target_scale),      # 反标准化用
    }
    # 空间图（可选）：构建成功则加入环境状态，Substep 通过字符串路径读取
    if dataset.has_spatial:
        env_state["edge_index"] = _long_property("edge_index", dataset.edge_index)
    if dataset.has_mobility:
        env_state["edge_index_mobility"] = _long_property(
            "edge_index_mobility", dataset.edge_index_mobility
        )

    # MovePolicy 的输入：基础特征 + 可选的空间图
    # 优先使用 2km k-NN 图（通勤尺度），其次退化为 80m 邻接图，最后不用空间注意力
    move_policy_inputs = {"block_features": "environment/block_features"}
    if dataset.has_mobility:
        # k-NN 图覆盖 2km 范围，能捕捉商业/住宅功能区聚集等中尺度模式
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
    # 注意：AggregateVitality 不加入 edge_index 到 aggregate_inputs。
    # 局部 softmax（受限于 2km 邻居内路由）经实验证明会损害高活力街坊的预测，
    # 因为华强北、深圳湾等地标的访客来自全市，不只是 2km 内。
    # 当前设计使用全局 softmax，edge_index 只传给 MovePolicy 用于特征增强。

    return {
        "simulation_metadata": {
            "calibration": False,                           # 不使用贝叶斯校准模式
            "device": device,                               # "auto" / "cpu" / "cuda"
            "num_agents": n_agents,                         # N_blocks × 4
            "num_blocks": dataset.num_blocks,               # 街坊总数（约 3023）
            "num_features": dataset.num_features,           # 特征维度 F（建筑+POI+OD+人口）
            "has_spatial":   dataset.has_spatial,           # 是否有 Neighbor80 图
            "has_mobility":  dataset.has_mobility,          # 是否有 k-NN 通勤图
            "num_demo_groups": N_DEMO_GROUPS,               # 4
            "num_targets": len(dataset.target_names),       # 48（工作日24 + 周末24）
            "hidden_dim": hidden_dim,                       # attract_net 隐藏层维度
            "temporal_rank": 8,                             # 保留字段（当前未使用）
            "num_episodes": 1,                              # 单轮训练 = 1 个 episode
            "num_steps_per_episode": 1,                     # 每 episode 只执行 1 步
            "num_substeps_per_step": 1,                     # 每步只有 1 个 substep
        },
        "state": {
            "environment": env_state,
            "agents": {
                "residents": {
                    "number": n_agents,
                    "properties": {
                        "home_block": _property("home_block", home_block),   # 家街坊行索引
                        "demo_group": _property("demo_group", demo_group),   # 人口群体编号
                        "weight":     _property("weight",     weight),       # 人口数量
                    },
                }
            },
            "objects": None,   # 本模型无静态对象
            "network": {},     # 本模型无显式网络结构（空间图在 env_state 里）
        },
        "substeps": {
            "0": {
                "name": "仿真居民移动",
                "description": (
                    "各居民群体决定留家还是外出，外出人口按吸引力路由到各街坊；"
                    "活力是所有居民行为的涌现聚合结果。"
                ),
                "active_agents": ["residents"],
                "observation": {"residents": None},   # 居民无需观测（直接从状态读特征）
                "policy": {
                    "residents": {
                        "move_policy": {
                            "generator": "MovePolicy",   # 注册名，对应 get_registry() 里的注册
                            "input_variables": move_policy_inputs,
                            "output_variables": ["p_home", "attract_logits", "block_log_scale"],
                            "arguments": None,
                        }
                    }
                },
                "transition": {
                    "aggregate_vitality": {
                        "generator": "AggregateVitality",   # 注册名
                        "input_variables": aggregate_inputs,
                        "output_variables": [
                            "predicted_vitality",           # 写入 environment/predicted_vitality
                            "predicted_vitality_scaled",    # 写入 environment/predicted_vitality_scaled
                        ],
                        "arguments": None,
                    }
                },
                "reward": None,   # 无强化学习奖励
            }
        },
    }
