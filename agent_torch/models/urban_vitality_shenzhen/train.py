"""深圳城市活力模型的训练与评估工具函数。

本模块提供的函数：
    train_model         — 主训练函数：创建 Runner、配置优化器、运行训练循环
    train_multi_seed    — 多随机种子重复实验，报告均值 ± 标准差
    train_district_sweep— 对深圳 10 个行政区逐一做空间留出验证
    run_baselines       — 训练并对比 Ridge / GBT / MLP 基线模型
    diagnose_errors     — 打印误差最大的 N 个街坊，辅助问题排查
    explain_feature_groups — 特征组消融：哪类特征对验证集 MAE 影响最大
    save_predictions    — 把预测结果导出为 CSV
    prediction_frame    — 把预测结果整理为 DataFrame（供导出或进一步分析）

评估指标说明：
    MAE         — 平均绝对误差（单位：LBS 人口数），越低越好，是主要指标
    RMSE        — 均方根误差，对大误差更敏感
    corr        — Pearson 相关系数（预测值 vs 观测值），反映趋势吻合程度
    spearman    — Spearman 秩相关，反映街坊活力排名的一致性
    pairwise_accuracy — 随机抽两个街坊，判断哪个更活跃的准确率
    top20%      — 预测的前 20% 高活力街坊与实测前 20% 的重叠率
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import create_runner


# ─────────────────────────── 评估指标函数 ────────────────────────────────────

def _masked_metrics(prediction, target, mask) -> Dict:
    """计算某个数据集划分（训练集或验证集）的全套评估指标。

    参数：
        prediction: (N_blocks, 48) 预测的原始人口数（非标准化）
        target:     (N_blocks, 48) 实测的原始人口数
        mask:       (N_blocks,) bool 张量，True = 属于本划分

    返回字典包含：
        mae         — 平均绝对误差（所有街坊、所有时段的平均）
        rmse        — 均方根误差
        median_ae   — 中位绝对误差（对异常值更鲁棒）
        corr        — Pearson 相关系数（展平为一维后计算）
        mae_by_tier — 按活力四分位数分层的 MAE（low/medium/high/top 各 25%）
        slot_mae    — 48 个时段各自的 MAE（列表，用于时间分析）
        rank        — 排名指标子字典（spearman / kendall / pairwise / topk）
    """
    if not bool(mask.any()):
        return {"mae": float("nan"), "rmse": float("nan"),
                "median_ae": float("nan"), "corr": float("nan"),
                "mae_by_tier": {}}

    pred_m = prediction[mask]    # (N_split, 48) 仅取本划分的街坊
    tgt_m  = target[mask]

    # 展平为一维，计算整体误差
    error   = pred_m - tgt_m
    abs_err = error.abs()

    # Pearson 相关系数：衡量预测值与实测值在数值上的线性一致性
    p_flat = pred_m.reshape(-1)
    t_flat = tgt_m.reshape(-1)
    if p_flat.std() > 1e-6 and t_flat.std() > 1e-6:
        # torch.corrcoef 返回 2×2 相关矩阵，取 [0,1] 元素是两变量的相关系数
        corr = float(torch.corrcoef(torch.stack([p_flat, t_flat]))[0, 1].item())
    else:
        corr = float("nan")   # 方差近零（如预测全为同一值）时相关系数无意义

    # 按街坊的平均活力水平分成四个层级，分层计算 MAE
    # 这能揭示模型在不同活力区段的表现差异（高活力区往往误差更大）
    mean_obs = tgt_m.mean(dim=1)   # (N_split,) 每个街坊的 48 时段平均活力
    q1, q2, q3 = mean_obs.quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()
    tier_mae = {}
    for tier_label, tier_mask in [
        ("low",    mean_obs <= q1),                        # 最低 25% 活力街坊
        ("medium", (mean_obs > q1) & (mean_obs <= q2)),    # 25–50%
        ("high",   (mean_obs > q2) & (mean_obs <= q3)),    # 50–75%
        ("top",    mean_obs > q3),                         # 最高 25%
    ]:
        if tier_mask.any():
            tier_mae[tier_label] = float(abs_err[tier_mask].mean().item())
        else:
            tier_mae[tier_label] = float("nan")

    # 各时段的 MAE：48 个数值，反映模型在哪些时间段表现更差
    slot_mae = abs_err.mean(dim=0).tolist()

    return {
        "mae":        float(abs_err.mean().item()),
        "rmse":       float(error.square().mean().sqrt().item()),
        "median_ae":  float(abs_err.median().item()),
        "corr":       corr,
        "mae_by_tier": tier_mae,
        "slot_mae":    slot_mae,
        "rank":        _rank_metrics(pred_m, tgt_m),
    }


def _rank_metrics(pred_m: torch.Tensor, tgt_m: torch.Tensor) -> Dict:
    """计算街坊活力排名相关的评估指标（基于各街坊 48 时段均值）。

    对城市规划来说，"哪些街坊活力更高"往往比绝对数值更重要，
    因此排名指标是重要的补充评估维度。

    指标说明：
        spearman        — Spearman 秩相关（-1 到 1），不受绝对误差影响
        kendall         — Kendall Tau，对一致/不一致对数量的稳健衡量
        pairwise_accuracy — 随机选两个街坊，正确判断哪个活力更高的概率
        topk_hit_rate   — 预测的前 K 名与实测前 K 名的重叠率
                          (top10, top20, top50, top100, top10%, top20%, top25%)
    """
    pred_rank = pred_m.mean(dim=1).detach().cpu().numpy()   # (N,) 每街坊均值预测
    tgt_rank  = tgt_m.mean(dim=1).detach().cpu().numpy()    # (N,) 每街坊均值实测
    n = len(tgt_rank)

    # 至少需要 2 个样本，且预测值需有方差（否则排名无意义）
    if n < 2 or np.std(pred_rank) < 1e-9 or np.std(tgt_rank) < 1e-9:
        return {
            "spearman": float("nan"),
            "kendall": float("nan"),
            "pairwise_accuracy": float("nan"),
            "topk_hit_rate": {},
        }

    try:
        from scipy.stats import kendalltau, spearmanr
        spearman = float(spearmanr(tgt_rank, pred_rank).statistic)
        kendall  = float(kendalltau(tgt_rank, pred_rank).statistic)
    except Exception:
        # scipy 不可用时退化为 numpy 手算 Spearman（Kendall 留 nan）
        obs_order  = pd.Series(tgt_rank).rank(method="average").to_numpy()
        pred_order = pd.Series(pred_rank).rank(method="average").to_numpy()
        spearman = float(np.corrcoef(obs_order, pred_order)[0, 1])
        kendall  = float("nan")

    # Pairwise accuracy：枚举所有街坊对 (i, j)，统计预测排序与实测排序一致的比例
    # 跳过实测活力相同的对（没有正确答案），d_obs != 0 是过滤条件
    correct = 0
    total   = 0
    for i in range(n):
        d_obs  = tgt_rank[i + 1:] - tgt_rank[i]
        d_pred = pred_rank[i + 1:] - pred_rank[i]
        mask   = d_obs != 0
        correct += int(((d_obs[mask] > 0) == (d_pred[mask] > 0)).sum())
        total   += int(mask.sum())

    # Top-K 命中率：预测前 K 名与实测前 K 名的交集大小 / K
    topk = {}
    for k in [10, 20, 50, 100]:
        if n >= k:
            obs_top  = set(np.argsort(tgt_rank)[-k:])
            pred_top = set(np.argsort(pred_rank)[-k:])
            topk[f"top{k}"] = len(obs_top & pred_top) / k
    for pct in [0.10, 0.20, 0.25]:
        k = max(1, int(round(n * pct)))
        obs_top  = set(np.argsort(tgt_rank)[-k:])
        pred_top = set(np.argsort(pred_rank)[-k:])
        topk[f"top{int(pct * 100)}pct"] = len(obs_top & pred_top) / k

    return {
        "spearman":         spearman,
        "kendall":          kendall,
        "pairwise_accuracy": correct / total if total else float("nan"),
        "topk_hit_rate":    topk,
    }


# ─────────────────────────── 基线对比 ────────────────────────────────────────

def run_baselines(dataset) -> Dict:
    """训练 Ridge / GBT / MLP 基线模型，与 AgentTorch 对比验证集 MAE。

    所有基线模型接受与 AgentTorch 相同的归一化特征矩阵，预测全部 48 个时段的活力。
    结果均为原始人口数尺度的 MAE，与 AgentTorch 的 MAE 直接可比。

    基线模型说明：
        naive_mean       — 最简基线：用训练集各时段均值作为所有街坊的预测
        ridge            — Ridge 回归（L2 正则化线性回归），alpha=1
        gradient_boosting— 梯度提升树（HistGBT），对每个时段单独训练一个模型
        mlp_no_agent     — 多层感知机，相同特征但无 Agent 仿真，用于消融 Agent 机制的贡献
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neural_network import MLPRegressor

    # 只取训练集/验证集各自的特征和目标（numpy 格式供 sklearn 使用）
    X_tr = dataset.features[dataset.train_mask].numpy()
    X_va = dataset.features[dataset.validation_mask].numpy()
    y_tr = dataset.vitality[dataset.train_mask].numpy()      # 原始人口数
    y_va = dataset.vitality[dataset.validation_mask].numpy()

    results: Dict = {}

    # 基线 1：训练集均值（每时段一个标量，不考虑街坊差异）
    naive_pred = y_tr.mean(axis=0, keepdims=True)
    results["naive_mean"] = {
        "val_mae": float(np.abs(y_va - naive_pred).mean()),
        "note": "各时段训练集均值，无空间区分",
    }

    # 基线 2：Ridge 回归（L2 正则线性回归，alpha=1 是经验默认值）
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_tr)   # sklearn Ridge 原生支持多输出（48 个目标列同时拟合）
    results["ridge"] = {
        "val_mae": float(np.abs(y_va - ridge.predict(X_va)).mean()),
        "note": "Ridge(alpha=1)",
    }

    # 基线 3：梯度提升树（每个时段独立一棵树，MultiOutputRegressor 并行训练）
    # HistGBT 比传统 GBT 快很多，适合特征较多时使用
    gbt = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=200, max_depth=4, random_state=42),
        n_jobs=-1,  # 使用全部 CPU 核心并行
    )
    gbt.fit(X_tr, y_tr)
    results["gradient_boosting"] = {
        "val_mae": float(np.abs(y_va - gbt.predict(X_va)).mean()),
        "note": "HistGBT(max_iter=200, depth=4) × 48 个时段",
    }

    # 基线 4：无 Agent 机制的 MLP（与 AgentTorch 的 attract_net 结构相近，但无仿真环节）
    # 用于消融实验：如果 AgentTorch 不优于此 MLP，说明 Agent 仿真没有额外贡献
    mlp = MLPRegressor(hidden_layer_sizes=(128,), max_iter=500, random_state=42,
                       early_stopping=True, n_iter_no_change=20)
    mlp.fit(X_tr, y_tr)
    results["mlp_no_agent"] = {
        "val_mae": float(np.abs(y_va - mlp.predict(X_va)).mean()),
        "note": "MLP(128) — 相同特征，无 Agent 聚合",
    }

    return results


# ─────────────────────────── 特征重要性分析 ──────────────────────────────────

def _masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """计算给定掩码下的 MAE（单一数值），供消融实验使用。"""
    if not bool(mask.any()):
        return float("nan")
    return float((prediction[mask] - target[mask]).abs().mean().item())


def _feature_group_indices(feature_names: List[str]) -> Dict[str, List[int]]:
    """把特征列名按类型分组，返回各组的列索引。

    分组逻辑：
        od_flow        — 以 "od_" 开头的列（OD 到达/出发流量，96 列）
        poi            — 以 "poi_" 开头的列（12 类 POI 计数，log1p 变换后）
        portrait       — 来自街坊_人口画像.csv 的人口统计列（15 列）
        building_landuse — 其余所有列（建筑面积、用地类型、密度、交通可达性等）
    """
    portrait = {
        "CZZL20_24", "CZZL25_29", "CZZL30_34", "CZZL35_39", "CZZL40_44",
        "CZZL45_49", "CZZL50_54", "CZZL55_59", "XB1CZRK", "XB2CZRK",
        "就业人", "CZRKMD", "FHJCZRKZSL", "JZSJD1RKSL", "JZSJD5RKSL",
    }
    groups = {
        "od_flow": [],
        "poi": [],
        "portrait": [],
        "building_landuse": [],
    }
    for i, name in enumerate(feature_names):
        if name.startswith("od_"):
            groups["od_flow"].append(i)
        elif name.startswith("poi_"):
            groups["poi"].append(i)
        elif name in portrait:
            groups["portrait"].append(i)
        else:
            groups["building_landuse"].append(i)
    return {k: v for k, v in groups.items() if v}   # 去掉空组


def explain_feature_groups(runner, dataset) -> pd.DataFrame:
    """特征组消融实验：逐组把特征置零，观察验证集 MAE 上升多少。

    方法：
        不重新训练，只是把某组特征的归一化值替换为 0（等于训练集均值，
        因为 Z-score 归一化后均值对应 0），然后用当前权重做推断。
        MAE 上升越多，说明模型对该特征组的依赖越强。

    这是一种"置换重要性"的简化版本，不考虑特征组间的交互效应，
    但计算成本很低（无需重训练）且结论通常可靠。

    返回：
        DataFrame，列为 feature_group / num_features / val_mae / delta_mae，
        按 delta_mae 降序排列（最重要的特征组排在最前）。
    """
    dev = runner.initializer.device
    # 先用完整特征跑一次，获得基准 MAE
    runner.reset_state()
    with torch.no_grad():
        runner.step(1)
    observed = runner.state["environment"]["observed_vitality"].detach().cpu()
    baseline = runner.state["environment"]["predicted_vitality"].detach().cpu()
    base_mae = _masked_mae(baseline, observed, dataset.validation_mask)

    rows = []
    for group, indices in _feature_group_indices(dataset.feature_names).items():
        modified = dataset.features.clone()
        modified[:, indices] = 0.0   # 把该组特征的归一化值置零（= 用训练集均值替代）
        runner.reset_state()
        runner.state["environment"]["block_features"] = modified.to(dev)
        with torch.no_grad():
            runner.step(1)
        pred = runner.state["environment"]["predicted_vitality"].detach().cpu()
        mae  = _masked_mae(pred, observed, dataset.validation_mask)
        rows.append({
            "feature_group": group,
            "num_features":  len(indices),
            "val_mae":       round(mae, 1),
            "delta_mae":     round(mae - base_mae, 1),   # 正值 = MAE 上升 = 该组重要
        })

    # 恢复完整特征状态
    runner.reset_state()
    with torch.no_grad():
        runner.step(1)
    return pd.DataFrame(rows).sort_values("delta_mae", ascending=False).reset_index(drop=True)


def diagnose_errors(runner, dataset, top_n: int = 20) -> pd.DataFrame:
    """列出验证集中误差最大的 top_n 个街坊，辅助排查模型失效案例。

    输出列说明：
        block_id      — 街坊编号
        mean_observed — 该街坊 48 时段实测平均活力（反映街坊活力水平）
        mae           — 该街坊的平均绝对误差（越大越需要关注）
        bias          — 预测偏差（正值 = 系统性高估，负值 = 系统性低估）
        mae_pct       — MAE 占实测均值的百分比（相对误差）
        district      — 所属行政区（如果数据可用）
    """
    prediction = runner.state["environment"]["predicted_vitality"].detach().cpu().numpy()
    observed   = runner.state["environment"]["observed_vitality"].detach().cpu().numpy()

    val_idx   = dataset.validation_mask.numpy()
    block_ids = dataset.block_ids[dataset.validation_mask].numpy()
    pred_val  = prediction[val_idx]
    obs_val   = observed[val_idx]

    per_block_mae = np.abs(pred_val - obs_val).mean(axis=1)          # 每街坊的 MAE
    mean_obs      = obs_val.mean(axis=1)                              # 每街坊的实测均值
    bias          = (pred_val - obs_val).mean(axis=1)                 # 每街坊的系统偏差

    df = pd.DataFrame({
        "block_id":      block_ids,
        "mean_observed": mean_obs.round(1),
        "mae":           per_block_mae.round(1),
        "bias":          bias.round(1),
        "mae_pct":       (per_block_mae / (mean_obs + 1e-6) * 100).round(1),
    })

    if dataset.districts is not None:
        df["district"] = dataset.districts[val_idx]

    return df.nlargest(top_n, "mae").reset_index(drop=True)


# ─────────────────────────── 主训练函数 ──────────────────────────────────────

def train_model(
    data_dir="data_shenzhen",
    epochs: int = 400,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    validation_fraction: float = 0.2,
    seed: int = 42,
    device: str = "auto",
    cosine_lr: bool = False,
    early_stop_patience: int = 60,
    split_strategy: str = "random",
    holdout_district=None,
):
    """训练 AgentTorch 深圳活力模型，返回训练好的 runner 和评估指标。

    参数说明：
        data_dir            — 数据目录（默认 "data_shenzhen"）
        epochs              — 最大训练轮数（提前满足早停条件时实际轮数更少）
        learning_rate       — Adam 基础学习率（各参数组有独立倍率，见下方）
        hidden_dim          — attract_net 隐藏层维度（同时写入 config，供 MovePolicy 使用）
        validation_fraction — 随机划分时验证集占比（0.2 = 20%）
        seed                — 随机种子，同时控制数据划分和权重初始化
        device              — "auto"（优先 GPU）/ "cpu" / "cuda"
        cosine_lr           — 是否使用余弦退火学习率调度；False = 恒定学习率，
                              在 400 轮预算内收敛更稳定
        early_stop_patience — 验证集 MAE 连续多少轮无改善则停止（0 = 禁用）
        split_strategy      — "random"（随机划分）或 "district"（行政区留出）
        holdout_district    — 当 split_strategy="district" 时指定留出的行政区名称

    返回：
        runner  — 已训练的 AgentTorch Runner（含最优参数）
        dataset — ShenzhenVitalityDataset
        metrics — 字典，包含 train/validation 的各项指标
        history — 各轮训练损失的列表
    """
    torch.manual_seed(seed)
    runner, dataset = create_runner(
        data_dir=data_dir,
        hidden_dim=hidden_dim,
        validation_fraction=validation_fraction,
        seed=seed,
        device=device,
        split_strategy=split_strategy,
        holdout_district=holdout_district,
    )
    runner.to(runner.initializer.device)
    dev = runner.initializer.device

    train_mask = dataset.train_mask.to(dev)

    # ── 参数分组与学习率设置 ──────────────────────────────────────────────
    # 不同参数有不同的学习率，原因：
    #   home_logits：直接控制留家概率，梯度信号强，5× LR 加速收敛
    #   attract_net：控制空间路由，weight_decay=0.1 防止过拟合（L2 正则）
    #   scale_net：  控制规模修正，2× LR 匹配 log_scale 的更新速度；weight_decay 约束修正幅度
    #   log_scale：  全局时段修正，2× LR 使其与 scale_net 保持同步

    move_policy = runner.initializer.policy_function["0"]["residents"]["move_policy"]
    param_groups = [
        {"params": move_policy.home_logits,                "lr": learning_rate * 5},
        {"params": move_policy.attract_net.parameters(),   "lr": learning_rate,
         "weight_decay": 0.1},
        {"params": move_policy.scale_net.parameters(),     "lr": learning_rate * 2,
         "weight_decay": 0.1},
    ]
    # 如果空间注意力存在（需要 geopandas + shapefile），也加入优化
    if move_policy.spatial_attn is not None:
        param_groups.append(
            {"params": move_policy.spatial_attn.parameters(), "lr": learning_rate,
             "weight_decay": 1e-4}
        )
    aggregate = runner.initializer.transition_function["0"]["aggregate_vitality"]
    param_groups.append(
        {"params": aggregate.log_scale, "lr": learning_rate * 2}
    )

    optimizer = torch.optim.Adam(param_groups, lr=learning_rate)

    # 余弦退火调度：训练结束时 LR 降至 eta_min=1e-5
    # 实验发现恒定 LR 在 400 轮预算内收敛更好，cosine_lr 默认关闭
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        if cosine_lr else None
    )

    # Huber 损失：误差 < delta=1.0 时用 MSE（平滑），误差 > 1.0 时用 MAE（鲁棒）
    # 在标准化对数空间运行，delta=1.0 大约对应 e-1 ≈ 1.7 倍人口数的误差
    # 相比纯 MSE 对异常街坊更鲁棒，相比纯 MAE 梯度更稳定
    loss_fn = nn.HuberLoss(delta=1.0)

    history: List[float] = []
    best_val_mae   = float("inf")
    best_state: Optional[dict] = None
    patience_counter = 0
    val_mask_dev   = dataset.validation_mask.to(dev)

    # ── 训练循环 ──────────────────────────────────────────────────────────
    for epoch in range(epochs):
        runner.reset_state()     # 把所有状态张量重置回初始值（预测活力 = 全零）
        optimizer.zero_grad()    # 清空上一轮的梯度缓存

        runner.step(1)           # 运行 1 步仿真：MovePolicy → AggregateVitality

        # 从仿真状态中取出标准化预测值和目标值（在对数空间计算损失）
        predicted = runner.state["environment"]["predicted_vitality_scaled"]
        observed  = runner.state["environment"]["observed_vitality_scaled"]

        # 只在训练集街坊上计算损失（验证集不参与参数更新）
        loss = loss_fn(predicted[train_mask], observed[train_mask])
        loss.backward()          # 反向传播：计算所有可训练参数的梯度
        optimizer.step()         # Adam 用梯度更新参数
        if scheduler is not None:
            scheduler.step()     # 更新学习率（余弦退火）
        history.append(float(loss.detach().cpu().item()))

        # ── 早停：监控验证集 MAE，保存最优参数快照 ─────────────────────
        if early_stop_patience > 0 and dataset.validation_mask.any():
            with torch.no_grad():
                # 取原始人口数（非标准化）用于早停判断，单位与最终报告一致
                pred_raw = runner.state["environment"]["predicted_vitality"].detach().cpu()
                obs_raw  = runner.state["environment"]["observed_vitality"].detach().cpu()
            current_val_mae = float(
                (pred_raw[dataset.validation_mask] - obs_raw[dataset.validation_mask])
                .abs().mean().item()
            )
            if current_val_mae < best_val_mae - 1.0:
                # 改善超过 1 人（避免噪声抖动触发保存）时更新最优快照
                best_val_mae = current_val_mae
                import copy
                # 只保存参数张量（不保存完整 runner 状态），节省内存
                best_state = copy.deepcopy({
                    k: v.detach().clone()
                    for k, v in runner.named_parameters()
                })
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break   # 连续 early_stop_patience 轮无改善，提前终止

    # ── 恢复最优参数 ──────────────────────────────────────────────────────
    # 若早停触发，用保存的最优快照覆盖当前参数（防止最后几轮过拟合）
    if best_state is not None:
        for name, param in runner.named_parameters():
            if name in best_state:
                param.data.copy_(best_state[name])

    # ── 最终评估 ──────────────────────────────────────────────────────────
    runner.reset_state()
    with torch.no_grad():
        runner.step(1)
    prediction = runner.state["environment"]["predicted_vitality"].detach().cpu()
    observed   = runner.state["environment"]["observed_vitality"].detach().cpu()
    metrics = {
        "training_loss": history[-1] if history else float("nan"),
        "num_features":  dataset.num_features,
        "num_blocks":    dataset.num_blocks,
        "train":      _masked_metrics(prediction, observed, dataset.train_mask),
        "validation": _masked_metrics(prediction, observed, dataset.validation_mask),
    }
    return runner, dataset, metrics, history


# ─────────────────────────── 多种子 / 分区 sweep ─────────────────────────────

def train_multi_seed(
    seeds: List[int] = (42, 123, 456),
    data_dir: str = "data_shenzhen",
    epochs: int = 400,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    validation_fraction: float = 0.2,
    device: str = "auto",
    cosine_lr: bool = False,
    early_stop_patience: int = 60,
    split_strategy: str = "random",
    holdout_district=None,
) -> Dict:
    """用多个随机种子重复训练，报告指标的均值、标准差、最优和最差。

    作用：
        单次训练结果受随机划分和参数初始化影响，
        多种子实验能评估模型稳定性，排除"运气好的单次结果"。

    返回字典结构：
        {
          "per_seed": [每个种子的完整 metrics 字典, ...],
          "summary":  {val_mae_mean, val_mae_std, val_mae_best, val_mae_worst,
                       val_rmse_mean, val_corr_mean, val_spearman_mean}
        }
    """
    per_seed: List[Dict] = []
    for seed in seeds:
        print(f"\n[seed={seed}]")
        _, _, metrics, _ = train_model(
            data_dir=data_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            validation_fraction=validation_fraction,
            seed=seed,
            device=device,
            cosine_lr=cosine_lr,
            early_stop_patience=early_stop_patience,
            split_strategy=split_strategy,
            holdout_district=holdout_district,
        )
        per_seed.append({"seed": seed, **metrics})
        rank = metrics["validation"].get("rank", {})
        print(
            f"  val MAE={metrics['validation']['mae']:.0f}  "
            f"val RMSE={metrics['validation']['rmse']:.0f}  "
            f"corr={metrics['validation']['corr']:.3f}  "
            f"spearman={rank.get('spearman', float('nan')):.3f}"
        )

    # 过滤掉 nan（某些种子可能因数据问题得到无效结果）
    val_maes  = [r["validation"]["mae"]  for r in per_seed if not np.isnan(r["validation"]["mae"])]
    val_rmses = [r["validation"]["rmse"] for r in per_seed if not np.isnan(r["validation"]["rmse"])]
    val_corrs = [r["validation"]["corr"] for r in per_seed if not np.isnan(r["validation"]["corr"])]
    val_spearmans = [
        r["validation"].get("rank", {}).get("spearman")
        for r in per_seed
        if not np.isnan(r["validation"].get("rank", {}).get("spearman", float("nan")))
    ]

    summary = {
        "val_mae_mean":      float(np.mean(val_maes)),
        "val_mae_std":       float(np.std(val_maes)),
        "val_mae_best":      float(np.min(val_maes)),   # 最好种子的 MAE
        "val_mae_worst":     float(np.max(val_maes)),   # 最差种子的 MAE
        "val_rmse_mean":     float(np.mean(val_rmses)),
        "val_corr_mean":     float(np.mean(val_corrs)),
        "val_spearman_mean": float(np.mean(val_spearmans)) if val_spearmans else float("nan"),
    }
    print(
        f"\n[多种子汇总] "
        f"val MAE = {summary['val_mae_mean']:.0f} ± {summary['val_mae_std']:.0f}  "
        f"(最优={summary['val_mae_best']:.0f}, 最差={summary['val_mae_worst']:.0f})  "
        f"corr={summary['val_corr_mean']:.3f}  "
        f"spearman={summary['val_spearman_mean']:.3f}"
    )
    return {"per_seed": per_seed, "summary": summary}


def train_district_sweep(
    districts: Optional[List[str]] = None,
    data_dir: str = "data_shenzhen",
    epochs: int = 400,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    device: str = "auto",
    cosine_lr: bool = False,
    early_stop_patience: int = 60,
) -> pd.DataFrame:
    """对深圳 10 个行政区逐一做空间留出验证，返回汇总 DataFrame。

    每次把一个行政区的所有街坊作为验证集，其余街坊训练模型，
    相当于对空间泛化能力做 10 折交叉验证（Leave-One-District-Out）。

    这比随机划分更严格，能检验模型在从未见过的区域的预测能力，
    适合评估模型是否真正学到了可迁移的城市规律，而非记忆局部模式。

    返回的 DataFrame 列：
        district       — 行政区名称
        val_blocks     — 该区街坊数量
        val_mae        — 验证集 MAE
        val_rmse       — 验证集 RMSE
        corr           — Pearson 相关系数
        spearman       — Spearman 秩相关
        top20pct_hit   — 预测前 20% 高活力街坊的命中率
    """
    if districts is None:
        districts = ["福田区", "南山区", "罗湖区", "宝安区", "龙岗区",
                     "龙华区", "光明区", "盐田区", "坪山区", "大鹏新区"]
    rows = []
    for district in districts:
        print(f"\n[district={district}]")
        _, dataset, metrics, _ = train_model(
            data_dir=data_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            validation_fraction=0.2,   # district 模式下此参数被忽略
            seed=42,
            device=device,
            cosine_lr=cosine_lr,
            early_stop_patience=early_stop_patience,
            split_strategy="district",
            holdout_district=district,
        )
        val  = metrics["validation"]
        rank = val.get("rank", {})
        row = {
            "district":     district,
            "val_blocks":   int(dataset.validation_mask.sum().item()),
            "val_mae":      round(val["mae"], 1),
            "val_rmse":     round(val["rmse"], 1),
            "corr":         round(val["corr"], 4),
            "spearman":     round(rank.get("spearman", float("nan")), 4),
            "top20pct_hit": round(rank.get("topk_hit_rate", {}).get("top20pct", float("nan")), 4),
        }
        rows.append(row)
        print(
            f"  val MAE={row['val_mae']:.0f}  corr={row['corr']:.3f}  "
            f"spearman={row['spearman']:.3f}  top20%={row['top20pct_hit']:.3f}"
        )
    return pd.DataFrame(rows)


# ─────────────────────────── 预测导出 ────────────────────────────────────────

def prediction_frame(runner, dataset):
    """把当前 runner 状态的预测结果整理为 DataFrame，供导出或可视化使用。

    输出列说明：
        Block_ID                   — 街坊编号
        split                      — 0=训练集，1=验证集
        observed_weekday_vitality  — 实测工作日均值活力（24 时段平均）
        predicted_weekday_vitality — 预测工作日均值活力
        observed_weekend_vitality  — 实测周末均值活力
        predicted_weekend_vitality — 预测周末均值活力
        predicted_WD_C_00 ...      — 各时段的逐小时预测值（48 列）
    """
    predicted = runner.state["environment"]["predicted_vitality"].detach().cpu().numpy()
    observed  = runner.state["environment"]["observed_vitality"].detach().cpu().numpy()
    result = pd.DataFrame({"Block_ID": dataset.block_ids.numpy()})
    result["split"] = dataset.validation_mask.numpy().astype("int8")
    result["observed_weekday_vitality"]  = observed[:, :24].mean(axis=1)
    result["predicted_weekday_vitality"] = predicted[:, :24].mean(axis=1)
    result["observed_weekend_vitality"]  = observed[:, 24:].mean(axis=1)
    result["predicted_weekend_vitality"] = predicted[:, 24:].mean(axis=1)
    for index, name in enumerate(dataset.target_names):
        result[f"predicted_{name}"] = predicted[:, index]
    return result


def save_predictions(runner, dataset, output_path):
    """把预测结果保存为 CSV 文件。

    自动创建父目录（如 outputs/ 不存在时会建立）。
    CSV 使用 utf-8-sig 编码（带 BOM），确保在 Excel 中直接打开不乱码。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame(runner, dataset).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
