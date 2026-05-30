# Urban Vitality Shenzhen — 进度记录

## 当前最佳结果（seed=42，Phase 3）

| 指标 | 数值 |
|------|------|
| 验证集 MAE | **1388** |
| 朴素基线 MAE | 4285 |
| Ridge 基线 MAE | 1494 |
| GBT 基线 MAE | 1067 |
| MLP（无 Agent）基线 MAE | 1234 |
| 超越朴素基线 | **68%** |
| 超越 Ridge | **7%** |
| 验证集相关系数 | 0.923 |
| 验证集中位绝对误差 | 619 |
| 训练集 MAE | 1183 |
| 特征数 | 174（78 静态 + 96 OD） |

## 多 Seed 验证

### Phase 3（174 特征，含 OD）

| Seed | 验证 MAE | 验证 RMSE | 相关系数 |
|------|----------|-----------|---------|
| 42   | 1368     | 2801      | 0.926   |
| 123  | 1236     | 2556      | 0.938   |
| 456  | 1408     | 2742      | 0.928   |
| **均值±标准差** | **1337 ± 74** | 2700 | **0.930** |

> ✅ 方差从 ±530 降至 ±74（−86%），seed 间浮动从 1207 缩至 172。
> 模型性能不再依赖"有利划分"，体现真实泛化能力。

### 与 Phase 2 对比

| | Phase 2（78 特征，无 OD） | Phase 3（174 特征，含 OD） | 改善 |
|---|---|---|---|
| 均值 MAE | 2593 | **1337** | **−48%** |
| 标准差 | ±530 | **±74** | **−86%** |
| best / worst | 1852 / 3059 | 1236 / 1408 | — |

## 按活力分层的 MAE（seed=42，Phase 3）

| 活力层 | Phase 2 MAE | Phase 3 MAE | 改善 |
|-------|------------|------------|------|
| 低（Q1 以下） | 329 | **227** | −31% |
| 中低（Q1–Q2） | 872 | **719** | −18% |
| 中高（Q2–Q3） | 1348 | **1299** | −4% |
| 高（Q3 以上） | 4535 | **3315** | −27% |

## 完成的工作

### Phase 0 — 可复现性 ✅

- 多 seed 验证函数 `train_multi_seed()`，报告 mean/std/best/worst
- 早停（patience=60）防止过拟合
- 完整输出（CSV + 可视化）

### Phase 1 — 验证可靠性 ✅

- **空间分块验证**：`--split-strategy district --holdout-district <行政区>` CLI 参数
  - 可用行政区：福田区 南山区 罗湖区 宝安区 龙岗区 龙华区 光明区 盐田区 坪山区 大鹏新区
- **强基线对比**：`--baselines` 输出 naive_mean / Ridge / GBT / MLP vs Agent 对比表
- **高误差街坊诊断**：`--diagnose N` 输出 top-N 高误差街坊（含 district 标签）
- `ShenzhenVitalityDataset` 新增 `districts` 字段

### Phase 2 — POI 数据补全 ✅

- 新增 8 类 POI（餐饮 / 购物 / 生活服务 / 交通设施 / 公司企业 / 体育休闲 / 住宿 / 科教文化）
- 特征数 70 → 78，hidden_dim 64 → 128

### Phase 3 — OD 流量特征 ✅

**新增组件**：

- `scale_net: Linear(n_feat, n_time, bias=False)` — per-block 时序 log 尺度修正
  - `bias=False`：与全局 `log_scale(48)` 非冗余（各司其职）
  - 零初始化，weight_decay=0.1，lr=2×
  - 输出 `block_log_scale: (N_blocks, 48)` 作为 MovePolicy 第 3 个输出
- `scale(j,t) = exp(log_scale_global(t) + block_log_scale(j,t))`
- Loss 换为 `HuberLoss(delta=1.0)`（比 MSELoss 更对齐 MAE 评估）

**OD 特征聚合**（`_aggregate_od_to_blocks`）：

- 源数据：90,417 个 ~159m geohash 网格，工作日/周末 × 到达/出发 × 24 小时
- 空间连接：网格质心 → 街坊多边形，匹配率 89.8%，全部 3023 个街坊均有覆盖（无覆盖填零）
- 输出 96 列（log1p 变换）：`od_arr_wd_h{00..23}` / `od_dep_wd_h{00..23}` / `od_arr_we_h{00..23}` / `od_dep_we_h{00..23}`
- 加载耗时：~22s（空间连接，一次性）

**已验证的失败路径**（勿重复）：

1. **时序因子化路由**：`attract_net→(N,T)` + global softmax → 48× 零和过拟合容量 → val MAE 143422
2. **冗余 scale 参数**：`log_scale + block_scale + temporal_logits` 三者相加 → 优化不稳定 → val MAE 2782
3. **替换 log_scale 的无 bias scale_net**：丢失全局 LBS 校正 → val MAE 2076
4. **Local softmax（任何条件下）**：高活力中心吸引全市人流，k=30 邻居窗口截断全市级吸引力
   - 无 OD 特征：1766 → 1870（更差）
   - 有 OD 特征：1388 → 1698（更差）
   - Root cause：Local softmax 需要真实 OD 流向对才有意义，现有数据只有总到达/出发量

## 架构概览（当前）

```
MovePolicy 输出：
  p_home:          (4, 48)       — 时序先验 + 梯度学习
  attract_logits:  (N_blocks, 1) — 标量吸引力，用于全局 softmax 路由
  block_log_scale: (N_blocks, 48)— per-block 时序尺度修正（scale_net 输出）

AggregateVitality：
  scale(j,t) = exp(log_scale_global(t) + block_log_scale(j,t))
  predicted(j,t) = (home_vitality(j,t) + away_vitality(j,t)) × scale(j,t)

训练参数组：
  home_logits:         lr×5
  attract_net:         lr×1,  weight_decay=0.1
  scale_net:           lr×2,  weight_decay=0.1
  spatial_attn:        lr×1,  weight_decay=1e-4
  aggregate.log_scale: lr×2
  loss: HuberLoss(delta=1.0)
```

## 文件结构

```
agent_torch/models/urban_vitality_shenzhen/
├── __init__.py          # get_registry(), create_runner()（含 split_strategy 参数）
├── data.py              # 数据加载、_aggregate_od_to_blocks、build_config()
├── train.py             # train_model(), run_baselines(), diagnose_errors(), train_multi_seed()
├── main.py              # CLI（--split-strategy, --holdout-district, --baselines, --diagnose）
└── substeps/
    ├── move.py          # MovePolicy: home_logits + spatial_attn + attract_net + scale_net
    └── aggregate.py     # AggregateVitality: log_scale + global softmax
tests/
└── test_urban_vitality_shenzhen.py   # 6 个测试，全部通过
data_shenzhen/
├── 街坊_数据连接.csv / 街坊_LBS统计.csv / 街坊_人口画像.csv
├── 街坊范围shp/
├── POI2026/
└── LBS原始数据/          # OD 原始数据（工作日/周末 × 到达/出发 网格）
```

## 运行方式

```bash
cd /home/ubuntu/agenttorch

# 单次训练
python3.12 -m agent_torch.models.urban_vitality_shenzhen.main --epochs 600

# 多 seed 验证
python3.12 -m agent_torch.models.urban_vitality_shenzhen.main \
  --seeds 42 123 456 --epochs 600

# 行政区空间分块验证 + 基线对比 + 误差诊断
python3.12 -m agent_torch.models.urban_vitality_shenzhen.main \
  --epochs 600 --split-strategy district --holdout-district 福田区 \
  --baselines --diagnose 20

# 运行测试
python3.12 -m pytest tests/test_urban_vitality_shenzhen.py -v
```

## 推送到 GitHub

- Remote：`github → git@github.com:yueZ2046/AgentTorch.git`
- SSH 密钥：`~/.ssh/id_ed25519_github`（GitHub 账号内名称：4agenttorch）
- SSH config：`~/.ssh/config` 已配置，直接可用

```bash
git push github master:main
```

> `data_shenzhen/` 和 `outputs/` 已在 `.gitignore` 中，永远不会上传。

## 下一步计划

### Phase 4 — 干预场景仿真

- [ ] 定义干预变量语义（增加商业体 / 公共空间 / 交通可达性）
- [ ] 将干预参数引入模型状态
- [ ] 构建场景对比 API（baseline vs intervention A vs intervention B）
- [ ] 输出时序影响、空间影响、群组影响、溢出效应

## 完整实验记录（验证集 MAE，seed=42）

| 版本 | val MAE | 说明 |
|------|---------|------|
| 朴素基线（训练均值） | 4285 | baseline |
| 全局 softmax，无空间 | ~9000 | 早期，严重过拟合 |
| + Neighbor80 spatial_proj | 2660 | 一跳均值聚合 |
| + POI 特征（4 类） | 2588 | — |
| + 注意力 k-NN 2km | 2571 | — |
| + log_scale 尺度因子 | 1886 | 修复 LBS 采样偏差 |
| + 新 8 类 POI + hidden=128 + 早停 | 1769 | Phase 2 最佳 |
| 多 seed 均值（42/123/456） | 2593 ± 530 | Phase 2 真实期望性能 |
| + scale_net（无 OD） | 1766 | Phase 3 第一步，微幅改善 |
| + local softmax（无 OD） | 1870 | 更差，已弃用 |
| **+ OD 聚合特征（96 列）+ scale_net** | **1388** | **Phase 3 完成，超越 Ridge** |
| + local softmax（有 OD） | 1698 | 仍更差，已弃用 |
| **多 seed 均值（42/123/456，Phase 3）** | **1337 ± 74** | **真实泛化性能，方差−86%** |
