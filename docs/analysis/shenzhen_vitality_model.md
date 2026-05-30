---
name: shenzhen-vitality-model
description: 深圳城市活力仿真模型完整技术细节：当前架构(scale_net+OD特征)、性能(MAE=1388超越Ridge)、已验证的失败路径、Phase 1-3完成状态
metadata: 
  node_type: memory
  type: project
  originSessionId: 4cae5642-aebe-443d-85db-f1f65d31e291
---

## 模型定位

`agent_torch/models/urban_vitality_shenzhen/` — 基于AgentTorch可微分框架的深圳街坊级LBS人流活力预测原型。

> 当前描述：**深圳街坊级LBS人流活力预测模型，已超越Ridge基线，是所有结构化模型中最优**。

---

## Agent设计

| 维度 | 数值 |
|---|---|
| 空间单元 | 3,023 深圳街坊 |
| 人口群体 | 4个人口学队组（青少年与儿童/青年/中年/老年） |
| 总Agent数 | 12,092 居民Agent |
| 目标变量 | LBS在场人口，48个时间段（工作日24h + 周末24h） |

---

## 当前架构（Phase 3完成后）

### MovePolicy — `substeps/move.py`

```python
# 三个输出：p_home, attract_logits, block_log_scale
home_logits:    nn.Parameter(4, 48)           # 时序先验初始化
spatial_attn:   SpatialAttentionAggregation   # k-NN 2km 图注意力
attract_net:    Linear(n_feat,hidden) →ReLU→Linear(hidden,1)  # 标量吸引力，用于路由
scale_net:      Linear(n_feat, n_time, bias=False)             # per-block时序尺度修正
```

**scale_net 设计要点**：
- `bias=False`：避免与 `log_scale` 冗余（log_scale 负责全局时序，scale_net 负责街块间偏差）
- 零初始化权重：训练起点与原始模型相同（scale=1）
- weight_decay=0.1，lr=2×：与原 log_scale 参数设置对齐
- 输出 `block_log_scale: (N_blocks, n_time)` — 每块每时段的独立 log 尺度修正

**为何不用时序分解（temporal factorization）路由**：
- 将时序变化引入 global softmax 路由 → 48× 过拟合容量 → train MAE=1579 但 val MAE=143422
- 时序表达能力必须作用在 **scale**（乘性修正）而非 **routing**（softmax 竞争）

### AggregateVitality — `substeps/aggregate.py`

```python
# 无可学习参数（log_scale 已移入 MovePolicy.scale_net）
scale(j,t) = exp(log_scale_global(t) + block_log_scale(j,t))

predicted(j,t) = (home_vitality(j,t) + away_vitality(j,t)) × scale(j,t)
```

`log_scale_global(48)` 仍保留在 `AggregateVitality`，负责全局 LBS 采样偏差修正；
`block_log_scale` 来自 MovePolicy.scale_net，负责街块特定的时序偏差。

**Global softmax（当前生产配置）**：
```python
away_vitality(j,t) = softmax(attract_logits, dim=0)[j] × total_away(t)
```

**Local softmax 结论（已验证有害）**：
- 随机split：1766→1870（-104），+OD特征后：1388→1698（-310）
- 根本原因：高活力商业中心吸引全市人流，k=30 邻居窗口截断全市级吸引力
- Top 层 MAE 从 3315 → 4431 最明显
- **Local softmax 只在有真实 OD 流向对时才有意义**（知道从A→B的具体流量）

### 训练配置（`train.py`）

```python
param_groups = [
    {"params": home_logits,            "lr": lr*5},
    {"params": attract_net.parameters(),"lr": lr,   "weight_decay": 0.1},
    {"params": scale_net.parameters(), "lr": lr*2,  "weight_decay": 0.1},
    {"params": spatial_attn.parameters(),"lr": lr,  "weight_decay": 1e-4},
    {"params": aggregate.log_scale,    "lr": lr*2},
]
loss_fn = nn.HuberLoss(delta=1.0)  # 比 MSELoss 更好对齐 MAE 评估指标
early_stop_patience = 60
```

---

## 数据层（`data.py`）

### 特征构成（共174列，Phase 3完成后）

| 类别 | 来源 | 维度 |
|---|---|---|
| 建筑特征 | 街坊_数据连接.csv | ~51列 |
| 人口画像 | 街坊_人口画像.csv | 15列 |
| POI（SHP，4类） | POI2026/POI_SHP/ | 4列（医疗/风景/汽车/摩托车） |
| POI（CSV，8类） | POI2026/POI_CSV/ | 8列（餐饮/购物/生活/交通/公司/体育/住宿/科教） |
| **OD流特征（Phase 3新增）** | LBS原始数据/ | **96列** |

**OD特征详情（96列）**：
- `od_arr_wd_h{00..23}`：工作日每时段到达量（24列，log1p）
- `od_dep_wd_h{00..23}`：工作日每时段出发量（24列，log1p）
- `od_arr_we_h{00..23}`：周末每时段到达量（24列，log1p）
- `od_dep_we_h{00..23}`：周末每时段出发量（24列，log1p）

**为何OD特征有效**：到达/出发时序剖面直接编码每个街坊的实测活动节律（几点来人、几点走人），scale_net 拿到后基本上是"将实测流量曲线映射到LBS在场量曲线"，而非从建筑/POI静态特征推断。

**OD数据技术细节**：
- 源数据：90,417个~159m geohash 网格，grid_id为字符串
- 空间连接：网格质心 → 街坊多边形（inner join），匹配率89.8%，2809/3023街坊有OD数据
- 无OD覆盖的214个街坊填零
- 加载耗时：~22s（空间连接一次性计算）
- 到达/实时人口相关系数 r=0.82，规模比≈9.8%（流量 vs 存量概念不同）

### Phase 1 新增 CLI 功能

```bash
# 行政区空间分块验证
python3.12 -m ... --split-strategy district --holdout-district 福田区

# 基线对比（Ridge/GBT/MLP）
python3.12 -m ... --baselines

# 高误差街坊诊断
python3.12 -m ... --diagnose 20

# 可用行政区：福田区 南山区 罗湖区 宝安区 龙岗区 龙华区 光明区 盐田区 坪山区 大鹏新区
```

`ShenzhenVitalityDataset` 新增 `districts: Optional[np.ndarray]` 字段。

---

## 当前性能（seed=42）

| 版本 | val MAE | 说明 |
|---|---|---|
| 朴素基线 | 4285 | 训练集均值 |
| Ridge | 1494 | 线性基线 |
| MLP (无Agent) | 1234 | 同特征，无结构 |
| GBT | 1067 | 非结构化最强基线 |
| 原始Agent（78特征） | 1769 | Phase 2最佳 |
| + scale_net（无OD） | 1766 | 微幅改善 |
| **+ OD特征（174特征）+ scale_net** | **1388** | **Phase 3，超越Ridge** |

**相关系数**：0.857 → 0.923  
**中位绝对误差**：764 → 619  
**按活力分层（Phase 3）**：low=227 / medium=719 / high=1299 / top=3315

### 多Seed验证（Phase 3）

| | Phase 2（无OD） | Phase 3（+OD+scale_net） |
|---|---|---|
| 均值 MAE | 2593 | **1337** |
| 标准差 | ±530 | **±74** |
| seed 42/123/456 | 1852/2868/3059 | 1368/1236/1408 |

方差从 ±530 降至 ±74（**-86%**）：OD特征使模型对随机split不再敏感，体现真实泛化能力。

---

## 已验证的失败路径（勿重复）

1. **时序因子化用于路由**：attract_net→(N,T)然后global softmax → 全局零和竞争×48时段 → 灾难性过拟合（val MAE 143422）
2. **冗余scale参数**：log_scale + block_scale + temporal_logits 三者相加 → 优化不稳定，val MAE 2782
3. **替换log_scale（无bias scale_net）**：丢失全局LBS校正 → val MAE 2076
4. **Local softmax（任何条件下）**：没有真实OD流向对时，截断全市级吸引力 → top层显著变差

---

## 文件结构

```
agent_torch/models/urban_vitality_shenzhen/
├── __init__.py     # get_registry(), create_runner()（支持split_strategy参数）
├── data.py         # 数据加载、OD聚合(_aggregate_od_to_blocks)、图构建、build_config()
├── train.py        # train_model(), run_baselines(), diagnose_errors(), train_multi_seed()
├── main.py         # CLI入口（--split-strategy, --holdout-district, --baselines, --diagnose）
└── substeps/
    ├── move.py     # MovePolicy（home_logits + spatial_attn + attract_net + scale_net）
    └── aggregate.py # AggregateVitality（log_scale + global softmax）
tests/test_urban_vitality_shenzhen.py  # 6个测试，全部通过
```

---

## 路线图（当前状态）

| 阶段 | 状态 | 核心成果 |
|---|---|---|
| Phase 0 | ✅ 完成 | 多seed验证，可复现性报告 |
| Phase 1 | ✅ 完成 | 空间分块验证、基线对比、误差诊断 |
| Phase 2 | ✅ 完成 | 8类新POI特征（78列） |
| Phase 3 | ✅ 完成 | OD聚合特征（174列），MAE 1769→1388 |
| Phase 4 | 🔲 未开始 | 干预场景变量 |
| Phase 5 | 🔲 未开始 | 多轮演化 |

**Why:** OD特征提供了静态建筑/POI特征无法推断的时序活动剖面，scale_net 将其直接映射为 per-block 时序尺度修正，这是当前最有效的改进方向。
**How to apply:** 当前架构已稳定，下一步优先Phase 4（干预场景），需要先定义干预变量语义再写代码。
