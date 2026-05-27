# Urban Vitality Shenzhen — 进度记录

## 当前最佳结果

| 指标 | 数值 |
|------|------|
| 验证集 MAE | **1886** |
| 朴素基线 MAE | 4285 |
| 超越基线 | **56%** |
| 训练集 MAE | 2957 |
| 训练损失（400 epoch） | 0.1556 |

## 已完成的工作

### 模型架构（Agent-based）
- **Agent**：3023 街坊 × 4 人口群体 = 12092 个居民 Agent
- **涌现机制**：活力 = 居家人口（scatter_add）+ 外出人口（全局 softmax）
- **MovePolicy**：`home_logits(4×48)` + `SpatialAttentionAggregation` + `attract_net(n_feat→1)`
- **AggregateVitality**：散射聚合 + `log_scale(48)` 尺度因子

### 特征工程
- 基础特征：建筑面积、容积率等（约 51 列）
- 人口画像：CZZL 年龄段、就业、性别、密度等（+15 列）
- POI 特征：医疗/风景/汽车销售/摩托车（+4 列，log1p 变换）
- **总特征数：70**

### 空间网络
- **Neighbor80**（80m）：解析 `街坊范围shp/深圳_街坊_Pro.shp` 的 `Neighbor80` 字段，55738 条边 → 已弃用于 MovePolicy
- **k-NN 2km**（k=30）：scipy cKDTree 计算重心距离，90690 条边 → 当前用于 MovePolicy 注意力聚合
- **局部 softmax 实验**：尝试过将外出人口限制在邻域内分配，结果更差（无 OD 数据时不如全局 softmax）

### 关键设计决策
1. `home_logits` 用时序先验初始化（夜间高 p_home，白天低）
2. `attract_net` 输出 (N_blocks, 1) 而非 (N_blocks, 48)，避免吸收时序信号
3. `attract_net` weight_decay=0.1（防止全局 softmax 零和竞争过拟合）
4. `home_logits` 使用 5× 学习率（梯度比 attract_net 小约 14 倍）
5. `log_scale(48)` 用 2× 学习率，修复 LBS 采样与居住人口的尺度不匹配

## 文件结构

```
agent_torch/models/urban_vitality_shenzhen/
├── __init__.py          # get_registry(), create_runner()
├── data.py              # 数据加载、图构建、build_config()
├── train.py             # train_model(), prediction_frame()
├── main.py              # CLI 入口
└── substeps/
    ├── move.py          # MovePolicy: home_logits + SpatialAttentionAggregation + attract_net
    └── aggregate.py     # AggregateVitality: scatter + log_scale
tests/
└── test_urban_vitality_shenzhen.py   # 6 个测试全部通过
outputs/
├── vitality_overview.png    # 工作日/周末 预测 vs 实测
├── vitality_timeslots.png   # 8h/13h/20h 分时段对比
└── vitality_eval.png        # 散点图 + 分时段 MAE 曲线
```

## 运行方式

```bash
cd /home/ubuntu/agenttorch

# 快速训练（约 2-4 分钟，CPU）
python3.12 -m agent_torch.models.urban_vitality_shenzhen.main --epochs 400

# 运行测试
python3.12 -m pytest tests/test_urban_vitality_shenzhen.py -v

# 自定义训练
python3.12 - <<'EOF'
import sys, torch; sys.path.insert(0, ".")
from agent_torch.models.urban_vitality_shenzhen.train import train_model
runner, dataset, metrics, history = train_model("data_shenzhen", epochs=400)
print(f"val MAE={metrics['validation']['mae']:.0f}")
EOF
```

## 下一步计划

### 优先级高
- [ ] **补全 POI 数据**（用户提供）：购物、餐饮、地铁站等 → 这是当前最大的特征缺口
  - 补完后重新训练，预期验证集 MAE 进一步下降
  - POI 数据到位后重新评估局部 softmax（用 `edge_index_mobility`）

### 优先级中
- [ ] **利用 LBS 原始数据**（`data_shenzhen/LBS原始数据/`）：网格级到达/出发人口
  - 可提取真实 OD 流，为局部 softmax 提供路由约束
  - 文件：`工作日实时人口_网格粒度.shp`、`工作日到达人口_网格.shp` 等

### 优先级低
- [ ] **更多 epoch + 余弦 LR 调度**：当前 400 epoch 损失仍在下降
- [ ] **2-hop GNN**：目前是 1-hop 注意力，可在 k-NN 图上叠加第二层

## 实验记录（验证集 MAE）

| 版本 | val MAE | 说明 |
|------|---------|------|
| 朴素基线（训练均值） | 4285 | 预测所有街坊取训练集均值 |
| 全局 softmax，无空间 | ~9000 | 早期版本，严重过拟合 |
| + Neighbor80 spatial_proj | 2660 | 一跳均值聚合 |
| + POI 特征（4类，log1p） | 2588 | 医疗/风景/汽车/摩托车 |
| + 注意力 k-NN 2km | 2571 | GAT 替换均值聚合 |
| **+ log_scale 尺度因子** | **1886** | 修复 LBS 采样偏差，当前最佳 |
