# AgentTorch 框架核心架构解析

## 五个核心文件总览

### 文件架构图

```
用户代码
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│  runner.py ← 总指挥 / 入口                               │
│  - runner.init()       → 启动 Initializer               │
│  - runner.reset_state()→ 重置状态（不重建模块，用于训练） │
│  - runner.step()       → 驱动 Controller 循环            │
└────────────────────┬────────────────────────────────────┘
                     │ 调用
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌──────────────────────────────────┐
│  controller.py   │   │  initializer.py                  │
│  ← 调度员        │   │  ← 建造者                        │
│  .observe()      │   │  从 YAML 读配置，                 │
│  .act()          │   │  用 Registry 查找类，             │
│  .progress()     │   │  实例化 state + 所有 substep 模块 │
└──────────────────┘   └──────────────────────────────────┘
          │ 调用                  │ 依赖
          ▼                      ▼
┌──────────────────┐   ┌──────────────────────────────────┐
│  substep.py      │   │  registry.py                     │
│  ← 业务逻辑基类  │   │  ← 插件注册表                    │
│  SubstepObser.   │   │  registry.register(              │
│  SubstepAction   │   │    MovePolicy, "MovePolicy",     │
│  SubstepTransit. │   │    "policy")                     │
└──────────────────┘   └──────────────────────────────────┘
```

---

## 每个文件的主要作用

### 1. `registry.py` — 插件注册表（最先运行）

**文件路径**：`agent_torch/core/registry.py`

**解决的问题**：YAML 配置文件里写的是字符串 `"MovePolicy"`，代码怎么知道去实例化哪个 Python 类？

```python
registry = Registry()
registry.register(MovePolicy, "MovePolicy", "policy")
# 现在 "MovePolicy" 这个字符串就和 MovePolicy 类绑定了
```

**注册表的 5 个槽位**：

| 槽位名 | 存放内容 |
|--------|----------|
| `transition` | SubstepTransition 的子类（状态转移） |
| `observation` | SubstepObservation 的子类（观察） |
| `policy` | SubstepAction 的子类（策略/行动） |
| `initialization` | 状态初始化函数（如从 CSV 读数据） |
| `network` | 图/网络构建函数（如生成 k-NN 图） |

---

### 2. `initializer.py` — 建造者（`runner.init()` 时运行一次）

**文件路径**：`agent_torch/core/initializer.py`

**两件事**：
1. **构建初始 `state` 字典**（从 YAML config 读取，构建 Tensor）
2. **实例化所有 substep 模块**（从 Registry 查找类，按 YAML 参数初始化）

**构建完成后 state 的结构**：

```python
state = {
    "current_step": 0,
    "current_substep": "0",
    "environment": {
        "vitality": Tensor,
        "block_features": Tensor,
        ...
    },
    "agents": {
        "residents": {
            "home_block": Tensor,
            "weight": Tensor,
            ...
        }
    },
    "network": { ... },
    "parameters": nn.ParameterDict
}
```

**关键方法**：

| 方法 | 作用 |
|------|------|
| `initialize()` | 完整初始化 state + substep 模块 |
| `reset_state()` | 只重置 state 张量，不重建 substep 模块（训练用） |
| `simulator()` | 依次调用 init_environment / init_agents / init_objects / init_network |
| `substeps()` | 从 Registry 查找并实例化所有 Observation / Policy / Transition 模块 |

---

### 3. `substep.py` — 业务逻辑的"接口"（你写代码继承它）

**文件路径**：`agent_torch/core/substep.py`

三个抽象基类，必须继承并实现 `forward()`：

| 基类 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `SubstepObservation` | 智能体"看"世界 | `state` | observation dict |
| `SubstepAction` | 智能体"决策" | `state + observation` | action dict |
| `SubstepTransition` | 世界"更新" | `state + action` | 新状态变量 dict |

另有 `SubstepTransitionMessagePassing`，用于图结构（PyTorch Geometric）上的消息传递式状态转移。

**深圳城市活力模型的具体实现例子**：

| 文件 | 继承自 | 职责 |
|------|--------|------|
| `substeps/move.py → MovePolicy` | `SubstepAction` | 计算留家概率 / 街坊吸引力 / 规模修正 |
| `substeps/aggregate.py → AggregateVitality` | `SubstepTransition` | 把人口散射到街坊，输出活力预测值 |

---

### 4. `controller.py` — 调度员（每个 substep 调用一次）

**文件路径**：`agent_torch/core/controller.py`

**只负责"在正确时机调用正确函数"，不含业务逻辑**。

每个 substep 的执行顺序：

```
controller.observe()   → 调 SubstepObservation.forward(state)
controller.act()       → 调 SubstepAction.forward(state, observation)
controller.progress()  → 调 SubstepTransition.forward(state, action)
                       → 把返回值按 YAML 路径写回 state
```

**`progress()` 的关键细节**：
- 先 `copy_module(state)` 生成 state 副本再修改，保证梯度图不被破坏，支持反向传播
- `current_substep` 在此处自动递增（环形，到头回 0）
- transition 的返回值按 YAML config 中定义的路径写回 state（如 `"environment/vitality"`）

---

### 5. `runner.py` — 总指挥（用户代码的直接调用对象）

**文件路径**：`agent_torch/core/runner.py`

```python
runner = Runner(config, registry)
runner.init()                 # 调 Initializer，构建 state + substep 模块

for epoch in range(N):
    runner.reset_state()      # 只重置 state 张量，不重建 substep 模块
                              # 可学习参数保持不变（优化器继续跟踪）
    runner.step()             # 跑完一个 episode：num_steps × num_substeps

    loss = compute_loss(runner.state_trajectory)
    loss.backward()
    optimizer.step()
```

**`reset()` vs `reset_state()` 的重要区别**：

| 方法 | 行为 | 适用场景 |
|------|------|----------|
| `reset()` | 完整重建（等同于重新 init） | 完全重跑仿真 |
| `reset_state()` | 只重置 state 张量，substep 模块不动 | 训练循环（梯度保留） |

**`state_trajectory`**：每个 substep 结束后，state 被 `to_cpu()` 拷贝一份存入列表，用于后续损失计算或分析。

---

## 一次完整 step 的数据流（以深圳模型为例）

```
runner.step()
  └── for time_step in range(num_steps):
        └── for substep in ["0"]:   ← 深圳模型只有 1 个 substep
              │
              ├── controller.observe("residents")
              │     └── GetFeatures.forward(state)
              │           返回: {"block_features": Tensor(3023, F)}
              │
              ├── controller.act("residents", observation)
              │     └── MovePolicy.forward(state, observation)
              │           返回: {
              │             "p_home":           Tensor(4, 48),   ← 4个人口群体×48时段留家概率
              │             "attract_logits":   Tensor(3023, 1), ← 各街坊吸引力
              │             "block_log_scale":  Tensor(3023, 48) ← 各街坊时段规模修正
              │           }
              │
              └── controller.progress(state, action)
                    └── AggregateVitality.forward(state, action)
                          返回: {
                            "predicted_vitality":        Tensor(3023, 48), ← 原始人口数
                            "predicted_vitality_scaled": Tensor(3023, 48)  ← 标准化对数值
                          }
                    → 写回 state["environment"]["vitality"]
```

---

## 深圳模型的核心数学公式

```
V(j, t) = [home_vitality(j,t) + away_vitality(j,t)] × scale(j,t)

home_vitality(j, t)  = Σ_{i∈block j} weight_i × p_home[demo_i, t]
                     → 街坊 j 在 t 时段"留在本街坊"的居民人口数

away_vitality(j, t)  = softmax(attract_logits)[j] × Σ_i weight_i × (1 - p_home[demo_i, t])
                     → 全市外出人口中，被吸引到街坊 j 的那部分（全局 softmax 路由）

scale(j, t)          = exp(log_scale_global[t] + block_log_scale[j, t])
                     → LBS 采样率修正
```

---

## 总结

> **Registry 注册类 → Initializer 用类构建 state 和模块 → Runner 驱动 Controller 循环 → Controller 按顺序调 substep 的三个 forward() → 你的业务逻辑在 substep 里实现**

**你实际需要写代码的只有 substep**（继承 `SubstepAction` / `SubstepTransition` / `SubstepObservation`），其余四个文件是框架固定的"骨架"，一般不需要修改。

---

*生成日期：2026-06-07*
