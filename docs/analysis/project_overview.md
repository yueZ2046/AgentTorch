---
name: project-overview
description: AgentTorch完整架构深度解析：核心层、LLM子系统、可微分分布、向量化、所有内置模型及关键设计决策
metadata: 
  node_type: memory
  type: project
  originSessionId: 4cae5642-aebe-443d-85db-f1f65d31e291
---

## 项目定位

AgentTorch（MIT Media Lab, Ayush Chopra）是一个**大规模人口模型(LPM)**框架，定位为"面向Agent仿真的PyTorch"。

**四大设计原则**：
1. **Scalability** — 百万级Agent，商用硬件到集群均可
2. **Differentiability** — 梯度可穿过随机动态和条件干预
3. **Composition** — 可与LLM/神经网络/其他LPM组合
4. **Generalization** — 流行病、经济、城市、生态等多域通用

---

## 核心运行时（5个关键组件）

### 1. `core/initializer.py` — 状态与模块构建

- 解析YAML config，初始化 `environment / agents / objects / networks` 张量
- 构建 `observation_function / policy_function / transition_function / reward_function`（均为 `nn.ModuleDict`）
- 设备管理：`cfg_dev = "auto"` 时自动选择 CUDA；写回 `config["simulation_metadata"]["device"]`
- CUDA优化：`_num_streams=4` 流并发传输（pin_memory + non_blocking）
- `reset_state()` — **只重置状态张量，不重建substep模块**（训练循环专用，保留learnable参数的梯度追踪）

### 2. `core/runner.py` — 主仿真循环

继承 `nn.Module`，核心分支：

```
step() → GPU: _step_gpu_optimized()
       → CPU: _step_cpu_base()
```

**CPU路径**：标准 observe→act→progress，每substep后 `to_cpu(state)` 存入 `state_trajectory`

**GPU路径优化**：
- 内存池(`memory_pool`) — 按(shape,dtype,device)为key，避免反复分配
- CUDA流(`_snapshot_stream`) — 异步CPU←GPU快照
- `_leased_tensors` — 租出的池张量，substep结束后自动归还
- `_compress_state_for_snapshot()` — 只快照environment+元数据，float降到fp32，int64→int32，bool→uint8
- `_wire_transition_buffer_allocator()` — 将池分配器注入transition模块的 `_get_buffer`

`_set_parameters()` / `step_from_params()` — 支持外部注入参数（校准用）

### 3. `core/controller.py` — 单步执行调度

```python
observe(state, observation_function, agent_type) -> dict
act(state, observation, policy_function, agent_type) -> dict
progress(state, action_profile, transition_function) -> next_state
```

- `progress()` 用 `copy_module(state)` 深拷贝（保留梯度图），`set_by_path()` 按路径写回状态
- `progress_inplace()` — 原地更新，避免拷贝（某些场景更快）
- `_obs_keys_cache / _policy_keys_cache` — 按(substep, agent_type)缓存键列表，减少热路径Python开销

### 4. `core/registry.py` — 函数注册中心

```python
Registry.helpers = {
    "transition": {}, "observation": {}, "policy": {},
    "initialization": {}, "network": {}
}

# 两种注册方式：
registry.register(cls, name, key)
@Registry.register_helper(name, key)  # 类方法装饰器
```

### 5. `core/substep.py` — Substep抽象基类

- `SubstepObservation(nn.Module, ABC)` — 抽象 `forward(self, state)`
- `SubstepAction(nn.Module, ABC)` — 抽象 `forward(self, state, observation)`
- `SubstepTransition(nn.Module, ABC)` — 抽象 `forward(self, state, action)`
- `SubstepTransitionMessagePassing(MessagePassing, ABC)` — 图消息传递版本（`aggr="add"`）

所有基类统一处理 `learnable_args / fixed_args`；当 `calibration=True` 时自动创建 `calibrate_<key>` 属性（requires_grad）。

---

## LLM子系统（`core/llm/`）

### 调用层次

```
Archetype.sample()
  └── Behavior.sample()
        ├── Template.get_grouped_prompts()  [Template路径]
        └── PromptManager.get_prompt_list() [base路径]
              └── LLMArchetype.__call__()
                    └── LLMBackend.prompt()
```

### 关键类

**`LLMBackend`**（抽象）：
- `DspyLLM` — DSPy + OpenAI（默认gpt-4o-mini），线程池并发调用
- `MockLLM` — 测试用，返回[low,high]随机值

**`Archetype`（门面）**：
```python
arch = Archetype(prompt, llm, n_arch=1)
arch.broadcast(population, match_on=..., group_on=...)  # 绑定人口
arch.configure(external_df=df, split=100)              # 配置外部数据
result = arch.sample(kwargs, verbose=True)             # 返回 (n_agents,) 张量
```

**`Behavior`**：
- Template路径：按grouping_logic分组 → 每组一个prompt → 平均n_arch个archetype输出 → scatter回(n_agents,1)
- 支持P3O：`template.create_slots()` → 对learnable Variable采样 → `template.set_optimized_slots()`
- `export_memory_to_file()` — 保存对话历史

**`Template`(dataclass)**：
- `src` — 外部数据文件路径
- `grouping_logic` — 按哪些列分组（str或list）
- `Variable` descriptor — `{field_name, learnable=True/False}`
- P3O slot优化：`get_slot_parameters()` 供外部优化器使用

**`Variable`**：可学习的prompt slot，支持REINFORCE梯度估计。

---

## 可微分分布（`core/distributions/`）

使用**stochastic triple trick**实现离散采样的可微分梯度估计：

| 类 | 前向 | 反向（梯度权重ws） |
|---|---|---|
| `StraightThroughBernoulli` | bernoulli采样 | ws=1（直通） |
| `Bernoulli` | bernoulli采样 | ws = 1/(2p) 或 1/(2(1-p)) |
| `Binomial` | binomial采样 | wminus/wplus平均 |
| `Geometric` | 逆CDF采样 | 跳上/跳下权重 |
| `Categorical` | multinomial采样 | one_hot × w_chosen |

所有分布均设置 `generate_vmap_rule = True`，支持 `torch.vmap`。

---

## 向量化工具（`core/helpers/vmap.py`）

```python
@vmap(agent_args=["position", "alive"],
      shared_args=["sugar_grid"],
      outputs=["position"])
class AgentMovement(SubstepTransition):
    def forward(self, state, action=None):
        # state 为单个Agent！
        position = state["position"]   # [2]
        sugar_grid = state["sugar_grid"]  # [H, W] 共享
        ...
        return {"position": new_position}
```

- `torch.vmap(..., randomness='different')` — 每Agent独立随机数
- `compile=True` 参数可额外启用 `torch.compile`
- `sample_grid(grid, position)` — 双线性插值采样2D网格（内部使用 `F.grid_sample`）

---

## Config系统（`config/`）

YAML结构（与`build_config()`对应）：
```yaml
simulation_metadata:
  num_agents: ...
  num_episodes: ...
  num_steps_per_episode: ...
  num_substeps_per_step: ...
  device: auto
  calibration: false
state:
  environment: {prop: {name, shape, dtype, learnable, value, initialization_function}}
  agents:
    <type>:
      properties: {prop: ...}
  objects: ...
  network: ...
substeps:
  "0":
    name: ...
    active_agents: [<type>]
    observation:
      <type>: {func: {generator, input_variables, output_variables, arguments}}
    policy: ...
    transition: ...
```

`config/state_builder.py` 和 `substep_builder.py` 提供编程式构建API（`config_api`教程）。

---

## 内置模型

### COVID（`models/covid/`）
- SEIRM流行病模型，NYC Astoria区 37,518人
- Substeps：`new_transmission / quarantine / seirm_progression / testing`
- `calibnn.py` — 神经网络校准器，输入特征→感染率参数
- 支持 `llm/utils/llm.py` 实现LLM驱动的隔离决策

### Macro Economics（`models/macro_economics/`）
- NYC五区消费/收入多Agent模型
- 用于分析刺激政策对失业率的影响（stimulus vs no-stimulus对比数据）

### Boids（`examples/models/boids/`）
- 经典集群涌现仿真（alignment/cohesion/separation）
- 全GPU向量化实现

### Movement（`examples/models/movement/`）
- 随机游走基准模型，带分布式版本（`run_movement_sim_distributed.py`）

---

## 在研模型：Shenzhen Urban Vitality

见 [[shenzhen-vitality-model]] 详细记录。

---

**Why:** 框架核心权衡是可微分性（梯度优化参数）与大规模并发（GPU向量化）的统一，Runner对两者分别优化。
**How to apply:** 新功能优先保持可微分；新模型按substep三段式拆分；训练循环用 `reset_state()` 而非 `reset()` 保留参数追踪。
