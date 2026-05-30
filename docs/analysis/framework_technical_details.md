---
name: framework-technical-details
description: AgentTorch框架底层技术细节：可微分分布、vmap向量化、YAML配置规范、LLM后端、数据加载、分析器
metadata: 
  node_type: memory
  type: project
  originSessionId: 4cae5642-aebe-443d-85db-f1f65d31e291
---

## 可微分离散分布（`core/distributions/distributions.py`）

使用**stochastic triple trick**，对离散采样实现无偏梯度估计：

```python
from agent_torch.core.distributions import Bernoulli, Binomial, Geometric, Categorical

# 直通估计器（更激进）
result = StraightThroughBernoulli.apply(p)  # backward: ws=1

# 随机三元组（无偏）
result = Bernoulli.apply(p)  # backward: ws = 1/(2p) 或 1/(2(1-p))
```

所有分布设 `generate_vmap_rule = True`，与 `torch.vmap` 兼容。

---

## 向量化工具（`core/helpers/vmap.py`）

### @vmap 装饰器

单Agent逻辑自动向量化到全部Agent：

```python
from agent_torch.core.substep import SubstepTransition, vmap
from agent_torch.core.helpers.vmap import sample_grid

@vmap(
    agent_args=["position", "alive"],   # 按dim=0向量化
    shared_args=["sugar_grid"],         # 不向量化（环境共享）
    outputs=["position"],
    compile=False                        # 可选 torch.compile
)
class AgentMovement(SubstepTransition):
    def forward(self, state, action=None):
        # state 为单个Agent的数据（无batch维度）
        position = state["position"]   # shape [2]
        sugar_grid = state["sugar_grid"]  # shape [H, W]
        return {"position": new_position}
```

底层：`torch.vmap(fn, randomness='different')`

### sample_grid

```python
value = sample_grid(grid, position)  # 双线性插值，返回标量
```

内部用 `F.grid_sample`，坐标归一化到 [-1,1]。

---

## LLM后端（`core/llm/backend.py`）

### DspyLLM

```python
from agent_torch.core.llm.backend import DspyLLM
llm = DspyLLM(openai_api_key="...", qa=MyQA, cot=MyCoT, model="gpt-4o-mini")
llm.initialize_llm()  # 配置 dspy.settings
```

特性：`ThreadPoolExecutor` 并发调用，返回 `{"text": answer}` 列表。

### MockLLM（测试用）

```python
from agent_torch.core.llm.mock_llm import MockLLM
llm = MockLLM(low=0.1, high=0.9)  # 返回均匀随机值
```

### 自定义后端

继承 `LLMBackend`，实现 `prompt(prompt_list) -> List[{"text": str}]`。

---

## Config编程API（`config/`）

### 通过Python代码构建Config

```python
from agent_torch.config import state_builder, substep_builder

# 定义environment变量
state = state_builder.build_state(
    environment={"temperature": {"name": "temperature", "shape": [1], ...}},
    agents={"citizens": {"properties": {...}}},
)

# 定义substep
substep = substep_builder.build_substep(
    name="update",
    active_agents=["citizens"],
    observation={"citizens": {"observe_temp": {...}}},
    policy={...},
    transition={...},
)
```

`config/example_usage.py` 有完整示例；`docs/tutorials/config_api/` 有Jupyter notebook。

---

## 数据加载（`core/dataloader.py`）

```python
from agent_torch.core.dataloader import LoadPopulation

pop = LoadPopulation(region)         # 加载pickle/parquet人口文件
pop.population_size                  # 总人口数
pop.data                             # DataFrame
```

`LinkPopulation` — 将人口数据链接到模拟配置。

---

## 分析器（`core/analyzer/`）

用于仿真后分析：

- `simulation_analyzer.py` — 分析完整轨迹
- `agent_graph.py` — Agent交互图分析
- `retriever.py` — 从state_trajectory中提取数据
- `datamodels.py` — 分析数据模型

---

## 装饰器工具（`core/decorators.py`）

为substep/agent提供辅助装饰器，简化注册流程。

---

## 向量化Runner（`core/vectorized_runner.py`）

基础Runner的向量化变体，适合内存受限场景，见 `docs/tutorials/runner_optimizations/`。

---

## 分布式Runner（`core/distributed_runner.py`）

多进程/多机分布式仿真，见示例 `agent_torch/examples/run_movement_sim_distributed.py`。

---

## 环境助手（`core/helpers/environment.py`）

```python
from agent_torch.core.environment import envs

runner = envs.create(model=movement, population=astoria)
```

高层封装，一行代码创建Runner。

---

## 关键工具函数（`core/helpers/general.py`）

- `get_by_path(obj, path_list)` / `set_by_path(obj, path_list, value)` — 路径式访问嵌套dict
- `copy_module(module)` — 深拷贝（保留tensor梯度图，不断开计算图）
- `to_cpu(state)` — 递归将state中所有tensor移到CPU
- `get_var(state, path)` — 从state按路径取变量

---

## P3O提示优化（`core/llm/Variable.py`，`core/llm/template.py`）

P3O（Prompt Policy Optimization）：用REINFORCE估计优化prompt的呈现方式：

```python
from agent_torch.core.llm.Variable import Variable
from agent_torch.core.llm.template import Template

class MyTemplate(Template):
    presentation_style = Variable(choices=["tabular", "narrative"], learnable=True)
```

- `Variable.sample_index(template)` → `(idx, log_prob, entropy)`
- optimizer对 `arch.parameters()` 做梯度更新

---

## 测试结构

```
tests/
├── fixtures/behavior.py    # Behavior测试夹具
├── fixtures/executor.py    # Executor测试夹具
├── mocks/llm.py            # MockLLM
├── test_behavior.py        # Behavior/Archetype测试
├── test_executor.py        # Executor测试
├── test_initializer.py     # Initializer测试
├── test_urban_vitality_shenzhen.py  # 6个测试，全部通过
└── test_vmap_substeps.py   # vmap装饰器测试
```

运行：`python3.12 -m pytest tests/ -v`

---

**Why:** 这些底层工具是构建新模型时的可复用积木，了解它们避免重复实现。
**How to apply:** 写新substep时优先考虑 `@vmap` 向量化；需要离散概率采样时用 `core/distributions/` 的可微分版本，不要用原生 `torch.bernoulli`。
