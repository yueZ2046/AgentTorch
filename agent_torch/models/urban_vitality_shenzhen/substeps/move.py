"""居民移动决策模块（MovePolicy）。

这是仿真的"大脑"，负责在每步仿真时输出三个核心量：
    1. p_home          (n_demo, 48)  — 各人口群体在各时段的留家概率
    2. attract_logits  (N_blocks, 1) — 各街坊对外来人口的吸引力（未经归一化的 logit）
    3. block_log_scale (N_blocks, 48)— 各街坊各时段的活力规模修正（对数空间）

这三个量传递给 AggregateVitality，完成人口的空间路由和活力计算。
"""

import math

import torch
from torch import nn

from agent_torch.core.helpers import get_var
from agent_torch.core.substep import SubstepAction


def _temporal_prior(n_demo: int, n_time: int) -> torch.Tensor:
    """生成留家倾向的初始先验，形状为 (n_demo, n_time)。

    用余弦函数模拟城市居民典型的昼夜节律：
        - 凌晨（0–5时）：留家概率高 → 余弦值为正
        - 白天（8–18时）：外出为主  → 余弦值为负
        - 工作日（前24列）幅度 2.2，周末（后24列）幅度 1.2
          （周末早晨更多人在家，但外出时间更灵活）

    这只是训练的起点，不是硬编码约束，会通过反向传播更新。
    所有人口群体（青少年/青年/中年/老年）初始使用相同先验，
    训练后各组会学到不同的时段模式。

    参数：
        n_demo: 人口群体数量（默认 4）
        n_time: 时间槽总数（默认 48 = 工作日24 + 周末24）
    """
    n_half = n_time // 2                        # 24（工作日或周末各一半）
    hours  = torch.arange(n_half, dtype=torch.float32)

    # 工作日：峰值在凌晨 2 点（hour=2），表示此时留家概率最高
    wd = 2.2 * torch.cos(2 * math.pi * (hours - 2.0) / n_half)

    # 周末：峰值在凌晨 3 点（晚睡晚起），幅度较小（更多人可能在家休息）
    we = 1.2 * torch.cos(2 * math.pi * (hours - 3.0) / n_half)

    prior_1d = torch.cat([wd, we])              # (48,) 工作日+周末拼接
    # expand 后 clone 确保每个群体有独立的参数副本，不共享内存
    return prior_1d.unsqueeze(0).expand(n_demo, -1).clone()


class SpatialAttentionAggregation(nn.Module):
    """单头图注意力：让每个街坊的特征融合其 k-NN 邻居的信息。

    为什么需要空间注意力？
        单个街坊的静态特征（建筑面积、POI 数）只反映本地情况，
        但实际活力受周边功能区影响——
        例如：紧邻大型商圈的居住街坊，其居民也会被吸引到商圈去。
        2km k-NN 图（edge_index_mobility）捕捉通勤尺度的功能区聚集效应。

    机制（单头缩放点积注意力）：
        对每条边 (center → neighbor)：
            score = dot(Q[center], K[neighbor]) / sqrt(d)
        对每个中心节点归一化 → 注意力权重 attn
        输出 = 原始特征 + 线性变换（加权邻居值的聚合）

    注意：out 层权重用很小的初始值（std=1e-3），bias 初始化为 0，
    保证训练初期注意力几乎不改变特征（稳定训练开始时的梯度）。
    """

    def __init__(self, n_feat: int):
        super().__init__()
        d = max(n_feat // 4, 16)               # 注意力内部维度，比特征维度小以降低参数量
        self.query = nn.Linear(n_feat, d, bias=False)   # 查询向量：描述"我想聚合什么"
        self.key   = nn.Linear(n_feat, d, bias=False)   # 键向量：描述"邻居能提供什么"
        self.value = nn.Linear(n_feat, n_feat, bias=False)  # 值向量：实际传递的内容
        self.out   = nn.Linear(n_feat, n_feat)          # 输出投影（有 bias）
        nn.init.normal_(self.out.weight, std=1e-3)       # 小初始值，保证训练初期稳定
        nn.init.zeros_(self.out.bias)
        self._scale = d ** -0.5                          # 缩放因子，防止点积值过大导致梯度消失

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        参数：
            features:   (N_blocks, n_feat) 街坊特征矩阵
            edge_index: (2, E) 稀疏图的边列表，edge_index[0]=中心节点，edge_index[1]=邻居节点

        返回：
            (N_blocks, n_feat) 融合了邻居信息的增强特征（残差结构）
        """
        center   = edge_index[0]               # (E,) 每条边的中心街坊索引
        nbr      = edge_index[1]               # (E,) 每条边的邻居街坊索引
        n_blocks = features.shape[0]

        q = self.query(features)               # (N_blocks, d)
        k = self.key(features)                 # (N_blocks, d)
        v = self.value(features)               # (N_blocks, n_feat)

        # 计算每条边的注意力得分（clamp 防止 exp 溢出）
        score = (q[center] * k[nbr]).sum(-1) * self._scale   # (E,)
        exp_s = torch.exp(score.clamp(-20, 20))              # (E,) softmax 的分子

        # 对每个中心节点，把其所有邻居的 exp_s 相加（用于归一化）
        sum_s = torch.zeros(n_blocks, device=features.device)
        sum_s.scatter_add_(0, center, exp_s)

        # 归一化得到注意力权重（分母加 1e-10 防止除以零）
        attn = exp_s / (sum_s[center] + 1e-10)               # (E,)

        # 加权聚合邻居的值向量
        v_weighted = v[nbr] * attn.unsqueeze(1)              # (E, n_feat)
        agg = torch.zeros_like(features)
        agg.scatter_add_(0, center.unsqueeze(1).expand_as(v_weighted), v_weighted)  # (N_blocks, n_feat)

        # 残差连接：原始特征 + 聚合结果经过线性变换
        return features + self.out(agg)


class MovePolicy(SubstepAction):
    """居民移动决策网络，学习三个互补的活力组成部分。

    组件 1：home_logits (n_demo, 48) — 可训练参数
        每个人口群体在每个时段"留家"的倾向（log-odds 形式）。
        sigmoid(home_logits) = p_home 留家概率（值域 0~1）。
        这捕捉了不同年龄群体的日常节律差异（老年人更多待家，青年外出多）。

    组件 2：attract_net → (N_blocks, 1) — 两层 MLP
        把街坊特征映射到标量吸引力分数。
        通过全局 softmax 决定外出人口如何分配到各街坊。
        捕捉功能区差异：商业街坊吸引力高，纯居住街坊吸引力低。

    组件 3：scale_net: Linear(n_feat, 48) — 无偏置线性层
        把街坊特征直接映射到 48 个时段的对数规模修正值。
        无偏置设计避免与 AggregateVitality 里的全局 log_scale 参数重叠。
        权重初始化为全零 → 训练开始时 scale=1（不修正），
        逐步学到商业街坊中午活力高、住宅区深夜活力高等模式。

    为什么把 scale 独立出来而不合并进 attract_net？
        attract_net 控制的是人口的空间路由比例（相对量），
        scale_net  控制的是绝对规模（LBS 采样率的时空异质性修正），
        两者物理意义不同，分开有助于训练稳定性和可解释性。
    """

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        meta   = config["simulation_metadata"]
        n_demo = int(meta["num_demo_groups"])   # 人口群体数，默认 4
        n_time = int(meta["num_targets"])        # 时段总数，默认 48
        n_feat = int(meta["num_features"])       # 特征维度（训练数据决定）
        hidden = int(meta.get("hidden_dim", 64)) # attract_net 隐藏层维度

        # 留家倾向参数：用时段先验初始化，后续反向传播更新
        self.home_logits = nn.Parameter(_temporal_prior(n_demo, n_time))

        # 如果配置中有 edge_index 输入（k-NN 图），启用空间注意力
        if "edge_index" in input_variables:
            self.spatial_attn = SpatialAttentionAggregation(n_feat)
        else:
            self.spatial_attn = None

        # 吸引力网络：街坊特征 → 标量吸引力
        # ReLU 激活层引入非线性，允许模型捕捉特征间的复杂交互
        self.attract_net = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # 规模修正网络：街坊特征 → 48 个时段的对数修正值
        # 无偏置（bias=False）：避免与 AggregateVitality 的全局 log_scale 参数功能重叠
        # 全零初始化：训练开始时各街坊修正量为 0（exp(0)=1，即不修正）
        self.scale_net = nn.Linear(n_feat, n_time, bias=False)
        nn.init.zeros_(self.scale_net.weight)

    def forward(self, state, observation):
        """仿真每步被调用一次，计算并返回三个输出量。

        参数：
            state:       当前仿真状态字典（包含 environment 和 agents 的所有张量）
            observation: 当前步的观测（此模型未使用观测输入）

        返回字典（键名与 output_variables 配置一一对应）：
            p_home:           (n_demo, 48)  各群体各时段的留家概率
            attract_logits:   (N_blocks, 1) 各街坊的未归一化吸引力
            block_log_scale:  (N_blocks, 48)各街坊各时段的规模修正（对数空间）
        """
        # 从仿真状态中读取标准化后的街坊特征（形状：N_blocks × F）
        features = get_var(state, self.input_variables["block_features"])

        # sigmoid 把 log-odds 转为概率（值域严格在 0~1 之间）
        p_home = torch.sigmoid(self.home_logits)   # (n_demo, 48)

        # 如果有空间注意力，先用邻居信息增强特征；否则直接使用原始特征
        h = features
        if self.spatial_attn is not None and "edge_index" in self.input_variables:
            # edge_index 存储为 float（框架限制），这里转回 long 用作索引
            edge_index = get_var(state, self.input_variables["edge_index"]).long()
            h = self.spatial_attn(features, edge_index)

        attract_logits  = self.attract_net(h)      # (N_blocks, 1)  街坊吸引力
        block_log_scale = self.scale_net(h)         # (N_blocks, 48) 规模修正

        return {
            self.output_variables[0]: p_home,           # → "p_home"
            self.output_variables[1]: attract_logits,    # → "attract_logits"
            self.output_variables[2]: block_log_scale,   # → "block_log_scale"
        }
