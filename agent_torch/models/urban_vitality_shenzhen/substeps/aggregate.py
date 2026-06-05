"""活力聚合模块（AggregateVitality）。

这是仿真的"物理核心"，把 MovePolicy 输出的概率转化为具体的人口数值。

核心数学公式：
    V(j, t) = [home_vitality(j,t) + away_vitality(j,t)] × scale(j,t)

    其中：
        home_vitality(j, t)  = Σ_{i∈ block j} weight_i × p_home[demo_i, t]
            → 街坊 j 在 t 时段"留在本街坊"的居民人口数

        away_vitality(j, t)  = attract_prob(j) × Σ_i weight_i × (1 - p_home[demo_i, t])
            → 全市外出人口中，被吸引到街坊 j 的那部分（全局 softmax 路由）

        scale(j, t) = exp( log_scale_global(t) + block_log_scale(j, t) )
            → LBS 采样率修正：log_scale_global 捕捉时段整体采样率变化，
              block_log_scale 捕捉不同功能区（商业/住宅）在特定时段的采样偏差

    两个 log_scale 项非冗余：
        log_scale_global(t)    ：所有街坊共享同一曲线，学习时间维度的系统偏差
        block_log_scale(j, t)  ：每个街坊独立修正，学习空间维度的功能差异
"""

import torch
from torch import nn

from agent_torch.core.helpers import get_var
from agent_torch.core.substep import SubstepTransition


class AggregateVitality(SubstepTransition):
    """把居民权重散射到城市街坊，产生每小时活力预测值。

    参数（可训练）：
        log_scale: (48,) 全局每时段对数规模修正，初始化为全零（即初始 scale=1）
    """

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        n_time = int(config["simulation_metadata"]["num_targets"])   # 48 个时段
        # 全局时段修正：初始为 0 → exp(0)=1 → 不做修正，训练后学到各时段采样率
        self.log_scale = nn.Parameter(torch.zeros(n_time))

    @staticmethod
    def _local_softmax_away(
        attract_logits: torch.Tensor,   # (N_blocks, 1) 各街坊吸引力
        away_per_block: torch.Tensor,   # (N_blocks, n_time) 各街坊各时段的外出人口汇总
        edge_index: torch.Tensor,       # (2, E) 局部图的边
        n_blocks: int,
        n_time: int,
    ) -> torch.Tensor:
        """局部 softmax 路由（备用，当前代码路径未激活）。

        注意：经过实验验证，局部 softmax 对高活力街坊预测有害。
        高活力地标（如华强北、深圳湾）吸引的人口来自全市范围，
        而不只是 2km 邻居；限制到局部图会系统性低估这类街坊的活力。
        当前模型使用全局 softmax（见 forward 的 else 分支），
        此方法保留供未来研究对比使用。

        机制：
            对每条边 (src_block → dst_block)，计算 dst_block 的吸引力指数，
            在 src_block 的所有邻居中归一化，得到局部路由概率。
            然后把 src_block 的外出人口按此概率分配到邻居街坊。
        """
        src_blocks = edge_index[0]
        dst_blocks = edge_index[1]
        edge_logits = attract_logits.squeeze(-1)[dst_blocks]          # 目标街坊的吸引力
        edge_exp    = torch.exp(edge_logits.clamp(-20, 20))           # 防止溢出

        # 对每个源街坊，把所有邻居的 exp 值相加（用于归一化）
        sum_exp = torch.zeros(n_blocks, device=attract_logits.device)
        sum_exp.scatter_add_(0, src_blocks, edge_exp)

        # 局部路由概率（每个源街坊的邻居中，各邻居占多少份额）
        local_prob = edge_exp / (sum_exp[src_blocks] + 1e-10)

        # 把源街坊的外出人口按局部概率分配到目标街坊
        src_away  = away_per_block[src_blocks]                        # (E, n_time)
        edge_flow = local_prob.unsqueeze(1) * src_away                # (E, n_time)
        away_vitality = torch.zeros(n_blocks, n_time, device=attract_logits.device)
        away_vitality.scatter_add_(
            0, dst_blocks.unsqueeze(1).expand(-1, n_time), edge_flow
        )
        return away_vitality

    def forward(self, state, action):
        """仿真每步执行一次，将人口散射到街坊并输出预测活力。

        参数：
            state:  当前仿真状态（含 environment 和 agents 属性）
            action: MovePolicy 的输出（含 p_home / attract_logits / block_log_scale）

        返回字典：
            predicted_vitality:        (N_blocks, 48) 原始人口数预测（用于评估和导出）
            predicted_vitality_scaled: (N_blocks, 48) 标准化后的对数值（用于训练损失）
        """
        # ── 读取 Agent 属性 ──────────────────────────────────────────────────
        # home_block: 每个 Agent 所属的街坊行索引（不是 Block_ID）
        home_block   = get_var(state, self.input_variables["home_block"]).long()   # (N_agents,)
        # demo_group: 每个 Agent 所属的人口群体编号（0=青少年, 1=青年, 2=中年, 3=老年）
        demo_group   = get_var(state, self.input_variables["demo_group"]).long()   # (N_agents,)
        # weight: 每个 Agent 代表的人口数量（该街坊该群体的实际人数）
        weight       = get_var(state, self.input_variables["weight"])               # (N_agents,)

        # 用于把预测值反标准化的统计量（load_shenzhen_vitality_data 中计算）
        target_mean  = get_var(state, self.input_variables["target_mean"])          # (48,)
        target_scale = get_var(state, self.input_variables["target_scale"])         # (48,)

        # ── 读取 MovePolicy 的输出 ───────────────────────────────────────────
        p_home_mat      = action["residents"]["p_home"]              # (n_demo, 48) 留家概率
        attract_logits  = action["residents"]["attract_logits"]      # (N_blocks, 1) 吸引力
        block_log_scale = action["residents"]["block_log_scale"]     # (N_blocks, 48) 规模修正

        n_blocks = attract_logits.shape[0]   # 街坊总数（3023）
        n_time   = p_home_mat.shape[1]       # 时段总数（48）

        # ── 第一步：计算留家人口 ─────────────────────────────────────────────
        # 把 (n_demo, 48) 的留家概率矩阵展开到 (N_agents, 48)
        # demo_group[i] 告诉我们 Agent i 属于哪个群体，对应取那一行
        p_home_agents = p_home_mat[demo_group]                       # (N_agents, 48)

        # 每个 Agent 在各时段"留在家里"的人口贡献
        home_contrib  = weight.unsqueeze(1) * p_home_agents          # (N_agents, 48)

        # scatter_add_: 把所有属于同一街坊的 Agent 的留家人口加在一起
        # 相当于 for each agent i: home_vitality[home_block[i]] += home_contrib[i]
        # 但完全向量化（无 Python 循环），且对参数可微
        home_vitality = torch.zeros(n_blocks, n_time, device=weight.device)
        home_vitality.scatter_add_(
            0,
            home_block.unsqueeze(1).expand(-1, n_time),   # 广播到 (N_agents, 48)
            home_contrib
        )

        # ── 第二步：计算外出人口并路由到各街坊 ─────────────────────────────
        # 每个 Agent 在各时段"外出"的人口贡献
        away_contrib = weight.unsqueeze(1) * (1.0 - p_home_agents)   # (N_agents, 48)

        if "edge_index" in self.input_variables:
            # 局部路由（备用路径，参见 _local_softmax_away 注释）
            away_per_block = torch.zeros(n_blocks, n_time, device=weight.device)
            away_per_block.scatter_add_(
                0, home_block.unsqueeze(1).expand(-1, n_time), away_contrib
            )
            edge_index = get_var(state, self.input_variables["edge_index"]).long()
            away_vitality = self._local_softmax_away(
                attract_logits, away_per_block, edge_index, n_blocks, n_time
            )
        else:
            # 全局 softmax 路由（当前默认使用此路径）
            # total_away: (48,) 全市每个时段的总外出人口
            total_away    = away_contrib.sum(dim=0)

            # softmax 把 (N_blocks, 1) 的 logit 转为路由概率，所有街坊之和 = 1
            # 高吸引力街坊（商业区）获得更大比例的外出人口
            attract_probs = torch.softmax(attract_logits, dim=0)     # (N_blocks, 1)

            # 广播乘法：每个街坊的路由概率 × 对应时段的总外出人口
            away_vitality = attract_probs * total_away.unsqueeze(0)  # (N_blocks, 48)

        # ── 第三步：应用规模修正 ─────────────────────────────────────────────
        # log_scale_global(t)   ：广播到所有街坊，捕捉 LBS 时段整体采样率变化
        # block_log_scale(j, t) ：每街坊独立修正，捕捉功能区采样差异
        # 两者相加后取指数，得到大于 0 的乘性修正因子
        scale = torch.exp(self.log_scale.unsqueeze(0) + block_log_scale)  # (N_blocks, 48)

        predicted_vitality = (home_vitality + away_vitality) * scale       # (N_blocks, 48)

        # ── 第四步：转换为标准化对数形式（用于训练损失计算）───────────────
        # log1p(x) = ln(x+1)：把人口数映射到对数空间，压缩极端值
        # clamp_min(0) 保证对数参数非负（实际上 predicted_vitality 应始终 ≥ 0）
        predicted_log    = torch.log1p(predicted_vitality.clamp_min(0.0))
        # 用训练集统计量标准化（与 load_shenzhen_vitality_data 中的处理方式对称）
        predicted_scaled = (predicted_log - target_mean) / target_scale     # (N_blocks, 48)

        return {
            "predicted_vitality":        predicted_vitality,    # 原始人口数，用于报告
            "predicted_vitality_scaled": predicted_scaled,      # 标准化值，用于损失函数
        }
