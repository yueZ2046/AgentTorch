"""Vitality aggregation transition for Shenzhen resident agent groups."""

import torch
from torch import nn

from agent_torch.core.helpers import get_var
from agent_torch.core.substep import SubstepTransition


class AggregateVitality(SubstepTransition):
    """Scatter resident weights onto city blocks to produce hourly vitality.

    For each time slot t the predicted vitality of block j is:

        V(j, t) = (home_vitality(j, t) + away_vitality(j, t)) × scale(j, t)

    scale(j, t) = exp( log_scale_global(t) + block_log_scale(j, t) )

    log_scale_global(t): global per-time-slot LBS correction (original design).
    block_log_scale(j,t): per-block deviation from MovePolicy.scale_net (no bias).

    The two terms are non-redundant: log_scale_global captures the time-varying
    LBS sampling rate shared by all blocks; scale_net captures block-specific
    deviations (e.g. commercial blocks have higher LBS penetration at noon).
    """

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        n_time = int(config["simulation_metadata"]["num_targets"])
        self.log_scale = nn.Parameter(torch.zeros(n_time))

    @staticmethod
    def _local_softmax_away(
        attract_logits: torch.Tensor,   # (N_blocks, 1)
        away_per_block: torch.Tensor,   # (N_blocks, n_time)
        edge_index: torch.Tensor,
        n_blocks: int,
        n_time: int,
    ) -> torch.Tensor:
        src_blocks = edge_index[0]
        dst_blocks = edge_index[1]
        edge_logits = attract_logits.squeeze(-1)[dst_blocks]
        edge_exp    = torch.exp(edge_logits.clamp(-20, 20))
        sum_exp = torch.zeros(n_blocks, device=attract_logits.device)
        sum_exp.scatter_add_(0, src_blocks, edge_exp)
        local_prob = edge_exp / (sum_exp[src_blocks] + 1e-10)
        src_away  = away_per_block[src_blocks]
        edge_flow = local_prob.unsqueeze(1) * src_away
        away_vitality = torch.zeros(n_blocks, n_time, device=attract_logits.device)
        away_vitality.scatter_add_(
            0, dst_blocks.unsqueeze(1).expand(-1, n_time), edge_flow
        )
        return away_vitality

    def forward(self, state, action):
        home_block   = get_var(state, self.input_variables["home_block"]).long()
        demo_group   = get_var(state, self.input_variables["demo_group"]).long()
        weight       = get_var(state, self.input_variables["weight"])
        target_mean  = get_var(state, self.input_variables["target_mean"])
        target_scale = get_var(state, self.input_variables["target_scale"])

        p_home_mat      = action["residents"]["p_home"]             # (n_demo, n_time)
        attract_logits  = action["residents"]["attract_logits"]     # (N_blocks, 1)
        block_log_scale = action["residents"]["block_log_scale"]    # (N_blocks, n_time)

        n_blocks = attract_logits.shape[0]
        n_time   = p_home_mat.shape[1]

        p_home_agents = p_home_mat[demo_group]

        home_contrib  = weight.unsqueeze(1) * p_home_agents
        home_vitality = torch.zeros(n_blocks, n_time, device=weight.device)
        home_vitality.scatter_add_(
            0, home_block.unsqueeze(1).expand(-1, n_time), home_contrib
        )

        away_contrib = weight.unsqueeze(1) * (1.0 - p_home_agents)

        if "edge_index" in self.input_variables:
            away_per_block = torch.zeros(n_blocks, n_time, device=weight.device)
            away_per_block.scatter_add_(
                0, home_block.unsqueeze(1).expand(-1, n_time), away_contrib
            )
            edge_index = get_var(state, self.input_variables["edge_index"]).long()
            away_vitality = self._local_softmax_away(
                attract_logits, away_per_block, edge_index, n_blocks, n_time
            )
        else:
            total_away    = away_contrib.sum(dim=0)
            attract_probs = torch.softmax(attract_logits, dim=0)
            away_vitality = attract_probs * total_away.unsqueeze(0)

        # Global time correction + per-block deviation (no bias → non-redundant)
        scale = torch.exp(self.log_scale.unsqueeze(0) + block_log_scale)
        predicted_vitality = (home_vitality + away_vitality) * scale

        predicted_log    = torch.log1p(predicted_vitality.clamp_min(0.0))
        predicted_scaled = (predicted_log - target_mean) / target_scale

        return {
            "predicted_vitality":        predicted_vitality,
            "predicted_vitality_scaled": predicted_scaled,
        }
