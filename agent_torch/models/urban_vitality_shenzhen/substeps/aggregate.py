"""Vitality aggregation transition for Shenzhen resident agent groups."""

import torch
from torch import nn

from agent_torch.core.helpers import get_var
from agent_torch.core.substep import SubstepTransition


class AggregateVitality(SubstepTransition):
    """Scatter resident weights onto city blocks to produce hourly vitality.

    For each time slot t the predicted vitality of block j is:

        V(j, t) = home_vitality(j, t) + away_vitality(j, t)

    home_vitality(j, t) = Σ_{i: home(i)=j}  weight(i) × p_home(demo(i), t)

    away_vitality is computed in one of two modes:

    Local softmax (when edge_index is provided — preferred):
        For each source block s, its away population is distributed among
        s's spatial neighbours proportional to their attractiveness:

            away_vitality(d, t) = Σ_{s: d ∈ neighbours(s)}
                                    local_prob(s→d) × away_pop(s, t)

        where local_prob(s→d) = exp(attract(d)) / Σ_{n ∈ nbrs(s)} exp(attract(n))

        This removes the zero-sum global competition: validation blocks are no
        longer starved of away population by training blocks on the other side
        of the city.

    Global softmax fallback (no edge_index):
        away_vitality(j, t) = softmax(attract_logits)[j] × Σ_i away_pop(i, t)
    """

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        n_time = int(config["simulation_metadata"]["num_targets"])
        # Per-time-slot log-scale: corrects the systematic bias caused by
        # LBS sampling not matching residential population totals.
        # Initialized to 0 (scale=1, no adjustment) so training starts unchanged.
        self.log_scale = nn.Parameter(torch.zeros(n_time))

    @staticmethod
    def _local_softmax_away(
        attract_logits: torch.Tensor,   # (N_blocks, 1)
        away_per_block: torch.Tensor,   # (N_blocks, n_time)
        edge_index: torch.Tensor,       # (2, E) long — [src_block, dst_block]
        n_blocks: int,
        n_time: int,
    ) -> torch.Tensor:
        """Route away population to spatial neighbours via per-source softmax."""
        src_blocks = edge_index[0]   # (E,) home/source block
        dst_blocks = edge_index[1]   # (E,) destination block

        # Attractiveness of each edge's destination; clamp for numerical safety
        edge_logits = attract_logits.squeeze(-1)[dst_blocks]          # (E,)
        edge_exp    = torch.exp(edge_logits.clamp(-20, 20))           # (E,)

        # Sum exp per source block → normalisation denominator
        sum_exp = torch.zeros(n_blocks, device=attract_logits.device)
        sum_exp.scatter_add_(0, src_blocks, edge_exp)
        local_prob = edge_exp / (sum_exp[src_blocks] + 1e-10)         # (E,)

        # Flow along each edge: prob × source block's away population
        src_away  = away_per_block[src_blocks]                        # (E, n_time)
        edge_flow = local_prob.unsqueeze(1) * src_away                # (E, n_time)

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

        p_home_mat     = action["residents"]["p_home"]           # (n_demo, n_time)
        attract_logits = action["residents"]["attract_logits"]   # (N_blocks, 1)

        n_blocks = attract_logits.shape[0]
        n_time   = p_home_mat.shape[1]

        p_home_agents = p_home_mat[demo_group]                   # (N_agents, n_time)

        # Home component: weight × p_home accumulated at each agent's home block
        home_contrib  = weight.unsqueeze(1) * p_home_agents      # (N_agents, n_time)
        home_vitality = torch.zeros(n_blocks, n_time, device=weight.device)
        home_vitality.scatter_add_(
            0, home_block.unsqueeze(1).expand(-1, n_time), home_contrib
        )

        # Away component
        away_contrib = weight.unsqueeze(1) * (1.0 - p_home_agents)   # (N_agents, n_time)

        if "edge_index" in self.input_variables:
            # Local softmax: route per-source-block away pop to spatial neighbours
            away_per_block = torch.zeros(n_blocks, n_time, device=weight.device)
            away_per_block.scatter_add_(
                0, home_block.unsqueeze(1).expand(-1, n_time), away_contrib
            )
            edge_index = get_var(state, self.input_variables["edge_index"]).long()
            away_vitality = self._local_softmax_away(
                attract_logits, away_per_block, edge_index, n_blocks, n_time
            )
        else:
            # Global softmax fallback (no spatial data available)
            total_away    = away_contrib.sum(dim=0)               # (n_time,)
            attract_probs = torch.softmax(attract_logits, dim=0)  # (N_blocks, 1)
            away_vitality = attract_probs * total_away.unsqueeze(0)

        # Apply per-time-slot scale to bridge population units → LBS sampling units
        scale = torch.exp(self.log_scale).unsqueeze(0)          # (1, n_time)
        predicted_vitality = (home_vitality + away_vitality) * scale

        predicted_log    = torch.log1p(predicted_vitality.clamp_min(0.0))
        predicted_scaled = (predicted_log - target_mean) / target_scale

        return {
            "predicted_vitality":        predicted_vitality,
            "predicted_vitality_scaled": predicted_scaled,
        }
