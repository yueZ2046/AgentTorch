"""Movement policy for Shenzhen resident agent groups."""

import math

import torch
from torch import nn

from agent_torch.core.helpers import get_var
from agent_torch.core.substep import SubstepAction


def _temporal_prior(n_demo: int, n_time: int) -> torch.Tensor:
    """Return (n_demo, n_time) home-logit prior encoding typical urban rhythms.

    Positive logit → high p_home (resident likely at home).
    Negative logit → low p_home (resident likely away).

    Weekday pattern: away peak at 13h, home peak at 2h (amplitude 2.2).
    Weekend pattern: away peak at 14h, home peak at 3h (amplitude 1.2, shallower).
    All four demographic groups start from the same prior; gradients then
    learn group-specific deviations from this baseline.
    """
    n_half = n_time // 2
    hours  = torch.arange(n_half, dtype=torch.float32)
    wd     = 2.2 * torch.cos(2 * math.pi * (hours - 2.0) / n_half)
    we     = 1.2 * torch.cos(2 * math.pi * (hours - 3.0) / n_half)
    prior_1d = torch.cat([wd, we])
    return prior_1d.unsqueeze(0).expand(n_demo, -1).clone()


class SpatialAttentionAggregation(nn.Module):
    """Single-head dot-product attention over a spatial graph.

    For each block b, attends to its graph neighbours and aggregates their
    features with learned attention weights:

        attn(b, n) = softmax_n [ query(h_b) · key(h_n) / sqrt(d) ]
        h_b_new    = h_b + out_proj( Σ_n  attn(b, n) · value(h_n) )

    out_proj is initialised near-zero so the model starts at the same point
    as without spatial data and learns to use the context gradually.
    """

    def __init__(self, n_feat: int):
        super().__init__()
        d = max(n_feat // 4, 16)   # attention head dimension
        self.query = nn.Linear(n_feat, d, bias=False)
        self.key   = nn.Linear(n_feat, d, bias=False)
        self.value = nn.Linear(n_feat, n_feat, bias=False)
        self.out   = nn.Linear(n_feat, n_feat)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)
        self._scale = d ** -0.5

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        features:   (N_blocks, n_feat)
        edge_index: (2, E)  — [0]=center block, [1]=neighbour block
        returns:    (N_blocks, n_feat) enriched features
        """
        center   = edge_index[0]          # (E,) blocks receiving aggregated info
        nbr      = edge_index[1]          # (E,) neighbour blocks sending info
        n_blocks = features.shape[0]

        q = self.query(features)          # (N_blocks, d)
        k = self.key(features)            # (N_blocks, d)
        v = self.value(features)          # (N_blocks, n_feat)

        # Per-edge attention score
        score = (q[center] * k[nbr]).sum(-1) * self._scale   # (E,)

        # Per-center softmax (numerically stable via clamp)
        exp_s = torch.exp(score.clamp(-20, 20))
        sum_s = torch.zeros(n_blocks, device=features.device)
        sum_s.scatter_add_(0, center, exp_s)
        attn = exp_s / (sum_s[center] + 1e-10)               # (E,)

        # Weighted value aggregation → skip connection
        v_weighted = v[nbr] * attn.unsqueeze(1)              # (E, n_feat)
        agg = torch.zeros_like(features)
        agg.scatter_add_(0, center.unsqueeze(1).expand_as(v_weighted), v_weighted)

        return features + self.out(agg)


class MovePolicy(SubstepAction):
    """Learns two complementary components of resident mobility:

    1. home_logits (n_demo × n_time) — log-odds of staying home by
       demographic group and time slot.  Sigmoid gives p_home.
       Initialised from a temporal prior (night=high, daytime=low).

    2. spatial_attn + attract_net — dot-product attention over the k-NN
       graph (≈2 km) enriches block features with neighbourhood context
       before a scalar attractiveness score is computed.  Falls back to
       raw features when no graph is available.
    """

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        meta   = config["simulation_metadata"]
        n_demo = int(meta["num_demo_groups"])
        n_time = int(meta["num_targets"])
        n_feat = int(meta["num_features"])
        hidden = int(meta.get("hidden_dim", 64))

        self.home_logits = nn.Parameter(_temporal_prior(n_demo, n_time))

        # Build spatial attention only when a graph will be provided at runtime.
        # We detect this from input_variables (set by build_config) rather than
        # a metadata flag, so the module stays in sync with the actual data.
        if "edge_index" in input_variables:
            self.spatial_attn = SpatialAttentionAggregation(n_feat)
        else:
            self.spatial_attn = None

        # One scalar attractiveness per block; temporal patterns live in home_logits.
        self.attract_net = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, observation):
        features = get_var(state, self.input_variables["block_features"])
        p_home   = torch.sigmoid(self.home_logits)

        h = features
        if self.spatial_attn is not None and "edge_index" in self.input_variables:
            edge_index = get_var(state, self.input_variables["edge_index"]).long()
            h = self.spatial_attn(features, edge_index)

        attract_logits = self.attract_net(h)    # (N_blocks, 1)
        return {
            self.output_variables[0]: p_home,
            self.output_variables[1]: attract_logits,
        }
