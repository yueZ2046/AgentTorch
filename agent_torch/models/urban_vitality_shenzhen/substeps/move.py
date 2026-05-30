"""Movement policy for Shenzhen resident agent groups."""

import math

import torch
from torch import nn

from agent_torch.core.helpers import get_var
from agent_torch.core.substep import SubstepAction


def _temporal_prior(n_demo: int, n_time: int) -> torch.Tensor:
    """Return (n_demo, n_time) home-logit prior encoding typical urban rhythms."""
    n_half = n_time // 2
    hours  = torch.arange(n_half, dtype=torch.float32)
    wd     = 2.2 * torch.cos(2 * math.pi * (hours - 2.0) / n_half)
    we     = 1.2 * torch.cos(2 * math.pi * (hours - 3.0) / n_half)
    prior_1d = torch.cat([wd, we])
    return prior_1d.unsqueeze(0).expand(n_demo, -1).clone()


class SpatialAttentionAggregation(nn.Module):
    """Single-head dot-product attention over a spatial graph."""

    def __init__(self, n_feat: int):
        super().__init__()
        d = max(n_feat // 4, 16)
        self.query = nn.Linear(n_feat, d, bias=False)
        self.key   = nn.Linear(n_feat, d, bias=False)
        self.value = nn.Linear(n_feat, n_feat, bias=False)
        self.out   = nn.Linear(n_feat, n_feat)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)
        self._scale = d ** -0.5

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        center   = edge_index[0]
        nbr      = edge_index[1]
        n_blocks = features.shape[0]
        q = self.query(features)
        k = self.key(features)
        v = self.value(features)
        score = (q[center] * k[nbr]).sum(-1) * self._scale
        exp_s = torch.exp(score.clamp(-20, 20))
        sum_s = torch.zeros(n_blocks, device=features.device)
        sum_s.scatter_add_(0, center, exp_s)
        attn = exp_s / (sum_s[center] + 1e-10)
        v_weighted = v[nbr] * attn.unsqueeze(1)
        agg = torch.zeros_like(features)
        agg.scatter_add_(0, center.unsqueeze(1).expand_as(v_weighted), v_weighted)
        return features + self.out(agg)


class MovePolicy(SubstepAction):
    """Learns two complementary components of resident mobility:

    1. home_logits (n_demo × n_time) — log-odds of staying home by demographic
       group and time slot.  Initialised from a temporal prior.

    2. attract_net → scalar attract per block — routes away population via
       global/local softmax.  Unchanged from the original design.

    3. scale_net: Linear(n_feat, n_time) — produces per-block, per-time log-scale
       corrections in one non-redundant step, replacing the original global
       log_scale(n_time) vector.

       Initialised with weight=0, bias=0 so scale=1 everywhere at the start
       (same as the original model's log_scale=0 initialisation).
       Learns block-specific temporal activity profiles (office morning peak,
       restaurant noon peak, etc.) through a single linear map from 78 features
       to 48 log-scale values — equivalent to Ridge regression in log-space —
       without interfering with routing or creating parameter redundancy.
    """

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        meta   = config["simulation_metadata"]
        n_demo = int(meta["num_demo_groups"])
        n_time = int(meta["num_targets"])
        n_feat = int(meta["num_features"])
        hidden = int(meta.get("hidden_dim", 64))

        self.home_logits = nn.Parameter(_temporal_prior(n_demo, n_time))

        if "edge_index" in input_variables:
            self.spatial_attn = SpatialAttentionAggregation(n_feat)
        else:
            self.spatial_attn = None

        # Scalar attract for routing — preserves the original stable design.
        self.attract_net = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Per-block log-scale correction (no bias — avoids redundancy with the
        # global log_scale in AggregateVitality).
        # weight: (n_time, n_feat) maps block features to per-time deviations.
        # Zero-init → zero correction at training start; weight_decay keeps it small.
        self.scale_net = nn.Linear(n_feat, n_time, bias=False)
        nn.init.zeros_(self.scale_net.weight)

    def forward(self, state, observation):
        features = get_var(state, self.input_variables["block_features"])
        p_home   = torch.sigmoid(self.home_logits)

        h = features
        if self.spatial_attn is not None and "edge_index" in self.input_variables:
            edge_index = get_var(state, self.input_variables["edge_index"]).long()
            h = self.spatial_attn(features, edge_index)

        attract_logits  = self.attract_net(h)      # (N_blocks, 1)
        block_log_scale = self.scale_net(h)        # (N_blocks, n_time)

        return {
            self.output_variables[0]: p_home,
            self.output_variables[1]: attract_logits,
            self.output_variables[2]: block_log_scale,
        }
