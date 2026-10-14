"""Plan–Action bilinear interaction for Investment PPO (Step 3B-9).

The sole sampled Action is the marginal family. Candidate Plans remain latent;
their probabilities weight independently masked, plan-conditioned families.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

from .hierarchical_l3_shadow_step3b5 import (
    L3DecisionFacts, ROUTE_ORDER, differentiable_soft_plan,
)
from .hierarchical_shadow_controller_step3b4 import FAMILY_ORDER
from .runtime_context import RUNTIME_CONTEXT_SIZE
from .victory_plan_policy import (
    CANDIDATE_FEATURE_DIM, GLOBAL_FEATURE_DIM, MAX_PLAN_CANDIDATES,
    VictoryPlanShadowHeads, VictoryPlanShadowOutput,
)


SCHEMA = "plan_action_interaction_step3b9_v1"
STATE_FIXED_DIM = (RUNTIME_CONTEXT_SIZE + GLOBAL_FEATURE_DIM + len(ROUTE_ORDER)
                   + 2 * len(FAMILY_ORDER) + 1)
PLAN_INPUT_DIM = CANDIDATE_FEATURE_DIM + 4


@dataclass(frozen=True)
class PlanActionOutput:
    raw_logits: torch.Tensor  # [B,7], log marginal Family probability
    masked_logits: torch.Tensor  # [B,7]
    probabilities: torch.Tensor  # [B,7], the only PPO Action distribution
    entropy: torch.Tensor  # [B]
    available: torch.Tensor  # [B]
    comparison_mask: torch.Tensor  # [B,7]
    per_plan_probabilities: torch.Tensor  # [B,36,7], invalid Plan slots zero
    per_plan_logits: torch.Tensor  # [B,36,7], before Action mask
    base_logits: torch.Tensor  # [B,7]
    interaction_logits: torch.Tensor  # [B,36,7]
    state_embedding_rms: torch.Tensor  # [B]
    plan_embedding_rms: torch.Tensor  # [B], valid candidates only
    action_query_rms: torch.Tensor  # [B]


class PlanActionInteractionActor(nn.Module):
    """Small state query × candidate Plan embedding; no family-specific rules."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64,
                 interaction_dim: int = 32, initial_scale: float = .1):
        super().__init__()
        self.latent_dim = latent_dim
        self.interaction_dim = interaction_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(latent_dim + STATE_FIXED_DIM, hidden_dim), nn.Tanh(),
            nn.LayerNorm(hidden_dim),
        )
        self.base_head = nn.Linear(hidden_dim, len(FAMILY_ORDER))
        self.action_query = nn.Linear(hidden_dim,
                                      len(FAMILY_ORDER) * interaction_dim, bias=False)
        self.plan_encoder = nn.Sequential(
            nn.Linear(PLAN_INPUT_DIM, interaction_dim), nn.Tanh(),
            nn.Linear(interaction_dim, interaction_dim),
            nn.LayerNorm(interaction_dim),
        )
        # Symmetric random initialization: no built-in Action/Plan ranking.
        nn.init.normal_(self.base_head.weight, 0., .01)
        nn.init.zeros_(self.base_head.bias)
        nn.init.xavier_uniform_(self.action_query.weight)
        self.interaction_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, latent: torch.Tensor, runtime: torch.Tensor,
                global_features: torch.Tensor, candidates: torch.Tensor,
                plan: VictoryPlanShadowOutput, facts: L3DecisionFacts,
                *, plan_probabilities: torch.Tensor | None = None,
                priorities: torch.Tensor | None = None) -> PlanActionOutput:
        batch = latent.shape[0]
        n = MAX_PLAN_CANDIDATES
        families = len(FAMILY_ORDER)
        if (latent.shape != (batch, self.latent_dim) or
                runtime.shape != (batch, RUNTIME_CONTEXT_SIZE) or
                global_features.shape != (batch, GLOBAL_FEATURE_DIM) or
                candidates.shape != (batch, n, CANDIDATE_FEATURE_DIM) or
                plan.candidate_mask.shape != (batch, n) or
                plan.priorities.shape != (batch, n, 4) or
                facts.route_one_hot.shape != (batch, len(ROUTE_ORDER)) or
                facts.comparison_family_mask.shape != (batch, families)):
            raise ValueError("Plan–Action batch shape mismatch")
        weights = plan.probabilities if plan_probabilities is None else plan_probabilities
        values = plan.priorities if priorities is None else priorities
        valid = plan.candidate_mask.bool()
        if weights.shape != (batch, n) or values.shape != (batch, n, 4):
            raise ValueError("Plan/Priority intervention shape mismatch")
        if torch.any(weights[~valid] != 0):
            raise ValueError("Impossible/padded Plan has probability mass")
        if torch.any(plan.plan_available & ((weights.sum(-1) - 1).abs() > 1e-5)):
            raise ValueError("Plan weights must sum to one on available rows")
        mask = (facts.comparison_family_mask.bool() & facts.legal_family_mask.bool()
                & facts.l3_route_available.bool().unsqueeze(-1))
        available = mask.any(-1)
        dtype = runtime.dtype
        state_input = torch.cat((
            latent.detach(), runtime, global_features, facts.route_one_hot,
            facts.legal_family_mask.to(dtype), facts.comparison_family_mask.to(dtype),
            facts.construction_commit,
        ), -1)
        state = self.state_encoder(state_input)
        base = self.base_head(state)
        query = self.action_query(state).reshape(batch, families, self.interaction_dim)
        embedding = self.plan_encoder(torch.cat((candidates, values), -1))
        interaction = (self.interaction_scale / sqrt(self.interaction_dim) *
                       torch.einsum("bah,bnh->bna", query, embedding))
        per_plan_logits = base.unsqueeze(1) + interaction
        conditioned = torch.softmax(per_plan_logits.masked_fill(
            ~mask.unsqueeze(1), -1e8), -1)
        conditioned = conditioned * mask.unsqueeze(1).to(dtype) * valid.unsqueeze(-1).to(dtype)
        marginal = (weights.unsqueeze(-1) * conditioned).sum(1)
        # Plan unavailable is explicit; base-only is a legal Action fallback,
        # never a fictional zero Plan or a forced EndTurn.
        fallback = torch.softmax(base.masked_fill(~mask, -1e8), -1) * mask.to(dtype)
        marginal = torch.where(plan.plan_available.unsqueeze(-1), marginal, fallback)
        marginal = torch.where(available.unsqueeze(-1), marginal, torch.zeros_like(marginal))
        marginal = marginal * mask.to(dtype)
        marginal = marginal / marginal.sum(-1, keepdim=True).clamp_min(1e-30)
        raw = marginal.clamp_min(1e-30).log()
        masked = raw.masked_fill(~mask, -1e8)
        entropy = -(marginal * raw).sum(-1)
        state_rms = state.square().mean(-1).sqrt()
        plan_squared = embedding.square().mean(-1)
        plan_rms = ((plan_squared * valid.to(dtype)).sum(-1) /
                    valid.sum(-1).clamp_min(1)).sqrt()
        query_rms = query.square().mean((-1, -2)).sqrt()
        return PlanActionOutput(raw, masked, marginal, entropy, available, mask,
                                conditioned, per_plan_logits, base, interaction,
                                state_rms, plan_rms, query_rms)


class PlanActionInvestmentPolicy(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.plan_heads = VictoryPlanShadowHeads(latent_dim)
        self.l3_actor = PlanActionInteractionActor(latent_dim)

    def forward(self, latent: torch.Tensor, runtime: torch.Tensor,
                global_features: torch.Tensor, candidates: torch.Tensor,
                plan_mask: torch.Tensor, facts: L3DecisionFacts):
        plan = self.plan_heads(latent.detach(), runtime, global_features,
                               candidates, plan_mask)
        bridge = differentiable_soft_plan(plan, candidates)
        action = self.l3_actor(latent, runtime, global_features, candidates, plan, facts)
        return plan, bridge, action
