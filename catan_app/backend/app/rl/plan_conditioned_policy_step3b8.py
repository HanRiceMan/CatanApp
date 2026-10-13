"""Plan-conditioned Investment actor (Step 3B-8).

The sampled PPO action is the *marginal* family distribution. A plan is never
sampled, and invalid plans contribute exactly zero probability.
"""

from __future__ import annotations

from dataclasses import dataclass

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


SCHEMA = "plan_conditioned_investment_step3b8_v1"
_FIXED_DIM = (RUNTIME_CONTEXT_SIZE + GLOBAL_FEATURE_DIM +
              CANDIDATE_FEATURE_DIM + 4 + 1 + len(ROUTE_ORDER) +
              2 * len(FAMILY_ORDER) + 1)


@dataclass(frozen=True)
class PlanConditionedOutput:
    raw_logits: torch.Tensor  # [B, 7], log marginal probabilities
    masked_logits: torch.Tensor  # [B, 7]
    probabilities: torch.Tensor  # [B, 7], final PPO distribution
    entropy: torch.Tensor  # [B]
    available: torch.Tensor  # [B]
    comparison_mask: torch.Tensor  # [B, 7]
    per_plan_probabilities: torch.Tensor  # [B, 36, 7]; invalid slots zero
    fallback_probabilities: torch.Tensor  # [B, 7]; only for unavailable Plan


class PlanConditionedL3Actor(nn.Module):
    """One shared scorer per plan; family probabilities are mixed, not logits."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim + _FIXED_DIM, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.family_head = nn.Linear(hidden_dim, len(FAMILY_ORDER))
        # Small, family-symmetric random initialization permits plan sensitivity
        # and upstream gradients without a handcrafted family preference.
        nn.init.normal_(self.family_head.weight, mean=0., std=.01)
        nn.init.zeros_(self.family_head.bias)

    def forward(self, latent: torch.Tensor, runtime: torch.Tensor,
                global_features: torch.Tensor, candidates: torch.Tensor,
                plan: VictoryPlanShadowOutput, facts: L3DecisionFacts,
                *, plan_probabilities: torch.Tensor | None = None,
                priorities: torch.Tensor | None = None) -> PlanConditionedOutput:
        batch = latent.shape[0]
        if (latent.shape != (batch, self.latent_dim) or
                runtime.shape != (batch, RUNTIME_CONTEXT_SIZE) or
                global_features.shape != (batch, GLOBAL_FEATURE_DIM) or
                candidates.shape != (batch, MAX_PLAN_CANDIDATES, CANDIDATE_FEATURE_DIM) or
                plan.candidate_mask.shape != (batch, MAX_PLAN_CANDIDATES) or
                facts.comparison_family_mask.shape != (batch, len(FAMILY_ORDER))):
            raise ValueError("Plan-conditioned L3 batch shape mismatch")
        weights = plan.probabilities if plan_probabilities is None else plan_probabilities
        values = plan.priorities if priorities is None else priorities
        if weights.shape != plan.probabilities.shape or values.shape != plan.priorities.shape:
            raise ValueError("Plan intervention shape mismatch")
        valid = plan.candidate_mask.bool()
        if torch.any(weights[~valid] != 0):
            raise ValueError("Impossible/padded Plan has nonzero weight")
        if torch.any(plan.plan_available & ((weights.sum(-1) - 1).abs() > 1e-5)):
            raise ValueError("Available Plan weights must sum to one")
        n = MAX_PLAN_CANDIDATES
        dtype = runtime.dtype
        mask = (facts.comparison_family_mask.bool() & facts.legal_family_mask.bool()
                & facts.l3_route_available.bool().unsqueeze(-1))
        available = mask.any(-1)
        repeated = lambda x: x.unsqueeze(1).expand(-1, n, -1)
        features = torch.cat((
            repeated(latent.detach()), repeated(runtime), repeated(global_features),
            candidates, values,
            plan.plan_available.to(dtype).view(batch, 1, 1).expand(-1, n, -1),
            repeated(facts.route_one_hot),
            repeated(facts.legal_family_mask.to(dtype)),
            repeated(facts.comparison_family_mask.to(dtype)),
            repeated(facts.construction_commit),
        ), -1)
        logits = self.family_head(self.encoder(features))
        plan_action = torch.softmax(logits.masked_fill(~mask.unsqueeze(1), -1e8), -1)
        plan_action = plan_action * mask.unsqueeze(1).to(dtype) * valid.unsqueeze(-1).to(dtype)
        marginal = (weights.unsqueeze(-1) * plan_action).sum(1)
        # A plan-unavailable state still has a legal-action distribution, but
        # this independent fallback is not interpreted as selecting a zero Plan.
        fallback_features = torch.cat((
            latent.detach(), runtime, global_features,
            candidates.new_zeros(batch, CANDIDATE_FEATURE_DIM),
            candidates.new_full((batch, 4), .5),
            candidates.new_zeros(batch, 1), facts.route_one_hot,
            facts.legal_family_mask.to(dtype),
            facts.comparison_family_mask.to(dtype), facts.construction_commit,
        ), -1)
        fallback_logits = self.family_head(self.encoder(fallback_features))
        fallback = torch.softmax(fallback_logits.masked_fill(~mask, -1e8), -1)
        fallback = fallback * mask.to(dtype)
        marginal = torch.where(plan.plan_available.unsqueeze(-1), marginal, fallback)
        marginal = torch.where(available.unsqueeze(-1), marginal, torch.zeros_like(marginal))
        marginal = marginal * mask.to(dtype)
        # No manually weighted Plan/action scores: this is only a numerical
        # normalization of the exact mixture distribution.
        marginal = marginal / marginal.sum(-1, keepdim=True).clamp_min(1e-30)
        raw = marginal.clamp_min(1e-30).log()
        masked = raw.masked_fill(~mask, -1e8)
        entropy = -(marginal * raw).sum(-1)
        return PlanConditionedOutput(raw, masked, marginal, entropy, available,
                                     mask, plan_action, fallback)


class PlanConditionedInvestmentPolicy(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.plan_heads = VictoryPlanShadowHeads(latent_dim)
        self.l3_actor = PlanConditionedL3Actor(latent_dim)

    def forward(self, latent: torch.Tensor, runtime: torch.Tensor,
                global_features: torch.Tensor, candidates: torch.Tensor,
                plan_mask: torch.Tensor, facts: L3DecisionFacts):
        plan = self.plan_heads(latent.detach(), runtime, global_features,
                               candidates, plan_mask)
        bridge = differentiable_soft_plan(plan, candidates)
        l3 = self.l3_actor(latent, runtime, global_features, candidates, plan, facts)
        return plan, bridge, l3
