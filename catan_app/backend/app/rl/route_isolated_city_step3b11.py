"""Frozen 3B-9 base with a City-only, Plan-conditioned residual Actor."""

from __future__ import annotations

import torch
from torch import nn

from .hierarchical_l3_shadow_step3b5 import L3DecisionFacts, ROUTE_ORDER
from .hierarchical_shadow_controller_step3b4 import FAMILY_ORDER, HierarchicalRoute
from .plan_action_interaction_step3b9 import PlanActionInvestmentPolicy, PlanActionOutput
from .victory_plan_policy import CANDIDATE_FEATURE_DIM, VictoryPlanShadowOutput


SCHEMA = "route_isolated_city_step3b11_v1"
CITY_INDEX = ROUTE_ORDER.index(HierarchicalRoute.CITY_VALUE_COMPARISON)


class CityPlanAdapter(nn.Module):
    """Shared small residual over candidate Plans; zero output at initialization."""

    def __init__(self, state_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + CANDIDATE_FEATURE_DIM + 4, hidden_dim),
            nn.Tanh(),
        )
        self.output = nn.Linear(hidden_dim, len(FAMILY_ORDER))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor, candidates: torch.Tensor,
                priorities: torch.Tensor) -> torch.Tensor:
        count = candidates.shape[1]
        repeated = state.unsqueeze(1).expand(-1, count, -1)
        features = torch.cat((repeated, candidates, priorities), -1)
        return self.output(self.encoder(features))


class RouteIsolatedCityPolicy(PlanActionInvestmentPolicy):
    """Investment is exactly the frozen base; City uses only adapter residuals."""

    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)
        self.city_adapter = CityPlanAdapter()
        self.freeze_base()

    def freeze_base(self) -> None:
        for module in (self.plan_heads, self.l3_actor):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def load_base(self, parent: PlanActionInvestmentPolicy) -> None:
        self.plan_heads.load_state_dict(parent.plan_heads.state_dict())
        self.l3_actor.load_state_dict(parent.l3_actor.state_dict())
        self.freeze_base()

    def _city_output(self, latent: torch.Tensor, runtime: torch.Tensor,
                     global_features: torch.Tensor, candidates: torch.Tensor,
                     plan: VictoryPlanShadowOutput, facts: L3DecisionFacts,
                     base: PlanActionOutput, *,
                     plan_probabilities: torch.Tensor | None = None,
                     priorities: torch.Tensor | None = None) -> PlanActionOutput:
        actor = self.l3_actor
        dtype = runtime.dtype
        state_input = torch.cat((
            latent.detach(), runtime, global_features, facts.route_one_hot,
            facts.legal_family_mask.to(dtype), facts.comparison_family_mask.to(dtype),
            facts.construction_commit,
        ), -1)
        with torch.no_grad():
            state = actor.state_encoder(state_input)
        weights = plan.probabilities if plan_probabilities is None else plan_probabilities
        values = plan.priorities if priorities is None else priorities
        if (weights.shape != plan.probabilities.shape or
                values.shape != plan.priorities.shape):
            raise ValueError("City Plan/Priority intervention shape mismatch")
        if torch.any(weights[~plan.candidate_mask.bool()] != 0):
            raise ValueError("Invalid Plan has probability mass")
        if torch.any(plan.plan_available & ((weights.sum(-1) - 1).abs() > 1e-5)):
            raise ValueError("City Plan weights must sum to one")
        delta = self.city_adapter(state, candidates, values)
        mask = base.comparison_mask
        valid = plan.candidate_mask.bool()
        logits = base.per_plan_logits + delta
        conditional = torch.softmax(logits.masked_fill(~mask.unsqueeze(1), -1e8), -1)
        conditional = conditional * mask.unsqueeze(1).to(dtype) * valid.unsqueeze(-1).to(dtype)
        marginal = (weights.unsqueeze(-1) * conditional).sum(1)
        # The base-only fallback has no selected Plan and must not create one.
        marginal = torch.where(plan.plan_available.unsqueeze(-1), marginal,
                               base.probabilities)
        marginal = torch.where(base.available.unsqueeze(-1), marginal,
                               torch.zeros_like(marginal))
        marginal = marginal * mask.to(dtype)
        marginal = marginal / marginal.sum(-1, keepdim=True).clamp_min(1e-30)
        raw = marginal.clamp_min(1e-30).log()
        entropy = -(marginal * raw).sum(-1)
        return PlanActionOutput(
            raw, raw.masked_fill(~mask, -1e8), marginal, entropy,
            base.available, mask, conditional, logits, base.base_logits,
            base.interaction_logits + delta, base.state_embedding_rms,
            base.plan_embedding_rms, base.action_query_rms,
        )

    def forward(self, latent: torch.Tensor, runtime: torch.Tensor,
                global_features: torch.Tensor, candidates: torch.Tensor,
                plan_mask: torch.Tensor, facts: L3DecisionFacts, *,
                plan_probabilities: torch.Tensor | None = None,
                priorities: torch.Tensor | None = None):
        # Base/Plan/Priority are frozen even during City PPO. The only Actor
        # gradient path is through city_adapter.
        with torch.no_grad():
            plan, bridge, base = super().forward(
                latent, runtime, global_features, candidates, plan_mask, facts)
        city = facts.route_one_hot[:, CITY_INDEX].bool()
        if not torch.any(city):
            return plan, bridge, base
        if not torch.all(city):
            raise ValueError("City and non-City routes must be batched separately")
        return plan, bridge, self._city_output(
            latent, runtime, global_features, candidates, plan, facts, base,
            plan_probabilities=plan_probabilities, priorities=priorities)
