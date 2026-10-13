"""Differentiable L1/L2 bridge and action-inert seven-family L3 shadow actor.

Only rule facts become masks. Victory Plan and Priority values are features,
never Action prohibitions or handcrafted family preferences.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .hierarchical_shadow_controller_step3b4 import (
    FAMILY_ORDER, HierarchicalDecisionContext, HierarchicalRoute,
)
from .runtime_context import RUNTIME_CONTEXT_SIZE
from .victory_plan_policy import (
    GLOBAL_FEATURE_DIM, VictoryPlanShadowOutput,
)


SCHEMA = "hierarchical_l3_shadow_step3b5_v1"
ROUTE_ORDER = tuple(HierarchicalRoute)
SOFT_PLAN_FEATURE_ORDER = (
    "settlement_gain_count", "city_upgrade_count", "acquire_longest_road",
    "acquire_largest_army", "planned_point_gain", "overshoot_points",
)
# Existing 3B-2 fixed rule-based candidate scales, reversed only to preserve
# the 3B-4 diagnostic API's raw six feature values in the Tensor bridge.
_BRIDGE_RAW_SCALES = (10.0, 10.0, 1.0, 1.0, 11.0, 1.0)
L3_SOFT_FEATURE_SCALES = (10.0, 10.0, 1.0, 1.0, 11.0, 1.0)
L3_INPUT_FIXED_DIM = (RUNTIME_CONTEXT_SIZE + GLOBAL_FEATURE_DIM + 6 + 4 + 1
                      + len(ROUTE_ORDER) + len(FAMILY_ORDER) * 2 + 1)
_NORMAL_ROUTES = {
    HierarchicalRoute.CONSTRUCTION_COMMIT,
    HierarchicalRoute.CITY_VALUE_COMPARISON,
    HierarchicalRoute.DUAL_CONSTRUCTION_COMPARISON,
    HierarchicalRoute.INVESTMENT,
}


@dataclass(frozen=True)
class DifferentiableSoftPlan:
    available: torch.Tensor  # [B] bool; zero-filled features are not a Plan
    weighted_features: torch.Tensor  # [B, 6], raw 3B-4 diagnostic units
    expected_priorities: torch.Tensor  # [B, 4], independent sigmoid values


def differentiable_soft_plan(
    output: VictoryPlanShadowOutput, candidate_features: torch.Tensor,
) -> DifferentiableSoftPlan:
    """Preserve the Plan Scorer and Priority Head computation graph.

    Padded and IMPOSSIBLE slots contribute exactly zero. No Python scalar
    conversion, detach, or renormalized fictional Plan is introduced here.
    """
    if candidate_features.ndim != 3 or candidate_features.shape[:2] != output.probabilities.shape:
        raise ValueError("Candidate feature / probability batch mismatch")
    if candidate_features.shape[-1] < 6 or output.priorities.shape != (*output.probabilities.shape, 4):
        raise ValueError("Candidate or Priority feature shape mismatch")
    if output.candidate_mask.shape != output.probabilities.shape:
        raise ValueError("Candidate mask shape mismatch")
    scales = candidate_features.new_tensor(_BRIDGE_RAW_SCALES)
    raw_six = candidate_features[..., :6] * scales
    probabilities = output.probabilities * output.candidate_mask.to(output.probabilities.dtype)
    valid = output.plan_available.bool()
    probabilities = torch.where(valid.unsqueeze(-1), probabilities,
                                torch.zeros_like(probabilities))
    weighted = (probabilities.unsqueeze(-1) * raw_six).sum(dim=1)
    expected = (probabilities.unsqueeze(-1) * output.priorities).sum(dim=1)
    return DifferentiableSoftPlan(valid, weighted, expected)


@dataclass(frozen=True)
class L3DecisionFacts:
    route_one_hot: torch.Tensor  # [B, 8]
    legal_family_mask: torch.Tensor  # [B, 7], diagnostic only
    comparison_family_mask: torch.Tensor  # [B, 7]
    construction_commit: torch.Tensor  # [B, 1], state-local fact
    l3_route_available: torch.Tensor  # [B] bool


def encode_l3_decision_facts(
    decisions: tuple[HierarchicalDecisionContext, ...] | list[HierarchicalDecisionContext],
    *, device: torch.device | str | None = None,
    allow_no_use_fallback: bool = False,
) -> L3DecisionFacts:
    """Encode deterministic routing facts; never sample a Dev usage choice.

    ACTION_DEV_USAGE can expose its normal-route mask only when the caller has
    explicitly taken the hypothetical NO_USE fallback. PRE_ROLL_DEV_USAGE and
    forced/immediate-win boundaries never enter L3.
    """
    if not decisions:
        raise ValueError("L3 decision batch cannot be empty")
    effective_routes = [
        item.fallback_route if (allow_no_use_fallback
                                and item.route is HierarchicalRoute.ACTION_DEV_USAGE
                                and item.fallback_route in _NORMAL_ROUTES)
        else item.route for item in decisions
    ]
    routes = torch.tensor([ROUTE_ORDER.index(route) for route in effective_routes],
                          dtype=torch.long, device=device)
    route_one_hot = torch.nn.functional.one_hot(routes, len(ROUTE_ORDER)).to(dtype=torch.float32)
    legal = torch.tensor([item.legal_family_mask for item in decisions],
                         dtype=torch.bool, device=device)
    comparison = torch.tensor([item.comparison_family_mask for item in decisions],
                              dtype=torch.bool, device=device)
    commit = torch.tensor([[item.construction_commit_active] for item in decisions],
                          dtype=torch.float32, device=device)
    route_available = torch.tensor([
        item.route in _NORMAL_ROUTES or (
            allow_no_use_fallback and item.route is HierarchicalRoute.ACTION_DEV_USAGE
            and item.fallback_route in _NORMAL_ROUTES)
        for item in decisions], dtype=torch.bool, device=device)
    if torch.any(comparison & ~legal):
        raise AssertionError("L3 comparison family is not rule-available")
    return L3DecisionFacts(route_one_hot, legal, comparison, commit, route_available)


@dataclass(frozen=True)
class L3ShadowOutput:
    raw_logits: torch.Tensor  # [B, 7]
    masked_logits: torch.Tensor  # [B, 7]
    probabilities: torch.Tensor  # [B, 7]
    entropy: torch.Tensor  # [B]
    available: torch.Tensor  # [B] bool
    comparison_mask: torch.Tensor  # [B, 7] bool


class L3ShadowActor(nn.Module):
    """One shared small MLP; final family layer is neutral until learned."""

    def __init__(self, frozen_actor_latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.frozen_actor_latent_dim = frozen_actor_latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(frozen_actor_latent_dim + L3_INPUT_FIXED_DIM, hidden_dim),
            nn.Tanh(),
        )
        self.family_head = nn.Linear(hidden_dim, len(FAMILY_ORDER))
        nn.init.zeros_(self.family_head.weight)
        nn.init.zeros_(self.family_head.bias)

    def forward(
        self, frozen_actor_latent: torch.Tensor, runtime_context: torch.Tensor,
        plan_global_features: torch.Tensor, soft_plan: DifferentiableSoftPlan,
        facts: L3DecisionFacts,
    ) -> L3ShadowOutput:
        batch = frozen_actor_latent.shape[0]
        if (frozen_actor_latent.shape != (batch, self.frozen_actor_latent_dim)
                or runtime_context.shape != (batch, RUNTIME_CONTEXT_SIZE)
                or plan_global_features.shape != (batch, GLOBAL_FEATURE_DIM)
                or soft_plan.weighted_features.shape != (batch, 6)
                or soft_plan.expected_priorities.shape != (batch, 4)
                or soft_plan.available.shape != (batch,)
                or facts.route_one_hot.shape != (batch, len(ROUTE_ORDER))
                or facts.legal_family_mask.shape != (batch, len(FAMILY_ORDER))
                or facts.comparison_family_mask.shape != (batch, len(FAMILY_ORDER))
                or facts.construction_commit.shape != (batch, 1)
                or facts.l3_route_available.shape != (batch,)):
            raise ValueError("L3 batch feature shape mismatch")
        scales = soft_plan.weighted_features.new_tensor(L3_SOFT_FEATURE_SCALES)
        inputs = torch.cat((
            frozen_actor_latent.detach(), runtime_context, plan_global_features,
            soft_plan.weighted_features / scales, soft_plan.expected_priorities,
            soft_plan.available.to(soft_plan.weighted_features.dtype).unsqueeze(-1),
            facts.route_one_hot,
            facts.legal_family_mask.to(soft_plan.weighted_features.dtype),
            facts.comparison_family_mask.to(soft_plan.weighted_features.dtype),
            facts.construction_commit,
        ), dim=-1)
        raw = self.family_head(self.encoder(inputs))
        mask = facts.comparison_family_mask & facts.legal_family_mask & facts.l3_route_available.unsqueeze(-1)
        available = mask.any(dim=-1)
        masked = raw.masked_fill(~mask, -1e8)
        probabilities = torch.softmax(masked, dim=-1) * mask.to(raw.dtype)
        probabilities = torch.where(available.unsqueeze(-1), probabilities,
                                    torch.zeros_like(probabilities))
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
        return L3ShadowOutput(raw, masked, probabilities, entropy, available, mask)
