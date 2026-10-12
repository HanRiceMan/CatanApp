"""Untrained, action-independent Victory Plan and Priority shadow heads.

The candidate scorer sees only frozen Champion representation, deterministic
runtime context, public/self-known plan facts, and one candidate at a time.
No Plan is selected or applied by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .runtime_context import RUNTIME_CONTEXT_SIZE
from .victory_plan_context import PlanFeasibility, VictoryPlanContext


SCHEMA_VERSION = "victory_plan_shadow_heads_v1"
MAX_PLAN_CANDIDATES = 36
GLOBAL_FEATURE_DIM = 30
CANDIDATE_FEATURE_DIM = 9
PRIORITY_ORDER = ("settlement", "city", "longest_road", "largest_army")

GLOBAL_FEATURE_ORDER = (
    "current_score", "remaining_points",
    "settlement.current_count", "settlement.pieces_available",
    "settlement.distance_rule_empty_count", "settlement.structurally_reachable_count",
    "settlement.currently_connected_count", "settlement.max_additional_roads",
    "city.current_count", "city.pieces_available", "city.upgradeable_count",
    "longest.currently_owned", "longest.current_length",
    "longest.public_best_opponent_length", "longest.acquisition_gap",
    "longest.defense_gap", "longest.structurally_extendable",
    "longest.holder_self", "longest.holder_opponent", "longest.holder_none",
    "army.currently_owned", "army.used_knights",
    "army.public_best_opponent_used_knights", "army.plays_needed",
    "army.development_deck_remaining", "army.known_usable_knights",
    "army.known_new_knights", "army.holder_self", "army.holder_opponent",
    "army.holder_none",
)
GLOBAL_FEATURE_SCALES = (
    10, 10, 5, 5, 54, 54, 54, 15, 4, 4, 5,
    1, 15, 15, 15, 15, 72, 1, 1, 1,
    1, 14, 14, 14, 25, 14, 14, 1, 1, 1,
)
CANDIDATE_FEATURE_ORDER = (
    "settlement_gain_count", "city_upgrade_count", "acquire_longest_road",
    "acquire_largest_army", "planned_point_gain", "overshoot_points",
    "feasibility.FEASIBLE", "feasibility.UNKNOWN", "feasibility.IMPOSSIBLE",
)
CANDIDATE_FEATURE_SCALES = (10, 10, 1, 1, 11, 1, 1, 1, 1)
assert len(GLOBAL_FEATURE_ORDER) == len(GLOBAL_FEATURE_SCALES) == GLOBAL_FEATURE_DIM
assert len(CANDIDATE_FEATURE_ORDER) == len(CANDIDATE_FEATURE_SCALES) == CANDIDATE_FEATURE_DIM


@dataclass(frozen=True)
class EncodedVictoryPlan:
    global_features: np.ndarray
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    candidate_ids: tuple[tuple[int, int, bool, bool], ...]
    plan_available: bool


def _holder_flags(holder_id: int | None, player_id: int) -> tuple[int, int, int]:
    return (int(holder_id == player_id),
            int(holder_id is not None and holder_id != player_id),
            int(holder_id is None))


def encode_victory_plan_features(context: VictoryPlanContext) -> EncodedVictoryPlan:
    """Pad deterministic facts only; never inject Planner ranks or value scores."""
    s, c, r, a = (context.settlement, context.city,
                  context.longest_road, context.largest_army)
    raw_global = (
        min(context.current_score, 10), context.remaining_points,
        s.current_count, s.pieces_available, s.distance_rule_empty_count,
        s.structurally_reachable_count, s.currently_connected_count,
        s.max_additional_roads, c.current_count, c.pieces_available,
        c.upgradeable_count, int(r.currently_owned), r.current_length,
        r.public_best_opponent_length, r.acquisition_gap, r.defense_gap,
        r.structurally_extendable, *_holder_flags(r.current_holder, context.player_id),
        int(a.currently_owned), a.used_knights,
        a.public_best_opponent_used_knights, a.plays_needed,
        a.development_deck_remaining, a.known_usable_knights,
        a.known_new_knights, *_holder_flags(a.current_holder, context.player_id),
    )
    global_features = np.asarray(raw_global, dtype=np.float32) / np.asarray(
        GLOBAL_FEATURE_SCALES, dtype=np.float32)
    if len(context.candidates) > MAX_PLAN_CANDIDATES:
        raise ValueError("Victory Plan candidate count exceeds fixed padding")
    candidate_features = np.zeros((MAX_PLAN_CANDIDATES, CANDIDATE_FEATURE_DIM),
                                  dtype=np.float32)
    candidate_mask = np.zeros(MAX_PLAN_CANDIDATES, dtype=np.bool_)
    for index, candidate in enumerate(context.candidates):
        raw_candidate = (
            candidate.settlement_gain_count, candidate.city_upgrade_count,
            int(candidate.acquire_longest_road), int(candidate.acquire_largest_army),
            candidate.planned_point_gain, context.overshoot_points(candidate),
            int(candidate.feasibility is PlanFeasibility.FEASIBLE),
            int(candidate.feasibility is PlanFeasibility.UNKNOWN),
            int(candidate.feasibility is PlanFeasibility.IMPOSSIBLE),
        )
        candidate_features[index] = np.asarray(raw_candidate, dtype=np.float32) / np.asarray(
            CANDIDATE_FEATURE_SCALES, dtype=np.float32)
        candidate_mask[index] = candidate.valid
    if not np.isfinite(global_features).all() or not np.isfinite(candidate_features).all():
        raise AssertionError("Victory Plan features must be finite")
    if (global_features < 0).any() or (global_features > 1).any() or \
            (candidate_features < 0).any() or (candidate_features > 1).any():
        raise AssertionError("Victory Plan features exceed rule-based fixed scales")
    for array in (global_features, candidate_features, candidate_mask):
        array.setflags(write=False)
    return EncodedVictoryPlan(
        global_features, candidate_features, candidate_mask,
        tuple(candidate.candidate_id for candidate in context.candidates),
        bool(candidate_mask.any()) and context.remaining_points > 0,
    )


class VictoryPlanFeatureEncoder(nn.Module):
    """One global encoding and one permutation-equivariant shared candidate MLP."""

    def __init__(self, frozen_actor_latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.global_encoder = nn.Sequential(
            nn.Linear(frozen_actor_latent_dim + RUNTIME_CONTEXT_SIZE + GLOBAL_FEATURE_DIM,
                      hidden_dim), nn.Tanh())
        self.candidate_encoder = nn.Sequential(
            nn.Linear(CANDIDATE_FEATURE_DIM, hidden_dim), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh())

    def forward(self, frozen_latent: torch.Tensor, runtime_context: torch.Tensor,
                global_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        global_code = self.global_encoder(torch.cat(
            (frozen_latent.detach(), runtime_context, global_features), dim=-1))
        candidate_code = self.candidate_encoder(candidate_features)
        expanded = global_code.unsqueeze(1).expand(-1, candidate_code.shape[1], -1)
        return self.fusion(torch.cat((expanded, candidate_code), dim=-1))


class VictoryPlanScorer(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, candidate_code: torch.Tensor) -> torch.Tensor:
        return self.output(candidate_code).squeeze(-1)


class StrategicPriorityHead(nn.Module):
    """Four independent sigmoid priorities, not a four-way softmax."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.output = nn.Linear(hidden_dim, len(PRIORITY_ORDER))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, candidate_code: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.output(candidate_code))


@dataclass(frozen=True)
class VictoryPlanShadowOutput:
    raw_logits: torch.Tensor
    probabilities: torch.Tensor
    candidate_mask: torch.Tensor
    plan_available: torch.Tensor
    entropy: torch.Tensor
    priorities: torch.Tensor


class VictoryPlanShadowHeads(nn.Module):
    def __init__(self, frozen_actor_latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_encoder = VictoryPlanFeatureEncoder(frozen_actor_latent_dim, hidden_dim)
        self.plan_scorer = VictoryPlanScorer(hidden_dim)
        self.priority_head = StrategicPriorityHead(hidden_dim)

    def forward(self, frozen_latent: torch.Tensor, runtime_context: torch.Tensor,
                global_features: torch.Tensor, candidate_features: torch.Tensor,
                candidate_mask: torch.Tensor) -> VictoryPlanShadowOutput:
        if runtime_context.shape[-1] != RUNTIME_CONTEXT_SIZE:
            raise ValueError("Runtime Context must have 79 dimensions")
        if candidate_features.shape[-2:] != (MAX_PLAN_CANDIDATES, CANDIDATE_FEATURE_DIM):
            raise ValueError("Candidate features must use 36-slot padding")
        if candidate_mask.shape != candidate_features.shape[:-1]:
            raise ValueError("Candidate mask shape mismatch")
        code = self.feature_encoder(
            frozen_latent, runtime_context, global_features, candidate_features)
        logits = self.plan_scorer(code)
        priorities = self.priority_head(code)
        mask = candidate_mask.bool()
        available = mask.any(dim=-1)
        masked_logits = logits.masked_fill(~mask, -1e8)
        probabilities = torch.softmax(masked_logits, dim=-1) * mask
        probabilities = torch.where(available.unsqueeze(-1), probabilities,
                                    torch.zeros_like(probabilities))
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
        return VictoryPlanShadowOutput(logits, probabilities, mask, available,
                                       entropy, priorities)
