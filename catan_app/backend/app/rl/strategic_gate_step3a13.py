"""Zero-impact six-family Gate primitives for the Step 3A-13 audit.

No method in this module applies a Gate proposal to the live GameState.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite, log
from typing import Any

import numpy as np
import torch
from torch import nn

from app.domain.actions import (Action, BankTradeAction, BuildRoadAction,
                                BuyDevelopmentAction, EndTurnAction,
                                UseDevelopmentAction)
from app.domain.game import (GameActionError, GameState, RESOURCES,
                             _longest_road_length, apply_action)

from .action_families import ACTION_FAMILY_INDEX
from .action_space import (ACTION_CATALOG, ACTION_SPACE_SIZE, action_to_id,
                           get_action_mask)
from .expansion_planner import analyze_expansion_plan
from .hierarchical_distribution import FAMILY_COUNT
from .observation import encode_observation, get_observation
from .reward import actual_score
from .runtime_context import RUNTIME_CONTEXT_SIZE, encode_runtime_context
from .settlement_planning_agent import SettlementPlanningAgent
from .strategic_action_surface import FAMILIES, family_of


PRIOR_VERSION = "strategic_gate_prior_step3a13_v1"
_OLD_FAMILIES = ("settlement", "city", "road", "development_purchase",
                 "bank_trade", "end_turn")
_ACTION_IDS = {family: tuple(item.id for item in ACTION_CATALOG
                             if family_of(item.action) == family)
               for family in FAMILIES}


@dataclass(frozen=True)
class FrozenSnapshot:
    actor_latent: torch.Tensor
    critic_value: torch.Tensor
    candidate_logits: np.ndarray
    old_family_logits: np.ndarray
    runtime_context: np.ndarray


def frozen_snapshot(policy_agent, game: GameState, player_id: int,
                    *, max_plan_roads: int = 3) -> FrozenSnapshot:
    """Read frozen v2 Actor/Critic and deterministic v7 suffix once."""
    observation = encode_observation(get_observation(game, player_id, version="v2"))
    policy = policy_agent.model.policy
    with torch.no_grad():
        tensor, _ = policy.obs_to_tensor(observation)
        latent = policy.mlp_extractor.forward_actor(tensor).detach()
        value = policy.predict_values(tensor).detach()
        all_logits = policy.action_net(latent)[0].detach().cpu().numpy().copy()
    if all_logits.shape != (ACTION_SPACE_SIZE + FAMILY_COUNT,):
        raise AssertionError("Frozen Champion must expose 377 + 15 logits")
    context = encode_runtime_context(game, player_id,
                                     max_plan_roads=max_plan_roads)
    if context.shape != (RUNTIME_CONTEXT_SIZE,) or not np.isfinite(context).all():
        raise AssertionError("Invalid v7 runtime context")
    return FrozenSnapshot(latent, value, all_logits[:ACTION_SPACE_SIZE],
                          all_logits[ACTION_SPACE_SIZE:], context)


def _masked_softmax(scores: np.ndarray, available: tuple[bool, ...]) -> np.ndarray:
    mask = np.asarray(available, dtype=bool)
    if mask.shape != (len(FAMILIES),) or not mask.any():
        raise ValueError("At least one of six families must be available")
    shifted = scores[mask] - np.max(scores[mask])
    weights = np.exp(shifted)
    result = np.zeros(len(FAMILIES), dtype=np.float64)
    result[mask] = weights / weights.sum()
    return result


def family_priors(snapshot: FrozenSnapshot, action_mask: np.ndarray,
                  available: tuple[bool, ...]) -> dict[str, Any]:
    """Candidate max, cardinality-corrected log-mean-exp, native family head."""
    if action_mask.shape != (ACTION_SPACE_SIZE,):
        raise ValueError("Invalid 377 legal mask")
    legal_counts = []
    max_scores = np.full(len(FAMILIES), -np.inf, dtype=np.float64)
    lme_scores = np.full(len(FAMILIES), -np.inf, dtype=np.float64)
    native_scores = np.asarray([
        snapshot.old_family_logits[ACTION_FAMILY_INDEX[name]]
        for name in _OLD_FAMILIES], dtype=np.float64)
    for family_index, family in enumerate(FAMILIES):
        legal_ids = [index for index in _ACTION_IDS[family] if action_mask[index]]
        legal_counts.append(len(legal_ids))
        if bool(legal_ids) != available[family_index]:
            raise AssertionError("Strategic and 377 candidate masks disagree")
        if legal_ids:
            values = snapshot.candidate_logits[legal_ids].astype(np.float64)
            maximum = float(values.max())
            max_scores[family_index] = maximum
            lme_scores[family_index] = (
                maximum + log(float(np.exp(values - maximum).sum()))
                - log(len(legal_ids)))
    priors = {
        "uniform": _masked_softmax(np.zeros(len(FAMILIES)), available),
        "max_legal_logit": _masked_softmax(max_scores, available),
        "log_mean_exp": _masked_softmax(lme_scores, available),
        "native_family_logits": _masked_softmax(native_scores, available),
    }
    return {
        "version": PRIOR_VERSION,
        "legal_action_counts": legal_counts,
        "raw_scores": {"max_legal_logit": [
                           float(value) if np.isfinite(value) else None
                           for value in max_scores],
                       "log_mean_exp": [
                           float(value) if np.isfinite(value) else None
                           for value in lme_scores],
                       "native_family_logits": native_scores.tolist()},
        "probabilities": {name: value.tolist() for name, value in priors.items()},
    }


class StrategicActionGate(nn.Module):
    """Frozen family-head prior plus zero-initialized trainable six-way residual."""

    def __init__(self, frozen_actor_latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(
            frozen_actor_latent_dim + RUNTIME_CONTEXT_SIZE + len(FAMILIES),
            hidden_dim), nn.Tanh())
        self.residual = nn.Linear(hidden_dim, len(FAMILIES))
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, frozen_latent: torch.Tensor, context: torch.Tensor,
                candidate_mask: torch.Tensor,
                frozen_family_logits: torch.Tensor,
                temperature: float = 1.0) -> torch.Tensor:
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("Strategic prior temperature must be finite and positive")
        mask = candidate_mask.bool()
        if not mask.any(dim=-1).all():
            raise ValueError("Strategic Gate requires a nonempty family mask")
        features = torch.cat((frozen_latent.detach(), context, mask.to(context.dtype)),
                             dim=-1)
        prior = (frozen_family_logits.detach() if temperature == 1.0
                 else frozen_family_logits.detach() / temperature)
        logits = prior + self.residual(self.encoder(features))
        return logits.masked_fill(~mask, -1e8)


class StrategicSurfaceCritic(nn.Module):
    """Separate value residual; the frozen base Critic is not a new-policy oracle."""

    def __init__(self, frozen_actor_latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(
            frozen_actor_latent_dim + RUNTIME_CONTEXT_SIZE + len(FAMILIES),
            hidden_dim), nn.Tanh())
        self.residual = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, base_value: torch.Tensor, frozen_latent: torch.Tensor,
                context: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        features = torch.cat((frozen_latent.detach(), context,
                              candidate_mask.to(context.dtype)), dim=-1)
        return base_value.detach() + self.residual(self.encoder(features))


@dataclass(frozen=True)
class StrategicTransition:
    """Future semi-MDP record; Step 3A-13 collects none of these."""

    family_index: int
    candidate_mask: tuple[bool, ...]
    gate_log_prob: float
    value: float
    reward_accumulated: float
    macro_duration: int
    done: bool
    next_actual_delegated: bool
    next_candidate_mask: tuple[bool, ...] | None = None
    truncated: bool = False
    delegated: bool = True

    def __post_init__(self) -> None:
        if (len(self.candidate_mask) != len(FAMILIES)
                or self.family_index not in range(len(FAMILIES))
                or not self.candidate_mask[self.family_index]
                or not self.delegated or self.macro_duration < 1):
            raise ValueError("Invalid on-policy Strategic transition")
        if self.done and (self.next_actual_delegated or self.truncated):
            raise ValueError("Terminal cannot bootstrap or be truncation")
        if not self.done and not self.truncated and not self.next_actual_delegated:
            raise ValueError("Nonterminal transitions close at the next delegated Gate")
        if not self.done and self.next_actual_delegated != (
                self.next_candidate_mask is not None):
            raise ValueError("Bootstrap only at the next actually delegated Gate")
        if self.next_candidate_mask is not None and len(self.next_candidate_mask) != len(FAMILIES):
            raise ValueError("Next Gate mask must have six entries")

    def bootstrap_discount(self, gamma: float) -> float:
        return 0.0 if self.done else gamma ** self.macro_duration


@dataclass(frozen=True)
class ExecutorResult:
    family: str
    method: str
    action: Action | None
    failure: str | None

    @property
    def success(self) -> bool:
        return self.action is not None and self.failure is None


class FrozenStrategicExecutors:
    """No family selection, Champion actual action, Intent, or Reason input."""

    def __init__(self, champion: SettlementPlanningAgent):
        self.champion = champion

    def _raw_action(self, game: GameState, player_id: int, family: str,
                    method: str, mask: np.ndarray,
                    snapshot: FrozenSnapshot) -> Action | None:
        if family == "BUILD_SETTLEMENT":
            return self.champion._best_settlement(game, player_id)
        if family == "BUILD_CITY":
            return self.champion._best_city(game, player_id)
        if family == "BUY_DEVELOPMENT":
            return BuyDevelopmentAction()
        if family == "END_TURN":
            return EndTurnAction()
        if family == "BANK_TRADE" and method == "A_PLANNER_EXTRACTION":
            # All existing helpers require a Planner-selected goal/budget.
            return None
        if family == "BUILD_ROAD" and method == "A_PLANNER_EXTRACTION":
            plan = analyze_expansion_plan(
                game, player_id, max_additional_roads=self.champion.max_plan_roads)
            roads = self.champion._purposeful_road_candidates(
                game, player_id, plan, affordable_only=True)
            if not roads:
                return None
            before = _longest_road_length(game, player_id)

            def road_rank(road: BuildRoadAction):
                game.roads[road.edge_id] = player_id
                try:
                    after = _longest_road_length(game, player_id)
                finally:
                    del game.roads[road.edge_id]
                return (self.champion._road_is_expansion_step(plan, road),
                        after - before, after,
                        snapshot.candidate_logits[action_to_id(road)],
                        -road.edge_id)

            return max(roads, key=road_rank)
        if family in {"BUILD_ROAD", "BANK_TRADE"} and method == "B_FROZEN_PPO":
            legal = [index for index in _ACTION_IDS[family] if mask[index]]
            if not legal:
                return None
            selected = max(legal, key=lambda index: (
                float(snapshot.candidate_logits[index]), -index))
            return ACTION_CATALOG[selected].action
        raise ValueError(f"No executor method {method} for {family}")

    def execute(self, game: GameState, player_id: int, family: str,
                *, method: str = "B_FROZEN_PPO",
                mask: np.ndarray | None = None,
                snapshot: FrozenSnapshot | None = None) -> ExecutorResult:
        if family not in FAMILIES:
            raise ValueError("Unknown Strategic family")
        mask = get_action_mask(game, player_id) if mask is None else mask
        if not any(mask[index] for index in _ACTION_IDS[family]):
            return ExecutorResult(family, method, None, "FAMILY_UNAVAILABLE")
        if family == "BANK_TRADE" and method == "A_PLANNER_EXTRACTION":
            return ExecutorResult(family, method, None,
                                  "UNSAFE_GOAL_BUDGET_DEPENDENCY")
        snapshot = snapshot or frozen_snapshot(
            self.champion.policy, game, player_id,
            max_plan_roads=self.champion.max_plan_roads)
        before = deepcopy(game)
        try:
            first = self._raw_action(game, player_id, family, method, mask, snapshot)
            second = self._raw_action(game, player_id, family, method, mask, snapshot)
            if first != second:
                raise ValueError("NONDETERMINISTIC_EXECUTOR")
            if game != before:
                raise ValueError("EXECUTOR_MUTATED_GAMESTATE")
            if first is None:
                return ExecutorResult(family, method, None, "NO_FAMILY_ACTION")
            if family_of(first) != family:
                raise ValueError("EXECUTOR_RETURNED_WRONG_FAMILY")
            if not mask[action_to_id(first)]:
                raise ValueError("EXECUTOR_RETURNED_NONLEGAL_ACTION")
            clone = deepcopy(game)
            resources_before = {
                resource: clone.bank[resource] + sum(
                    player.resources[resource] for player in clone.players)
                for resource in RESOURCES
            } if family == "BANK_TRADE" else None
            apply_action(clone, player_id, first, clone.revision,
                         "strategic-executor-audit")
            if resources_before is not None and any(
                resources_before[resource] != clone.bank[resource] + sum(
                    player.resources[resource] for player in clone.players)
                or clone.bank[resource] < 0
                or any(player.resources[resource] < 0 for player in clone.players)
                for resource in RESOURCES
            ):
                raise ValueError("BANK_RESOURCE_OR_INVENTORY_INVARIANT")
            return ExecutorResult(family, method, first, None)
        except (GameActionError, ValueError, AssertionError) as error:
            if game != before:
                raise AssertionError("Failed executor mutated source GameState") from error
            return ExecutorResult(family, method, None, str(error))


@dataclass(frozen=True)
class ImmediateWinResult:
    action: Action | None
    winning_action_ids: tuple[int, ...]


def resolve_immediate_win(game: GameState, player_id: int,
                          mask: np.ndarray | None = None) -> ImmediateWinResult:
    """Copy-apply deterministic scoring actions; lowest catalog ID breaks ties."""
    if game.phase not in {"action", "turn_pre_roll"} or actual_score(game, player_id) < 8:
        return ImmediateWinResult(None, ())
    mask = get_action_mask(game, player_id) if mask is None else mask
    kinds = {"BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD"}
    winning: list[int] = []
    for item in ACTION_CATALOG:
        if not mask[item.id]:
            continue
        if family_of(item.action) not in kinds and item.action != UseDevelopmentAction("knight"):
            continue
        clone = deepcopy(game)
        apply_action(clone, player_id, item.action, clone.revision,
                     "immediate-win-audit")
        if clone.winner_id == player_id and actual_score(clone, player_id) >= 10:
            winning.append(item.id)
    if not winning:
        return ImmediateWinResult(None, ())
    return ImmediateWinResult(ACTION_CATALOG[min(winning)].action,
                              tuple(winning))
