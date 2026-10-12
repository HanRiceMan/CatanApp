"""Side-effect-free Step 3B-4 routing and soft Victory Plan representation.

This module classifies a decision boundary. It never chooses, recommends, or
applies a concrete Action, and its masks are not wired to the game controller.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isclose, isfinite

from app.domain.actions import (
    BankTradeAction, BuildCityAction, BuildRoadAction, BuildSettlementAction,
    BuyDevelopmentAction, EndTurnAction, UseDevelopmentAction,
)
from app.domain.game import GameState, legal_actions

from .strategic_gate_step3a13 import resolve_immediate_win
from .victory_plan_context import VictoryPlanContext
from .victory_plan_policy import VictoryPlanShadowOutput


SCHEMA = "hierarchical_shadow_controller_step3b4_v1"
FAMILY_ORDER = (
    "BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "BUY_DEVELOPMENT",
    "BANK_TRADE", "PLAYER_TRADE", "END_TURN",
)
DEVELOPMENT_ORDER = ("KNIGHT", "MONOPOLY", "YEAR_OF_PLENTY", "ROAD_BUILDING")
DEVELOPMENT_USAGE_ORDER = ("NO_USE",) + DEVELOPMENT_ORDER
_CARD_TO_NAME = {"knight": "KNIGHT", "monopoly": "MONOPOLY",
                 "year_of_plenty": "YEAR_OF_PLENTY", "road_building": "ROAD_BUILDING"}
_ACTION_TYPES = (
    BuildSettlementAction, BuildCityAction, BuildRoadAction,
    BuyDevelopmentAction, BankTradeAction, None, EndTurnAction,
)


class HierarchicalRoute(str, Enum):
    IMMEDIATE_WIN = "IMMEDIATE_WIN"
    FORCED_PHASE = "FORCED_PHASE"
    PRE_ROLL_DEV_USAGE = "PRE_ROLL_DEV_USAGE"
    ACTION_DEV_USAGE = "ACTION_DEV_USAGE"
    CONSTRUCTION_COMMIT = "CONSTRUCTION_COMMIT"
    CITY_VALUE_COMPARISON = "CITY_VALUE_COMPARISON"
    DUAL_CONSTRUCTION_COMPARISON = "DUAL_CONSTRUCTION_COMPARISON"
    INVESTMENT = "INVESTMENT"


@dataclass(frozen=True)
class SoftPlanRepresentation:
    available: bool
    candidate_ids: tuple[tuple[int, int, bool, bool], ...]
    candidate_mask: tuple[bool, ...]
    candidate_probabilities: tuple[float, ...]
    candidate_priorities: tuple[tuple[float, float, float, float], ...]
    # S gain, C gain, P(Longest), P(Army), planned gain, overshoot.
    weighted_features: tuple[float, float, float, float, float, float] | None
    expected_priorities: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class HierarchicalDecisionContext:
    schema: str
    route: HierarchicalRoute
    fallback_route: HierarchicalRoute | None
    phase: str
    build_class: str | None
    settlement_buildable: bool
    city_buildable: bool
    usable_development: tuple[bool, bool, bool, bool]
    development_usage_mask: tuple[bool, bool, bool, bool, bool]
    legal_family_mask: tuple[bool, bool, bool, bool, bool, bool, bool]
    comparison_family_mask: tuple[bool, bool, bool, bool, bool, bool, bool]
    player_trade_rule_available: bool
    construction_commit_active: bool
    forced_reason: str | None
    soft_plan: SoftPlanRepresentation | None = None


def _player_trade_rule_available(game: GameState, player_id: int) -> bool:
    """Rule-level proposal possibility, not a protected-PlayerTrade value score."""
    return (game.phase == "action" and game.current_player_id == player_id
            and not game.free_road_remaining
            and game.pending_trade is None and game.trade_count < 5
            and any(player.id != player_id for player in game.players)
            and any(game.players[player_id - 1].resources.values()))


def _development_mask(legal: list) -> tuple[bool, bool, bool, bool]:
    names = {_CARD_TO_NAME[action.card] for action in legal
             if isinstance(action, UseDevelopmentAction)
             and action.card in _CARD_TO_NAME}
    return tuple(name in names for name in DEVELOPMENT_ORDER)


def _family_mask(legal: list, player_trade: bool) -> tuple[bool, ...]:
    return tuple(player_trade if action_type is None else
                 any(isinstance(action, action_type) for action in legal)
                 for action_type in _ACTION_TYPES)


def _build_class(settlement: bool, city: bool) -> str:
    return ("BOTH" if settlement and city else
            "SETTLEMENT_ONLY" if settlement else
            "CITY_ONLY" if city else "NONE")


def classify_hierarchical_route(
    game: GameState, player_id: int, *, legal: list | None = None,
    immediate_win_available: bool | None = None,
) -> HierarchicalDecisionContext:
    """Classify using only the pre-action state and deterministic legal facts.

    ``immediate_win_available`` is injectable for tests/observers that have
    already run the resolver; it never comes from the Champion's chosen Action.
    """
    phase = game.phase
    if legal is None:
        legal = legal_actions(game, player_id)
    if immediate_win_available is None:
        immediate_win_available = (resolve_immediate_win(game, player_id).action is not None
                                   if phase in {"action", "turn_pre_roll"} else False)
    settlement = any(isinstance(action, BuildSettlementAction) for action in legal)
    city = any(isinstance(action, BuildCityAction) for action in legal)
    dev = _development_mask(legal)
    player_trade = _player_trade_rule_available(game, player_id)
    families = _family_mask(legal, player_trade)
    build_class = _build_class(settlement, city) if phase == "action" and not game.free_road_remaining else None
    fallback = None
    reason = None
    if immediate_win_available:
        route = HierarchicalRoute.IMMEDIATE_WIN
    elif phase == "action" and game.free_road_remaining:
        route, reason = HierarchicalRoute.FORCED_PHASE, "FREE_ROAD"
    elif phase == "turn_pre_roll" and game.current_player_id == player_id:
        if any(dev):
            route = HierarchicalRoute.PRE_ROLL_DEV_USAGE
        else:
            route, reason = HierarchicalRoute.FORCED_PHASE, "ROLL_ONLY"
    elif phase != "action" or game.current_player_id != player_id or game.pending_trade is not None:
        route, reason = HierarchicalRoute.FORCED_PHASE, (
            "PENDING_TRADE" if game.pending_trade is not None else phase.upper())
    else:
        fallback = {
            "SETTLEMENT_ONLY": HierarchicalRoute.CONSTRUCTION_COMMIT,
            "CITY_ONLY": HierarchicalRoute.CITY_VALUE_COMPARISON,
            "BOTH": HierarchicalRoute.DUAL_CONSTRUCTION_COMPARISON,
            "NONE": HierarchicalRoute.INVESTMENT,
        }[build_class]
        route = HierarchicalRoute.ACTION_DEV_USAGE if any(dev) else fallback
    normal_route = fallback if route is HierarchicalRoute.ACTION_DEV_USAGE else route
    if normal_route is HierarchicalRoute.DUAL_CONSTRUCTION_COMPARISON:
        comparison = tuple(available and index < 2
                           for index, available in enumerate(families))
    elif normal_route is HierarchicalRoute.INVESTMENT:
        comparison = tuple(available and index >= 2
                           for index, available in enumerate(families))
    elif normal_route is HierarchicalRoute.CITY_VALUE_COMPARISON:
        comparison = tuple(available and index != 0
                           for index, available in enumerate(families))
    elif normal_route is HierarchicalRoute.CONSTRUCTION_COMMIT:
        # Diagnostic availability only. No hard commitment family mask exists.
        comparison = families
    else:
        comparison = (False,) * len(FAMILY_ORDER)
    return HierarchicalDecisionContext(
        SCHEMA, route, fallback, phase, build_class, settlement, city, dev,
        ((True,) + dev if route in {HierarchicalRoute.PRE_ROLL_DEV_USAGE,
                                   HierarchicalRoute.ACTION_DEV_USAGE}
         else (False,) * len(DEVELOPMENT_USAGE_ORDER)),
        families, comparison, player_trade,
        normal_route is HierarchicalRoute.CONSTRUCTION_COMMIT, reason,
    )


def soft_plan_representation(
    context: VictoryPlanContext, output: VictoryPlanShadowOutput,
) -> SoftPlanRepresentation:
    """Expectation over candidate facts; no Plan is sampled or selected."""
    n = len(context.candidates)
    probabilities = tuple(float(value) for value in output.probabilities[0, :n].detach())
    priorities = tuple(tuple(float(value) for value in row)
                       for row in output.priorities[0, :n].detach())
    mask = context.candidate_mask
    if tuple(bool(value) for value in output.candidate_mask[0, :n]) != mask:
        raise AssertionError("Plan Head/context candidate mask mismatch")
    if len(probabilities) != n or len(priorities) != n:
        raise AssertionError("Plan Head/candidate alignment mismatch")
    if any(not isfinite(value) or value < 0 for value in probabilities):
        raise AssertionError("Invalid Plan probability")
    if any(probabilities[i] != 0 for i, valid in enumerate(mask) if not valid):
        raise AssertionError("IMPOSSIBLE Plan has nonzero probability")
    available = bool(any(mask)) and context.remaining_points > 0
    if available != bool(output.plan_available.item()):
        raise AssertionError("Plan availability disagreement")
    if not available:
        if any(probabilities):
            raise AssertionError("Unavailable Plan has a distribution")
        features = expected = None
    else:
        if not isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-6):
            raise AssertionError("Plan probability does not sum to one")
        raw = tuple((float(candidate.settlement_gain_count),
                     float(candidate.city_upgrade_count),
                     float(candidate.acquire_longest_road),
                     float(candidate.acquire_largest_army),
                     float(candidate.planned_point_gain),
                     float(context.overshoot_points(candidate)))
                    for candidate in context.candidates)
        features = tuple(sum(probabilities[i] * raw[i][j] for i in range(n))
                         for j in range(6))
        expected = tuple(sum(probabilities[i] * priorities[i][j] for i in range(n))
                         for j in range(4))
        if not all(isfinite(value) for value in (*features, *expected)):
            raise AssertionError("Nonfinite soft Plan representation")
    return SoftPlanRepresentation(
        available, tuple(candidate.candidate_id for candidate in context.candidates),
        mask, probabilities, priorities, features, expected,
    )


def with_soft_plan(decision: HierarchicalDecisionContext,
                   soft: SoftPlanRepresentation) -> HierarchicalDecisionContext:
    """Attach read-only soft context without changing routing or any legal mask."""
    return replace(decision, soft_plan=soft)
