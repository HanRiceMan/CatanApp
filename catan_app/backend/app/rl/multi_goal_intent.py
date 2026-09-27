"""現行Plannerから独立したMulti-Goal Intentを非介入で診断する。

各Goalは排他的ではない。``UNKNOWN`` は「否定材料がない」ではなく、現行の
source-level情報だけではactive/inactiveを安全に確定できないことを表す。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from app.domain.actions import BuildRoadAction, BuyDevelopmentAction, UseDevelopmentAction
from app.domain.game import BUILD_COSTS, GameState, _longest_road_length, player_for

from .development_strategy import assess_development_purchase
from .expansion_planner import analyze_expansion_plan, estimated_build_wait
from .macro_goal import ObjectiveGoal
from .reward import actual_score
from .trade_strategy import construction_trade_goal
from .victory_race import public_score

if TYPE_CHECKING:
    from .settlement_planning_agent import SettlementPlanningAgent


MULTI_GOAL_DIAGNOSTIC_VERSION = "multi_goal_intent_v1"
MULTI_GOALS = (
    ObjectiveGoal.SETTLEMENT,
    ObjectiveGoal.CITY,
    ObjectiveGoal.DEVELOPMENT,
    ObjectiveGoal.ROAD_TITLE,
    ObjectiveGoal.KNIGHT_TITLE,
)


class IntentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNKNOWN = "unknown"


class IntentReasonCode(str, Enum):
    SETTLEMENT_PLAN_ACTIVE = "SETTLEMENT_PLAN_ACTIVE"
    SETTLEMENT_PIECES_EXHAUSTED = "SETTLEMENT_PIECES_EXHAUSTED"
    CITY_ACTION_AVAILABLE = "CITY_ACTION_AVAILABLE"
    CITY_CONSTRUCTION_GOAL = "CITY_CONSTRUCTION_GOAL"
    CITY_PIECES_EXHAUSTED = "CITY_PIECES_EXHAUSTED"
    DEVELOPMENT_PURCHASE_ELIGIBLE = "DEVELOPMENT_PURCHASE_ELIGIBLE"
    DEVELOPMENT_STALLED = "DEVELOPMENT_STALLED"
    DEVELOPMENT_ROBBER_RELEASE = "DEVELOPMENT_ROBBER_RELEASE"
    DEVELOPMENT_DECK_EMPTY = "DEVELOPMENT_DECK_EMPTY"
    ROAD_TITLE_ACTION_AVAILABLE = "ROAD_TITLE_ACTION_AVAILABLE"
    ROAD_TITLE_IMMEDIATE_WIN = "ROAD_TITLE_IMMEDIATE_WIN"
    ROAD_TITLE_NOT_PURSUED_BY_CONFIG = "ROAD_TITLE_NOT_PURSUED_BY_CONFIG"
    ROAD_PIECES_EXHAUSTED = "ROAD_PIECES_EXHAUSTED"
    KNIGHT_TITLE_ACTION_AVAILABLE = "KNIGHT_TITLE_ACTION_AVAILABLE"
    KNIGHT_TITLE_DEVELOPMENT_PURCHASE = "KNIGHT_TITLE_DEVELOPMENT_PURCHASE"
    KNIGHT_TITLE_NOT_PURSUED_BY_CONFIG = "KNIGHT_TITLE_NOT_PURSUED_BY_CONFIG"
    NO_SAFE_SOURCE_LEVEL_DECISION = "NO_SAFE_SOURCE_LEVEL_DECISION"


@dataclass(frozen=True)
class GoalIntentSignal:
    status: IntentStatus
    reason_codes: tuple[IntentReasonCode, ...]
    source_values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason_codes": [reason.value for reason in self.reason_codes],
            "source_values": dict(self.source_values),
        }


@dataclass(frozen=True)
class MultiGoalIntentReport:
    state_id: str
    player_id: int
    turn: int
    phase: str
    signals: dict[ObjectiveGoal, GoalIntentSignal]
    diagnostic_version: str = MULTI_GOAL_DIAGNOSTIC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "player_id": self.player_id,
            "turn": self.turn,
            "phase": self.phase,
            "signals": {
                goal.value: self.signals[goal].to_dict() for goal in MULTI_GOALS
            },
            "diagnostic_version": self.diagnostic_version,
        }


def _unknown(**source_values: Any) -> GoalIntentSignal:
    return GoalIntentSignal(
        IntentStatus.UNKNOWN,
        (IntentReasonCode.NO_SAFE_SOURCE_LEVEL_DECISION,),
        source_values,
    )


def _active(reason: IntentReasonCode, *extra: IntentReasonCode,
            **source_values: Any) -> GoalIntentSignal:
    return GoalIntentSignal(IntentStatus.ACTIVE, (reason, *extra), source_values)


def _inactive(reason: IntentReasonCode, **source_values: Any) -> GoalIntentSignal:
    return GoalIntentSignal(IntentStatus.INACTIVE, (reason,), source_values)


def _road_gap(game: GameState, player_id: int) -> tuple[int, int, int]:
    own = _longest_road_length(game, player_id)
    rival = max(
        (_longest_road_length(game, other.id)
         for other in game.players if other.id != player_id),
        default=0,
    )
    return own, rival, max(0, max(5, rival + 1) - own)


def _source_report(agent: "SettlementPlanningAgent", game: GameState,
                   player_id: int) -> MultiGoalIntentReport:
    player = player_for(game, player_id)
    signals: dict[ObjectiveGoal, GoalIntentSignal] = {}
    if game.phase != "action":
        signals = {goal: _unknown(phase=game.phase) for goal in MULTI_GOALS}
        return MultiGoalIntentReport(
            f"{game.id}:{game.revision}:{player_id}", player_id,
            game.turn_number, game.phase, signals,
        )

    plan = analyze_expansion_plan(
        game, player_id, max_additional_roads=agent.max_plan_roads,
    )
    target = plan.selected_target
    site_count = player.settlements + player.cities
    score = actual_score(game, player_id)
    city_wait_for_comparison = (
        float(estimated_build_wait(game, player_id, BUILD_COSTS["city"]))
        if player.settlements > 0 and player.cities < 4 else 99.0
    )
    settlement_context = bool(
        target is not None
        and (
            site_count < agent.target_sites
            or (score >= 9 and target.wait_rounds < city_wait_for_comparison)
            or (
                agent.post_target_connected_reserve_max_wait
                and player.settlements < 5 and player.cities > 0
                and agent.post_target_connected_reserve_min_score <= score
                <= agent.post_target_connected_reserve_max_score
                and target.additional_roads == 0
                and target.wait_rounds
                <= agent.post_target_connected_reserve_max_wait
            )
            or (
                agent.endgame_point_reserve_score
                and score >= agent.endgame_point_reserve_score
                and target.wait_rounds < city_wait_for_comparison
            )
        )
    )
    settlement_hard_exhausted = player.settlements >= 5 and player.cities >= 4
    if (settlement_context and player.settlements < 5
            and target.wait_rounds <= agent.max_wait_rounds):
        signals[ObjectiveGoal.SETTLEMENT] = _active(
            IntentReasonCode.SETTLEMENT_PLAN_ACTIVE,
            target_vertex=target.vertex_id,
            wait_rounds=float(target.wait_rounds),
            additional_roads=int(target.additional_roads),
            within_target_site_phase=(site_count < agent.target_sites),
        )
    elif settlement_hard_exhausted:
        signals[ObjectiveGoal.SETTLEMENT] = _inactive(
            IntentReasonCode.SETTLEMENT_PIECES_EXHAUSTED,
            settlements=player.settlements,
            cities=player.cities,
        )
    else:
        signals[ObjectiveGoal.SETTLEMENT] = _unknown(
            target_available=target is not None,
            wait_rounds=(float(target.wait_rounds) if target is not None else None),
        )

    construction = construction_trade_goal(
        game, player_id,
        target_sites=agent.target_sites,
        max_plan_roads=agent.max_plan_roads,
        max_wait_rounds=agent.max_wait_rounds,
        max_city_wait_rounds=agent.max_city_wait_rounds,
    )
    best_city = agent._best_city(game, player_id)
    city_wait = (city_wait_for_comparison
                 if city_wait_for_comparison < 99.0 else None)
    if best_city is not None:
        signals[ObjectiveGoal.CITY] = _active(
            IntentReasonCode.CITY_ACTION_AVAILABLE,
            target_vertex=getattr(best_city, "vertex_id", None),
            wait_rounds=city_wait,
        )
    elif construction is not None and construction[0] == "city":
        signals[ObjectiveGoal.CITY] = _active(
            IntentReasonCode.CITY_CONSTRUCTION_GOAL,
            wait_rounds=city_wait,
        )
    elif player.cities >= 4:
        signals[ObjectiveGoal.CITY] = _inactive(
            IntentReasonCode.CITY_PIECES_EXHAUSTED,
            cities=player.cities,
        )
    else:
        signals[ObjectiveGoal.CITY] = _unknown(wait_rounds=city_wait)

    development = assess_development_purchase(
        game, player_id,
        max_plan_roads=agent.max_plan_roads,
        stall_wait_rounds=agent.development_stall_wait_rounds,
        max_build_delay=agent.development_max_build_delay,
        tactical_score=agent.development_tactical_score,
        title_horizon=agent.title_horizon,
    )
    development_reasons = tuple(development["reasons"])
    if development["eligible"]:
        extras = []
        if "construction_stall" in development_reasons:
            extras.append(IntentReasonCode.DEVELOPMENT_STALLED)
        if "robber_relief" in development_reasons:
            extras.append(IntentReasonCode.DEVELOPMENT_ROBBER_RELEASE)
        signals[ObjectiveGoal.DEVELOPMENT] = _active(
            IntentReasonCode.DEVELOPMENT_PURCHASE_ELIGIBLE,
            *extras,
            reasons=list(development_reasons),
            build_wait_before=float(development["build_wait_before"]),
            affordable=bool(development["affordable"]),
        )
    elif not game.development_deck:
        signals[ObjectiveGoal.DEVELOPMENT] = _inactive(
            IntentReasonCode.DEVELOPMENT_DECK_EMPTY,
        )
    else:
        signals[ObjectiveGoal.DEVELOPMENT] = _unknown(
            reasons=list(development_reasons),
            affordable=bool(development["affordable"]),
            eligible=False,
        )

    own_road, rival_road, road_gap = _road_gap(game, player_id)
    score = public_score(game, player_id)
    opponent_near_win = max(
        (public_score(game, other.id) for other in game.players
         if other.id != player_id),
        default=0,
    ) >= 8
    title_gate = bool(
        agent.prioritize_immediate_titles
        and (score >= agent.title_min_score or opponent_near_win)
    )
    road_title_gate = bool(
        title_gate and player.settlements + player.cities >= agent.target_sites
    )
    title_road = (
        agent._longest_road_progress(game, player_id) if road_title_gate else None
    )
    immediate_win_road = None
    if title_road is not None and agent._road_wins_now(game, player_id, title_road):
        immediate_win_road = title_road
    if isinstance(title_road, BuildRoadAction):
        reason = (IntentReasonCode.ROAD_TITLE_IMMEDIATE_WIN
                  if immediate_win_road is not None
                  else IntentReasonCode.ROAD_TITLE_ACTION_AVAILABLE)
        signals[ObjectiveGoal.ROAD_TITLE] = _active(
            reason,
            edge_id=title_road.edge_id,
            own_length=own_road,
            rival_length=rival_road,
            gap=road_gap,
        )
    elif sum(owner == player_id for owner in game.roads.values()) >= 15:
        signals[ObjectiveGoal.ROAD_TITLE] = _inactive(
            IntentReasonCode.ROAD_PIECES_EXHAUSTED,
            own_length=own_road,
        )
    elif (game.longest_road_holder_id == player_id and not agent.defend_titles):
        signals[ObjectiveGoal.ROAD_TITLE] = _inactive(
            IntentReasonCode.ROAD_TITLE_NOT_PURSUED_BY_CONFIG,
            holding=True,
        )
    else:
        signals[ObjectiveGoal.ROAD_TITLE] = _unknown(
            title_gate=road_title_gate,
            own_length=own_road,
            rival_length=rival_road,
            gap=road_gap,
        )

    title_knight = agent._largest_army_progress(game, player_id) if title_gate else None
    knight_purchase_intent = bool(
        development["eligible"] and "largest_army_race" in development_reasons
    )
    if isinstance(title_knight, (UseDevelopmentAction, BuyDevelopmentAction)):
        signals[ObjectiveGoal.KNIGHT_TITLE] = _active(
            IntentReasonCode.KNIGHT_TITLE_ACTION_AVAILABLE,
            action_type=type(title_knight).__name__,
            largest_army_gap=int(development["largest_army_gap"]),
        )
    elif knight_purchase_intent:
        signals[ObjectiveGoal.KNIGHT_TITLE] = _active(
            IntentReasonCode.KNIGHT_TITLE_DEVELOPMENT_PURCHASE,
            largest_army_gap=int(development["largest_army_gap"]),
        )
    elif (game.largest_army_holder_id == player_id and not agent.defend_titles):
        signals[ObjectiveGoal.KNIGHT_TITLE] = _inactive(
            IntentReasonCode.KNIGHT_TITLE_NOT_PURSUED_BY_CONFIG,
            holding=True,
        )
    else:
        signals[ObjectiveGoal.KNIGHT_TITLE] = _unknown(
            title_gate=title_gate,
            largest_army_gap=int(development["largest_army_gap"]),
            development_reasons=list(development_reasons),
        )

    return MultiGoalIntentReport(
        f"{game.id}:{game.revision}:{player_id}", player_id,
        game.turn_number, game.phase, signals,
    )


def diagnose_multi_goal_intents(
    agent: "SettlementPlanningAgent", game: GameState, player_id: int,
) -> MultiGoalIntentReport:
    """GameStateとRNGを変更せず、5 Goalを独立したtri-stateで返す。"""
    return _source_report(agent, deepcopy(game), player_id)
