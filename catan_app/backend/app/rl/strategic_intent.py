"""新taxonomyのStrategic Intentを、Action選択へ介入せず診断する。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from app.domain.actions import (Action, BankTradeAction, BuildCityAction,
                                BuildRoadAction, BuildSettlementAction,
                                BuyDevelopmentAction, EndTurnAction,
                                ProposeTradeAction, UseDevelopmentAction)
from app.domain.game import BUILD_COSTS, GameState, _longest_road_length, player_for

from .development_strategy import (_blocked_production_pips,
                                   assess_development_purchase)
from .expansion_planner import analyze_expansion_plan, estimated_build_wait
from .reward import actual_score
from .trade_strategy import construction_trade_goal
from .victory_race import public_score

if TYPE_CHECKING:
    from .settlement_planning_agent import SettlementPlanningAgent


STRATEGIC_INTENT_SCHEMA_VERSION = "strategic_intent_v1"


class StrategicIntent(str, Enum):
    SETTLEMENT = "SETTLEMENT"
    CITY = "CITY"
    ROAD_TITLE = "ROAD_TITLE"
    KNIGHT_TITLE = "KNIGHT_TITLE"
    ROBBER_RELIEF = "ROBBER_RELIEF"


STRATEGIC_INTENTS = tuple(StrategicIntent)


class IntentState(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNKNOWN = "unknown"


class ExecutionSource(str, Enum):
    PLANNER_ADAPTIVE_DEVELOPMENT = "PLANNER_ADAPTIVE_DEVELOPMENT"
    TITLE_PLANNER = "TITLE_PLANNER"
    SETTLEMENT_PLANNER = "SETTLEMENT_PLANNER"
    CITY_PLANNER = "CITY_PLANNER"
    TRADE_PLANNER = "TRADE_PLANNER"
    PLANNER_GUARD = "PLANNER_GUARD"
    ROBBER_PLANNER = "ROBBER_PLANNER"
    SETUP_PLANNER = "SETUP_PLANNER"
    BASE_PPO = "BASE_PPO"
    UNKNOWN = "UNKNOWN"


class DiagnosticExecution(str, Enum):
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"
    BUILD_CITY = "BUILD_CITY"
    BUILD_ROAD = "BUILD_ROAD"
    BUY_DEVELOPMENT = "BUY_DEVELOPMENT"
    PLAY_KNIGHT = "PLAY_KNIGHT"
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"
    PLAY_MONOPOLY = "PLAY_MONOPOLY"
    PLAY_VICTORY_POINT = "PLAY_VICTORY_POINT"
    TRADE_PLAYER = "TRADE_PLAYER"
    TRADE_BANK = "TRADE_BANK"
    END_TURN = "END_TURN"
    OTHER = "OTHER"


class StrategicReasonCode(str, Enum):
    SETTLEMENT_TARGET_AVAILABLE = "SETTLEMENT_TARGET_AVAILABLE"
    SETTLEMENT_RACE = "SETTLEMENT_RACE"
    SETTLEMENT_RESOURCE_SHORTAGE = "SETTLEMENT_RESOURCE_SHORTAGE"
    CITY_PRODUCTION_GAIN = "CITY_PRODUCTION_GAIN"
    CITY_RESOURCE_SHORTAGE = "CITY_RESOURCE_SHORTAGE"
    ROAD_TITLE_OPPORTUNITY = "ROAD_TITLE_OPPORTUNITY"
    ROAD_TITLE_DEFENSE = "ROAD_TITLE_DEFENSE"
    ROAD_TITLE_IMMEDIATE_WIN = "ROAD_TITLE_IMMEDIATE_WIN"
    LARGEST_ARMY_RACE = "LARGEST_ARMY_RACE"
    LARGEST_ARMY_DEFENSE = "LARGEST_ARMY_DEFENSE"
    LARGEST_ARMY_IMMEDIATE_WIN = "LARGEST_ARMY_IMMEDIATE_WIN"
    ROBBER_BLOCKING_IMPORTANT_TILE = "ROBBER_BLOCKING_IMPORTANT_TILE"
    STALL_BREAK = "STALL_BREAK"
    GENERAL_EXPECTED_VALUE = "GENERAL_EXPECTED_VALUE"
    EXPECTED_KNIGHT_VALUE = "EXPECTED_KNIGHT_VALUE"
    EXPECTED_RESOURCE_GAIN = "EXPECTED_RESOURCE_GAIN"
    ROAD_BUILDING_CARD_SYNERGY = "ROAD_BUILDING_CARD_SYNERGY"
    HAND_OVERFLOW = "HAND_OVERFLOW"
    RESOURCE_REBALANCE = "RESOURCE_REBALANCE"
    TRADE_COMPLETES_GOAL = "TRADE_COMPLETES_GOAL"
    TRADE_IMPROVES_GOAL_ETA = "TRADE_IMPROVES_GOAL_ETA"
    PIECES_EXHAUSTED = "PIECES_EXHAUSTED"
    RULED_OUT_BY_CONFIG = "RULED_OUT_BY_CONFIG"
    NO_SAFE_SOURCE_EVIDENCE = "NO_SAFE_SOURCE_EVIDENCE"


@dataclass(frozen=True)
class ActionSelectionTrace:
    execution_source: ExecutionSource
    contributed_intents: tuple[StrategicIntent, ...] = ()
    contributed_reasons: tuple[StrategicReasonCode, ...] = ()
    development_eligible: bool | None = None
    new_development_cards_blocked: bool | None = None


@dataclass(frozen=True)
class StrategicIntentSignal:
    state: IntentState
    reason_codes: tuple[StrategicReasonCode, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reason_codes": [item.value for item in self.reason_codes],
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class StrategicIntentReport:
    signals: dict[StrategicIntent, StrategicIntentSignal]
    selected_execution: DiagnosticExecution
    execution_source: ExecutionSource
    reason_contributed_to_selection: dict[StrategicIntent, bool | None]
    selection_reason_codes: tuple[StrategicReasonCode, ...]
    development_eligible: bool | None
    new_development_cards_blocked: bool | None
    new_development_cards_blocked_purchase: bool | None
    new_knight_blocked_use: bool | None
    schema_version: str = STRATEGIC_INTENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        targets = [
            int(self.signals[intent].state == IntentState.ACTIVE)
            for intent in STRATEGIC_INTENTS
        ]
        known_mask = [
            int(self.signals[intent].state != IntentState.UNKNOWN)
            for intent in STRATEGIC_INTENTS
        ]
        return {
            "strategic_intent_order": [item.value for item in STRATEGIC_INTENTS],
            "strategic_intent_targets": targets,
            "strategic_intent_known_mask": known_mask,
            "strategic_intent_states": {
                intent.value: self.signals[intent].state.value
                for intent in STRATEGIC_INTENTS
            },
            "strategic_intent_reason_codes": {
                intent.value: [reason.value for reason in self.signals[intent].reason_codes]
                for intent in STRATEGIC_INTENTS
            },
            "strategic_intent_evidence": {
                intent.value: dict(self.signals[intent].evidence)
                for intent in STRATEGIC_INTENTS
            },
            "selected_execution": self.selected_execution.value,
            "execution_source": self.execution_source.value,
            "reason_contributed_to_selection": {
                intent.value: self.reason_contributed_to_selection[intent]
                for intent in STRATEGIC_INTENTS
            },
            "selection_reason_codes": [item.value for item in self.selection_reason_codes],
            "development_eligible": self.development_eligible,
            "new_development_cards_blocked": self.new_development_cards_blocked,
            "new_development_cards_blocked_purchase": (
                self.new_development_cards_blocked_purchase
            ),
            "new_knight_blocked_use": self.new_knight_blocked_use,
            "strategic_intent_schema_version": self.schema_version,
        }


def diagnostic_execution(action: Action) -> DiagnosticExecution:
    if isinstance(action, BuildSettlementAction):
        return DiagnosticExecution.BUILD_SETTLEMENT
    if isinstance(action, BuildCityAction):
        return DiagnosticExecution.BUILD_CITY
    if isinstance(action, BuildRoadAction):
        return DiagnosticExecution.BUILD_ROAD
    if isinstance(action, BuyDevelopmentAction):
        return DiagnosticExecution.BUY_DEVELOPMENT
    if isinstance(action, UseDevelopmentAction):
        return {
            "knight": DiagnosticExecution.PLAY_KNIGHT,
            "road_building": DiagnosticExecution.PLAY_ROAD_BUILDING,
            "year_of_plenty": DiagnosticExecution.PLAY_YEAR_OF_PLENTY,
            "monopoly": DiagnosticExecution.PLAY_MONOPOLY,
            "victory_point": DiagnosticExecution.PLAY_VICTORY_POINT,
        }.get(action.card, DiagnosticExecution.OTHER)
    if isinstance(action, ProposeTradeAction):
        return DiagnosticExecution.TRADE_PLAYER
    if isinstance(action, BankTradeAction):
        return DiagnosticExecution.TRADE_BANK
    if isinstance(action, EndTurnAction):
        return DiagnosticExecution.END_TURN
    return DiagnosticExecution.OTHER


def _signal(state: IntentState, reason: StrategicReasonCode,
            **evidence: Any) -> StrategicIntentSignal:
    return StrategicIntentSignal(state, (reason,), evidence)


def _road_lengths(game: GameState, player_id: int) -> tuple[int, int, int]:
    own = _longest_road_length(game, player_id)
    rival = max((_longest_road_length(game, item.id) for item in game.players
                 if item.id != player_id), default=0)
    return own, rival, max(0, max(5, rival + 1) - own)


def _diagnose(agent: "SettlementPlanningAgent", game: GameState, player_id: int,
              action: Action, trace: ActionSelectionTrace) -> StrategicIntentReport:
    player = player_for(game, player_id)
    signals: dict[StrategicIntent, StrategicIntentSignal] = {}
    development: dict[str, Any] | None = None
    if game.phase != "action":
        signals = {
            intent: _signal(IntentState.UNKNOWN,
                            StrategicReasonCode.NO_SAFE_SOURCE_EVIDENCE,
                            phase=game.phase)
            for intent in STRATEGIC_INTENTS
        }
    else:
        plan = analyze_expansion_plan(
            game, player_id, max_additional_roads=agent.max_plan_roads,
        )
        target = plan.selected_target
        site_count = player.settlements + player.cities
        score = actual_score(game, player_id)
        city_wait = (estimated_build_wait(game, player_id, BUILD_COSTS["city"])
                     if player.settlements > 0 and player.cities < 4 else 99.0)
        settlement_context = bool(
            target is not None and target.wait_rounds <= agent.max_wait_rounds and (
                site_count < agent.target_sites
                or (score >= 9 and target.wait_rounds < city_wait)
                or (agent.endgame_point_reserve_score
                    and score >= agent.endgame_point_reserve_score
                    and target.wait_rounds < city_wait)
                or (agent.post_target_connected_reserve_max_wait
                    and player.settlements < 5 and player.cities > 0
                    and agent.post_target_connected_reserve_min_score <= score
                    <= agent.post_target_connected_reserve_max_score
                    and target.additional_roads == 0
                    and target.wait_rounds
                    <= agent.post_target_connected_reserve_max_wait)
            )
        )
        if ((settlement_context and player.settlements < 5)
                or (StrategicIntent.SETTLEMENT in trace.contributed_intents
                    and not (player.settlements >= 5 and player.cities >= 4))):
            signals[StrategicIntent.SETTLEMENT] = _signal(
                IntentState.ACTIVE, StrategicReasonCode.SETTLEMENT_TARGET_AVAILABLE,
                target_vertex=(target.vertex_id if target is not None else
                               action.vertex_id if isinstance(action, BuildSettlementAction)
                               else None),
                wait_rounds=(float(target.wait_rounds) if target is not None else None),
                additional_roads=(int(target.additional_roads)
                                  if target is not None else None),
                selected_by_source=(StrategicIntent.SETTLEMENT
                                    in trace.contributed_intents),
                settlement_piece_available=player.settlements < 5,
                city_can_recycle_piece=(player.settlements >= 5 and player.cities < 4),
            )
        elif player.settlements >= 5 and player.cities >= 4:
            signals[StrategicIntent.SETTLEMENT] = _signal(
                IntentState.INACTIVE, StrategicReasonCode.PIECES_EXHAUSTED,
                settlements=player.settlements, cities=player.cities,
            )
        else:
            signals[StrategicIntent.SETTLEMENT] = _signal(
                IntentState.UNKNOWN, StrategicReasonCode.NO_SAFE_SOURCE_EVIDENCE,
                target_available=target is not None,
            )

        construction = construction_trade_goal(
            game, player_id, target_sites=agent.target_sites,
            max_plan_roads=agent.max_plan_roads,
            max_wait_rounds=agent.max_wait_rounds,
            max_city_wait_rounds=agent.max_city_wait_rounds,
        )
        best_city = agent._best_city(game, player_id)
        if best_city is not None:
            signals[StrategicIntent.CITY] = _signal(
                IntentState.ACTIVE, StrategicReasonCode.CITY_PRODUCTION_GAIN,
                target_vertex=best_city.vertex_id, wait_rounds=float(city_wait),
            )
        elif construction is not None and construction[0] == "city":
            signals[StrategicIntent.CITY] = _signal(
                IntentState.ACTIVE, StrategicReasonCode.CITY_RESOURCE_SHORTAGE,
                wait_rounds=float(city_wait),
            )
        elif player.cities >= 4:
            signals[StrategicIntent.CITY] = _signal(
                IntentState.INACTIVE, StrategicReasonCode.PIECES_EXHAUSTED,
                cities=player.cities,
            )
        else:
            signals[StrategicIntent.CITY] = _signal(
                IntentState.UNKNOWN, StrategicReasonCode.NO_SAFE_SOURCE_EVIDENCE,
                wait_rounds=(float(city_wait) if city_wait < 99 else None),
            )

        public = public_score(game, player_id)
        opponent_near_win = max((public_score(game, item.id) for item in game.players
                                 if item.id != player_id), default=0) >= 8
        title_gate = bool(
            agent.prioritize_immediate_titles
            and (public >= agent.title_min_score or opponent_near_win)
        )
        # 道路称号の分岐は騎士のscore gateと異なり、拠点数だけで開く。
        road_gate = agent.prioritize_immediate_titles and site_count >= agent.target_sites
        road_action = agent._longest_road_progress(game, player_id) if road_gate else None
        own_road, rival_road, road_gap = _road_lengths(game, player_id)
        if (isinstance(road_action, BuildRoadAction)
                or StrategicIntent.ROAD_TITLE in trace.contributed_intents):
            selected_road = road_action if isinstance(road_action, BuildRoadAction) else action
            reason = (StrategicReasonCode.ROAD_TITLE_IMMEDIATE_WIN
                      if isinstance(selected_road, BuildRoadAction)
                      and agent._road_wins_now(game, player_id, selected_road)
                      else StrategicReasonCode.ROAD_TITLE_OPPORTUNITY)
            signals[StrategicIntent.ROAD_TITLE] = _signal(
                IntentState.ACTIVE, reason,
                edge_id=(selected_road.edge_id
                         if isinstance(selected_road, BuildRoadAction) else None),
                own_length=own_road, rival_length=rival_road, gap=road_gap,
                selected_by_source=(StrategicIntent.ROAD_TITLE
                                    in trace.contributed_intents),
            )
        elif sum(owner == player_id for owner in game.roads.values()) >= 15:
            signals[StrategicIntent.ROAD_TITLE] = _signal(
                IntentState.INACTIVE, StrategicReasonCode.PIECES_EXHAUSTED,
                own_length=own_road,
            )
        elif game.longest_road_holder_id == player_id and not agent.defend_titles:
            signals[StrategicIntent.ROAD_TITLE] = _signal(
                IntentState.INACTIVE, StrategicReasonCode.RULED_OUT_BY_CONFIG,
                holding=True,
            )
        else:
            signals[StrategicIntent.ROAD_TITLE] = _signal(
                IntentState.UNKNOWN, StrategicReasonCode.NO_SAFE_SOURCE_EVIDENCE,
                title_gate=road_gate, gap=road_gap,
            )

        development = assess_development_purchase(
            game, player_id, max_plan_roads=agent.max_plan_roads,
            stall_wait_rounds=agent.development_stall_wait_rounds,
            max_build_delay=agent.development_max_build_delay,
            tactical_score=agent.development_tactical_score,
            title_horizon=agent.title_horizon,
        )
        knight_action = agent._largest_army_progress(game, player_id) if title_gate else None
        largest_race = "largest_army_race" in development["reasons"]
        if knight_action is not None or largest_race:
            signals[StrategicIntent.KNIGHT_TITLE] = _signal(
                IntentState.ACTIVE, StrategicReasonCode.LARGEST_ARMY_RACE,
                candidate_action=(type(knight_action).__name__
                                  if knight_action is not None else None),
                gap=int(development["largest_army_gap"]),
                development_eligible=bool(development["eligible"]),
            )
        elif game.largest_army_holder_id == player_id and not agent.defend_titles:
            signals[StrategicIntent.KNIGHT_TITLE] = _signal(
                IntentState.INACTIVE, StrategicReasonCode.RULED_OUT_BY_CONFIG,
                holding=True,
            )
        else:
            usable_knight = (player.development_cards.count("knight")
                             > player.new_development_cards.count("knight"))
            if not game.development_deck and not usable_knight:
                signals[StrategicIntent.KNIGHT_TITLE] = _signal(
                    IntentState.INACTIVE, StrategicReasonCode.PIECES_EXHAUSTED,
                    development_deck_empty=True, usable_knight=False,
                )
            else:
                signals[StrategicIntent.KNIGHT_TITLE] = _signal(
                    IntentState.UNKNOWN, StrategicReasonCode.NO_SAFE_SOURCE_EVIDENCE,
                    gap=int(development["largest_army_gap"]),
                )

        blocked_pips = int(_blocked_production_pips(game, player_id))
        if blocked_pips >= 4:
            signals[StrategicIntent.ROBBER_RELIEF] = _signal(
                IntentState.ACTIVE,
                StrategicReasonCode.ROBBER_BLOCKING_IMPORTANT_TILE,
                blocked_production_pips=blocked_pips,
            )
        elif blocked_pips == 0:
            signals[StrategicIntent.ROBBER_RELIEF] = _signal(
                IntentState.INACTIVE,
                StrategicReasonCode.ROBBER_BLOCKING_IMPORTANT_TILE,
                blocked_production_pips=0,
            )
        else:
            signals[StrategicIntent.ROBBER_RELIEF] = _signal(
                IntentState.UNKNOWN, StrategicReasonCode.NO_SAFE_SOURCE_EVIDENCE,
                blocked_production_pips=blocked_pips,
            )

    contributions: dict[StrategicIntent, bool | None] = {}
    contributed = set(trace.contributed_intents)
    for intent in STRATEGIC_INTENTS:
        state = signals[intent].state
        if intent in contributed:
            contributions[intent] = True
        elif state == IntentState.INACTIVE:
            contributions[intent] = False
        elif state == IntentState.ACTIVE and trace.execution_source not in {
            ExecutionSource.BASE_PPO, ExecutionSource.UNKNOWN,
        }:
            contributions[intent] = False
        else:
            contributions[intent] = None
    if development is None:
        development_eligible = trace.development_eligible
        new_cards_blocked_purchase = trace.new_development_cards_blocked
        new_knight_blocked_use = None
    else:
        development_eligible = bool(development["eligible"])
        new_cards_blocked_purchase = bool(
            development["already_bought_this_turn"]
            and development["affordable"]
            and not development["immediate_construction"]
            and development["reasons"]
            and development["build_delay"] <= development["allowed_build_delay"]
            and development["expected_card_value"] >= 0.9
        )
        new_knight_blocked_use = bool(
            title_gate
            and not game.used_development_this_turn
            and player.new_development_cards.count("knight") > 0
            and player.development_cards.count("knight")
            == player.new_development_cards.count("knight")
            and 0 < int(development["largest_army_gap"]) <= agent.title_horizon
        )
    new_cards_blocked = (
        None if new_cards_blocked_purchase is None and new_knight_blocked_use is None
        else bool(new_cards_blocked_purchase or new_knight_blocked_use)
    )
    return StrategicIntentReport(
        signals=signals,
        selected_execution=diagnostic_execution(action),
        execution_source=trace.execution_source,
        reason_contributed_to_selection=contributions,
        selection_reason_codes=trace.contributed_reasons,
        development_eligible=development_eligible,
        new_development_cards_blocked=new_cards_blocked,
        new_development_cards_blocked_purchase=new_cards_blocked_purchase,
        new_knight_blocked_use=new_knight_blocked_use,
    )


def build_strategic_intent_report(
    agent: "SettlementPlanningAgent", game: GameState, player_id: int,
    action: Action, trace: ActionSelectionTrace,
) -> StrategicIntentReport:
    return _diagnose(agent, deepcopy(game), player_id, action, trace)
