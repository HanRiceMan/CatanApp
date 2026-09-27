"""現行Plannerの判断を、学習に介入せず構造化して観測する診断API。

``objective_goal`` は最終的に達成したい戦略目標、``execution_mode`` は
そのために現在実行する手段を表す。現行Plannerは全Goalを同一尺度で採点して
いないため、この版の ``goal_scores`` は異種の診断値であり比較不可とする。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING, Any

from app.domain.actions import (Action, BankTradeAction, BuildCityAction,
                                BuildRoadAction, BuildSettlementAction,
                                BuyDevelopmentAction, DiscardHalfAction,
                                DiscardResourcesAction, EndTurnAction,
                                MoveRobberAction, PlaceInitialRoadAction,
                                PlaceInitialSettlementAction,
                                ProposeCounterTradeAction, ProposeTradeAction,
                                RespondToTradeAction, RollOrderAction,
                                RollTurnDiceAction, StartGameAction,
                                StealResourceAction, UseDevelopmentAction)
from app.domain.game import (BUILD_COSTS, RESOURCES, GameState,
                             _longest_road_length, _port_ratio, player_for)

from .action_space import action_to_id
from .development_strategy import assess_development_purchase
from .expansion_planner import analyze_expansion_plan, estimated_build_wait
from .reward import actual_score

if TYPE_CHECKING:
    from .settlement_planning_agent import SettlementPlanningAgent
    from .trade_strategy import TradeDecisionDiagnostic


PLANNER_DIAGNOSTIC_VERSION = "macro_goal_diagnostics_v2"


class ObjectiveGoal(str, Enum):
    """Plannerが最終的に実現しようとしている戦略目標。"""

    SETTLEMENT = "SETTLEMENT"
    CITY = "CITY"
    DEVELOPMENT = "DEVELOPMENT"
    ROAD_TITLE = "ROAD_TITLE"
    KNIGHT_TITLE = "KNIGHT_TITLE"
    HOLD = "HOLD"


class ExecutionMode(str, Enum):
    """選ばれたActionが、現在何を実行するものか。"""

    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"
    BUILD_CITY = "BUILD_CITY"
    BUILD_ROAD = "BUILD_ROAD"
    BUY_DEVELOPMENT = "BUY_DEVELOPMENT"
    TRADE_PLAYER = "TRADE_PLAYER"
    TRADE_BANK = "TRADE_BANK"
    END_TURN = "END_TURN"
    PLAY_DEVELOPMENT = "PLAY_DEVELOPMENT"
    INITIAL_SETTLEMENT = "INITIAL_SETTLEMENT"
    INITIAL_ROAD = "INITIAL_ROAD"
    ROLL_DICE = "ROLL_DICE"
    START_GAME = "START_GAME"
    DISCARD = "DISCARD"
    MOVE_ROBBER = "MOVE_ROBBER"
    STEAL_RESOURCE = "STEAL_RESOURCE"
    RESPOND_TRADE = "RESPOND_TRADE"
    COUNTER_TRADE = "COUNTER_TRADE"
    OTHER = "OTHER"


class PlannerReasonCode(str, Enum):
    """分析で集計できる、自由文ではない判断理由。"""

    NON_MACRO_PHASE = "NON_MACRO_PHASE"
    SETTLEMENT_IMMEDIATE = "SETTLEMENT_IMMEDIATE"
    SETTLEMENT_TARGET_AVAILABLE = "SETTLEMENT_TARGET_AVAILABLE"
    SETTLEMENT_ROAD_STEP = "SETTLEMENT_ROAD_STEP"
    SETTLEMENT_RESOURCE_WAIT = "SETTLEMENT_RESOURCE_WAIT"
    CITY_IMMEDIATE = "CITY_IMMEDIATE"
    CITY_HIGH_PRODUCTION_GAIN = "CITY_HIGH_PRODUCTION_GAIN"
    CITY_RESOURCE_WAIT = "CITY_RESOURCE_WAIT"
    DEVELOPMENT_PURCHASE = "DEVELOPMENT_PURCHASE"
    DEVELOPMENT_STALLED = "DEVELOPMENT_STALLED"
    DEVELOPMENT_ROBBER_RELEASE = "DEVELOPMENT_ROBBER_RELEASE"
    DEVELOPMENT_CARD_USE = "DEVELOPMENT_CARD_USE"
    ROAD_TITLE_OPPORTUNITY = "ROAD_TITLE_OPPORTUNITY"
    ROAD_TITLE_IMMEDIATE_WIN = "ROAD_TITLE_IMMEDIATE_WIN"
    ROAD_TITLE_DEFENSE = "ROAD_TITLE_DEFENSE"
    KNIGHT_TITLE_OPPORTUNITY = "KNIGHT_TITLE_OPPORTUNITY"
    KNIGHT_TITLE_DEFENSE = "KNIGHT_TITLE_DEFENSE"
    BANK_TRADE_REACHABLE = "BANK_TRADE_REACHABLE"
    PLAYER_TRADE_REACHABLE = "PLAYER_TRADE_REACHABLE"
    TRADE_FOR_SETTLEMENT = "TRADE_FOR_SETTLEMENT"
    TRADE_FOR_CITY = "TRADE_FOR_CITY"
    TRADE_HAND_OVERFLOW = "TRADE_HAND_OVERFLOW"
    TRADE_RESOURCE_REBALANCE = "TRADE_RESOURCE_REBALANCE"
    TRADE_FUTURE_VALUE = "TRADE_FUTURE_VALUE"
    TRADE_OBJECTIVE_AMBIGUOUS = "TRADE_OBJECTIVE_AMBIGUOUS"
    END_TURN_RESOURCE_WAIT = "END_TURN_RESOURCE_WAIT"
    NO_REACHABLE_GOAL = "NO_REACHABLE_GOAL"
    GOAL_ACTION_AMBIGUOUS = "GOAL_ACTION_AMBIGUOUS"
    IMMEDIATE_WIN = "IMMEDIATE_WIN"


@dataclass(frozen=True)
class PlannerDecisionReport:
    """一つのPlanner判断に対応する、保存可能なTeacher診断レコード。"""

    state_id: str
    player_id: int
    turn: int
    phase: str
    is_macro_decision: bool
    objective_goal: ObjectiveGoal | None
    execution_mode: ExecutionMode
    goal_scores: dict[str, float | None]
    goal_score_semantics: dict[str, str]
    scores_comparable: bool
    best_goal: ObjectiveGoal | None
    second_goal: ObjectiveGoal | None
    margin: float | None
    target_vertex: int | None
    target_edge: int | None
    target_player: int | None
    reason_codes: tuple[PlannerReasonCode, ...]
    estimated_turns_to_goal: dict[str, float | None]
    selected_action: dict[str, Any]
    selected_action_id: int | None
    available_goal_mask: dict[str, bool]
    trade_objective: ObjectiveGoal | None = None
    trade_reason_codes: tuple[PlannerReasonCode, ...] = ()
    trade_target_player: int | None = None
    wanted_resource: str | None = None
    offered_resource: str | None = None
    offered_resources: tuple[str, ...] = ()
    offer_ratio: int | None = None
    source_level_objective_available: bool = False
    target_goal: ObjectiveGoal | None = None
    immediate_goal_reachable: bool | None = None
    goal_after_trade: ObjectiveGoal | None = None
    shortage_before_trade: int | None = None
    shortage_after_trade: int | None = None
    shortage_basis: str | None = None
    planner_version: str = PLANNER_DIAGNOSTIC_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Enumを文字列へ変換し、JSON Linesへそのまま保存できる形にする。"""
        result = asdict(self)
        result["objective_goal"] = (
            self.objective_goal.value if self.objective_goal is not None else None
        )
        result["execution_mode"] = self.execution_mode.value
        result["best_goal"] = self.best_goal.value if self.best_goal is not None else None
        result["second_goal"] = (
            self.second_goal.value if self.second_goal is not None else None
        )
        result["reason_codes"] = [reason.value for reason in self.reason_codes]
        result["trade_objective"] = (
            self.trade_objective.value if self.trade_objective is not None else None
        )
        result["trade_reason_codes"] = [
            reason.value for reason in self.trade_reason_codes
        ]
        result["target_goal"] = (
            self.target_goal.value if self.target_goal is not None else None
        )
        result["goal_after_trade"] = (
            self.goal_after_trade.value if self.goal_after_trade is not None else None
        )
        return result


def _execution_mode(action: Action) -> ExecutionMode:
    mapping = (
        (BuildSettlementAction, ExecutionMode.BUILD_SETTLEMENT),
        (BuildCityAction, ExecutionMode.BUILD_CITY),
        (BuildRoadAction, ExecutionMode.BUILD_ROAD),
        (BuyDevelopmentAction, ExecutionMode.BUY_DEVELOPMENT),
        (ProposeTradeAction, ExecutionMode.TRADE_PLAYER),
        (BankTradeAction, ExecutionMode.TRADE_BANK),
        (EndTurnAction, ExecutionMode.END_TURN),
        (UseDevelopmentAction, ExecutionMode.PLAY_DEVELOPMENT),
        (PlaceInitialSettlementAction, ExecutionMode.INITIAL_SETTLEMENT),
        (PlaceInitialRoadAction, ExecutionMode.INITIAL_ROAD),
        ((RollOrderAction, RollTurnDiceAction), ExecutionMode.ROLL_DICE),
        (StartGameAction, ExecutionMode.START_GAME),
        ((DiscardHalfAction, DiscardResourcesAction), ExecutionMode.DISCARD),
        (MoveRobberAction, ExecutionMode.MOVE_ROBBER),
        (StealResourceAction, ExecutionMode.STEAL_RESOURCE),
        (RespondToTradeAction, ExecutionMode.RESPOND_TRADE),
        (ProposeCounterTradeAction, ExecutionMode.COUNTER_TRADE),
    )
    for action_type, mode in mapping:
        if isinstance(action, action_type):
            return mode
    return ExecutionMode.OTHER


def _serialize_action(action: Action) -> dict[str, Any]:
    fields = asdict(action) if is_dataclass(action) else {}
    return {"type": type(action).__name__, **fields}


def _safe_action_id(action: Action) -> int | None:
    try:
        return action_to_id(action)
    except ValueError:
        return None


def _road_title_gap(game: GameState, player_id: int) -> tuple[int, int, int]:
    own = _longest_road_length(game, player_id)
    rival = max(
        (_longest_road_length(game, other.id)
         for other in game.players if other.id != player_id),
        default=0,
    )
    target = max(5, rival + 1)
    return own, rival, max(0, target - own)


def _army_title_gap(game: GameState, player_id: int) -> tuple[int, int, int]:
    player = player_for(game, player_id)
    own = player.played_knights + player.development_cards.count("knight")
    rival = max((other.played_knights for other in game.players
                 if other.id != player_id), default=0)
    target = max(3, rival + 1)
    return own, rival, max(0, target - own)


def _required_goal_for_trade(agent: "SettlementPlanningAgent", game: GameState,
                             player_id: int) -> tuple[ObjectiveGoal, dict[str, int]] | None:
    from .trade_strategy import construction_trade_goal

    goal = construction_trade_goal(
        game, player_id,
        target_sites=agent.target_sites,
        max_plan_roads=agent.max_plan_roads,
        max_wait_rounds=agent.max_wait_rounds,
        max_city_wait_rounds=agent.max_city_wait_rounds,
    )
    if goal is None:
        return None
    name, required = goal
    objective = ObjectiveGoal.SETTLEMENT if name == "settlement" else ObjectiveGoal.CITY
    return objective, required


def _reasons_for_development(assessment: dict[str, Any]) -> list[PlannerReasonCode]:
    reasons = [PlannerReasonCode.DEVELOPMENT_PURCHASE]
    if "construction_stall" in assessment["reasons"]:
        reasons.append(PlannerReasonCode.DEVELOPMENT_STALLED)
    if "robber_relief" in assessment["reasons"]:
        reasons.append(PlannerReasonCode.DEVELOPMENT_ROBBER_RELEASE)
    return reasons


def _trade_objective(value: str | None) -> ObjectiveGoal | None:
    return {
        "settlement": ObjectiveGoal.SETTLEMENT,
        "city": ObjectiveGoal.CITY,
        "development": ObjectiveGoal.DEVELOPMENT,
        "knight_title": ObjectiveGoal.KNIGHT_TITLE,
    }.get(value)


def _source_trade_reasons(
    diagnostic: "TradeDecisionDiagnostic",
) -> list[PlannerReasonCode]:
    objective = _trade_objective(diagnostic.source_objective)
    reasons = []
    if objective == ObjectiveGoal.SETTLEMENT:
        reasons.append(PlannerReasonCode.TRADE_FOR_SETTLEMENT)
    elif objective == ObjectiveGoal.CITY:
        reasons.append(PlannerReasonCode.TRADE_FOR_CITY)
    purpose_map = {
        "HAND_OVERFLOW": PlannerReasonCode.TRADE_HAND_OVERFLOW,
        "RESOURCE_REBALANCE": PlannerReasonCode.TRADE_RESOURCE_REBALANCE,
        "FUTURE_VALUE": PlannerReasonCode.TRADE_FUTURE_VALUE,
        "OBJECTIVE_AMBIGUOUS": PlannerReasonCode.TRADE_OBJECTIVE_AMBIGUOUS,
    }
    reasons.extend(
        purpose_map[reason.value]
        for reason in diagnostic.reasons if reason.value in purpose_map
    )
    return list(dict.fromkeys(reasons))


def _trade_report_fields(
    game: GameState,
    player_id: int,
    action: Action,
    diagnostic: "TradeDecisionDiagnostic | None",
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "trade_objective": None,
        "trade_reason_codes": (),
        "trade_target_player": None,
        "wanted_resource": None,
        "offered_resource": None,
        "offered_resources": (),
        "offer_ratio": None,
        "source_level_objective_available": False,
        "target_goal": None,
        "immediate_goal_reachable": None,
        "goal_after_trade": None,
        "shortage_before_trade": None,
        "shortage_after_trade": None,
        "shortage_basis": None,
    }
    if isinstance(action, BankTradeAction):
        fields.update(
            wanted_resource=action.receive_resource,
            offered_resource=action.give_resource,
            offered_resources=(action.give_resource,),
            offer_ratio=_port_ratio(game, player_id, action.give_resource),
        )
    elif isinstance(action, ProposeTradeAction):
        offered = tuple(dict.fromkeys(
            resource for offer in action.offers for resource in offer.give
        ))
        wanted = tuple(dict.fromkeys(
            resource for offer in action.offers for resource in offer.want
        ))
        fields.update(
            trade_target_player=action.target_id,
            wanted_resource=wanted[0] if len(wanted) == 1 else None,
            offered_resource=offered[0] if offered else None,
            offered_resources=offered,
        )
    if diagnostic is None:
        return fields
    objective = _trade_objective(diagnostic.source_objective)
    fields.update(
        trade_objective=objective,
        trade_reason_codes=tuple(_source_trade_reasons(diagnostic)),
        trade_target_player=diagnostic.target_player,
        wanted_resource=diagnostic.wanted_resource,
        offered_resource=(
            diagnostic.offered_resources[0]
            if diagnostic.offered_resources else None
        ),
        offered_resources=diagnostic.offered_resources,
        offer_ratio=diagnostic.offer_ratio,
        source_level_objective_available=objective is not None,
        target_goal=_trade_objective(diagnostic.target_goal),
        immediate_goal_reachable=diagnostic.immediate_goal_reachable,
        goal_after_trade=_trade_objective(diagnostic.goal_after_trade),
        shortage_before_trade=diagnostic.shortage_before_trade,
        shortage_after_trade=diagnostic.shortage_after_trade,
        shortage_basis=diagnostic.shortage_basis,
    )
    return fields


def _goal_for_action(
    agent: "SettlementPlanningAgent",
    game: GameState,
    player_id: int,
    action: Action,
    *,
    plan,
    development: dict[str, Any],
    trade_diagnostic: "TradeDecisionDiagnostic | None",
) -> tuple[ObjectiveGoal | None, list[PlannerReasonCode]]:
    player = player_for(game, player_id)
    target = plan.selected_target
    score = actual_score(game, player_id)

    if isinstance(action, BuildSettlementAction):
        reasons = [PlannerReasonCode.SETTLEMENT_IMMEDIATE]
        if score >= 9:
            reasons.append(PlannerReasonCode.IMMEDIATE_WIN)
        return ObjectiveGoal.SETTLEMENT, reasons
    if isinstance(action, BuildCityAction):
        reasons = [PlannerReasonCode.CITY_IMMEDIATE,
                   PlannerReasonCode.CITY_HIGH_PRODUCTION_GAIN]
        if score >= 9:
            reasons.append(PlannerReasonCode.IMMEDIATE_WIN)
        return ObjectiveGoal.CITY, reasons
    if isinstance(action, BuildRoadAction):
        if agent._road_wins_now(game, player_id, action):
            return ObjectiveGoal.ROAD_TITLE, [
                PlannerReasonCode.ROAD_TITLE_IMMEDIATE_WIN,
                PlannerReasonCode.IMMEDIATE_WIN,
            ]
        expansion_step = bool(
            target is not None and action.edge_id in target.first_road_ids
        )
        increases_longest = agent._road_increases_longest(game, player_id, action)
        site_count = player.settlements + player.cities
        if expansion_step and site_count < agent.target_sites:
            return ObjectiveGoal.SETTLEMENT, [PlannerReasonCode.SETTLEMENT_ROAD_STEP]
        if increases_longest:
            reason = (PlannerReasonCode.ROAD_TITLE_DEFENSE
                      if game.longest_road_holder_id == player_id
                      else PlannerReasonCode.ROAD_TITLE_OPPORTUNITY)
            return ObjectiveGoal.ROAD_TITLE, [reason]
        if expansion_step:
            return ObjectiveGoal.SETTLEMENT, [PlannerReasonCode.SETTLEMENT_ROAD_STEP]
        return None, [PlannerReasonCode.GOAL_ACTION_AMBIGUOUS]
    if isinstance(action, BuyDevelopmentAction):
        if "largest_army_race" in development["reasons"]:
            return ObjectiveGoal.KNIGHT_TITLE, [
                PlannerReasonCode.KNIGHT_TITLE_OPPORTUNITY,
                *_reasons_for_development(development),
            ]
        return ObjectiveGoal.DEVELOPMENT, _reasons_for_development(development)
    if isinstance(action, (BankTradeAction, ProposeTradeAction)):
        construction = _required_goal_for_trade(agent, game, player_id)
        reason = (PlannerReasonCode.BANK_TRADE_REACHABLE
                  if isinstance(action, BankTradeAction)
                  else PlannerReasonCode.PLAYER_TRADE_REACHABLE)
        if isinstance(action, ProposeTradeAction) and trade_diagnostic is not None:
            source_objective = _trade_objective(
                trade_diagnostic.source_objective
            )
            source_reasons = _source_trade_reasons(trade_diagnostic)
            if source_objective is not None:
                return source_objective, [reason, *source_reasons]
            return None, [
                reason, *source_reasons,
                PlannerReasonCode.GOAL_ACTION_AMBIGUOUS,
            ]
        if construction is not None:
            objective, required = construction
            wanted = (action.receive_resource if isinstance(action, BankTradeAction)
                      else next((resource for offer in action.offers
                                 for resource, count in offer.want.items() if count), None))
            if wanted is not None and player.resources[wanted] < required.get(wanted, 0):
                return objective, [reason]
        return None, [reason, PlannerReasonCode.GOAL_ACTION_AMBIGUOUS]
    if isinstance(action, UseDevelopmentAction):
        if action.card == "knight":
            _, _, gap = _army_title_gap(game, player_id)
            if 0 < gap <= agent.title_horizon:
                reason = (PlannerReasonCode.KNIGHT_TITLE_DEFENSE
                          if game.largest_army_holder_id == player_id
                          else PlannerReasonCode.KNIGHT_TITLE_OPPORTUNITY)
                return ObjectiveGoal.KNIGHT_TITLE, [reason]
        if (action.card == "road_building" and target is not None
                and target.additional_roads > 0):
            return ObjectiveGoal.SETTLEMENT, [PlannerReasonCode.SETTLEMENT_ROAD_STEP]
        return ObjectiveGoal.DEVELOPMENT, [PlannerReasonCode.DEVELOPMENT_CARD_USE]
    if isinstance(action, EndTurnAction):
        construction = _required_goal_for_trade(agent, game, player_id)
        if construction is not None:
            objective, _ = construction
            reason = (PlannerReasonCode.SETTLEMENT_RESOURCE_WAIT
                      if objective == ObjectiveGoal.SETTLEMENT
                      else PlannerReasonCode.CITY_RESOURCE_WAIT)
            return objective, [reason, PlannerReasonCode.END_TURN_RESOURCE_WAIT]
        return ObjectiveGoal.HOLD, [PlannerReasonCode.NO_REACHABLE_GOAL]
    return None, [PlannerReasonCode.GOAL_ACTION_AMBIGUOUS]


def _report_from_copy(
    agent: "SettlementPlanningAgent", game: GameState, player_id: int, action: Action,
    trade_diagnostic: "TradeDecisionDiagnostic | None" = None,
) -> PlannerDecisionReport:
    mode = _execution_mode(action)
    state_id = f"{game.id}:{game.revision}:{player_id}"
    if game.phase != "action":
        return PlannerDecisionReport(
            state_id=state_id, player_id=player_id, turn=game.turn_number,
            phase=game.phase, is_macro_decision=False, objective_goal=None,
            execution_mode=mode,
            goal_scores={goal.value: None for goal in ObjectiveGoal},
            goal_score_semantics={goal.value: "not_applicable" for goal in ObjectiveGoal},
            scores_comparable=False, best_goal=None, second_goal=None, margin=None,
            target_vertex=getattr(action, "vertex_id", None),
            target_edge=getattr(action, "edge_id", None),
            target_player=(getattr(action, "target_id", None)
                           or getattr(action, "victim_id", None)),
            reason_codes=(PlannerReasonCode.NON_MACRO_PHASE,),
            estimated_turns_to_goal={goal.value: None for goal in ObjectiveGoal},
            selected_action=_serialize_action(action),
            selected_action_id=_safe_action_id(action),
            available_goal_mask={goal.value: False for goal in ObjectiveGoal},
        )

    player = player_for(game, player_id)
    plan = analyze_expansion_plan(
        game, player_id, max_additional_roads=agent.max_plan_roads,
    )
    target = plan.selected_target
    development = assess_development_purchase(
        game, player_id,
        max_plan_roads=agent.max_plan_roads,
        stall_wait_rounds=agent.development_stall_wait_rounds,
        max_build_delay=agent.development_max_build_delay,
        tactical_score=agent.development_tactical_score,
        title_horizon=agent.title_horizon,
    )
    own_road, _, road_gap = _road_title_gap(game, player_id)
    own_army, _, army_gap = _army_title_gap(game, player_id)
    city_vertices = [vertex for vertex, owner in game.settlements.items()
                     if owner == player_id]
    city_value = max(
        (agent._city_vertex_value(game, player_id, vertex)
         for vertex in city_vertices),
        default=None,
    )
    settlement_available = bool(target is not None and player.settlements < 5)
    city_available = bool(city_vertices and player.cities < 4)
    road_available = bool(
        sum(owner == player_id for owner in game.roads.values()) < 15
        and (road_gap <= agent.title_horizon
             or game.longest_road_holder_id == player_id)
    )
    army_available = bool(
        army_gap <= agent.title_horizon
        or game.largest_army_holder_id == player_id
    )
    available = {
        ObjectiveGoal.SETTLEMENT.value: settlement_available,
        ObjectiveGoal.CITY.value: city_available,
        ObjectiveGoal.DEVELOPMENT.value: bool(game.development_deck),
        ObjectiveGoal.ROAD_TITLE.value: road_available,
        ObjectiveGoal.KNIGHT_TITLE.value: army_available,
        ObjectiveGoal.HOLD.value: True,
    }
    road_required = dict.fromkeys(RESOURCES, 0)
    road_required["wood"] = road_gap
    road_required["brick"] = road_gap
    turns = {
        ObjectiveGoal.SETTLEMENT.value: (
            float(target.wait_rounds) if settlement_available else None
        ),
        ObjectiveGoal.CITY.value: (
            float(estimated_build_wait(game, player_id, BUILD_COSTS["city"]))
            if city_available else None
        ),
        ObjectiveGoal.DEVELOPMENT.value: (
            float(estimated_build_wait(game, player_id, BUILD_COSTS["development"]))
            if game.development_deck else None
        ),
        ObjectiveGoal.ROAD_TITLE.value: (
            float(estimated_build_wait(game, player_id, road_required))
            if road_available and road_gap else (0.0 if road_available else None)
        ),
        # 騎士は山札から種類を選べないため、疑似ターン数を作らない。
        ObjectiveGoal.KNIGHT_TITLE.value: 0.0 if army_gap == 0 else None,
        ObjectiveGoal.HOLD.value: 0.0,
    }
    raw_scores = {
        ObjectiveGoal.SETTLEMENT.value: (
            float(target.plan_value) if settlement_available else None
        ),
        ObjectiveGoal.CITY.value: float(city_value) if city_value is not None else None,
        ObjectiveGoal.DEVELOPMENT.value: (
            float(development["expected_card_value"])
            if game.development_deck else None
        ),
        # 称号は到達率。上の値とは尺度が異なるため比較には使わない。
        ObjectiveGoal.ROAD_TITLE.value: (
            min(1.0, own_road / max(1, own_road + road_gap)) if road_available else None
        ),
        ObjectiveGoal.KNIGHT_TITLE.value: (
            min(1.0, own_army / max(1, own_army + army_gap)) if army_available else None
        ),
        ObjectiveGoal.HOLD.value: 0.0,
    }
    score_semantics = {
        ObjectiveGoal.SETTLEMENT.value: "expansion_plan.plan_value",
        ObjectiveGoal.CITY.value: "best_city_vertex_production_value",
        ObjectiveGoal.DEVELOPMENT.value: "expected_development_card_value",
        ObjectiveGoal.ROAD_TITLE.value: "current_progress_ratio_to_title_target",
        ObjectiveGoal.KNIGHT_TITLE.value: "known_knight_progress_ratio_to_title_target",
        ObjectiveGoal.HOLD.value: "constant_diagnostic_baseline",
    }
    objective, reasons = _goal_for_action(
        agent, game, player_id, action, plan=plan, development=development,
        trade_diagnostic=trade_diagnostic,
    )
    trade_fields = _trade_report_fields(
        game, player_id, action, trade_diagnostic,
    )
    target_vertex = getattr(action, "vertex_id", None)
    if target_vertex is None and objective == ObjectiveGoal.SETTLEMENT and target is not None:
        target_vertex = target.vertex_id
    target_edge = getattr(action, "edge_id", None)
    if (target_edge is None and objective == ObjectiveGoal.SETTLEMENT
            and target is not None and target.first_road_ids):
        target_edge = target.first_road_ids[0]
    target_player = (getattr(action, "target_id", None)
                     or getattr(action, "victim_id", None))
    return PlannerDecisionReport(
        state_id=state_id, player_id=player_id, turn=game.turn_number,
        phase=game.phase, is_macro_decision=True, objective_goal=objective,
        execution_mode=mode, goal_scores=raw_scores,
        goal_score_semantics=score_semantics,
        scores_comparable=False, best_goal=objective, second_goal=None, margin=None,
        target_vertex=target_vertex, target_edge=target_edge,
        target_player=target_player, reason_codes=tuple(dict.fromkeys(reasons)),
        estimated_turns_to_goal={
            goal: (value if value is None or isfinite(value) else None)
            for goal, value in turns.items()
        },
        selected_action=_serialize_action(action),
        selected_action_id=_safe_action_id(action),
        available_goal_mask=available,
        **trade_fields,
    )


def build_planner_decision_report(
    agent: "SettlementPlanningAgent", game: GameState, player_id: int, action: Action,
    *, trade_diagnostic: "TradeDecisionDiagnostic | None" = None,
) -> PlannerDecisionReport:
    """選択済みActionから副作用なしで診断レポートを作る。

    診断器内の一時的な探索や将来の変更がChampionのGameStateへ影響しないよう、
    常にGameStateのdeep copy上で解析する。
    """
    return _report_from_copy(
        agent, deepcopy(game), player_id, action,
        trade_diagnostic=trade_diagnostic,
    )
