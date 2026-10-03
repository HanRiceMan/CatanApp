"""Action適用前の公開盤面・本人手札から作る、非介入の構造化診断。

値の不存在は ``None`` とvalid maskで表す。Planner固有の価値評価は
決定的な事実と同じ入力候補として扱わない。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any

from app.domain.game import (BUILD_COSTS, RESOURCES, GameState,
                             _longest_road_length, _port_ratio,
                             legal_road_ids, player_for)

from .action_space import ACTION_CATALOG, get_action_mask
from .development_strategy import _blocked_production_pips
from .expansion_planner import (analyze_expansion_plan, estimated_build_wait,
                                player_production,
                                vertex_production)
from .reward import actual_score


CONTEXT_SCHEMA_VERSION = "structured_context_v1"
OBSERVED = "OBSERVED_IN_V2"
DERIVED = "DETERMINISTIC_FROM_V2"
MISSING = "MISSING_FROM_V2"
HISTORY = "REQUIRES_HISTORY"
BELIEF = "REQUIRES_BELIEF"
PLANNER = "PLANNER_DIAGNOSTIC_ONLY"


@dataclass(frozen=True)
class ContextSignal:
    value: Any
    valid: bool
    unit: str
    valid_range: str
    semantics: str
    provenance: str

    def __post_init__(self) -> None:
        if not self.valid and self.value is not None:
            raise ValueError("invalid signal must use value=None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value, "valid": self.valid, "unit": self.unit,
            "valid_range": self.valid_range, "semantics": self.semantics,
            "provenance": self.provenance,
        }


def _s(value: Any, unit: str, valid_range: str, semantics: str,
       provenance: str = DERIVED, *, valid: bool = True) -> ContextSignal:
    return ContextSignal(value if valid else None, valid, unit, valid_range,
                         semantics, provenance)


def _shortage(stock: dict[str, int], cost: dict[str, int]) -> dict[str, int]:
    return {resource: max(0, cost.get(resource, 0) - stock[resource])
            for resource in RESOURCES}


def _affords(stock: dict[str, int], cost: dict[str, int]) -> bool:
    return all(stock[resource] >= count for resource, count in cost.items())


def _legal_actions(game: GameState, player_id: int) -> list[Any]:
    mask = get_action_mask(game, player_id)
    return [item.action for item in ACTION_CATALOG if mask[item.id]]


@dataclass(frozen=True)
class StructuredContextReport:
    phase: str
    sections: dict[str, dict[str, ContextSignal]]
    tactical_facts: dict[str, bool | None]
    observability_gaps: dict[str, bool]
    schema_version: str = CONTEXT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_schema_version": self.schema_version,
            "strategic_context": {
                name: {key: signal.value for key, signal in signals.items()}
                for name, signals in self.sections.items() if name != "resource_risk"
            },
            "strategic_context_valid_mask": {
                name: {key: signal.valid for key, signal in signals.items()}
                for name, signals in self.sections.items() if name != "resource_risk"
            },
            "resource_risk_context": {
                key: signal.value
                for key, signal in self.sections["resource_risk"].items()
            },
            "resource_risk_valid_mask": {
                key: signal.valid
                for key, signal in self.sections["resource_risk"].items()
            },
            "tactical_facts": self.tactical_facts,
            "observability_gaps": self.observability_gaps,
        }

    def definitions(self) -> dict[str, dict[str, dict[str, str]]]:
        return {
            section: {
                key: {"unit": signal.unit, "valid_range": signal.valid_range,
                      "semantics": signal.semantics,
                      "provenance": signal.provenance}
                for key, signal in fields.items()
            }
            for section, fields in self.sections.items()
        }


def _build(game: GameState, player_id: int, max_plan_roads: int) -> StructuredContextReport:
    player = player_for(game, player_id)
    stock = player.resources
    actions = _legal_actions(game, player_id)
    action_types = {type(action).__name__ for action in actions}
    action_count: dict[str, int] = {}
    for action in actions:
        family = type(action).__name__
        action_count[family] = action_count.get(family, 0) + 1
    from app.domain.actions import (BuildCityAction, BuildRoadAction,
                                    BuildSettlementAction, BuyDevelopmentAction,
                                    UseDevelopmentAction)

    plan = analyze_expansion_plan(game, player_id,
                                  max_additional_roads=max_plan_roads)
    target = plan.selected_target if player.settlements < 5 else None
    settlement_cost = (dict(zip(RESOURCES, target.required_resources, strict=True))
                       if target is not None else BUILD_COSTS["settlement"])
    settlement_short = _shortage(stock, settlement_cost)
    settlement_eta = target.wait_rounds if target is not None else None
    settlement_eta_valid = (settlement_eta is not None and isfinite(settlement_eta)
                            and settlement_eta <= 12)
    settlement = {
        "candidate_exists": _s(bool(target), "bool", "false/true", "使用可能な開拓地候補がある"),
        "candidate_count": _s(len(plan.targets) if player.settlements < 5 else 0,
                              "vertices", "0..54", "指定道路距離内の開拓地候補数"),
        "min_additional_roads": _s(
            min(item.additional_roads for item in plan.targets) if plan.targets else None,
            "roads", "0..15", "候補のうち構造上最短の追加道路数。価値順位は含まない",
            valid=bool(plan.targets) and player.settlements < 5),
        "best_target_vertex": _s(target.vertex_id if target else None, "vertex_id",
                                 "0..53", "Plannerが選んだ候補の交差点ID",
                                 PLANNER, valid=target is not None),
        "required_additional_roads": _s(target.additional_roads if target else None,
                                         "roads", "0..15", "候補到達に必要な追加道路",
                                         PLANNER, valid=target is not None),
        "resource_shortage": _s(settlement_short if target else None, "cards_by_resource",
                                "0..19 each", "候補到達の道路と開拓地を合わせた不足",
                                PLANNER, valid=target is not None),
        "shortage_total": _s(sum(settlement_short.values()) if target else None,
                             "cards", "0..95", "候補完成までの資源不足合計",
                             PLANNER, valid=target is not None),
        "immediate_build_possible": _s("BuildSettlementAction" in action_types,
                                       "bool", "false/true", "今このphaseで建築合法"),
        "estimated_wait_rounds": _s(settlement_eta, "rounds", "0..12",
                                    "生産と港の決定的概算。99 sentinelは欠測",
                                    PLANNER, valid=settlement_eta_valid),
        "site_value": _s(target.site_value if target else None, "planner_value",
                         "unbounded", "Planner固有の立地評価。共通priorityではない",
                         PLANNER, valid=target is not None),
        "plan_value": _s(target.plan_value if target else None, "planner_value",
                         "unbounded", "Planner固有の道路・時間込み評価。共通priorityではない",
                         PLANNER, valid=target is not None),
    }

    future_cities = sorted(
        (vertex for vertex, owner in game.settlements.items() if owner == player_id),
        key=lambda vertex: (-sum(vertex_production(game, vertex).values()), vertex),
    ) if player.cities < 4 else []
    best_future_city = future_cities[0] if future_cities else None
    immediate_city_ids = [a.vertex_id for a in actions if isinstance(a, BuildCityAction)]
    best_immediate_city = min(
        immediate_city_ids,
        key=lambda vertex: (-sum(vertex_production(game, vertex).values()), vertex),
        default=None,
    )
    city_short = _shortage(stock, BUILD_COSTS["city"])
    city_eta = (estimated_build_wait(game, player_id, BUILD_COSTS["city"])
                if best_future_city is not None else None)
    city = {
        "upgradeable_count": _s(len(future_cities), "vertices", "0..5",
                                "将来都市化できる自分の開拓地数。購入可否と別"),
        "best_future_vertex": _s(best_future_city, "vertex_id", "0..53",
                                 "pip生産増分最大の将来候補。同点はvertex ID順",
                                 valid=best_future_city is not None),
        "best_future_production_gain": _s(
            sum(vertex_production(game, best_future_city).values())
            if best_future_city is not None else None,
            "pips", "0..15", "都市化による生産pip増分",
            valid=best_future_city is not None),
        "immediate_city_candidate_count": _s(len(immediate_city_ids),
                                              "vertices", "0..5", "現在合法で支払可能な都市候補数"),
        "best_immediate_city_vertex": _s(best_immediate_city, "vertex_id", "0..53",
                                         "今合法な都市候補のうちpip増分最大。PPO tie-breakは不使用",
                                         valid=best_immediate_city is not None),
        "resource_shortage": _s(city_short, "cards_by_resource", "0..19 each",
                                "都市費用に対する現在手札の不足"),
        "shortage_total": _s(sum(city_short.values()), "cards", "0..5", "都市資源不足合計"),
        "immediate_build_possible": _s("BuildCityAction" in action_types,
                                       "bool", "false/true", "今このphaseで都市化合法"),
        "estimated_wait_rounds": _s(city_eta, "rounds", "0..12",
                                    "都市資源を揃える決定的概算。99 sentinelは欠測",
                                    valid=city_eta is not None and isfinite(city_eta)
                                    and city_eta <= 12),
        "city_pieces_remaining": _s(4 - player.cities, "pieces", "0..4",
                                    "未使用の都市コマ数"),
    }

    own_length = _longest_road_length(game, player_id)
    rival_length = max((_longest_road_length(game, other.id)
                        for other in game.players if other.id != player_id), default=0)
    road_holder = game.longest_road_holder_id
    road_count = sum(owner == player_id for owner in game.roads.values())
    # 構造的延伸はphaseに依存しない状態量。合法Actionの個数とは分ける。
    original_phase, original_actor = game.phase, game.current_player_id
    game.phase, game.current_player_id = "action", player_id
    try:
        structural = (legal_road_ids(game, player_id, initial=False)
                      if road_count < 15 else [])
    finally:
        game.phase, game.current_player_id = original_phase, original_actor
    title_directed: list[int] = []
    immediate_title: list[int] = []
    # 1つの診断用deepcopy上で仮設・撤去する。実GameStateには一切触らない。
    for edge_id in structural:
        game.roads[edge_id] = player_id
        length = _longest_road_length(game, player_id)
        del game.roads[edge_id]
        if length > own_length:
            title_directed.append(edge_id)
            if (road_holder != player_id and length >= 5
                    and length > rival_length):
                immediate_title.append(edge_id)
    affordable_roads = {a.edge_id for a in actions if isinstance(a, BuildRoadAction)}
    immediate_acquisition = bool(affordable_roads.intersection(immediate_title))
    road_short = _shortage(stock, BUILD_COSTS["road"])
    road = {
        "current_length": _s(own_length, "edges", "0..15", "自分の最長連結道路長"),
        "holder_id": _s(road_holder, "player_id", "1..4", "現在の最長交易路保持者",
                        valid=road_holder is not None),
        "acquisition_gap": _s(max(0, max(5, rival_length + 1) - own_length),
                              "edges", "0..15", "非保持時、称号獲得に必要な連結長差",
                              valid=road_holder != player_id),
        "defense_gap": _s(max(0, own_length - max(5, rival_length)),
                          "edges", "0..15", "保持時の他者最長長との差",
                          valid=road_holder == player_id),
        "structurally_extendable_edge_count": _s(len(structural), "edges", "0..72",
                                                  "資源を無視した構造上の合法道路数"),
        "structurally_title_directed_edge_count": _s(len(title_directed), "edges", "0..72",
                                                     "仮設で自分の最長長が増える道路数"),
        "affordable_title_directed_edge_count": _s(
            len(affordable_roads.intersection(title_directed)), "edges", "0..72",
            "今合法かつ支払可能な最長長増加道路数", MISSING),
        "immediate_acquisition": _s(immediate_acquisition, "bool", "false/true",
                                    "道路1本で直ちに称号獲得可能"),
        "immediate_win": _s(immediate_acquisition and actual_score(game, player_id) + 2 >= 10,
                            "bool", "false/true", "道路1本と称号点で即勝利可能"),
        "resource_shortage": _s(road_short, "cards_by_resource", "0..1 each",
                                "有料道路1本の不足。無料道路とは区別"),
    }

    used = player.played_knights
    held = player.development_cards.count("knight")
    new = player.new_development_cards.count("knight")
    usable = max(0, held - new)
    army_holder = game.largest_army_holder_id
    rival_knights = max((other.played_knights for other in game.players
                         if other.id != player_id), default=0)
    plays_needed = max(0, max(3, rival_knights + 1) - used) if army_holder != player_id else 0
    draws_needed = max(0, plays_needed - held)
    playable_knight = any(isinstance(a, UseDevelopmentAction) and a.card == "knight"
                          for a in actions)
    development_legal = any(isinstance(a, BuyDevelopmentAction) for a in actions)
    knight = {
        "used_knights": _s(used, "cards", "0..14", "既に使用した騎士数"),
        "held_knights": _s(held, "cards", "0..14", "手札の騎士数。newも含む", OBSERVED),
        "usable_knights": _s(usable, "cards", "0..14", "今使用可能な騎士枚数の上限", MISSING),
        "holder_id": _s(army_holder, "player_id", "1..4", "現在の最大騎士力保持者",
                        valid=army_holder is not None),
        "plays_needed": _s(plays_needed, "plays", "0..14", "保持者を上回るのに必要な騎士使用回数"),
        "draws_needed": _s(draws_needed, "cards", "0..14", "既存手札を全使用しても不足する騎士枚数"),
        "development_deck_remaining": _s(len(game.development_deck), "cards", "0..25",
                                         "発展カード山札の残枚数", OBSERVED),
        "development_affordable": _s(_affords(stock, BUILD_COSTS["development"])
                                     and bool(game.development_deck), "bool", "false/true",
                                     "資源と山札から発展購入可能。phase合法性と別"),
        "development_legal_now": _s(development_legal, "bool", "false/true",
                                    "今このphaseで発展購入合法", MISSING),
        "knight_playable_now": _s(playable_knight, "bool", "false/true",
                                  "今このphaseで騎士使用合法", MISSING),
        "immediate_acquisition": _s(playable_knight and plays_needed == 1,
                                    "bool", "false/true", "騎士1枚を今使えば称号獲得可能"),
        "immediate_win": _s(playable_knight and plays_needed == 1
                            and actual_score(game, player_id) + 2 >= 10,
                            "bool", "false/true", "騎士1枚の称号点で即勝利可能"),
    }

    production = player_production(game, player_id)
    total_production = sum(production.values())
    blocked = _blocked_production_pips(game, player_id)
    tile = game.board.tiles[game.robber_tile_id]
    blocked_vector = dict.fromkeys(RESOURCES, 0)
    if tile.terrain in blocked_vector:
        blocked_vector[tile.terrain] = blocked
    robber = {
        "blocked_production_pips": _s(blocked, "pips", "0..30", "盗賊で失う自分の生産pip"),
        "blocked_production_ratio": _s(blocked / total_production if total_production else None,
                                       "ratio", "0..1", "総生産pip中の盗賊妨害割合",
                                       valid=total_production > 0),
        "blocked_resource_vector": _s(blocked_vector, "pips_by_resource", "0..30 each",
                                      "盗賊で妨害される資源別pip"),
        "blocked_tile_id": _s(game.robber_tile_id, "tile_id", "0..18", "盗賊所在hex ID",
                              OBSERVED),
        "playable_knight_available": _s(playable_knight, "bool", "false/true",
                                        "今合法な騎士使用Actionがある", MISSING),
        "development_purchase_affordable": _s(_affords(stock, BUILD_COSTS["development"])
                                            and bool(game.development_deck), "bool", "false/true",
                                            "発展購入の資源と山札がある"),
        "immediate_removal_possible": _s(playable_knight or game.phase == "robber_move",
                                         "bool", "false/true", "今騎士使用か盗賊移動phaseで解除可能",
                                         MISSING),
    }

    shortage = {kind: _shortage(stock, cost) for kind, cost in BUILD_COSTS.items()}
    port_ratios = {resource: _port_ratio(game, player_id, resource)
                   for resource in RESOURCES}
    bank_options = sum(stock[give] >= port_ratios[give] and game.bank[receive] > 0
                       for give in RESOURCES for receive in RESOURCES if give != receive)
    hand_size = sum(stock.values())
    immediate_build = any(isinstance(a, (BuildRoadAction, BuildSettlementAction,
                                        BuildCityAction)) for a in actions)
    resource = {
        "hand_size": _s(hand_size, "cards", "0..95", "自分の資源手札枚数", OBSERVED),
        "discard_exposure": _s(hand_size > 7, "bool", "false/true", "7なら半数破棄の対象"),
        "discard_count_if_seven": _s(hand_size // 2 if hand_size > 7 else 0,
                                     "cards", "0..47", "今7が出た場合の破棄枚数"),
        "resource_diversity": _s(sum(stock[r] > 0 for r in RESOURCES), "types", "0..5",
                                 "保有資源の種類数"),
        "resource_vector": _s(dict(stock), "cards_by_resource", "0..19 each",
                              "本人の資源手札内訳", OBSERVED),
        "build_affordability": _s({kind: _affords(stock, cost)
                                    for kind, cost in BUILD_COSTS.items()},
                                  "bool_by_build", "false/true each", "資源上の建設可否。コマやphaseは無視"),
        "build_shortage": _s(shortage, "cards_by_build_and_resource", "0..19 each",
                             "建設種別ごとの手札資源不足"),
        "available_bank_trade_options": _s(bank_options, "pairs", "0..20",
                                           "手札と銀行在庫で可能な交換資源ペア数"),
        "port_trade_ratios": _s(port_ratios, "cards_by_resource", "2..4 each",
                                "各資源を銀行で1枚に変える交換比率"),
        "legal_action_count": _s(len(actions), "actions", "0..377",
                                 "377固定Action catalog内の現在合法なAction数。対人交渉は対象外",
                                 MISSING),
        "legal_action_count_by_family": _s(action_count, "actions_by_family", "0..377 each",
                                           "固定Action catalog内のAction family別合法数", MISSING),
        "immediate_construction_available": _s(immediate_build, "bool", "false/true",
                                               "今合法な道・開拓地・都市建設がある"),
        "current_stall_duration": _s(None, "turns", "0..unbounded",
                                     "履歴なしでは停滞継続時間を確定できない", HISTORY,
                                     valid=False),
        "opponent_trade_flexibility": _s(None, "probability", "0..1",
                                        "相手手札推定なしでは受諾可能性を確定できない", BELIEF,
                                        valid=False),
    }
    facts = {
        "NO_IMMEDIATE_CONSTRUCTION": not immediate_build,
        "INSUFFICIENT_RESOURCES": not any(_affords(stock, cost)
                                           for cost in BUILD_COSTS.values()),
        "HAND_OVERFLOW": hand_size > 7,
        "DISCARD_EXPOSURE": hand_size > 7,
        "RESOURCE_REBALANCE": None,
        "FUTURE_VALUE": None,
    }
    gaps = {
        "new_development_cards_effect": new > 0 and usable < held,
        # 実際にActionを決めた原因とは別。後段でsource確認できない限り候補件数。
        "belief_potential_dependency": game.phase in {
            "action", "robber_move", "robber_steal", "trade_response",
            "trade_counter_offer"} and bool(game.resource_events),
        "trade_history_potential_dependency": game.phase == "action" and game.trade_count > 0,
    }
    return StructuredContextReport(
        phase=game.phase,
        sections={"settlement": settlement, "city": city,
                  "road_title": road, "knight_title": knight,
                  "robber": robber, "resource_risk": resource},
        tactical_facts=facts, observability_gaps=gaps,
    )


def build_structured_context_report(
    game: GameState, player_id: int, *, max_plan_roads: int = 3,
) -> StructuredContextReport:
    """Action前のGameStateのみを読み、全仮設計算をcopy上で行う。"""
    return _build(deepcopy(game), player_id, max_plan_roads)
