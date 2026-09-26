"""評価専用の記録。Policyへの入力や報酬には使用しない。"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from app.domain.actions import (Action, BankTradeAction, BuildCityAction,
                                BuildRoadAction, BuildSettlementAction,
                                BuyDevelopmentAction, EndTurnAction)
from app.domain.game import (BUILD_COSTS, RESOURCES, GameState,
                             legal_settlement_ids, player_for)

from .action_space import ACTION_CATALOG
from .expansion_planner import analyze_expansion_plan
from .reward import actual_score


def endgame_rank(scores: dict[int, int], player_id: int, winner_id: int | None) -> tuple[float | None, float]:
    """公式勝者を1位、残りを実VP順。同点は平均順位・2位枠を等分する。"""
    if winner_id is None:
        return None, 0.0
    if player_id == winner_id:
        return 1.0, 0.0
    losers = [pid for pid in scores if pid != winner_id]
    score = scores[player_id]
    higher = sum(scores[pid] > score for pid in losers)
    tied = sum(scores[pid] == score for pid in losers)
    return 2.0 + higher + (tied - 1) / 2, (1.0 / tied if higher == 0 else 0.0)


_BUILD_IDS = {
    "settlement": [item.id for item in ACTION_CATALOG if isinstance(item.action, BuildSettlementAction)],
    "city": [item.id for item in ACTION_CATALOG if isinstance(item.action, BuildCityAction)],
}

_NUMBER_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
                8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def _vertex_production(game: GameState, vertex_id: int) -> dict[str, int]:
    """公開盤面から、その交差点の資源別生産pipを計算する。"""
    production = dict.fromkeys(RESOURCES, 0)
    for tile_id in game.board.vertices[vertex_id].hex_ids:
        tile = game.board.tiles[tile_id]
        if tile.terrain in production:
            production[tile.terrain] += _NUMBER_PIPS.get(tile.number, 0)
    return production


def _player_production(game: GameState, player_id: int) -> dict[str, int]:
    """開拓地を1倍、都市を2倍として現在の資源別生産pipを返す。"""
    production = dict.fromkeys(RESOURCES, 0)
    for vertices, multiplier in ((game.settlements, 1), (game.cities, 2)):
        for vertex_id, owner_id in vertices.items():
            if owner_id != player_id:
                continue
            for resource, pips in _vertex_production(game, vertex_id).items():
                production[resource] += multiplier * pips
    return production


def _port_at_vertex(game: GameState, vertex_id: int):
    return next((port for port in game.board.ports
                 if vertex_id in game.board.edges[port.edge_id].vertex_ids), None)


def _can_afford_settlement(game: GameState, player_id: int) -> bool:
    resources = player_for(game, player_id).resources
    return all(resources[resource] >= amount
               for resource, amount in BUILD_COSTS["settlement"].items())


@dataclass
class EvaluationTelemetry:
    player_id: int
    titles: dict[str, set[int]] = field(default_factory=lambda: {"longest_road": set(), "largest_army": set()})
    trades: Counter = field(default_factory=Counter)
    opportunities: Counter = field(default_factory=Counter)
    strategy: Counter = field(default_factory=Counter)
    score_changes: list[dict] = field(default_factory=list)
    observation_min: float = 1.0
    observation_max: float = 0.0
    _last_score: int = 0
    _road_reachable_before: set[int] | None = None
    _road_endpoint_pips: int = 0
    _road_touches_port: bool = False
    _own_turns_seen: set[int] = field(default_factory=set)
    _budget_turns_seen: set[int] = field(default_factory=set)

    def check_observation(self, observation: np.ndarray) -> None:
        if not np.isfinite(observation).all() or np.any(observation < 0) or np.any(observation > 1):
            raise AssertionError("Observationに非有限値または0〜1範囲外の値があります。")
        self.observation_min = min(self.observation_min, float(observation.min()))
        self.observation_max = max(self.observation_max, float(observation.max()))

    def before_action(self, game: GameState, action: Action, mask: np.ndarray) -> None:
        if (game.current_player_id == self.player_id
                and game.phase in {"turn_pre_roll", "action"}):
            self._own_turns_seen.add(game.turn_number)
            self.strategy["own_turns_observed"] = len(self._own_turns_seen)
        if game.phase != "action":
            return
        if isinstance(action, EndTurnAction):
            own_turn = len(self._own_turns_seen)
            if own_turn in (10, 20):
                player = player_for(game, self.player_id)
                prefix = f"construction_turn{own_turn}"
                self.strategy[f"{prefix}_observed"] = 1
                self.strategy[f"{prefix}_sites"] = player.settlements + player.cities
                self.strategy[f"{prefix}_cities"] = player.cities
                self.strategy[f"{prefix}_building_points"] = player.settlements + 2 * player.cities
                self.strategy[f"{prefix}_production_pips"] = sum(
                    _player_production(game, self.player_id).values()
                )
        plan = analyze_expansion_plan(game, self.player_id)
        if plan.selected_target is not None:
            self.opportunities["planned_settlement_target_decisions"] += 1
            self.strategy["planned_target_pips_total"] += plan.selected_target.total_pips
            self.strategy["planned_target_road_distance_total"] += (
                plan.selected_target.additional_roads
            )
        if plan.best_connected_target is not None:
            self.opportunities["planner_connected_target_decisions"] += 1
        if plan.should_wait_for_settlement:
            self.opportunities["planner_wait_for_settlement_decisions"] += 1
            if isinstance(action, BuildRoadAction):
                self.opportunities["build_road_while_planner_waits"] += 1
        if isinstance(action, BuildRoadAction) and plan.recommended_road_ids:
            self.opportunities["planned_road_available_decisions"] += 1
            if action.edge_id in plan.recommended_road_ids:
                self.opportunities["planned_road_selected"] += 1
            else:
                self.opportunities["off_plan_road_selected"] += 1
        if isinstance(action, BuyDevelopmentAction):
            if plan.build_wait_rounds < 6.0:
                self.opportunities["buy_development_while_build_near"] += 1
            else:
                self.opportunities["buy_development_during_planned_stall"] += 1
        if (isinstance(action, BuildSettlementAction)
                and plan.selected_target is not None
                and action.vertex_id == plan.selected_target.vertex_id):
            self.opportunities["planned_settlement_selected"] += 1
        connected_sites = legal_settlement_ids(game, self.player_id, initial=False)
        player = player_for(game, self.player_id)
        if game.turn_number not in self._budget_turns_seen:
            self._budget_turns_seen.add(game.turn_number)
            if connected_sites and player.settlements < 5:
                self.opportunities["connected_site_turns"] += 1
                missing = {resource for resource, amount in BUILD_COSTS["settlement"].items()
                           if player.resources[resource] < amount}
                if missing & {"wood", "brick"}:
                    self.opportunities["connected_site_turns_missing_wood_brick"] += 1
                if missing & {"sheep", "wheat"}:
                    self.opportunities["connected_site_turns_missing_sheep_wheat"] += 1
                if missing and missing <= {"sheep", "wheat"}:
                    self.opportunities["connected_site_turns_missing_only_sheep_wheat"] += 1
        if isinstance(action, BuyDevelopmentAction):
            for kind, feasible in (("settlement", bool(connected_sites) and player.settlements < 5),
                                   ("city", player.settlements > 0 and player.cities < 4)):
                if not feasible:
                    continue
                cost = BUILD_COSTS[kind]
                before = sum(max(0, amount - player.resources[resource])
                             for resource, amount in cost.items())
                after = sum(max(0, amount - player.resources[resource]
                                + BUILD_COSTS["development"].get(resource, 0))
                            for resource, amount in cost.items())
                self.strategy[f"development_{kind}_deficit_increase"] += after - before
                if after > before:
                    self.opportunities[f"development_consumes_{kind}_reserve"] += 1
        if isinstance(action, BuildRoadAction) and not game.free_road_remaining and connected_sites:
            self.strategy["paid_road_settlement_deficit_increase"] += sum(
                max(0, 1 - player.resources[r] + 1) - max(0, 1 - player.resources[r])
                for r in ("wood", "brick")
            )
        if connected_sites:
            self.opportunities["connected_settlement_site_decisions"] += 1
            if not _can_afford_settlement(game, self.player_id):
                self.opportunities["connected_site_but_unaffordable_decisions"] += 1
            if isinstance(action, BuyDevelopmentAction):
                self.opportunities["buy_development_with_connected_site"] += 1
            if isinstance(action, BuildRoadAction):
                self.opportunities["build_road_with_connected_site"] += 1
            if isinstance(action, EndTurnAction):
                self.opportunities["end_turn_with_connected_site"] += 1
        for kind, ids in _BUILD_IDS.items():
            if mask[ids].any():
                self.opportunities[f"{kind}_available_decisions"] += 1
                if kind == "settlement" and isinstance(action, BuildSettlementAction):
                    self.opportunities["settlement_selected_when_available"] += 1
                if kind == "settlement" and isinstance(action, BuildRoadAction):
                    self.opportunities["build_road_with_settlement_available"] += 1
                if isinstance(action, EndTurnAction):
                    self.opportunities[f"end_turn_with_{kind}_available"] += 1
                if isinstance(action, BuyDevelopmentAction):
                    self.opportunities[f"buy_development_with_{kind}_available"] += 1
        player = player_for(game, self.player_id)
        if isinstance(action, BuyDevelopmentAction) and player.settlements + player.cities < 3:
            self.opportunities["buy_development_before_third_site"] += 1
            if not any(mask[ids].any() for ids in _BUILD_IDS.values()):
                self.opportunities["buy_development_while_builds_unavailable"] += 1
        self._road_reachable_before = None
        if isinstance(action, BuildRoadAction):
            self._road_reachable_before = set(connected_sites)
            endpoints = game.board.edges[action.edge_id].vertex_ids
            self._road_endpoint_pips = max(
                sum(_vertex_production(game, vertex_id).values()) for vertex_id in endpoints
            )
            self._road_touches_port = any(_port_at_vertex(game, vertex_id) is not None
                                          for vertex_id in endpoints)

        if isinstance(action, BuildSettlementAction):
            production_before = _player_production(game, self.player_id)
            production = _vertex_production(game, action.vertex_id)
            self.strategy["settlement_builds"] += 1
            self.strategy["settlement_pips_total"] += sum(production.values())
            self.strategy["settlement_resource_kinds_total"] += sum(
                pips > 0 for pips in production.values()
            )
            self.strategy["settlement_new_resource_kinds_total"] += sum(
                pips > 0 and production_before[resource] == 0
                for resource, pips in production.items()
            )
            port = _port_at_vertex(game, action.vertex_id)
            if port is not None:
                self.strategy["settlement_port_builds"] += 1
                if port.resource is not None and production_before.get(port.resource, 0) > 0:
                    self.strategy["settlement_matching_port_builds"] += 1

    def after_action(self, game: GameState, player_id: int, action: Action) -> None:
        # Opponentの操作中に一時的に取得/喪失した称号も落とさない。
        for name, holder in (("longest_road", game.longest_road_holder_id),
                             ("largest_army", game.largest_army_holder_id)):
            if holder is not None:
                self.titles[name].add(holder)
        if player_id == self.player_id and isinstance(action, BankTradeAction):
            ratio = sum(game.activity_log[-1]["bank"]["give"].values())
            self.trades[f"{ratio}:1"] += 1
        if player_id == self.player_id and isinstance(action, BuildRoadAction):
            reachable_before = self._road_reachable_before or set()
            reachable_after = set(legal_settlement_ids(game, self.player_id, initial=False))
            newly_reachable = reachable_after - reachable_before
            self.strategy["road_builds"] += 1
            self.strategy["road_endpoint_pips_total"] += self._road_endpoint_pips
            if self._road_touches_port:
                self.strategy["road_touches_port"] += 1
            if newly_reachable:
                self.strategy["road_opens_settlement_site"] += 1
                self.strategy["road_opened_site_pips_total"] += max(
                    sum(_vertex_production(game, vertex_id).values())
                    for vertex_id in newly_reachable
                )
            else:
                self.strategy["road_without_immediate_settlement_site"] += 1
            self._road_reachable_before = None
        if player_id == self.player_id and isinstance(action, BuildSettlementAction):
            player = player_for(game, self.player_id)
            if player.settlements + player.cities >= 3 and not self.strategy["third_site_reached"]:
                self.strategy["third_site_reached"] = 1
                self.strategy["third_site_own_turn"] = len(self._own_turns_seen)
        if player_id == self.player_id and isinstance(action, (BuildSettlementAction, BuildCityAction)):
            player = player_for(game, self.player_id)
            milestones = {
                "fourth_site": player.settlements + player.cities >= 4,
                "first_city": player.cities >= 1,
                "second_city": player.cities >= 2,
            }
            for name, reached in milestones.items():
                if reached and not self.strategy[f"{name}_reached"]:
                    self.strategy[f"{name}_reached"] = 1
                    self.strategy[f"{name}_own_turn"] = len(self._own_turns_seen)
        score = actual_score(game, self.player_id)
        if score != self._last_score:
            self.score_changes.append({"revision": game.revision, "turn": game.turn_number, "score": score})
            self._last_score = score
