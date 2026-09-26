"""中盤の高レベル目標を比較するための、公開情報中心の局面特徴量。"""

from __future__ import annotations

from typing import Any

from app.domain.game import (BUILD_COSTS, RESOURCES, GameState,
                             _longest_road_length, _port_ratio,
                             legal_city_ids, legal_settlement_ids, player_for)

from .expansion_planner import NUMBER_PIPS, analyze_expansion_plan, player_production
from .victory_race import public_score

GOAL_CONTEXT_VERSION = "v1"
GOAL_STRATEGIES = ("balanced", "fifth_site", "first_city",
                   "early_titles", "army_purchase")


def _resource_deficit(stock: dict[str, int], cost: dict[str, int]) -> int:
    return sum(max(0, amount - stock[resource]) for resource, amount in cost.items())


def goal_context(game: GameState, player_id: int) -> dict[str, Any]:
    """相手の非公開内訳を使わず、目標選択用の特徴量を返す。"""
    player = player_for(game, player_id)
    opponents = [other for other in game.players if other.id != player_id]
    production = player_production(game, player_id)
    plan = analyze_expansion_plan(game, player_id)
    own_road = _longest_road_length(game, player_id)
    rival_road = max(_longest_road_length(game, other.id) for other in opponents)
    rival_knights = max(other.played_knights for other in opponents)
    robber_pressure = sum(
        NUMBER_PIPS.get(game.board.tiles[tile_id].number, 0) * multiplier
        for buildings, multiplier in ((game.settlements, 1), (game.cities, 2))
        for vertex_id, owner in buildings.items()
        if owner == player_id
        for tile_id in game.board.vertices[vertex_id].hex_ids
        if tile_id == game.robber_tile_id
    )
    return {
        "version": GOAL_CONTEXT_VERSION,
        "score": public_score(game, player_id),
        "settlements": player.settlements,
        "cities": player.cities,
        "resources": {resource: player.resources[resource] for resource in RESOURCES},
        "resource_count": sum(player.resources.values()),
        "development_count": len(player.development_cards),
        "usable_knights": (player.development_cards.count("knight")
                           - player.new_development_cards.count("knight")),
        "production": production,
        "production_total": sum(production.values()),
        "port_ratios": {resource: _port_ratio(game, player_id, resource)
                        for resource in RESOURCES},
        "city_deficit": _resource_deficit(player.resources, BUILD_COSTS["city"]),
        "settlement_deficit": _resource_deficit(
            player.resources, BUILD_COSTS["settlement"]
        ),
        "legal_city_count": len(legal_city_ids(game, player_id)),
        "legal_settlement_count": len(legal_settlement_ids(game, player_id, initial=False)),
        "next_settlement_wait": (plan.selected_target.wait_rounds
                                 if plan.selected_target is not None else None),
        "next_settlement_roads": (plan.selected_target.additional_roads
                                  if plan.selected_target is not None else None),
        "longest_road_length": own_road,
        "longest_road_gap": max(0, max(5, rival_road + 1) - own_road),
        "played_knights": player.played_knights,
        "largest_army_gap": max(0, max(3, rival_knights + 1) - player.played_knights),
        "holds_longest_road": game.longest_road_holder_id == player_id,
        "holds_largest_army": game.largest_army_holder_id == player_id,
        "robber_pressure_pips": robber_pressure,
        "rival_max_score": max(public_score(game, other.id) for other in opponents),
        "rival_max_production": max(sum(player_production(game, other.id).values())
                                    for other in opponents),
        "rival_max_resource_count": max(sum(other.resources.values()) for other in opponents),
        "rival_max_development_count": max(len(other.development_cards)
                                           for other in opponents),
    }


class GoalContextRecordingAgent:
    """4拠点到達後の最初の行動局面を記録し、判断は元Agentへ委譲する。"""

    def __init__(self, delegate, snapshots: dict[int, dict[str, Any]]):
        self.delegate = delegate
        self.snapshots = snapshots

    def select_action(self, game: GameState, player_id: int):
        player = player_for(game, player_id)
        if (game.seed not in self.snapshots and game.phase == "action"
                and player.settlements + player.cities >= 4):
            self.snapshots[game.seed] = goal_context(game, player_id)
        return self.delegate.select_action(game, player_id)
