"""公開情報だけから相手の勝利脅威と、盗賊による妨害価値を評価する。"""

from __future__ import annotations

from app.domain.game import BUILD_COSTS, RESOURCES, GameState, player_for

from .expansion_planner import NUMBER_PIPS, player_production


def public_score(game: GameState, player_id: int) -> int:
    player = player_for(game, player_id)
    return (player.settlements + 2 * player.cities
            + 2 * (game.longest_road_holder_id == player_id)
            + 2 * (game.largest_army_holder_id == player_id))


def threat_value(game: GameState, player_id: int) -> float:
    """非公開VPを断定せず、公開得点・生産・未使用カード枚数から脅威を測る。"""
    player = player_for(game, player_id)
    production = sum(player_production(game, player_id).values())
    # 発展カードは内訳を見ず、勝利点・騎士の可能性として小さく加える。
    hidden_potential = 0.35 * len(player.development_cards)
    return public_score(game, player_id) * 4.0 + production * 0.18 + hidden_potential


def robber_tile_value(game: GameState, player_id: int, tile_id: int) -> float:
    tile = game.board.tiles[tile_id]
    pips = NUMBER_PIPS.get(tile.number, 0)
    if not pips:
        return -20.0
    value = 0.0
    for vertex in game.board.vertices:
        if tile_id not in vertex.hex_ids:
            continue
        owner = game.settlements.get(vertex.id, game.cities.get(vertex.id))
        if owner is None:
            continue
        multiplier = 2 if vertex.id in game.cities else 1
        if owner == player_id:
            value -= pips * multiplier * 2.2
        else:
            score_weight = 1.0 + max(0, public_score(game, owner) - 5) * 0.35
            value += pips * multiplier * score_weight
            if sum(player_for(game, owner).resources.values()) > 0:
                value += threat_value(game, owner) * 0.025
    return value


def best_robber_tile(game: GameState, player_id: int, legal_tile_ids) -> int:
    candidates = list(legal_tile_ids)
    if not candidates:
        raise ValueError("盗賊を移動できるタイルがありません。")
    return max(candidates, key=lambda tile_id: (robber_tile_value(game, player_id, tile_id),
                                                -tile_id))


def best_robber_victim(game: GameState, player_id: int, victim_ids) -> int:
    candidates = list(victim_ids)
    if not candidates:
        raise ValueError("盗める相手がいません。")
    return max(candidates, key=lambda victim_id: (
        threat_value(game, victim_id),
        sum(player_for(game, victim_id).resources.values()),
        -victim_id,
    ))


def desired_resource_weights(game: GameState, player_id: int) -> dict[str, float]:
    """自分の公開方針と正規の自分手札から、盗めると嬉しい資源を重み付けする。"""
    player = player_for(game, player_id)
    costs = []
    if player.settlements < 5:
        costs.append(BUILD_COSTS["settlement"])
    if player.settlements > 0 and player.cities < 4:
        costs.append(BUILD_COSTS["city"])
    if not costs:
        costs.append(BUILD_COSTS["development"])
    return {
        resource: 1.0 + max(
            max(0, cost.get(resource, 0) - player.resources[resource])
            for cost in costs
        )
        for resource in RESOURCES
    }


def belief_steal_value(belief, weights: dict[str, float]) -> float:
    return sum(belief.random_steal_probability(resource) * weights[resource]
               for resource in RESOURCES)


def belief_robber_tile_value(game: GameState, player_id: int, tile_id: int,
                             beliefs: dict) -> float:
    """盤面妨害に、首位優先と推定手札からの略奪価値を加える。"""
    value = robber_tile_value(game, player_id, tile_id)
    tile = game.board.tiles[tile_id]
    pips = NUMBER_PIPS.get(tile.number, 0)
    opponents = [other.id for other in game.players if other.id != player_id]
    leader_score = max(public_score(game, owner) for owner in opponents)
    adjacent: set[int] = set()
    for vertex in game.board.vertices:
        if tile_id not in vertex.hex_ids:
            continue
        owner = game.settlements.get(vertex.id, game.cities.get(vertex.id))
        if owner is None or owner == player_id:
            continue
        adjacent.add(owner)
        multiplier = 2 if vertex.id in game.cities else 1
        gap = leader_score - public_score(game, owner)
        leader_factor = 1.0 if gap == 0 else (0.45 if gap == 1 else 0.0)
        value += pips * multiplier * 0.65 * leader_factor
    weights = desired_resource_weights(game, player_id)
    steal_values = [belief_steal_value(beliefs[owner], weights)
                    for owner in adjacent
                    if beliefs[owner].expected_total > 0]
    if steal_values:
        value += 1.5 * max(steal_values)
    return value


def best_belief_robber_tile(game: GameState, player_id: int, legal_tile_ids,
                            beliefs: dict) -> int:
    candidates = list(legal_tile_ids)
    if not candidates:
        raise ValueError("盗賊を移動できるタイルがありません。")
    return max(candidates, key=lambda tile_id: (
        belief_robber_tile_value(game, player_id, tile_id, beliefs), -tile_id,
    ))


def best_belief_robber_victim(game: GameState, player_id: int, victim_ids,
                              beliefs: dict) -> int:
    candidates = list(victim_ids)
    if not candidates:
        raise ValueError("盗める相手がいません。")
    weights = desired_resource_weights(game, player_id)
    return max(candidates, key=lambda victim_id: (
        threat_value(game, victim_id)
        + belief_steal_value(beliefs[victim_id], weights) * 3.0,
        -victim_id,
    ))
