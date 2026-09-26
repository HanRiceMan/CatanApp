"""初期道路の先に残る拡張候補・競合・港を評価する。"""

from __future__ import annotations

from collections import deque

from app.domain.game import (BUILD_COSTS, GameState, _building_owner,
                             legal_road_ids, player_for)

from .expansion_planner import vertex_production


def _can_settle_later(game: GameState, vertex_id: int) -> bool:
    if _building_owner(game, vertex_id) is not None:
        return False
    for edge_id in game.board.vertices[vertex_id].edge_ids:
        a, b = game.board.edges[edge_id].vertex_ids
        neighbor = b if a == vertex_id else a
        if _building_owner(game, neighbor) is not None:
            return False
    return True


def _port_at_vertex(game: GameState, vertex_id: int):
    for port in game.board.ports:
        if vertex_id in game.board.edges[port.edge_id].vertex_ids:
            return port
    return None


def _target_value(game: GameState, vertex_id: int, additional_roads: int) -> float:
    production = vertex_production(game, vertex_id)
    pips = sum(production.values())
    kinds = sum(value > 0 for value in production.values())
    port = _port_at_vertex(game, vertex_id)
    port_value = 0.0
    if port is not None:
        # 3:1港は資源構成を問わず有効。2:1港は対応資源を産出できるほど高評価。
        port_value = (2.6 if port.resource is None
                      else 1.4 + min(1.8, production[port.resource] * 0.3))
    return pips + 0.55 * kinds + port_value - 1.8 * (additional_roads - 1)


def assess_initial_road(game: GameState, player_id: int, edge_id: int, *,
                        max_additional_roads: int = 3) -> dict:
    settlement_id = game.pending_settlement_id
    if settlement_id is None or edge_id not in legal_road_ids(game, player_id):
        raise ValueError("合法な初期道路だけを評価できます。")
    edge = game.board.edges[edge_id]
    far_id = edge.vertex_ids[1] if edge.vertex_ids[0] == settlement_id else edge.vertex_ids[0]
    incident = game.board.vertices[far_id].edge_ids
    opponent_edges = [
        candidate for candidate in incident
        if game.roads.get(candidate) not in {None, player_id}
    ]
    opponent_owners = sorted({game.roads[candidate] for candidate in opponent_edges})
    open_exits = [candidate for candidate in incident
                  if candidate != edge_id and candidate not in game.roads]

    # 最初の道路の先から、さらに1〜3本道路を置いた範囲だけを探索する。
    queue = deque([(far_id, 0, frozenset((settlement_id, far_id)))])
    best_distance: dict[int, int] = {far_id: 0}
    targets: list[dict] = []
    while queue:
        vertex_id, distance, visited = queue.popleft()
        if distance >= max_additional_roads:
            continue
        if vertex_id != far_id and _building_owner(game, vertex_id) not in {None, player_id}:
            continue
        for next_edge_id in game.board.vertices[vertex_id].edge_ids:
            if next_edge_id == edge_id or next_edge_id in game.roads:
                continue
            a, b = game.board.edges[next_edge_id].vertex_ids
            next_vertex = b if a == vertex_id else a
            if next_vertex in visited:
                continue
            next_distance = distance + 1
            if _can_settle_later(game, next_vertex):
                targets.append({
                    "vertex_id": next_vertex,
                    "additional_roads": next_distance,
                    "value": _target_value(game, next_vertex, next_distance),
                    "pips": sum(vertex_production(game, next_vertex).values()),
                    "port": ((_port_at_vertex(game, next_vertex).resource or "generic")
                             if _port_at_vertex(game, next_vertex) is not None else None),
                })
            if next_distance < best_distance.get(next_vertex, 99):
                best_distance[next_vertex] = next_distance
                queue.append((next_vertex, next_distance, visited | {next_vertex}))

    player = player_for(game, player_id)
    own_road_ready = all(player.resources[resource] >= count
                         for resource, count in BUILD_COSTS["road"].items())
    opponent_road_ready = any(
        all(player_for(game, owner).resources[resource] >= count
            for resource, count in BUILD_COSTS["road"].items())
        for owner in opponent_owners
    )
    best_target_value = max((target["value"] for target in targets), default=-20.0)
    competition_penalty = 4.0 * len(opponent_edges)
    if opponent_edges and not own_road_ready:
        competition_penalty += 3.0
    if opponent_road_ready:
        competition_penalty += 3.0
    if opponent_edges and not open_exits:
        competition_penalty += 30.0
    score = best_target_value - competition_penalty
    return {
        "edge_id": edge_id,
        "settlement_id": settlement_id,
        "far_vertex_id": far_id,
        "open_exit_count": len(open_exits),
        "opponent_road_count": len(opponent_edges),
        "opponent_road_owner_ids": opponent_owners,
        "own_road_ready": own_road_ready,
        "opponent_road_ready": opponent_road_ready,
        "target_count": len(targets),
        "best_target_value": best_target_value,
        "best_targets": sorted(targets, key=lambda target: (
            -target["value"], target["additional_roads"], target["vertex_id"]
        ))[:5],
        "score": score,
    }


def choose_initial_road(game: GameState, player_id: int, preferred_edge_id: int,
                        *, retain_preference_margin: float = 0.75) -> tuple[int, list[dict]]:
    assessments = [assess_initial_road(game, player_id, edge_id)
                   for edge_id in legal_road_ids(game, player_id)]
    if not assessments:
        raise ValueError("初期道路の合法候補がありません。")
    selected = max(assessments, key=lambda item: (
        item["score"] + (retain_preference_margin
                         if item["edge_id"] == preferred_edge_id else 0.0),
        item["best_target_value"], item["open_exit_count"], -item["edge_id"],
    ))
    return selected["edge_id"], assessments
