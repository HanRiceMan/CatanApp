"""盤面拡張の目標地点と、そこへ向かう最初の道路を評価する。

学習用の教師・診断で使う決定論的な計画器。ゲームの合法手は変更しない。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from math import inf

from app.domain.game import (BUILD_COSTS, RESOURCES, GameState, _port_ratio,
                             player_for)


NUMBER_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
               8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


@dataclass(frozen=True)
class SettlementTarget:
    vertex_id: int
    additional_roads: int
    first_road_ids: tuple[int, ...]
    production_pips: tuple[int, ...]
    total_pips: int
    new_resource_kinds: int
    port_resource: str | None
    port_ratio: int | None
    site_value: float
    plan_value: float
    required_resources: tuple[int, ...]
    wait_rounds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExpansionPlan:
    targets: tuple[SettlementTarget, ...]
    selected_target: SettlementTarget | None
    best_connected_target: SettlementTarget | None
    recommended_road_ids: tuple[int, ...]
    should_wait_for_settlement: bool
    settlement_wait_rounds: float
    city_wait_rounds: float
    build_wait_rounds: float

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["targets"] = [target.to_dict() for target in self.targets]
        return value


def vertex_production(game: GameState, vertex_id: int) -> dict[str, int]:
    production = dict.fromkeys(RESOURCES, 0)
    for tile_id in game.board.vertices[vertex_id].hex_ids:
        tile = game.board.tiles[tile_id]
        if tile.terrain in production:
            production[tile.terrain] += NUMBER_PIPS.get(tile.number, 0)
    return production


def player_production(game: GameState, player_id: int) -> dict[str, int]:
    production = dict.fromkeys(RESOURCES, 0)
    for buildings, multiplier in ((game.settlements, 1), (game.cities, 2)):
        for vertex_id, owner_id in buildings.items():
            if owner_id != player_id:
                continue
            for resource, pips in vertex_production(game, vertex_id).items():
                production[resource] += multiplier * pips
    return production


def _port_at_vertex(game: GameState, vertex_id: int) -> tuple[str | None, int] | None:
    for port in game.board.ports:
        if vertex_id in game.board.edges[port.edge_id].vertex_ids:
            return port.resource, port.ratio
    return None


def _settlement_candidates(game: GameState) -> tuple[int, ...]:
    occupied = set(game.settlements) | set(game.cities)
    candidates: list[int] = []
    for vertex in game.board.vertices:
        if vertex.id in occupied:
            continue
        neighbours = {
            other
            for edge_id in vertex.edge_ids
            for other in game.board.edges[edge_id].vertex_ids
            if other != vertex.id
        }
        if occupied.isdisjoint(neighbours):
            candidates.append(vertex.id)
    return tuple(candidates)


def _routes_from_network(game: GameState, player_id: int) -> dict[int, tuple[int, set[int]]]:
    """各交差点までの追加道路数と、最短経路の最初の道路候補を返す。"""
    buildings = {**game.settlements, **game.cities}
    sources = {
        vertex_id for vertex_id, owner_id in buildings.items() if owner_id == player_id
    }
    for edge_id, owner_id in game.roads.items():
        if owner_id == player_id:
            sources.update(game.board.edges[edge_id].vertex_ids)
    # (vertex, first_new_edge) を別状態にし、同距離の最初の一手をすべて残す。
    distances: dict[tuple[int, int], int] = {}
    queue: list[tuple[int, int, int]] = []
    for vertex_id in sources:
        if buildings.get(vertex_id) not in {None, player_id}:
            continue
        distances[(vertex_id, -1)] = 0
        heappush(queue, (0, vertex_id, -1))

    while queue:
        distance, vertex_id, first_edge = heappop(queue)
        if distances.get((vertex_id, first_edge)) != distance:
            continue
        # 相手の建物がある交差点では道路網が分断される。
        if buildings.get(vertex_id) not in {None, player_id}:
            continue
        for edge_id in game.board.vertices[vertex_id].edge_ids:
            owner_id = game.roads.get(edge_id)
            if owner_id not in {None, player_id}:
                continue
            edge = game.board.edges[edge_id]
            next_vertex = edge.vertex_ids[1] if edge.vertex_ids[0] == vertex_id else edge.vertex_ids[0]
            extra = int(owner_id is None)
            next_first = first_edge if first_edge >= 0 or not extra else edge_id
            next_distance = distance + extra
            state = (next_vertex, next_first)
            if next_distance < distances.get(state, 10_000):
                distances[state] = next_distance
                heappush(queue, (next_distance, next_vertex, next_first))

    result: dict[int, tuple[int, set[int]]] = {}
    for (vertex_id, first_edge), distance in distances.items():
        current_distance, first_edges = result.get(vertex_id, (10_000, set()))
        if distance < current_distance:
            result[vertex_id] = (distance, {first_edge} if first_edge >= 0 else set())
        elif distance == current_distance and first_edge >= 0:
            first_edges.add(first_edge)
    return result


def _site_value(game: GameState, player_id: int, vertex_id: int,
                current_production: dict[str, int]) -> tuple[float, dict[str, int], int,
                                                              tuple[str | None, int] | None]:
    production = vertex_production(game, vertex_id)
    total_pips = sum(production.values())
    new_resource_kinds = sum(
        pips > 0 and current_production[resource] == 0
        for resource, pips in production.items()
    )
    # 生産が薄い資源ほど、同じpipでも補完価値を高くする。
    balance_value = sum(
        pips / (1.0 + current_production[resource] / 5.0)
        for resource, pips in production.items()
    )
    port = _port_at_vertex(game, vertex_id)
    port_value = 0.0
    if port is not None:
        port_resource, ratio = port
        if port_resource is None:
            port_value = 1.0
        else:
            # その資源をよく生産するほど2:1港を活用しやすい。
            port_value = 0.75 + min(2.5, current_production[port_resource] / 3.0)
        affected_resources = RESOURCES if port_resource is None else (port_resource,)
        if not any(_port_ratio(game, player_id, resource) > ratio
                   for resource in affected_resources):
            port_value *= 0.25
    value = total_pips + 0.55 * balance_value + 1.25 * new_resource_kinds + port_value
    return value, production, new_resource_kinds, port


def _estimated_wait_rounds(game: GameState, player_id: int,
                           required: dict[str, int],
                           production: dict[str, int]) -> float:
    """共有した余剰資源を生産・港交換へ一度だけ使う概算ラウンド数。"""
    player = player_for(game, player_id)
    stock = {resource: float(player.resources[resource]) for resource in RESOURCES}
    # 1ラウンドに4人が1回ずつ振る前提。
    per_round = {resource: production[resource] * 4.0 / 36.0 for resource in RESOURCES}

    def affordable(rounds: float) -> bool:
        available = {resource: stock[resource] + rounds * per_round[resource]
                     for resource in RESOURCES}
        deficits = sum(max(0.0, required.get(resource, 0) - available[resource])
                       for resource in RESOURCES)
        trades = sum(max(0.0, available[resource] - required.get(resource, 0))
                     / _port_ratio(game, player_id, resource)
                     for resource in RESOURCES)
        return trades + 1e-6 >= deficits

    if affordable(0):
        return 0.0
    if not affordable(12):
        return 99.0
    low, high = 0.0, 12.0
    for _ in range(12):
        middle = (low + high) / 2
        if affordable(middle):
            high = middle
        else:
            low = middle
    return high


def settlement_target_cost(target: SettlementTarget, *, free_roads: int = 0) -> dict[str, int]:
    """目標地点までの残り道路と開拓地を合わせた資源予算。"""
    paid_roads = max(0, target.additional_roads - free_roads)
    result = dict.fromkeys(RESOURCES, 0)
    result.update(BUILD_COSTS["settlement"])
    result["wood"] += paid_roads
    result["brick"] += paid_roads
    return result


def estimated_build_wait(game: GameState, player_id: int,
                         required: dict[str, int]) -> float:
    """現在の生産と所有港から、指定した建設予算までの概算ラウンド数を返す。"""
    return _estimated_wait_rounds(game, player_id, required,
                                  player_production(game, player_id))


def analyze_expansion_plan(game: GameState, player_id: int, *,
                           max_additional_roads: int = 3,
                           alternative_margin: float = 2.0) -> ExpansionPlan:
    """最も有望な開拓地目標と、今作る意味のある道路を求める。"""
    player = player_for(game, player_id)
    production = player_production(game, player_id)
    routes = _routes_from_network(game, player_id)
    remaining_roads = 15 - sum(owner_id == player_id for owner_id in game.roads.values())
    targets: list[SettlementTarget] = []
    for vertex_id in _settlement_candidates(game):
        route = routes.get(vertex_id)
        if (route is None or route[0] > max_additional_roads
                or route[0] > remaining_roads):
            continue
        road_count, first_roads = route
        site_value, vertex_pips, new_kinds, port = _site_value(
            game, player_id, vertex_id, production
        )
        paid_roads = max(0, road_count - game.free_road_remaining)
        required = dict.fromkeys(RESOURCES, 0)
        required.update(BUILD_COSTS["settlement"])
        required["wood"] += paid_roads
        required["brick"] += paid_roads
        wait_rounds = _estimated_wait_rounds(game, player_id, required, production)
        # 道路費用と完成までの時間を同じ目標評価に含め、近く完成できる地点を優先する。
        plan_value = site_value - 2.4 * paid_roads - 0.5 * min(wait_rounds, 12)
        targets.append(SettlementTarget(
            vertex_id=vertex_id,
            additional_roads=road_count,
            first_road_ids=tuple(sorted(first_roads)),
            production_pips=tuple(vertex_pips[resource] for resource in RESOURCES),
            total_pips=sum(vertex_pips.values()),
            new_resource_kinds=new_kinds,
            port_resource=port[0] if port else None,
            port_ratio=port[1] if port else None,
            site_value=site_value,
            plan_value=plan_value,
            required_resources=tuple(required[resource] for resource in RESOURCES),
            wait_rounds=wait_rounds,
        ))
    targets.sort(key=lambda target: (-target.plan_value, target.additional_roads,
                                     -target.site_value, target.vertex_id))
    connected = [target for target in targets if target.additional_roads == 0]
    best_connected = max(connected, key=lambda target: target.site_value, default=None)
    selected = targets[0] if targets else None
    # 接続済み候補がある時は資源を待つのを基本とし、将来候補の純価値が十分高い時だけ
    # 道路を伸ばす例外を認める。
    if best_connected is not None and selected is not None and selected.additional_roads:
        if selected.plan_value < best_connected.site_value + alternative_margin:
            selected = best_connected
    recommended_roads = selected.first_road_ids if selected is not None else ()
    should_wait = best_connected is not None and selected is not None and not selected.additional_roads

    settlement_wait = (selected.wait_rounds
                       if selected is not None and player.settlements < 5 else inf)
    city_wait = (_estimated_wait_rounds(game, player_id, BUILD_COSTS["city"], production)
                 if player.settlements > 0 and player.cities < 4 else inf)
    return ExpansionPlan(
        targets=tuple(targets),
        selected_target=selected,
        best_connected_target=best_connected,
        recommended_road_ids=recommended_roads,
        should_wait_for_settlement=should_wait,
        settlement_wait_rounds=settlement_wait,
        city_wait_rounds=city_wait,
        build_wait_rounds=min(settlement_wait, city_wait),
    )
