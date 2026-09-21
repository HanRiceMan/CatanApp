"""19タイルの盤面を生成する。論理座標を使い、描画はNext.jsに任せる。"""

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from random import Random
from typing import Literal

Terrain = Literal["wood", "brick", "sheep", "wheat", "ore", "desert"]
TERRAIN_COUNTS = {"wood": 4, "brick": 3, "sheep": 4, "wheat": 4, "ore": 3, "desert": 1}
NUMBER_TOKENS = (2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12)
CORNER_OFFSETS = ((2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1))
# 12時から反時計回り。番号はタイルIDではなく、港の両端の交差点ID。
FIXED_PORTS = (
    ((21, 27), None), ((11, 7), None), ((4, 1), "sheep"),
    ((2, 6), None), ((15, 20), "ore"), ((37, 42), "wheat"),
    ((50, 53), None), ((48, 52), "wood"), ((43, 38), "brick"),
)


@dataclass(frozen=True)
class BoardRules:
    desert_center: bool = False
    red_numbers_not_adjacent: bool = True
    red_numbers_distinct_terrains: bool = False
    terrain_clusters_max_two: bool = False


@dataclass(frozen=True, order=True)
class HexCoord:
    q: int
    r: int


@dataclass(frozen=True)
class Vertex:
    id: int
    x: int
    y: int
    hex_ids: tuple[int, ...]
    edge_ids: tuple[int, ...]


@dataclass(frozen=True)
class Edge:
    id: int
    vertex_ids: tuple[int, int]
    hex_ids: tuple[int, ...]


@dataclass(frozen=True)
class Topology:
    hexes: tuple[HexCoord, ...]
    vertices: tuple[Vertex, ...]
    edges: tuple[Edge, ...]


@dataclass(frozen=True)
class Tile:
    id: int
    q: int
    r: int
    terrain: Terrain
    number: int | None


@dataclass(frozen=True)
class Port:
    id: int
    edge_id: int
    resource: Terrain | None  # None = 一般港。desert は使わない。
    ratio: int


@dataclass(frozen=True)
class Board:
    tiles: tuple[Tile, ...]
    vertices: tuple[Vertex, ...]
    edges: tuple[Edge, ...]
    ports: tuple[Port, ...]
    robber_tile_id: int


def are_neighbors(a: HexCoord, b: HexCoord) -> bool:
    dq, dr = a.q - b.q, a.r - b.r
    return max(abs(dq), abs(dr), abs(dq + dr)) == 1


@lru_cache(maxsize=1)
def create_topology() -> Topology:
    hexes = tuple(HexCoord(q, r) for q in range(-2, 3) for r in range(-2, 3)
                  if max(abs(q), abs(r), abs(q + r)) <= 2)
    vertex_hexes = defaultdict(list)
    edge_hexes = defaultdict(list)
    for tile_id, coord in enumerate(hexes):
        corners = tuple((3 * coord.q + dx, coord.q + 2 * coord.r + dy)
                        for dx, dy in CORNER_OFFSETS)
        for vertex in corners:
            vertex_hexes[vertex].append(tile_id)
        for index in range(6):
            # 順番を揃えることで、隣のタイルが持つ同じ辺を共有できる。
            key = tuple(sorted((corners[index], corners[(index + 1) % 6])))
            edge_hexes[key].append(tile_id)

    vertex_ids = {coord: index for index, coord in enumerate(sorted(vertex_hexes))}
    edges = tuple(Edge(index, (vertex_ids[a], vertex_ids[b]), tuple(edge_hexes[(a, b)]))
                  for index, (a, b) in enumerate(sorted(edge_hexes)))
    incident_edges = defaultdict(list)
    for edge in edges:
        for vertex_id in edge.vertex_ids:
            incident_edges[vertex_id].append(edge.id)
    vertices = tuple(Vertex(index, x, y, tuple(vertex_hexes[(x, y)]), tuple(incident_edges[index]))
                     for (x, y), index in vertex_ids.items())
    return Topology(hexes, vertices, edges)


@lru_cache(maxsize=19)
def red_token_positions(desert_id: int) -> tuple[tuple[int, ...], ...]:
    """6・8を置ける、互いに隣接しない4地点の全候補。無限リトライを避ける。"""
    hexes = create_topology().hexes
    land_ids = [index for index in range(19) if index != desert_id]
    return tuple(group for group in combinations(land_ids, 4)
                 if all(not are_neighbors(hexes[a], hexes[b]) for a, b in combinations(group, 2)))


def coastal_edges(topology: Topology) -> tuple[int, ...]:
    """海岸の30辺を連続する順に並べる。港を等間隔に置くために使う。"""
    neighbors = defaultdict(list)
    for edge in topology.edges:
        if len(edge.hex_ids) == 1:
            a, b = edge.vertex_ids
            neighbors[a].append((b, edge.id))
            neighbors[b].append((a, edge.id))
    start = min(neighbors)
    current, previous = start, None
    ordered = []
    while True:
        following, edge_id = next((v, e) for v, e in sorted(neighbors[current]) if v != previous)
        ordered.append(edge_id)
        previous, current = current, following
        if current == start:
            return tuple(ordered)


@lru_cache(maxsize=1)
def tile_neighbors() -> tuple[tuple[int, ...], ...]:
    hexes = create_topology().hexes
    return tuple(tuple(j for j, b in enumerate(hexes) if are_neighbors(a, b)) for a in hexes)


def terrain_clusters_are_small(terrains: list[str]) -> bool:
    """同種地形で辺を共有する集まりを最大2枚に制限。2枚組が複数でもよい。"""
    visited = set()
    for start in range(19):
        if start in visited:
            continue
        pending, component = [start], set()
        while pending:
            tile_id = pending.pop()
            if tile_id in component:
                continue
            component.add(tile_id)
            if len(component) > 2:
                return False
            pending.extend(j for j in tile_neighbors()[tile_id]
                           if terrains[j] == terrains[start] and j not in component)
        visited.update(component)
    return True


@lru_cache(maxsize=38)
def red_position_candidates(desert_id: int, non_adjacent: bool) -> tuple[tuple[int, ...], ...]:
    if non_adjacent:
        return red_token_positions(desert_id)
    return tuple(combinations((i for i in range(19) if i != desert_id), 4))


def create_board(seed: int, rules: BoardRules | None = None) -> Board:
    rng = Random(seed)  # 同じseedから同じ盤面を再現できる。
    rules = rules or BoardRules()
    topology = create_topology()
    terrains = [kind for kind, count in TERRAIN_COUNTS.items() for _ in range(count)]
    for _ in range(5000):
        rng.shuffle(terrains)
        desert_id = terrains.index("desert")
        if rules.desert_center:
            center_id = next(i for i, coord in enumerate(topology.hexes) if (coord.q, coord.r) == (0, 0))
            terrains[desert_id], terrains[center_id] = terrains[center_id], terrains[desert_id]
            desert_id = center_id
        if rules.terrain_clusters_max_two and not terrain_clusters_are_small(terrains):
            continue
        candidates = red_position_candidates(desert_id, rules.red_numbers_not_adjacent)
        if rules.red_numbers_distinct_terrains:
            candidates = tuple(group for group in candidates if len({terrains[i] for i in group}) == 4)
        if candidates:
            red_ids = rng.choice(candidates)
            break
    else:
        raise ValueError("指定条件の盤面を生成できませんでした。再度生成してください。")
    red_numbers = [6, 6, 8, 8]
    other_numbers = [number for number in NUMBER_TOKENS if number not in (6, 8)]
    rng.shuffle(red_numbers)
    rng.shuffle(other_numbers)
    numbers = dict(zip(red_ids, red_numbers))
    for index in range(19):
        if index != desert_id and index not in numbers:
            numbers[index] = other_numbers.pop()
    tiles = tuple(Tile(index, coord.q, coord.r, terrains[index], numbers.get(index))
                  for index, coord in enumerate(topology.hexes))

    edge_by_vertices = {tuple(sorted(edge.vertex_ids)): edge for edge in topology.edges}
    ports = tuple(Port(index, edge_by_vertices[tuple(sorted(vertices))].id,
                       resource, 3 if resource is None else 2)
                  for index, (vertices, resource) in enumerate(FIXED_PORTS))
    return Board(tiles, topology.vertices, topology.edges, ports, desert_id)
