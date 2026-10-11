"""Read-only, non-prescriptive Victory Plan candidates (Step 3B-1).

Feasibility describes physical possibility, not action legality, affordability,
probability, or strategic value.  No candidate is selected by this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from app.domain.game import GameState, _longest_road_length, player_for
from app.rl.reward import actual_score


class PlanFeasibility(str, Enum):
    FEASIBLE = "FEASIBLE"
    UNKNOWN = "UNKNOWN"
    IMPOSSIBLE = "IMPOSSIBLE"


@dataclass(frozen=True)
class SettlementPlanFacts:
    current_count: int
    pieces_available: int
    distance_rule_empty_count: int
    structurally_reachable_count: int
    currently_connected_count: int
    max_additional_roads: int


@dataclass(frozen=True)
class CityPlanFacts:
    current_count: int
    pieces_available: int
    upgradeable_count: int


@dataclass(frozen=True)
class LongestRoadPlanFacts:
    currently_owned: bool
    current_length: int
    current_holder: int | None
    public_best_opponent_length: int
    acquisition_gap: int
    defense_gap: int
    structurally_extendable: int


@dataclass(frozen=True)
class LargestArmyPlanFacts:
    currently_owned: bool
    used_knights: int
    current_holder: int | None
    public_best_opponent_used_knights: int
    plays_needed: int
    development_deck_remaining: int
    known_usable_knights: int
    known_new_knights: int


@dataclass(frozen=True)
class VictoryPlanCandidate:
    settlement_gain_count: int
    city_upgrade_count: int
    acquire_longest_road: bool
    acquire_largest_army: bool
    feasibility: PlanFeasibility
    valid: bool
    diagnostic_reasons: tuple[str, ...]

    @property
    def candidate_id(self) -> tuple[int, int, bool, bool]:
        return (self.settlement_gain_count, self.city_upgrade_count,
                self.acquire_longest_road, self.acquire_largest_army)

    @property
    def planned_point_gain(self) -> int:
        return (self.settlement_gain_count + self.city_upgrade_count
                + 2 * int(self.acquire_longest_road)
                + 2 * int(self.acquire_largest_army))


@dataclass(frozen=True)
class VictoryPlanContext:
    schema_version: str
    player_id: int
    current_score: int
    remaining_points: int
    settlement: SettlementPlanFacts
    city: CityPlanFacts
    longest_road: LongestRoadPlanFacts
    largest_army: LargestArmyPlanFacts
    candidates: tuple[VictoryPlanCandidate, ...]
    unresolved: bool

    @property
    def candidate_mask(self) -> tuple[bool, ...]:
        return tuple(candidate.valid for candidate in self.candidates)


SCHEMA_VERSION = "victory_plan_context_v1"


def _distance_rule_vertices(game: GameState) -> tuple[int, ...]:
    blocked = set(game.settlements) | set(game.cities)
    for vertex_id in tuple(blocked):
        for edge_id in game.board.vertices[vertex_id].edge_ids:
            blocked.update(game.board.edges[edge_id].vertex_ids)
    return tuple(vertex.id for vertex in game.board.vertices if vertex.id not in blocked)


def _road_network_distances(game: GameState, player_id: int) -> dict[int, int]:
    """Minimum *additional* roads from the player's present network.

    Occupied opponent edges cannot be traversed and opponent buildings cannot
    be used as through-vertices. This is an optimistic reachability bound: it
    assumes the player could eventually place all presently available roads.
    """
    opponent_buildings = {vertex for vertex, owner in {**game.settlements, **game.cities}.items()
                          if owner != player_id}
    sources = {vertex for vertex, owner in {**game.settlements, **game.cities}.items()
               if owner == player_id}
    for edge_id, owner in game.roads.items():
        if owner == player_id:
            sources.update(game.board.edges[edge_id].vertex_ids)
    sources -= opponent_buildings
    distance = {vertex: 0 for vertex in sources}
    queue = deque(sorted(sources))
    while queue:
        vertex = queue.popleft()
        for edge_id in game.board.vertices[vertex].edge_ids:
            owner = game.roads.get(edge_id)
            if owner is not None and owner != player_id:
                continue
            a, b = game.board.edges[edge_id].vertex_ids
            other = b if a == vertex else a
            if other in opponent_buildings:
                continue
            candidate = distance[vertex] + (0 if owner == player_id else 1)
            if candidate < distance.get(other, 10**9):
                distance[other] = candidate
                (queue.appendleft if owner == player_id else queue.append)(other)
    return distance


def _connected_independent_witness(game: GameState, vertices: tuple[int, ...], count: int) -> bool:
    """Conservative constructive witness; failure only means UNKNOWN."""
    if count == 0:
        return True
    if len(vertices) < count:
        return False
    neighbors = {
        vertex_id: {other for edge_id in game.board.vertices[vertex_id].edge_ids
                    for other in game.board.edges[edge_id].vertex_ids if other != vertex_id}
        for vertex_id in vertices
    }
    for order in (vertices, tuple(reversed(vertices))):
        chosen: list[int] = []
        for vertex_id in order:
            if all(vertex_id not in neighbors[other] for other in chosen):
                chosen.append(vertex_id)
                if len(chosen) >= count:
                    return True
    return False


def _structural_road_options(game: GameState, player_id: int) -> int:
    if sum(owner == player_id for owner in game.roads.values()) >= 15:
        return 0
    buildings = {**game.settlements, **game.cities}
    total = 0
    for edge in game.board.edges:
        if edge.id in game.roads:
            continue
        if any(buildings.get(vertex_id) == player_id or
               (vertex_id not in buildings and any(
                   game.roads.get(other) == player_id
                   for other in game.board.vertices[vertex_id].edge_ids))
               for vertex_id in edge.vertex_ids):
            total += 1
    return total


def _candidate_feasibility(
    settlement_gain: int, city_upgrade: int, road: bool, army: bool,
    settlement: SettlementPlanFacts, city: CityPlanFacts,
    longest: LongestRoadPlanFacts, largest: LargestArmyPlanFacts,
    connected_witness: bool,
) -> tuple[PlanFeasibility, tuple[str, ...]]:
    impossible: list[str] = []
    unknown: list[str] = []
    # City upgrades return settlement pieces. Evaluate the *joint* plan, not
    # today's inventory, and allow a new plan settlement to be upgraded.
    if city.current_count + city_upgrade > 4:
        impossible.append("CITY_PIECE_LIMIT")
    if city_upgrade > settlement.current_count + settlement_gain:
        impossible.append("CITY_REQUIRES_SETTLEMENT")
    if settlement.current_count + settlement_gain - city_upgrade > 5:
        impossible.append("SETTLEMENT_PIECE_LIMIT_AFTER_RECYCLING")
    if settlement_gain > settlement.distance_rule_empty_count:
        impossible.append("INSUFFICIENT_DISTANCE_RULE_SITES")
    if settlement_gain > settlement.structurally_reachable_count:
        # Even using every remaining road piece, too few present distance-rule
        # sites can be reached. Opponent pieces can only close, not open, paths.
        # Before initial placement, however, the future network is unknown.
        if settlement.current_count + city.current_count == 0:
            unknown.append("SETUP_NETWORK_NOT_YET_ESTABLISHED")
        else:
            impossible.append("INSUFFICIENT_STRUCTURAL_REACHABLE_SITES")
    elif settlement_gain and not connected_witness:
        unknown.append("SETTLEMENT_SEQUENCE_NOT_CURRENTLY_WITNESSED")
    if road:
        if longest.currently_owned:
            impossible.append("LONGEST_ROAD_ALREADY_OWNED")
        else:
            unknown.append("LONGEST_ROAD_FUTURE_ROUTE_UNCONFIRMED")
    if army:
        if largest.currently_owned:
            impossible.append("LARGEST_ARMY_ALREADY_OWNED")
        elif (largest.used_knights + largest.known_usable_knights
              + largest.known_new_knights + largest.development_deck_remaining
              < largest.used_knights + largest.plays_needed):
            impossible.append("INSUFFICIENT_KNIGHT_CARD_UPPER_BOUND")
        elif largest.known_usable_knights + largest.known_new_knights < largest.plays_needed:
            unknown.append("FUTURE_KNIGHT_DRAW_UNCERTAIN")
    if impossible:
        return PlanFeasibility.IMPOSSIBLE, tuple(impossible)
    if unknown:
        return PlanFeasibility.UNKNOWN, tuple(unknown)
    return PlanFeasibility.FEASIBLE, ()


def build_victory_plan_context(game: GameState, player_id: int) -> VictoryPlanContext:
    """Derive an immutable exact-gain candidate set from presently known facts.

    IMPOSSIBLE tuples remain in the ordered audit list with ``valid=False``;
    ``candidate_mask`` is the list for a future categorical head. No chosen
    plan, action, Planner judgement, RNG call or GameState write occurs here.
    """
    player = player_for(game, player_id)
    score = actual_score(game, player_id)
    remaining = max(0, 10 - score)
    distance_vertices = _distance_rule_vertices(game)
    road_distances = _road_network_distances(game, player_id)
    road_count = sum(owner == player_id for owner in game.roads.values())
    roads_left = 15 - road_count
    reachable = tuple(vertex for vertex in distance_vertices
                      if road_distances.get(vertex, 10**9) <= roads_left)
    connected = tuple(vertex for vertex in distance_vertices
                      if any(game.roads.get(edge) == player_id
                             for edge in game.board.vertices[vertex].edge_ids))
    settlement = SettlementPlanFacts(
        current_count=player.settlements, pieces_available=5-player.settlements,
        distance_rule_empty_count=len(distance_vertices),
        structurally_reachable_count=len(reachable),
        currently_connected_count=len(connected), max_additional_roads=roads_left)
    city = CityPlanFacts(player.cities, 4-player.cities, player.settlements)
    own_length = _longest_road_length(game, player_id)
    opponent_length = max((_longest_road_length(game, p.id) for p in game.players
                           if p.id != player_id), default=0)
    road_owned = game.longest_road_holder_id == player_id
    longest = LongestRoadPlanFacts(
        road_owned, own_length, game.longest_road_holder_id, opponent_length,
        0 if road_owned else max(0, max(5, opponent_length+1)-own_length),
        max(0, own_length-max(5, opponent_length)) if road_owned else 0,
        _structural_road_options(game, player_id))
    other_knights = max((p.played_knights for p in game.players if p.id != player_id), default=0)
    army_owned = game.largest_army_holder_id == player_id
    own_knights = player.development_cards.count("knight")
    new_knights = player.new_development_cards.count("knight")
    largest = LargestArmyPlanFacts(
        army_owned, player.played_knights, game.largest_army_holder_id, other_knights,
        0 if army_owned else max(0, max(3, other_knights+1)-player.played_knights),
        len(game.development_deck), max(0, own_knights-new_knights), new_knights)
    candidates: list[VictoryPlanCandidate] = []
    if remaining:
        # Lexicographic canonical order: (S, C, R, A).
        for s in range(remaining+1):
            for c in range(remaining+1):
                for r in (False, True):
                    for a in (False, True):
                        if s+c+2*int(r)+2*int(a) != remaining:
                            continue
                        feasibility, reasons = _candidate_feasibility(
                            s, c, r, a, settlement, city, longest, largest,
                            _connected_independent_witness(game, connected, s))
                        candidates.append(VictoryPlanCandidate(
                            s, c, r, a, feasibility,
                            feasibility is not PlanFeasibility.IMPOSSIBLE, reasons))
    result = VictoryPlanContext(
        SCHEMA_VERSION, player_id, score, remaining,
        settlement, city, longest, largest, tuple(candidates),
        unresolved=remaining > 0 and not any(item.valid for item in candidates))
    return result
