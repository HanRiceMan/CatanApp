"""Step 3B-1: physical plan enumeration, never a policy or action mask."""

from copy import deepcopy
from app.domain.game import (BUILD_COSTS, build_piece, create_game,
                             legal_road_ids, legal_settlement_ids,
                             use_development)
from app.rl.reward import actual_score
from app.rl.victory_plan_context import (PlanFeasibility,
                                         _is_minimal_winning_composition,
                                         build_victory_plan_context)


def _game_with_buildable_settlement():
    game = create_game(33010001)
    game.phase = "action"
    game.current_player_id = 1
    player = game.players[0]
    game.settlements[0] = 1
    player.settlements = 1
    # A connected two-edge path from the original building permits a new
    # settlement while respecting the distance rule.
    start = 0
    first_edge = game.board.vertices[start].edge_ids[0]
    game.roads[first_edge] = 1
    neighbor = next(v for v in game.board.edges[first_edge].vertex_ids if v != start)
    second_edge = next(e for e in game.board.vertices[neighbor].edge_ids
                       if e != first_edge)
    game.roads[second_edge] = 1
    distant = game.board.vertices[-1].id
    game.settlements[distant] = 1
    player.settlements = 2
    assert legal_settlement_ids(game, 1, initial=False)
    for resource in BUILD_COSTS["settlement"]:
        player.resources[resource] = 3
    for resource, count in BUILD_COSTS["city"].items():
        player.resources[resource] = max(player.resources[resource], count)
    return game


def test_exact_gain_stable_immutable_and_no_side_effects():
    game = _game_with_buildable_settlement()
    before = deepcopy(game)
    first = build_victory_plan_context(game, 1)
    second = build_victory_plan_context(game, 1)
    assert game == before
    assert game.random_source.counter == before.random_source.counter
    assert first == second
    assert first.candidates == tuple(sorted(first.candidates, key=lambda x: x.candidate_id))
    assert len({candidate.candidate_id for candidate in first.candidates}) == len(first.candidates)
    assert len(first.candidates) == 28  # score 2, exact remaining gain 8
    assert all(candidate.planned_point_gain == first.remaining_points
               for candidate in first.candidates)
    assert all(_is_minimal_winning_composition(*candidate.candidate_id,
                                              first.remaining_points)
               for candidate in first.candidates)
    assert first.candidate_mask == tuple(c.valid for c in first.candidates)
    assert all(c.valid == (c.feasibility is not PlanFeasibility.IMPOSSIBLE)
               for c in first.candidates)
    assert first.current_score == actual_score(game, 1)
    assert first.settlement.currently_connected_count == len(
        legal_settlement_ids(game, 1, initial=False))
    assert first.longest_road.structurally_extendable == len(
        legal_road_ids(game, 1, initial=False))
    assert not hasattr(first, "selected_plan")


def test_known_vp_card_updates_score_and_deck_order_is_not_read():
    game = _game_with_buildable_settlement()
    before = build_victory_plan_context(game, 1)
    game.players[0].development_cards.append("victory_point")
    after = build_victory_plan_context(game, 1)
    assert after.current_score == before.current_score + 1
    assert after.remaining_points == before.remaining_points - 1
    assert {c.candidate_id for c in after.candidates} != {c.candidate_id for c in before.candidates}
    game.development_deck.reverse()
    assert build_victory_plan_context(game, 1) == after


def test_out_of_plan_settlement_replans_without_masking_action():
    game = _game_with_buildable_settlement()
    game.players[0].development_cards.extend(["victory_point"] * 3)
    before = build_victory_plan_context(game, 1)
    assert any(c.settlement_gain_count == 0 and c.valid for c in before.candidates)
    vertex = legal_settlement_ids(game, 1, initial=False)[0]
    build_piece(game, 1, "settlement", vertex, game.revision, "plan-outside-settlement")
    after = build_victory_plan_context(game, 1)
    assert after.current_score == before.current_score + 1
    assert after.remaining_points == before.remaining_points - 1
    assert after.candidates != before.candidates


def test_city_upgrade_is_net_one_and_recycles_settlement_piece():
    game = _game_with_buildable_settlement()
    game.players[0].development_cards.extend(["victory_point"] * 3)
    before = build_victory_plan_context(game, 1)
    city_vertex = next(v for v, owner in game.settlements.items() if owner == 1)
    build_piece(game, 1, "city", city_vertex, game.revision, "plan-city")
    after = build_victory_plan_context(game, 1)
    assert after.current_score == before.current_score + 1
    assert after.remaining_points == before.remaining_points - 1
    assert after.settlement.pieces_available == before.settlement.pieces_available + 1
    assert after.city.pieces_available == before.city.pieces_available - 1
    assert any(c.settlement_gain_count > before.settlement.pieces_available
               and c.city_upgrade_count > 0 and c.valid for c in before.candidates)


def test_titles_gain_two_once_and_owned_titles_remain_invalid_at_score_nine():
    game = _game_with_buildable_settlement()
    before = build_victory_plan_context(game, 1)
    game.longest_road_holder_id = 1
    after_road = build_victory_plan_context(game, 1)
    assert after_road.current_score == before.current_score + 2
    assert after_road.remaining_points == before.remaining_points - 2
    assert all(not c.acquire_longest_road for c in after_road.candidates if c.valid)
    game.largest_army_holder_id = 1
    after_army = build_victory_plan_context(game, 1)
    assert after_army.current_score == after_road.current_score + 2
    assert after_army.remaining_points == after_road.remaining_points - 2
    assert all(not c.acquire_largest_army for c in after_army.candidates if c.valid)
    game.players[0].development_cards.extend(["victory_point"] * 3)
    nine = build_victory_plan_context(game, 1)
    assert nine.current_score == 9
    assert nine.remaining_points == 1
    assert all(not c.acquire_longest_road and not c.acquire_largest_army
               for c in nine.candidates if c.valid)


def test_actual_road_and_knight_actions_acquire_titles_and_replan():
    road_game = _game_with_buildable_settlement()
    road_game.roads.clear()
    path: list[int] = []

    def extend(vertex: int, seen: set[int]) -> bool:
        if len(path) == 5:
            return True
        for edge_id in road_game.board.vertices[vertex].edge_ids:
            a, b = road_game.board.edges[edge_id].vertex_ids
            next_vertex = b if a == vertex else a
            if next_vertex in seen:
                continue
            path.append(edge_id)
            if extend(next_vertex, seen | {next_vertex}):
                return True
            path.pop()
        return False

    assert extend(0, {0})
    for edge_id in path[:4]:
        road_game.roads[edge_id] = 1
    before_road = build_victory_plan_context(road_game, 1)
    build_piece(road_game, 1, "road", path[4], road_game.revision, "title-road")
    after_road = build_victory_plan_context(road_game, 1)
    assert road_game.longest_road_holder_id == 1
    assert after_road.current_score == before_road.current_score + 2
    assert after_road.remaining_points == before_road.remaining_points - 2
    assert all(not c.acquire_longest_road for c in after_road.candidates if c.valid)

    army_game = _game_with_buildable_settlement()
    army_game.players[0].played_knights = 2
    army_game.players[0].development_cards.append("knight")
    before_army = build_victory_plan_context(army_game, 1)
    use_development(army_game, 1, "knight", army_game.revision, "title-army")
    after_army = build_victory_plan_context(army_game, 1)
    assert army_game.largest_army_holder_id == 1
    assert after_army.current_score == before_army.current_score + 2
    assert after_army.remaining_points == before_army.remaining_points - 2
    assert all(not c.acquire_largest_army for c in after_army.candidates if c.valid)


def test_unknown_army_draw_vs_proven_impossible_and_no_hidden_opponents():
    game = _game_with_buildable_settlement()
    initial = build_victory_plan_context(game, 1)
    army_candidates = [c for c in initial.candidates if c.acquire_largest_army
                       and not c.acquire_longest_road and c.settlement_gain_count == 0]
    assert army_candidates
    assert all(c.feasibility is PlanFeasibility.UNKNOWN for c in army_candidates
               if c.valid)
    game.development_deck.clear()
    exhausted = build_victory_plan_context(game, 1)
    assert all(not c.valid for c in exhausted.candidates if c.acquire_largest_army)
    game.players[1].resources["ore"] = 19
    game.players[1].development_cards.append("victory_point")
    assert build_victory_plan_context(game, 1) == exhausted


def test_physical_piece_limits_and_unresolved():
    game = _game_with_buildable_settlement()
    game.players[0].development_cards.extend(["victory_point"] * 4)
    ctx = build_victory_plan_context(game, 1)
    assert any(c.feasibility is PlanFeasibility.IMPOSSIBLE for c in ctx.candidates)
    assert all(c.city_upgrade_count <= 4 - ctx.city.current_count for c in ctx.candidates if c.valid)
    assert all(c.settlement_gain_count + ctx.settlement.current_count
               - c.city_upgrade_count <= 5 for c in ctx.candidates if c.valid)
    game.winner_id = 1
    game.phase = "game_over"
    game.players[0].development_cards.extend(["victory_point"] * 5)
    won = build_victory_plan_context(game, 1)
    assert won.remaining_points == 0
    assert won.candidates == ()
    assert not won.unresolved


def test_blocked_network_is_impossible_but_unplaced_setup_is_unknown():
    game = _game_with_buildable_settlement()
    game.roads.clear()
    for vertex_id, owner in game.settlements.items():
        if owner == 1:
            for edge_id in game.board.vertices[vertex_id].edge_ids:
                game.roads[edge_id] = 2
    blocked = build_victory_plan_context(game, 1)
    assert blocked.settlement.distance_rule_empty_count > 0
    assert blocked.settlement.structurally_reachable_count == 0
    assert all(not c.valid for c in blocked.candidates if c.settlement_gain_count > 0)
    assert any("INSUFFICIENT_STRUCTURAL_REACHABLE_SITES" in c.diagnostic_reasons
               for c in blocked.candidates)

    setup = create_game(33010002)
    before_placement = build_victory_plan_context(setup, 1)
    assert before_placement.current_score == 0
    assert len(before_placement.candidates) == 36  # global maximum for score 0..10
    assert any(c.settlement_gain_count > 0 and c.feasibility is PlanFeasibility.UNKNOWN
               for c in before_placement.candidates if c.valid)


def test_score_nine_title_only_routes_resolve_when_plus_one_routes_fail():
    game = create_game(33010003)
    game.phase = "action"
    game.current_player_id = 1
    chosen: list[int] = []
    occupied_edges: set[int] = set()
    for vertex in game.board.vertices:
        if not occupied_edges.intersection(vertex.edge_ids):
            chosen.append(vertex.id)
            occupied_edges.update(vertex.edge_ids)
            if len(chosen) == 4:
                break
    assert len(chosen) == 4
    game.cities = {vertex: 1 for vertex in chosen}
    game.players[0].cities = 4
    game.players[0].development_cards.append("victory_point")
    game.roads = {edge: 2 for edge in occupied_edges}
    context = build_victory_plan_context(game, 1)
    assert context.current_score == 9
    assert context.remaining_points == 1
    assert len(context.candidates) == 4
    assert [candidate.candidate_id for candidate in context.candidates] == sorted(
        candidate.candidate_id for candidate in context.candidates)
    by_id = {candidate.candidate_id: candidate for candidate in context.candidates}
    assert not by_id[(1, 0, False, False)].valid
    assert not by_id[(0, 1, False, False)].valid
    assert by_id[(0, 0, True, False)].valid
    assert by_id[(0, 0, False, True)].valid
    assert by_id[(0, 0, True, False)].feasibility is PlanFeasibility.UNKNOWN
    assert by_id[(0, 0, False, True)].feasibility is PlanFeasibility.UNKNOWN
    assert context.overshoot_points(by_id[(0, 0, True, False)]) == 1
    assert not context.unresolved


def test_remaining_three_dual_title_minimality_and_exact_regression():
    game = _game_with_buildable_settlement()
    game.players[0].development_cards.extend(["victory_point"] * 5)
    context = build_victory_plan_context(game, 1)
    assert context.current_score == 7 and context.remaining_points == 3
    ids = {candidate.candidate_id for candidate in context.candidates}
    assert (0, 0, True, True) in ids  # 4 points, but either title is essential
    assert (1, 0, True, True) not in ids  # redundant +1 settlement
    assert (0, 1, True, True) not in ids  # redundant +1 city
    assert (1, 0, True, False) in ids
    assert (0, 1, False, True) in ids
    dual = next(candidate for candidate in context.candidates
                if candidate.candidate_id == (0, 0, True, True))
    assert dual.planned_point_gain == 4
    assert context.overshoot_points(dual) == 1
    assert all(context.overshoot_points(candidate) in (0, 1)
               for candidate in context.candidates)

    start = _game_with_buildable_settlement()
    original_exact = build_victory_plan_context(start, 1)
    assert (2, 2, True, True) in {candidate.candidate_id for candidate in original_exact.candidates}
    assert original_exact.remaining_points == 8
