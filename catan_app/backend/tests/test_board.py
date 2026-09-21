"""盤面の個数・接続・制約を、複数の乱数seedで検証する。"""

import unittest
from collections import Counter
from itertools import combinations, product

from app.domain.board import (NUMBER_TOKENS, TERRAIN_COUNTS, FIXED_PORTS, BoardRules,
                              are_neighbors, create_board, create_topology, terrain_clusters_are_small)
from app.domain.game import create_game


class BoardTests(unittest.TestCase):
    def test_topology_and_shared_corners(self):
        topology = create_topology()
        self.assertEqual((len(topology.hexes), len(topology.vertices), len(topology.edges)), (19, 54, 72))
        self.assertEqual(sum(len(edge.hex_ids) == 1 for edge in topology.edges), 30)
        self.assertEqual(len({(v.x, v.y) for v in topology.vertices}), 54)
        for a, b in combinations(range(19), 2):
            if are_neighbors(topology.hexes[a], topology.hexes[b]):
                self.assertEqual(sum(a in v.hex_ids and b in v.hex_ids for v in topology.vertices), 2)
                self.assertEqual(sum(a in e.hex_ids and b in e.hex_ids for e in topology.edges), 1)
        for edge in topology.edges:
            for vertex_id in edge.vertex_ids:
                self.assertIn(edge.id, topology.vertices[vertex_id].edge_ids)

    def test_random_board_invariants(self):
        for seed in range(300):
            with self.subTest(seed=seed):
                board = create_board(seed)
                self.assertEqual(Counter(tile.terrain for tile in board.tiles), TERRAIN_COUNTS)
                self.assertEqual(Counter(tile.number for tile in board.tiles if tile.number is not None),
                                 Counter(NUMBER_TOKENS))
                self.assertEqual(board.tiles[board.robber_tile_id].terrain, "desert")
                self.assertIsNone(board.tiles[board.robber_tile_id].number)
                red = [tile for tile in board.tiles if tile.number in (6, 8)]
                self.assertTrue(all(not are_neighbors(a, b) for a, b in combinations(red, 2)))
                self.assertEqual(Counter(port.resource for port in board.ports),
                                 {None: 4, "wood": 1, "brick": 1, "sheep": 1, "wheat": 1, "ore": 1})
                used_vertices = set()
                for port in board.ports:
                    edge = board.edges[port.edge_id]
                    self.assertEqual(len(edge.hex_ids), 1)
                    self.assertTrue(used_vertices.isdisjoint(edge.vertex_ids))
                    used_vertices.update(edge.vertex_ids)
                    self.assertEqual(port.ratio, 3 if port.resource is None else 2)

    def test_seed_reproduces_board(self):
        self.assertEqual(create_board(42), create_board(42))
        self.assertNotEqual(create_board(42).tiles, create_board(43).tiles)

    def test_four_local_players_start_before_placement(self):
        game = create_game(0)
        self.assertEqual(game.phase, "rolling_order")
        self.assertEqual([p.id for p in game.players], [1, 2, 3, 4])
        self.assertFalse(any(p.is_ai for p in game.players))
        self.assertEqual(game.setup_order, [])
        self.assertEqual(game.seat_order, [1, 2, 3, 4])
        self.assertEqual([p.color for p in game.players], ["red", "cyan", "purple", "yellow"])

    def test_all_sixteen_rule_combinations(self):
        for flags in product((False, True), repeat=4):
            rules = BoardRules(*flags)
            for seed in range(30):
                with self.subTest(flags=flags, seed=seed):
                    board = create_board(seed, rules)
                    self.assertEqual(Counter(t.terrain for t in board.tiles), TERRAIN_COUNTS)
                    self.assertEqual(Counter(t.number for t in board.tiles if t.number is not None),
                                     Counter(NUMBER_TOKENS))
                    red = [t for t in board.tiles if t.number in (6, 8)]
                    if rules.desert_center:
                        self.assertEqual(board.robber_tile_id, 9)
                    if rules.red_numbers_not_adjacent:
                        self.assertTrue(all(not are_neighbors(a, b) for a, b in combinations(red, 2)))
                    if rules.red_numbers_distinct_terrains:
                        self.assertEqual(len({t.terrain for t in red}), 4)
                    if rules.terrain_clusters_max_two:
                        # 別の探索で連結成分を検証する（判定関数自身だけを信用しない）。
                        remaining = set(range(19))
                        while remaining:
                            component = {remaining.pop()}
                            while True:
                                more = {j for j in remaining if any(
                                    board.tiles[j].terrain == board.tiles[i].terrain
                                    and are_neighbors(board.tiles[j], board.tiles[i]) for i in component)}
                                if not more:
                                    break
                                component.update(more)
                                remaining.difference_update(more)
                            self.assertLessEqual(len(component), 2)
                    for port, (vertices, resource) in zip(board.ports, FIXED_PORTS):
                        self.assertEqual(set(board.edges[port.edge_id].vertex_ids), set(vertices))
                        self.assertEqual(port.resource, resource)
                    self.assertEqual(board, create_board(seed, rules))

    def test_disabled_options_do_not_apply_constraints(self):
        boards = [create_board(seed, BoardRules(False, False, False, False)) for seed in range(50)]
        self.assertTrue(any(b.robber_tile_id != 9 for b in boards))
        self.assertTrue(any(any(are_neighbors(a, c) for a, c in combinations(
            [t for t in b.tiles if t.number in (6, 8)], 2)) for b in boards))
        self.assertTrue(any(len({t.terrain for t in b.tiles if t.number in (6, 8)}) < 4 for b in boards))
        self.assertTrue(any(not terrain_clusters_are_small([t.terrain for t in b.tiles]) for b in boards))


if __name__ == "__main__":
    unittest.main()
