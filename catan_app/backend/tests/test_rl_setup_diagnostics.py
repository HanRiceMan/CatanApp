import unittest

from app.domain.game import create_game
from app.rl.compare import PolicyCompatibleHeuristicAgent
from app.rl.setup_diagnostics import (GreedyPipsSettlementAgent, GreedyPipsSetupAgent, probe_setup, road_extension_pips,
                                      summarize, vertex_pips, vertex_production)


class SetupDiagnosticsTests(unittest.TestCase):
    def test_vertex_production_and_total_pips_agree(self):
        board = create_game(42).board
        for vertex in board.vertices:
            self.assertEqual(vertex_pips(board, vertex.id),
                             sum(vertex_production(board, vertex.id).values()))

    def test_road_must_touch_initial_settlement(self):
        game = create_game(42)
        edge = game.board.edges[0]
        other = next(vertex.id for vertex in game.board.vertices if vertex.id not in edge.vertex_ids)
        with self.assertRaises(ValueError):
            road_extension_pips(game, other, edge.id)

    def test_same_seed_reproduces_setup_and_has_two_pieces(self):
        agent = PolicyCompatibleHeuristicAgent()
        first = probe_setup(agent, 26001, 0)
        second = probe_setup(agent, 26001, 0)
        self.assertEqual(first, second)
        self.assertEqual(len(first["settlement_ids"]), 2)
        self.assertEqual(len(first["road_ids"]), 2)
        self.assertEqual(first["total_pips"], first["first_pips"] + first["second_pips"])
        for choice in first["settlement_choices"]:
            self.assertGreaterEqual(choice["best_legal_pips"], choice["selected_pips"])
        for choice in first["road_choices"]:
            self.assertGreaterEqual(choice["best_legal_extension_pips"], choice["selected_extension_pips"])
        summary = summarize([first])
        self.assertEqual(summary["games"], 1)
        self.assertEqual(summary["settlement_unique_vertices"], [1, 1])
        self.assertEqual(summary["settlement_most_common"][0]["share"], 1.0)

    def test_greedy_setup_chooses_highest_available_pips(self):
        row = probe_setup(GreedyPipsSetupAgent(), 26001, 0)
        for choice in row["settlement_choices"]:
            self.assertEqual(choice["selected_pips"], choice["best_legal_pips"])
        for choice in row["road_choices"]:
            self.assertEqual(choice["selected_extension_pips"], choice["best_legal_extension_pips"])

    def test_greedy_settlement_preserves_heuristic_road(self):
        row = probe_setup(GreedyPipsSettlementAgent(), 26001, 0)
        for choice in row["settlement_choices"]:
            self.assertEqual(choice["selected_pips"], choice["best_legal_pips"])
        self.assertEqual(len(row["road_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
