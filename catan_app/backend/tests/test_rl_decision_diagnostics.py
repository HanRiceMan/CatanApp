import unittest

from app.domain.actions import BuildRoadAction, MoveRobberAction
from app.domain.game import create_game, player_for
from app.rl.decision_diagnostics import diagnose_action


class DecisionDiagnosticsTests(unittest.TestCase):
    def test_road_diagnostic_does_not_change_game(self):
        game = create_game(20260929)
        game.phase = "action"
        game.current_player_id = 1
        game.last_roll = {"dice": [3, 4], "total": 7}
        game.roads[30] = 1
        player_for(game, 1).resources.update(wood=1, brick=1)
        before = dict(game.roads)

        result = diagnose_action(game, 1, BuildRoadAction(31), object())

        self.assertEqual(game.roads, before)
        self.assertEqual(result["kind"], "road")
        self.assertIn("best_longest_after_one_more", result)
        self.assertIn("blocked_endpoints", result)
        self.assertEqual(result["longest_increased"],
                         result["longest_after"] > result["longest_before"])

    def test_road_diagnostic_distinguishes_holding_from_taking_title(self):
        game = create_game(20260929)
        game.phase = "action"
        game.current_player_id = 1
        game.roads[30] = 1
        game.longest_road_holder_id = 1
        player_for(game, 1).resources.update(wood=1, brick=1)

        result = diagnose_action(game, 1, BuildRoadAction(31), object())

        self.assertTrue(result["held_longest_road_before"])
        self.assertFalse(result["takes_longest_road_now"])
        self.assertIn("holds_longest_road_after", result)

    def test_robber_diagnostic_contains_public_candidate_ranking(self):
        game = create_game(20260929)
        game.phase = "robber_move"
        game.current_player_id = 1
        tile_id = next(tile.id for tile in game.board.tiles
                       if tile.id != game.robber_tile_id)

        result = diagnose_action(game, 1, MoveRobberAction(tile_id), object())

        self.assertEqual(result["kind"], "robber")
        self.assertEqual(len(result["candidates"]), 18)
        self.assertGreaterEqual(result["public_threat_rank"], 1)


if __name__ == "__main__":
    unittest.main()
