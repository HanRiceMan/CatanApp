import unittest

from app.domain.game import create_game, player_for
from app.rl.goal_value import goal_context


class GoalValueTests(unittest.TestCase):
    def test_context_ignores_opponent_private_contents(self):
        game = create_game(20260930)
        opponent = player_for(game, 2)
        opponent.resources.update(wood=2, brick=1)
        opponent.development_cards.extend(["knight", "victory_point"])
        before = goal_context(game, 1)

        opponent.resources.update(wood=0, brick=0, sheep=1, wheat=2)
        opponent.development_cards[:] = ["monopoly", "road_building"]
        after = goal_context(game, 1)

        self.assertEqual(after, before)

    def test_context_contains_goal_distances(self):
        game = create_game(20260931)
        context = goal_context(game, 1)
        self.assertIn("city_deficit", context)
        self.assertIn("next_settlement_wait", context)
        self.assertIn("longest_road_gap", context)
        self.assertIn("largest_army_gap", context)


if __name__ == "__main__":
    unittest.main()
