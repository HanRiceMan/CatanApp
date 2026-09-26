import unittest

from app.domain.game import create_game, player_for
from app.rl.development_strategy import assess_development_purchase


def stalled_game():
    game = create_game(20261001)
    game.phase = "action"
    game.current_player_id = 1
    game.last_roll = (3, 4)
    game.roads[30] = 1
    player = player_for(game, 1)
    player.settlements = 2
    player.resources.update(sheep=1, wheat=1, ore=1)
    return game


class DevelopmentStrategyTests(unittest.TestCase):
    def test_two_sites_are_not_a_purchase_blocker(self):
        assessment = assess_development_purchase(stalled_game(), 1)

        self.assertEqual(assessment["site_count"], 2)
        self.assertIn("construction_stall", assessment["reasons"])
        self.assertTrue(assessment["eligible"])

    def test_opponent_hidden_card_type_does_not_change_probability(self):
        knight_game = stalled_game()
        player_for(knight_game, 2).development_cards.append("knight")
        point_game = stalled_game()
        player_for(point_game, 2).development_cards.append("victory_point")

        knight_view = assess_development_purchase(knight_game, 1)
        point_view = assess_development_purchase(point_game, 1)

        self.assertEqual(
            knight_view["unseen_card_probabilities"],
            point_view["unseen_card_probabilities"],
        )

    def test_nine_point_stall_remains_eligible_by_stall_not_score(self):
        game = stalled_game()
        player = player_for(game, 1)
        player.settlements = 3
        player.cities = 3

        assessment = assess_development_purchase(game, 1)

        self.assertEqual(assessment["score"], 9)
        self.assertIn("construction_stall", assessment["reasons"])
        self.assertNotIn("endgame_point_route", assessment["reasons"])
        self.assertTrue(assessment["eligible"])

    def test_nine_point_largest_army_route_can_still_buy(self):
        game = stalled_game()
        player = player_for(game, 1)
        player.settlements = 3
        player.cities = 3
        player.played_knights = 1

        assessment = assess_development_purchase(game, 1)

        self.assertIn("largest_army_race", assessment["reasons"])
        self.assertTrue(assessment["eligible"])


if __name__ == "__main__":
    unittest.main()
