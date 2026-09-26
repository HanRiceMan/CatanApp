import unittest

from app.domain.game import create_game, player_for
from app.rl.victory_race import (best_robber_victim, public_score,
                                 threat_value)


class VictoryRaceTests(unittest.TestCase):
    def test_public_score_excludes_hidden_victory_point_cards(self):
        game = create_game(20260924)
        player = player_for(game, 2)
        player.settlements = 3
        player.cities = 1
        player.development_cards.extend(["victory_point", "victory_point"])
        game.longest_road_holder_id = 2

        self.assertEqual(public_score(game, 2), 7)

    def test_threat_is_invariant_to_private_card_contents(self):
        game = create_game(20260925)
        player = player_for(game, 2)
        player.resources.update(wood=2, brick=1)
        player.development_cards.extend(["victory_point", "knight"])
        before = threat_value(game, 2)

        # 公開される合計枚数は同じまま、非公開の内訳だけを変える。
        player.resources.update(wood=0, brick=0, sheep=1, wheat=2)
        player.development_cards[:] = ["road_building", "monopoly"]

        self.assertEqual(threat_value(game, 2), before)

    def test_robber_victim_prefers_higher_public_threat(self):
        game = create_game(20260926)
        player_for(game, 2).settlements = 4
        player_for(game, 2).resources["wood"] = 1
        player_for(game, 3).settlements = 2
        player_for(game, 3).resources["wood"] = 5

        # 手札枚数が多い相手より、勝利に近い相手を優先する。
        self.assertEqual(best_robber_victim(game, 1, [2, 3]), 2)


if __name__ == "__main__":
    unittest.main()
