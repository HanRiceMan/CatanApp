import unittest

from app.domain.game import GameActionError, create_game, game_view, roll_for_order


class GameTests(unittest.TestCase):
    def setUp(self):
        self.game = create_game(42)

    def roll(self, player_id, dice):
        roll_for_order(self.game, player_id, self.game.revision, f"request-{self.game.revision}", dice)

    def test_rank_and_seat_only_once(self):
        for pid, dice in [(1, (2, 3)), (2, (6, 6)), (3, (4, 5))]:
            self.roll(pid, dice)
            self.assertEqual(self.game.seat_order, [1, 2, 3, 4])
        self.roll(4, (1, 1))
        self.assertEqual(self.game.phase, "setup_ready")
        self.assertEqual(self.game.seat_order, [2, 3, 1, 4])
        self.assertEqual(self.game.setup_order, [2, 3, 1, 4, 4, 1, 3, 2])
        with self.assertRaises(GameActionError):
            self.roll(2, (6, 6))
        self.assertEqual(self.game.seat_order, [2, 3, 1, 4])

    def test_ties_reroll_only_within_original_rank_groups(self):
        for pid, dice in [(1, (5, 5)), (2, (1, 1)), (3, (5, 5)), (4, (1, 1))]:
            self.roll(pid, dice)
        self.assertEqual(self.game.pending_rollers, [1, 3, 2, 4])
        self.assertEqual(self.game.round_number, 2)
        for pid, dice in [(1, (1, 1)), (3, (1, 1)), (2, (6, 6)), (4, (5, 5))]:
            self.roll(pid, dice)
        self.assertEqual(self.game.pending_rollers, [1, 3])
        self.roll(1, (1, 2))
        self.roll(3, (2, 2))
        self.assertEqual(self.game.seat_order, [3, 1, 2, 4])

    def test_wrong_turn_stale_revision_and_retries(self):
        with self.assertRaises(GameActionError):
            self.roll(2, (1, 1))
        self.roll(1, (3, 4))
        roll_for_order(self.game, 1, 0, "request-0", (6, 6))
        self.assertEqual(len(self.game.rolls), 1)
        self.assertEqual(self.game.rolls[0]["total"], 7)
        with self.assertRaises(GameActionError):
            roll_for_order(self.game, 2, 0, "stale-request", (1, 1))

    def test_private_hands_scores_and_awards(self):
        p1, p2 = self.game.players[:2]
        p1.resources.update(wood=3, sheep=2)
        p1.development_cards = ["knight", "victory_point"]
        p1.settlements = 2
        p2.development_cards = ["monopoly"]
        self.game.longest_road_holder_id = 1
        self.game.largest_army_holder_id = 2
        for viewer in (None, 1, 2, 3, 4):
            view = game_view(self.game, viewer)
            self.assertNotIn("player_tokens", view)
            for p in view["players"]:
                self.assertEqual("resources" in p, p["id"] == viewer)
                self.assertEqual("development_cards" in p, p["id"] == viewer)
                self.assertEqual("score" in p, p["id"] == viewer)
            self.assertEqual(view["players"][0]["resource_count"], 5)
            self.assertEqual(view["players"][0]["public_score"], 4)
            self.assertTrue(view["players"][0]["has_longest_road"])
            self.assertTrue(view["players"][1]["has_largest_army"])
        self.assertEqual(game_view(self.game, 1)["players"][0]["score"], 5)

    def test_no_invented_cards_or_awards_before_placement(self):
        view = game_view(self.game, 1)
        self.assertIsNone(view["longest_road_holder_id"])
        self.assertIsNone(view["largest_army_holder_id"])
        for p in view["players"]:
            self.assertEqual(p["resource_count"], 0)
            self.assertEqual(p["development_count"], 0)
            self.assertEqual(p["public_score"], 0)

    def test_ai_watch_reveals_every_hand_from_every_seat(self):
        game = create_game(77, ai_player_ids=(1, 2, 3, 4), ai_watch_mode=True)
        game.players[0].resources["wood"] = 2
        game.players[1].resources["ore"] = 1
        game.players[1].development_cards = ["knight", "victory_point"]
        game.players[1].settlements = 2

        master_view = game_view(game)
        for player in master_view["players"]:
            self.assertIn("resources", player)
            self.assertIn("development_cards", player)
            self.assertNotIn("score", player)
        self.assertEqual(master_view["players"][0]["resources"]["wood"], 2)
        self.assertEqual(master_view["players"][1]["development_cards"], ["knight", "victory_point"])
        self.assertEqual(master_view["players"][1]["public_score"], 2)

        player_view = game_view(game, 1)
        for player in player_view["players"]:
            self.assertIn("resources", player)
            self.assertIn("development_cards", player)
            self.assertEqual("score" in player, player["id"] == 1)
        self.assertEqual(game_view(game, 2)["players"][1]["score"], 3)


if __name__ == "__main__":
    unittest.main()
