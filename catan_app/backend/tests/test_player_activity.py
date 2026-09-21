import unittest

from app.domain.game import (create_game, discard_for_seven, game_view,
                             pending_player_actions, player_for, propose_trade,
                             propose_counter_trade, respond_to_trade, roll_for_order,
                             roll_turn_dice, start_normal_game)


class PlayerActivityTests(unittest.TestCase):
    def setUp(self):
        self.game = create_game(42)

    def start_turn(self):
        for pid, dice in [(1, (3, 3)), (2, (6, 6)), (3, (4, 4)), (4, (1, 1))]:
            roll_for_order(self.game, pid, self.game.revision, f"order-{pid}", dice)
        self.assertEqual(pending_player_actions(self.game), {2: "開拓地を置く"})
        self.game.phase = "ready_to_play"
        start_normal_game(self.game, 2, self.game.revision, "start")

    def test_order_setup_turn_and_game_over(self):
        self.assertEqual(pending_player_actions(self.game), {1: "順番決めのサイコロ"})
        self.start_turn()
        self.assertEqual(pending_player_actions(self.game), {2: "サイコロ・発展カード"})
        self.game.phase = "game_over"
        self.assertEqual(pending_player_actions(self.game), {})

    def test_multiple_discards_clear_individually_then_robber_actor(self):
        self.start_turn()
        for pid in (1, 3):
            player_for(self.game, pid).resources["wood"] = 8
        roll_turn_dice(self.game, 2, self.game.revision, "seven", (3, 4))
        self.assertEqual(set(pending_player_actions(self.game)), {1, 3})
        discard_for_seven(self.game, 1, {"wood": 4}, self.game.revision, "discard-1")
        self.assertEqual(set(pending_player_actions(self.game)), {3})
        discard_for_seven(self.game, 3, {"wood": 4}, self.game.revision, "discard-3")
        self.assertEqual(pending_player_actions(self.game), {2: "盗賊を移動"})

    def test_trade_recipient_counter_proposer_and_return_to_turn(self):
        self.start_turn()
        roll_turn_dice(self.game, 2, self.game.revision, "roll", (2, 3))
        player_for(self.game, 2).resources.update(wood=2, brick=1)
        player_for(self.game, 3).resources.update(wood=1, brick=2)
        propose_trade(self.game, 2, 3, [{"give": {"wood": 1}, "want": {"brick": 1}}], self.game.revision, "offer")
        self.assertEqual(pending_player_actions(self.game), {3: "交渉に返答"})
        respond_to_trade(self.game, 3, "counter", None, self.game.revision, "counter")
        self.assertEqual(pending_player_actions(self.game), {3: "逆提案を作る"})
        propose_counter_trade(self.game, 3, [{"give": {"brick": 1}, "want": {"wood": 1}}], self.game.revision, "counter-offer")
        self.assertEqual(pending_player_actions(self.game), {2: "交渉に返答"})
        respond_to_trade(self.game, 2, "decline", None, self.game.revision, "decline")
        self.assertEqual(pending_player_actions(self.game), {2: "手番の操作"})

    def test_activity_is_public_but_opponent_cards_remain_private(self):
        self.start_turn()
        player_for(self.game, 2).development_cards = ["victory_point"]
        player_for(self.game, 2).used_development_cards = ["knight", "monopoly"]
        view = game_view(self.game, 1)
        opponent = next(p for p in view["players"] if p["id"] == 2)
        self.assertEqual(opponent["pending_action"], "サイコロ・発展カード")
        self.assertNotIn("resources", opponent)
        self.assertNotIn("development_cards", opponent)
        self.assertNotIn("score", opponent)
        self.assertEqual(opponent["used_development_cards"], ["knight", "monopoly"])


if __name__ == "__main__":
    unittest.main()
