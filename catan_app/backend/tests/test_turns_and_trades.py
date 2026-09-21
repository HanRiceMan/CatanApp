import unittest

from app.domain.game import (GameActionError, bank_trade, build_piece, end_turn, game_view, move_robber, player_for,
                             propose_counter_trade, propose_trade, respond_to_trade, roll_for_order,
                             roll_turn_dice, start_normal_game, steal_with_robber, use_development)


def finish_order(game):
    for pid, dice in [(1, (3, 3)), (2, (6, 6)), (3, (4, 4)), (4, (1, 1))]:
        roll_for_order(game, pid, game.revision, f"order-{pid}", dice)


class TurnAndTradeTests(unittest.TestCase):
    def setUp(self):
        from app.domain.game import create_game
        self.game = create_game(42)
        finish_order(self.game)
        # 初期配置そのものはtest_setup.pyで検証済み。通常手番だけを最小状態で開始する。
        self.game.phase = "ready_to_play"
        start_normal_game(self.game, self.game.seat_order[0], self.game.revision, "start-normal")

    def action_phase(self):
        player_id = self.game.current_player_id
        roll_turn_dice(self.game, player_id, self.game.revision, f"roll-{self.game.revision}", (3, 3))
        self.assertEqual(self.game.phase, "action")
        return player_id

    def test_four_players_take_turns_in_fixed_seat_order(self):
        expected = self.game.seat_order * 2
        actual = []
        for index, player_id in enumerate(expected):
            self.assertEqual(self.game.current_player_id, player_id)
            actual.append(player_id)
            roll_turn_dice(self.game, player_id, self.game.revision, f"roll-{index}", (2, 3))
            end_turn(self.game, player_id, self.game.revision, f"end-{index}")
        self.assertEqual(actual, expected)
        self.assertEqual(self.game.phase, "turn_pre_roll")
        self.assertEqual(self.game.current_player_id, self.game.seat_order[0])

    def test_knight_before_roll_returns_to_dice_after_robber_and_random_steal(self):
        player_id = self.game.current_player_id
        victim_id = next(pid for pid in self.game.seat_order if pid != player_id)
        player_for(self.game, player_id).development_cards = ["knight"]
        player_for(self.game, victim_id).resources["wood"] = 1
        target = next(tile.id for tile in self.game.board.tiles if tile.id != self.game.robber_tile_id)
        vertex = next(vertex for vertex in self.game.board.vertices if target in vertex.hex_ids)
        self.game.settlements[vertex.id] = victim_id

        use_development(self.game, player_id, "knight", self.game.revision, "pre-roll-knight")
        self.assertEqual(self.game.phase, "robber_move")
        move_robber(self.game, player_id, target, self.game.revision, "move-after-knight")
        self.assertEqual(self.game.robber_victim_ids, [victim_id])
        self.assertEqual(self.game.phase, "robber_steal")
        steal_with_robber(self.game, player_id, victim_id, self.game.revision, "steal-after-knight")
        self.assertEqual(player_for(self.game, player_id).resources["wood"], 1)
        self.assertEqual(player_for(self.game, victim_id).resources["wood"], 0)
        self.assertEqual(self.game.phase, "turn_pre_roll")
        self.assertIsNone(self.game.last_roll)
        self.assertNotIn("木材", self.game.activity_log[-1]["text"])

    def test_road_building_before_roll_places_two_then_returns_to_dice(self):
        player_id = self.game.current_player_id
        player = player_for(self.game, player_id)
        player.development_cards = ["road_building"]
        first_edge = self.game.board.edges[0]
        self.game.roads[first_edge.id] = player_id

        use_development(self.game, player_id, "road_building", self.game.revision, "pre-roll-roads")
        self.assertEqual(self.game.phase, "action")
        self.assertEqual(self.game.free_road_remaining, 2)
        for index in range(2):
            target = self.game_view_legal_roads(player_id)[0]
            build_piece(self.game, player_id, "road", target, self.game.revision, f"free-road-{index}")
        self.assertEqual(self.game.free_road_remaining, 0)
        self.assertEqual(self.game.phase, "turn_pre_roll")
        with self.assertRaises(GameActionError):
            end_turn(self.game, player_id, self.game.revision, "end-before-roll")

    def game_view_legal_roads(self, player_id):
        from app.domain.game import legal_road_ids
        return legal_road_ids(self.game, player_id, initial=False)

    def test_year_of_plenty_can_be_used_before_roll(self):
        player_id = self.game.current_player_id
        player = player_for(self.game, player_id)
        player.development_cards = ["year_of_plenty"]
        use_development(self.game, player_id, "year_of_plenty", self.game.revision,
                        "pre-roll-plenty", resources=["wood", "ore"])
        self.assertEqual(self.game.phase, "turn_pre_roll")
        self.assertEqual(player.resources["wood"], 1)
        self.assertEqual(player.resources["ore"], 1)

    def test_used_cards_are_public_and_retry_does_not_duplicate_them(self):
        from app.domain.game import game_view
        pid = self.game.current_player_id
        player = player_for(self.game, pid)
        player.development_cards = ["year_of_plenty", "victory_point"]
        revision = self.game.revision
        for _ in range(2):
            use_development(self.game, pid, "year_of_plenty", revision, "public-card", resources=["wood", "ore"])
        for viewer in (None, 1, 2, 3, 4):
            public_player = next(p for p in game_view(self.game, viewer)["players"] if p["id"] == pid)
            self.assertEqual(public_player["used_development_cards"], ["year_of_plenty"])
            self.assertEqual("development_cards" in public_player, viewer == pid)
        events = [entry for entry in self.game.activity_log if entry["kind"] == "development"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["card"], "year_of_plenty")
        self.assertEqual(game_view(self.game)["activity_log"][-1], events[0])

    def test_old_development_card_can_be_played_after_buying_same_type(self):
        pid = self.game.current_player_id
        player = player_for(self.game, pid)
        player.development_cards = ["knight", "knight"]
        player.new_development_cards = ["knight"]
        use_development(self.game, pid, "knight", self.game.revision, "old-knight")
        self.assertEqual(player.used_development_cards, ["knight"])
        self.assertEqual(player.development_cards, ["knight"])
        self.assertEqual(player.new_development_cards, ["knight"])

    def test_dice_event_survives_end_turn_but_current_turn_dice_clears(self):
        from app.domain.game import game_view
        pid = self.action_phase()
        event = game_view(self.game, pid)["latest_dice_event"]
        self.assertEqual(event["dice"], [3, 3])
        self.assertEqual(event["kind"], "turn")
        self.assertEqual(game_view(self.game)["dice_history"], [{"player_id": pid, "dice": [3, 3], "total": 6, "turn_number": 1}])
        end_turn(self.game, pid, self.game.revision, "end-for-event")
        self.assertIsNone(game_view(self.game)["last_roll"])
        self.assertEqual(game_view(self.game)["latest_dice_event"], event)
        self.assertEqual(game_view(self.game)["dice_history"][0]["player_id"], pid)

    def test_trade_accepts_exactly_one_of_three_offers_and_moves_cards(self):
        proposer = self.action_phase()
        recipient = next(pid for pid in self.game.seat_order if pid != proposer)
        player_for(self.game, proposer).resources.update(wheat=3, brick=2, sheep=1)
        player_for(self.game, recipient).resources.update(wood=1, ore=1)
        offers = [
            {"give": {"wheat": 2, "brick": 1}, "want": {"wood": 1}},
            {"give": {"wheat": 2, "brick": 1}, "want": {"ore": 1}},
            {"give": {"wheat": 2, "sheep": 1}, "want": {"wood": 1}},
        ]
        propose_trade(self.game, proposer, recipient, offers, self.game.revision, "trade-one")
        self.assertEqual(self.game.phase, "trade_response")
        self.assertEqual(self.game.trade_count, 1)
        self.assertNotIn("小麦", self.game.activity_log[-1]["text"])
        self.assertNotIn("鉱石", self.game.activity_log[-1]["text"])
        proposal = game_view(self.game)["activity_log"][-1]["trade"]
        self.assertEqual((proposal["proposer_id"], proposal["recipient_id"]), (proposer, recipient))
        self.assertEqual(proposal["outcome"], "proposed")
        self.assertEqual(proposal["offers"][1]["want"]["ore"], 1)
        respond_to_trade(self.game, recipient, "accept", 1, self.game.revision, "accept-second")
        self.assertEqual(self.game.phase, "action")
        self.assertEqual(player_for(self.game, proposer).resources["ore"], 1)
        self.assertEqual(player_for(self.game, recipient).resources["wheat"], 2)
        self.assertEqual(player_for(self.game, recipient).resources["brick"], 1)
        result = self.game.activity_log[-1]
        self.assertEqual(result["kind"], "trade_accepted")
        self.assertIn(f"プレイヤー {proposer} がレンガ1枚・小麦2枚", result["text"])
        self.assertIn(f"プレイヤー {recipient} が鉱石1枚", result["text"])
        self.assertEqual(result["trade"]["outcome"], "accepted")
        self.assertEqual(result["trade"]["selected_offer_index"], 1)
        self.assertEqual(game_view(self.game)["activity_log"][-1], result)

    def test_bank_trade_and_declined_offer_have_public_cut_in_details(self):
        proposer = self.action_phase()
        recipient = next(pid for pid in self.game.seat_order if pid != proposer)
        player_for(self.game, proposer).resources["wood"] = 5
        bank_trade(self.game, proposer, "wood", "ore", self.game.revision, "bank-cut-in")
        bank_event = game_view(self.game, recipient)["activity_log"][-1]
        self.assertEqual(bank_event["bank"], {"give": {"wood": 4}, "want": {"ore": 1}})
        propose_trade(self.game, proposer, recipient, [{"give": {"wood": 1}, "want": {"brick": 1}}],
                      self.game.revision, "offer-cut-in")
        respond_to_trade(self.game, recipient, "decline", None, self.game.revision, "decline-cut-in")
        declined = game_view(self.game)["activity_log"][-1]["trade"]
        self.assertEqual(declined["outcome"], "declined")
        self.assertEqual(declined["offers"][0]["give"]["wood"], 1)

    def test_counter_trade_uses_same_count_and_original_player_can_accept(self):
        proposer = self.action_phase()
        recipient = next(pid for pid in self.game.seat_order if pid != proposer)
        player_for(self.game, proposer).resources.update(wood=2, sheep=1)
        player_for(self.game, recipient).resources.update(brick=2)
        propose_trade(self.game, proposer, recipient, [{"give": {"wood": 1}, "want": {"ore": 1}}], self.game.revision, "original")
        respond_to_trade(self.game, recipient, "counter", None, self.game.revision, "counter-choice")
        self.assertEqual(self.game.phase, "trade_counter_offer")
        self.assertEqual(self.game.trade_count, 1)
        self.assertEqual(self.game.activity_log[-1]["trade"]["outcome"], "counter_requested")
        propose_counter_trade(self.game, recipient, [{"give": {"brick": 1}, "want": {"sheep": 1}}], self.game.revision, "counter-offer")
        counter_event = self.game.activity_log[-1]["trade"]
        self.assertTrue(counter_event["is_counter"])
        self.assertEqual((counter_event["proposer_id"], counter_event["recipient_id"]), (recipient, proposer))
        respond_to_trade(self.game, proposer, "accept", 0, self.game.revision, "counter-accept")
        self.assertEqual(player_for(self.game, proposer).resources["brick"], 1)
        self.assertEqual(player_for(self.game, recipient).resources["sheep"], 1)

    def test_trade_rejects_short_cards_overlapping_resource_and_sixth_count(self):
        proposer = self.action_phase()
        recipient = next(pid for pid in self.game.seat_order if pid != proposer)
        player_for(self.game, proposer).resources["wood"] = 5
        with self.assertRaises(GameActionError):
            propose_trade(self.game, proposer, recipient, [{"give": {"wood": 1}, "want": {"wood": 1}}], self.game.revision, "same-resource")
        with self.assertRaises(GameActionError):
            propose_trade(self.game, proposer, recipient, [{"give": {"brick": 1}, "want": {"wood": 1}}], self.game.revision, "not-owned")
        for index in range(5):
            propose_trade(self.game, proposer, recipient, [{"give": {"wood": 1}, "want": {"ore": 1}}], self.game.revision, f"trade-{index}")
            respond_to_trade(self.game, recipient, "decline", None, self.game.revision, f"decline-{index}")
        with self.assertRaises(GameActionError):
            propose_trade(self.game, proposer, recipient, [{"give": {"wood": 1}, "want": {"ore": 1}}], self.game.revision, "trade-six")


if __name__ == "__main__":
    unittest.main()
