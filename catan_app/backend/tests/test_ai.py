import unittest

from fastapi.testclient import TestClient

from app.domain.ai import run_ai_until_pause
from app.domain.game import RESOURCES, _longest_road_length, create_game, game_view
from app.main import app, games


class AiPlayTests(unittest.TestCase):
    def test_ai_rolls_after_already_using_a_development_card(self):
        from app.domain.ai import play_ai_step
        game = create_game(42, ai_player_ids=(1, 2, 3, 4))
        game.phase = "turn_pre_roll"
        game.current_player_id = 1
        game.turn_number = 1
        game.used_development_this_turn = True
        game.players[0].development_cards = ["knight"]
        play_ai_step(game)
        self.assertIsNotNone(game.last_roll)
        self.assertEqual(game.players[0].development_cards, ["knight"])

    def test_four_ai_players_finish_a_rule_consistent_game(self):
        game = create_game(20260914, ai_player_ids=(1, 2, 3, 4))
        result = run_ai_until_pause(game, 100_000)

        self.assertEqual(result["status"], "game_over")
        self.assertEqual(game.phase, "game_over")
        self.assertIn(game.winner_id, (1, 2, 3, 4))
        self.assertLess(game.ai_action_number, 100_000)
        self.assertGreater(game.turn_number, 0)

        # 銀行と4人の手札を足すと、各資源は常に19枚。
        for resource in RESOURCES:
            self.assertEqual(game.bank[resource] + sum(player.resources[resource] for player in game.players), 19)

        buildings = set(game.settlements) | set(game.cities)
        self.assertEqual(len(buildings), len(game.settlements) + len(game.cities))
        for vertex_id in buildings:
            for edge_id in game.board.vertices[vertex_id].edge_ids:
                other = next(vertex for vertex in game.board.edges[edge_id].vertex_ids if vertex != vertex_id)
                self.assertNotIn(other, buildings, "開拓地・都市の距離ルール違反")

        for player in game.players:
            self.assertTrue(player.is_ai)
            self.assertEqual(player.settlements, sum(owner == player.id for owner in game.settlements.values()))
            self.assertEqual(player.cities, sum(owner == player.id for owner in game.cities.values()))
            self.assertLessEqual(player.settlements, 5)
            self.assertLessEqual(player.cities, 4)
            self.assertLessEqual(sum(owner == player.id for owner in game.roads.values()), 15)

        winner = game_view(game, game.winner_id)["players"][game.winner_id - 1]
        self.assertGreaterEqual(winner["score"], 10)
        if game.longest_road_holder_id is not None:
            self.assertGreaterEqual(_longest_road_length(game, game.longest_road_holder_id), 5)
        if game.largest_army_holder_id is not None:
            self.assertGreaterEqual(next(p.played_knights for p in game.players if p.id == game.largest_army_holder_id), 3)

        # 指定された交渉系統を、実際の対局中にすべて通す。
        self.assertTrue(any("候補で交渉" in message for message in game.ai_log))
        self.assertTrue(any("逆交渉を選択" in message for message in game.ai_log))
        self.assertTrue(any("を承諾" in message for message in game.ai_log))

    def test_three_ai_players_stop_when_human_input_is_needed(self):
        game = create_game(9, ai_player_ids=(2, 3, 4))
        result = run_ai_until_pause(game)
        self.assertEqual(result["status"], "waiting_for_human")
        self.assertEqual(game.phase, "rolling_order")
        self.assertEqual(game.pending_rollers[0], 1)
        self.assertEqual(game.ai_action_number, 0)

    def test_all_ai_api_creation_returns_completed_game(self):
        with TestClient(app) as client:
            response = client.post("/api/games", json={"seed": 20260914, "ai_player_ids": [1, 2, 3, 4]})
            self.assertEqual(response.status_code, 201, response.text)
            created = response.json()
            game_id = created["game"]["id"]
            try:
                self.assertEqual(created["ai_result"]["status"], "game_over")
                self.assertEqual(created["game"]["phase"], "game_over")
                self.assertGreaterEqual(created["game"]["ai"]["actions"], 1)
                self.assertIn(created["game"]["winner_id"], (1, 2, 3, 4))
            finally:
                games.pop(game_id, None)

    def test_watch_mode_advances_exactly_one_ai_action(self):
        with TestClient(app) as client:
            response = client.post("/api/games", json={
                "seed": 20260914, "ai_player_ids": [1, 2, 3, 4], "watch_ai": True})
            self.assertEqual(response.status_code, 201, response.text)
            created = response.json()
            game_id = created["game"]["id"]
            try:
                self.assertEqual(created["ai_result"]["status"], "watching")
                self.assertEqual(created["game"]["phase"], "rolling_order")
                self.assertEqual(created["game"]["ai"]["actions"], 0)
                self.assertTrue(created["game"]["ai"]["can_step"])
                stepped = client.post(f"/api/games/{game_id}/ai/run", json={"max_steps": 1})
                self.assertEqual(stepped.status_code, 200, stepped.text)
                self.assertEqual(stepped.json()["game"]["ai"]["actions"], 1)
                self.assertEqual(stepped.json()["result"]["steps"], 1)
            finally:
                games.pop(game_id, None)


if __name__ == "__main__":
    unittest.main()
