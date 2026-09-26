import unittest
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from app.main import app, games


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        response = self.client.post("/api/games", json={"seed": 42, "rules": {"desert_center": True}})
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.game_id = data["game"]["id"]
        self.tokens = {p["player_id"]: p["token"] for p in data["player_access"]}
        self.url = f"/api/games/{self.game_id}"

    def tearDown(self):
        games.pop(self.game_id, None)
        self.client.close()

    def headers(self, player_id):
        return {"X-Player-Token": self.tokens[player_id]}

    def test_views_and_private_cards(self):
        game = games[self.game_id]
        game.players[0].resources["wood"] = 4
        game.players[0].development_cards = ["victory_point", "knight"]
        for viewer in (None, 1, 2, 3, 4):
            response = self.client.get(self.url, headers=self.headers(viewer) if viewer else {})
            self.assertEqual(response.status_code, 200)
            view = response.json()
            self.assertEqual(view["viewer_id"], viewer)
            self.assertEqual(view["board"]["robber_tile_id"], 9)
            self.assertNotIn("player_access", view)
            self.assertNotIn("player_tokens", view)
            for player in view["players"]:
                self.assertEqual("resources" in player, player["id"] == viewer)
                self.assertEqual("development_cards" in player, player["id"] == viewer)
            self.assertEqual(view["players"][0]["resource_count"], 4)
            self.assertEqual(view["players"][0]["development_count"], 2)
        self.assertEqual(self.client.get(self.url, headers={"X-Player-Token": "wrong"}).status_code, 403)
        self.assertEqual(self.client.get("/api/games/missing").status_code, 404)

    def test_shared_state_auth_turns_and_idempotency(self):
        payload = {"expected_revision": 0, "request_id": "first-roll-request"}
        self.assertEqual(self.client.post(self.url + "/order-rolls", json=payload).status_code, 403)
        self.assertEqual(self.client.post(self.url + "/order-rolls", json=payload, headers=self.headers(2)).status_code, 409)
        first = self.client.post(self.url + "/order-rolls", json=payload, headers=self.headers(1))
        self.assertEqual(first.status_code, 200)
        repeated = self.client.post(self.url + "/order-rolls", json=payload, headers=self.headers(1))
        self.assertEqual(repeated.json()["rolls"], first.json()["rolls"])
        for pid in (1, 2, 3, 4):
            other_tab = self.client.get(self.url, headers=self.headers(pid)).json()
            self.assertEqual(other_tab["revision"], 1)
            self.assertEqual(other_tab["next_roller_id"], 2)
        # クライアントが出目を指定しても受け付けない。
        cheating = {"expected_revision": 1, "request_id": "forged-dice", "dice": [6, 6]}
        self.assertEqual(self.client.post(self.url + "/order-rolls", json=cheating, headers=self.headers(2)).status_code, 422)

    def test_simultaneous_requests_roll_only_once(self):
        def roll(index):
            return self.client.post(self.url + "/order-rolls", headers=self.headers(1), json={
                "expected_revision": 0, "request_id": f"concurrent-{index}"}).status_code
        with ThreadPoolExecutor(max_workers=2) as executor:
            statuses = list(executor.map(roll, (1, 2)))
        self.assertEqual(sorted(statuses), [200, 409])
        self.assertEqual(len(games[self.game_id].rolls), 1)

    def test_cors_and_invalid_input(self):
        for origin in ("http://localhost:3002", "http://192.168.0.128:3002"):
            with self.subTest(origin=origin):
                response = self.client.options(self.url, headers={
                    "Origin": origin, "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "X-Player-Token"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["access-control-allow-origin"], origin)
        self.assertEqual(self.client.post("/api/games", json={"seed": -1}).status_code, 422)

    def test_player_can_switch_own_controller_between_ai_and_human(self):
        game = games[self.game_id]
        game.ai_watch_mode = True
        response = self.client.post(self.url + "/players/1/controller", headers=self.headers(1), json={
            "expected_revision": game.revision, "request_id": "controller-to-ai", "is_ai": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["players"][0]["is_ai"])
        revision = response.json()["revision"]

        forbidden = self.client.post(self.url + "/players/1/controller", headers=self.headers(2), json={
            "expected_revision": revision, "request_id": "wrong-controller", "is_ai": False})
        self.assertEqual(forbidden.status_code, 403)

        response = self.client.post(self.url + "/players/1/controller", headers=self.headers(1), json={
            "expected_revision": revision, "request_id": "controller-to-human", "is_ai": False})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["players"][0]["is_ai"])

    def test_player_trade_and_counter_trade_api_use_domain_offers(self):
        game = games[self.game_id]
        game.phase = "action"
        game.current_player_id = 1
        game.players[0].resources["wood"] = 1
        game.players[1].resources["brick"] = 1

        proposal = self.client.post(
            self.url + "/player-trades", headers=self.headers(1), json={
                "expected_revision": 0,
                "request_id": "trade-proposal-1",
                "target_id": 2,
                "offers": [{"give": {"wood": 1}, "want": {"brick": 1}}],
            },
        )
        self.assertEqual(proposal.status_code, 200)
        self.assertEqual(proposal.json()["phase"], "trade_response")
        self.assertEqual(proposal.json()["pending_trade"]["offers"][0]["give"]["wood"], 1)

        response = self.client.post(
            self.url + "/player-trades/respond", headers=self.headers(2), json={
                "expected_revision": 1,
                "request_id": "trade-response-1",
                "decision": "counter",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["phase"], "trade_counter_offer")

        counter = self.client.post(
            self.url + "/player-trades/counter", headers=self.headers(2), json={
                "expected_revision": 2,
                "request_id": "trade-counter-1",
                "offers": [{"give": {"brick": 1}, "want": {"wood": 1}}],
            },
        )
        self.assertEqual(counter.status_code, 200)
        self.assertEqual(counter.json()["phase"], "trade_response")
        self.assertTrue(counter.json()["pending_trade"]["is_counter"])


if __name__ == "__main__":
    unittest.main()
