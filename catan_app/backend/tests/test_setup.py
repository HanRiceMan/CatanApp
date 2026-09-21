import unittest
from collections import Counter
from copy import deepcopy
from random import Random

from fastapi.testclient import TestClient

from app.domain.game import (GameActionError, create_game, game_view, legal_road_ids,
                             legal_settlement_ids, place_initial_piece, roll_for_order, setup_player_id)
from app.main import app, games


def finish_order(game):
    for pid, dice in [(1, (3, 3)), (2, (6, 6)), (3, (4, 4)), (4, (1, 1))]:
        roll_for_order(game, pid, game.revision, f"order-{pid}", dice)


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.game = create_game(42)
        finish_order(self.game)

    def place(self, kind, target, pid=None):
        place_initial_piece(self.game, pid or setup_player_id(self.game), kind, target,
                            self.game.revision, f"place-{self.game.revision}")

    def test_eight_pairs_resources_scores_bank_and_fixed_seats(self):
        for seed in range(40):
            game = create_game(seed)
            finish_order(game)
            rng = Random(seed)
            initial_seats = list(game.seat_order)
            expected_hands = {pid: Counter() for pid in initial_seats}
            for index, pid in enumerate(game.setup_order):
                self.assertEqual(setup_player_id(game), pid)
                self.assertEqual(game.setup_index, index)
                vertex_id = rng.choice(legal_settlement_ids(game))
                if index >= 4:
                    expected_hands[pid].update(game.board.tiles[t].terrain for t in game.board.vertices[vertex_id].hex_ids
                                               if game.board.tiles[t].terrain != "desert")
                place_initial_piece(game, pid, "settlement", vertex_id, game.revision, f"settlement-{index}")
                self.assertEqual(game.setup_index, index)  # 道を置くまで手番は移らない
                self.assertEqual(setup_player_id(game), pid)
                self.assertEqual(legal_settlement_ids(game), [])
                for edge_id in legal_road_ids(game):
                    self.assertIn(vertex_id, game.board.edges[edge_id].vertex_ids)
                road_id = rng.choice(legal_road_ids(game))
                place_initial_piece(game, pid, "road", road_id, game.revision, f"road-{index}")
                self.assertEqual(game.seat_order, initial_seats)
                for player in game.players:
                    self.assertEqual(Counter({k: v for k, v in player.resources.items() if v}), expected_hands[player.id])
            self.assertEqual(game.phase, "ready_to_play")
            self.assertEqual(len(game.settlements), 8)
            self.assertEqual(len(game.roads), 8)
            self.assertEqual(legal_settlement_ids(game), [])
            self.assertEqual(legal_road_ids(game), [])
            self.assertIsNone(setup_player_id(game))
            self.assertEqual(game_view(game)["first_player_id"], 2)
            for player in game.players:
                self.assertEqual(player.settlements, 2)
                self.assertEqual(sum(owner == player.id for owner in game.roads.values()), 2)
                self.assertEqual(game_view(game, player.id)["players"][player.id - 1]["score"], 2)
            for resource in game.bank:
                self.assertEqual(game.bank[resource] + sum(p.resources[resource] for p in game.players), 19)
            for edge in game.board.edges:
                self.assertFalse(all(v in game.settlements for v in edge.vertex_ids))
            with self.assertRaises(GameActionError):
                place_initial_piece(game, 2, "settlement", 0, game.revision, "after-completion")

    def test_invalid_actions_leave_state_unchanged(self):
        for kind, target, player in [("road", 0, 2), ("settlement", -1, 2),
                                     ("settlement", 54, 2), ("settlement", 0, 1), ("city", 0, 2)]:
            before = deepcopy(self.game)
            with self.assertRaises(GameActionError):
                self.place(kind, target, player)
            self.assertEqual(self.game, before)
        self.place("settlement", 0)
        before = deepcopy(self.game)
        for kind, target in [("settlement", 2), ("road", 71)]:
            with self.assertRaises(GameActionError):
                self.place(kind, target)
            self.assertEqual(self.game, before)

    def test_occupied_and_distance_rule(self):
        self.place("settlement", 0)
        self.place("road", legal_road_ids(self.game)[0])
        for vertex in (0, 3, 4):
            self.assertNotIn(vertex, legal_settlement_ids(self.game))
            with self.assertRaises(GameActionError):
                self.place("settlement", vertex)

    def test_road_must_connect_to_latest_settlement(self):
        first_vertex = None
        for index in range(4):
            vertex = legal_settlement_ids(self.game)[0]
            self.place("settlement", vertex)
            if index == 3:
                first_vertex = vertex
            self.place("road", legal_road_ids(self.game)[0])
        # 4番手は続けて2組目。ただし、以前の開拓地に道を足すことはできない。
        self.place("settlement", legal_settlement_ids(self.game)[0])
        old_edge = next(e for e in self.game.board.vertices[first_vertex].edge_ids if e not in self.game.roads)
        self.assertNotIn(old_edge, legal_road_ids(self.game))
        with self.assertRaises(GameActionError):
            self.place("road", old_edge)

    def test_idempotency_stale_revision_and_action_identity(self):
        rev = self.game.revision
        place_initial_piece(self.game, 2, "settlement", 0, rev, "same-request")
        after = deepcopy(self.game)
        place_initial_piece(self.game, 2, "settlement", 0, rev, "same-request")
        self.assertEqual(self.game, after)
        for target, revision, key in [(2, rev, "same-request"), (0, rev, "stale-request")]:
            with self.assertRaises(GameActionError):
                place_initial_piece(self.game, 2, "road", target, revision, key)
        with self.assertRaises(GameActionError):
            roll_for_order(self.game, 2, rev, "same-request")
        self.assertEqual(self.game, after)

    def test_only_current_player_gets_placement_targets(self):
        for viewer in (None, 1, 2, 3, 4):
            view = game_view(self.game, viewer)
            self.assertEqual(len(view["initial_placement"]["legal_vertex_ids"]), 54 if viewer == 2 else 0)
            self.assertEqual(view["initial_placement"]["player_id"], 2)


class SetupApiTests(unittest.TestCase):
    def test_authenticated_placement_sync_and_completion(self):
        with TestClient(app) as client:
            created = client.post("/api/games", json={"seed": 42}).json()
            game_id = created["game"]["id"]
            try:
                finish_order(games[game_id])
                headers = {a["player_id"]: {"X-Player-Token": a["token"]} for a in created["player_access"]}
                url = f"/api/games/{game_id}"
                payload = {"kind": "settlement", "target_id": 0, "expected_revision": 4, "request_id": "api-placement"}
                self.assertEqual(client.post(url + "/initial-placements", json=payload).status_code, 403)
                self.assertEqual(client.post(url + "/initial-placements", json=payload, headers=headers[1]).status_code, 409)
                self.assertEqual(client.post(url + "/initial-placements", json={**payload, "target_id": -1}, headers=headers[2]).status_code, 422)
                response = client.post(url + "/initial-placements", json=payload, headers=headers[2])
                self.assertEqual(response.status_code, 200)
                repeated = client.post(url + "/initial-placements", json=payload, headers=headers[2])
                self.assertEqual(repeated.json()["revision"], 5)
                self.assertEqual(repeated.json()["settlements"], [{"vertex_id": 0, "player_id": 2}])
                for i in range(15):
                    game = games[game_id]
                    pid = setup_player_id(game)
                    view = client.get(url, headers=headers[pid]).json()
                    placement = view["initial_placement"]
                    target = (placement["legal_vertex_ids"] or placement["legal_edge_ids"])[0]
                    response = client.post(url + "/initial-placements", headers=headers[pid], json={
                        "kind": placement["piece"], "target_id": target,
                        "expected_revision": view["revision"], "request_id": f"api-placement-{i}"})
                    self.assertEqual(response.status_code, 200, response.text)
                for pid in (1, 2, 3, 4):
                    view = client.get(url, headers=headers[pid]).json()
                    self.assertEqual(view["phase"], "ready_to_play")
                    self.assertEqual(len(view["settlements"]), 8)
                    self.assertEqual(len(view["roads"]), 8)
                    for p in view["players"]:
                        self.assertEqual("resources" in p, p["id"] == pid)
                        self.assertEqual(p["public_score"], 2)
            finally:
                games.pop(game_id, None)


if __name__ == "__main__":
    unittest.main()
