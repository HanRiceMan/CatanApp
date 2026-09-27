import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from app.agents.random_agent import RandomAgent
from app.agents.runner import get_agent_for_player, set_agent_for_player
from app.domain.game import create_game
from app.main import app, games
from app.rl.action_space import ACTION_SPACE_SIZE, ACTION_SPACE_VERSION
from app.rl.model_registry import ModelManifest, ModelRegistry
from app.rl.observation import OBSERVATION_VECTOR_SIZES


class RlPhaseFiveTests(unittest.TestCase):
    def setUp(self):
        games.clear()
        self.client = TestClient(app)

    def tearDown(self):
        games.clear()

    def test_agent_name_selects_random_agent_without_changing_game_rules(self):
        game = create_game(51, ai_agent_names={2: "random"})

        self.assertTrue(game.players[1].is_ai)
        self.assertEqual(game.players[1].agent_name, "random")
        self.assertIsInstance(get_agent_for_player(game, 2), RandomAgent)
        set_agent_for_player(game, 2, "heuristic")
        self.assertEqual(game.players[1].agent_name, "heuristic")

    def test_model_catalog_and_controller_api_keep_legacy_payloads_working(self):
        catalog = self.client.get("/api/ai-models")
        self.assertEqual(catalog.status_code, 200)
        self.assertIn("models", catalog.json())
        self.assertIn("ppo_runtime_available", catalog.json())
        planned = next(model for model in catalog.json()["models"]
                       if model["model_id"] == "ppo_gnn_board_65k_s03_exp_v003")
        self.assertEqual(planned["display_name"], "現行最強AI")
        self.assertTrue(planned["ui_planned_only"])
        self.assertEqual(planned["settlement_planning"]["target_sites"], 5)
        self.assertEqual(planned["settlement_planning"]["title_horizon"], 2)
        self.assertEqual(planned["settlement_planning"]["first_city_max_deficit"], 2)
        self.assertTrue(planned["settlement_planning"]["prioritize_immediate_win"])
        self.assertTrue(planned["settlement_planning"]["expansion_aware_setup"])
        self.assertEqual(planned["settlement_planning"]["endgame_point_reserve_score"], 8)
        self.assertTrue(planned["settlement_planning"]["prioritize_immediate_titles"])
        robber_candidate = next(
            model for model in catalog.json()["models"]
            if model["model_id"] == "ppo_robber_value_73k_s01_exp_v001"
        )
        self.assertEqual(robber_candidate["display_name"], "盗賊PPO移行AI（実験版）")
        self.assertTrue(robber_candidate["ui_planned_only"])
        self.assertEqual(
            robber_candidate["policy_architecture"],
            "robber_belief_gnn_family_hierarchical_candidate",
        )
        self.assertFalse(
            robber_candidate["settlement_planning"]["belief_aware_robber"]
        )

        created = self.client.post("/api/games", json={"seed": 52, "ai_agents": {"2": "random"}})
        self.assertEqual(created.status_code, 201)
        payload = created.json()
        game = payload["game"]
        self.assertEqual(next(player for player in game["players"] if player["id"] == 2)["agent_name"], "random")
        player_token = next(item["token"] for item in payload["player_access"] if item["player_id"] == 2)

        changed = self.client.post(
            f"/api/games/{game['id']}/players/2/controller",
            headers={"X-Player-Token": player_token},
            json={"is_ai": True, "agent_name": "heuristic", "expected_revision": game["revision"],
                  "request_id": "phase-five-controller-001"},
        )
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(next(player for player in changed.json()["players"] if player["id"] == 2)["agent_name"], "heuristic")

    def test_registry_requires_declared_gnn_setup_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "gnn_model"
            model.mkdir()
            (model / "model.zip").write_bytes(b"model")
            manifest = ModelManifest(
                model_id="gnn_model", algorithm="MaskablePPO", artifact_filename="model.zip",
                observation_version="v2", observation_size=OBSERVATION_VECTOR_SIZES["v2"],
                action_space_version=ACTION_SPACE_VERSION, action_space_size=ACTION_SPACE_SIZE,
                training_steps=1, training_seed=1, created_at="2026-09-23", evaluation={},
                initial_setup={"type": "gnn", "artifact_filename": "gnn_setup_policy.pt"},
            )
            (model / "metadata.json").write_text(
                json.dumps(manifest.to_dict()), encoding="utf-8"
            )
            registry = ModelRegistry(root)
            with self.assertRaisesRegex(ValueError, "初期配置モデル本体"):
                registry.load("gnn_model")
            (model / "gnn_setup_policy.pt").write_bytes(b"setup")
            self.assertEqual(registry.load("gnn_model").initial_setup["type"], "gnn")


if __name__ == "__main__":
    unittest.main()
