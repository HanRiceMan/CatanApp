import json
import tempfile
import unittest
from pathlib import Path

from app.agents.random_agent import RandomAgent
from app.domain.actions import (BuildCityAction, BuildRoadAction, BuildSettlementAction,
                                BuyDevelopmentAction, UseDevelopmentAction)
from app.rl.evaluate import _action_name, evaluate_agent
from app.rl.model_registry import (ACTION_SPACE_SIZE, ACTION_SPACE_VERSION, ModelRegistry,
                                   OBSERVATION_VECTOR_SIZE, OBSERVATION_VERSION)


class RlPhaseThreeTests(unittest.TestCase):
    def test_random_agent_completes_seeded_evaluation_without_illegal_actions(self):
        report = evaluate_agent(RandomAgent(), seeds=(101, 102, 103))

        self.assertEqual(len(report.episodes), 3)
        self.assertEqual(report.illegal_action_count, 0)
        self.assertTrue(all(not episode.truncated for episode in report.episodes))
        self.assertTrue(all(episode.winner_id in range(1, 5) for episode in report.episodes))
        self.assertEqual(report.to_dict()["summary"]["illegal_action_count"], 0)
        for episode in report.episodes:
            self.assertEqual(sum(episode.learner_actions.values()), episode.policy_steps)
            self.assertEqual(episode.learner_actions["initial_settlement"], 2)
            self.assertEqual(episode.learner_actions["initial_road"], 2)
            self.assertEqual(set(episode.learner_final_pieces), {"roads", "settlements", "cities"})
            self.assertEqual(len(episode.learner_initial_vertices), 2)
            self.assertGreater(episode.learner_initial_pips, 0)
            self.assertGreaterEqual(episode.learner_initial_resource_kinds, 1)
            self.assertLessEqual(episode.learner_initial_resource_kinds, 5)

    def test_evaluation_distinguishes_construction_and_development_actions(self):
        self.assertEqual(_action_name(BuildRoadAction(0)), "build_road_paid")
        self.assertEqual(_action_name(BuildRoadAction(0), free_road=True), "build_road_free")
        self.assertEqual(_action_name(BuildSettlementAction(0)), "build_settlement")
        self.assertEqual(_action_name(BuildCityAction(0)), "build_city")
        self.assertEqual(_action_name(BuyDevelopmentAction()), "buy_development")
        self.assertEqual(_action_name(UseDevelopmentAction("knight")), "use_development_knight")

    def test_model_registry_accepts_only_complete_compatible_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "ppo_v001"
            model.mkdir()
            (model / "model.zip").write_bytes(b"placeholder")
            (model / "metadata.json").write_text(json.dumps({
                "model_id": "ppo_v001",
                "algorithm": "MaskablePPO",
                "artifact_filename": "model.zip",
                "observation_version": OBSERVATION_VERSION,
                "observation_size": OBSERVATION_VECTOR_SIZE,
                "action_space_version": ACTION_SPACE_VERSION,
                "action_space_size": ACTION_SPACE_SIZE,
                "training_steps": 1000,
                "training_seed": 42,
                "created_at": "2026-09-21T00:00:00Z",
                "evaluation": {"win_rate": 0.25},
            }), encoding="utf-8")
            registry = ModelRegistry(root)

            manifest = registry.load("ppo_v001")

            self.assertTrue(manifest.is_runtime_compatible)
            self.assertEqual([item.model_id for item in registry.list_available()], ["ppo_v001"])
            (model / "model.zip").unlink()
            self.assertEqual(registry.list_available(), ())


if __name__ == "__main__":
    unittest.main()
