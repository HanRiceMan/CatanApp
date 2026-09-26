from copy import deepcopy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.domain.game import create_game, player_for
from app.domain.actions import RollOrderAction
from app.rl.env import CatanEnv, CatanEnvConfig
from app.rl.model_registry import ModelManifest
from app.rl.observation import (OBSERVATION_VECTOR_SIZES, RESOURCE_ORDER,
                                encode_observation, get_observation)


class ObservationV2Tests(unittest.TestCase):
    def test_v1_is_preserved_and_v2_is_fixed_normalized_size(self):
        game = create_game(42)
        v1 = encode_observation(get_observation(game, 1))
        v2_observation = get_observation(game, 1, version="v2")
        v2 = encode_observation(v2_observation)
        self.assertEqual(v1.shape, (1343,))
        self.assertEqual(v2.shape, (1667,))
        self.assertEqual(OBSERVATION_VECTOR_SIZES["v2"] - OBSERVATION_VECTOR_SIZES["v1"], 54 * 6)
        self.assertTrue(np.isfinite(v2).all())
        self.assertGreaterEqual(float(v2.min()), 0)
        self.assertLessEqual(float(v2.max()), 1)
        for vertex in v2_observation.vertices:
            expected = tuple(sum((6 - abs(7 - tile.number)) if tile.number else 0
                                 for tile_id in game.board.vertices[vertex.id].hex_ids
                                 for tile in (game.board.tiles[tile_id],) if tile.terrain == resource)
                             for resource in RESOURCE_ORDER)
            self.assertEqual(vertex.production_pips, expected)

    def test_v2_does_not_reveal_opponent_private_card_types(self):
        game = create_game(42)
        opponent = player_for(game, 2)
        opponent.resources.update(wood=1, brick=0)
        opponent.development_cards = ["knight"]
        first = encode_observation(get_observation(game, 1, version="v2"))
        opponent.resources.update(wood=0, brick=1)
        opponent.development_cards = ["victory_point"]
        second = encode_observation(get_observation(game, 1, version="v2"))
        np.testing.assert_array_equal(first, second)

    def test_env_and_registry_support_both_versions_but_reject_bad_size(self):
        for version, size in OBSERVATION_VECTOR_SIZES.items():
            env = CatanEnv(CatanEnvConfig(observation_version=version))
            try:
                observation, _ = env.reset(seed=42)
                self.assertEqual(observation.shape, (size,))
                self.assertEqual(env.observation_space.shape, (size,))
            finally:
                env.close()
            manifest = ModelManifest("sample", "MaskablePPO", "model.zip", version, size,
                                     "v1", 377, 100, 42, "2026-09-22", {})
            self.assertTrue(manifest.is_runtime_compatible)
            bad = deepcopy(manifest)
            object.__setattr__(bad, "observation_size", size + 1)
            self.assertFalse(bad.is_runtime_compatible)

    @unittest.skipUnless(importlib.util.find_spec("sb3_contrib"), "PPO runtime required")
    def test_v2_model_trains_saves_loads_and_selects_action(self):
        from app.agents.ppo import PPOAgent
        from app.rl.train import TrainConfig, train_maskable_ppo

        with tempfile.TemporaryDirectory() as directory:
            result = train_maskable_ppo(TrainConfig(
                model_id="v2_smoke", observation_version="v2", total_timesteps=32,
                seed=88, n_steps=32, batch_size=32, n_epochs=1,
                policy_net_arch=(32, 32), checkpoint_interval_steps=32,
                evaluation_seeds=(706,), device="cpu", models_root=Path(directory),
            ))
            self.assertEqual(result.manifest.observation_version, "v2")
            self.assertEqual(result.manifest.observation_size, 1667)
            self.assertEqual(result.evaluation.illegal_action_count, 0)
            agent = PPOAgent("v2_smoke", models_root=Path(directory), device="cpu")
            self.assertIsInstance(agent.select_action(create_game(88), 1), RollOrderAction)


if __name__ == "__main__":
    unittest.main()
