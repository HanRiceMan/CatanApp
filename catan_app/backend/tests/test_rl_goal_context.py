import unittest

import numpy as np
import torch
from gymnasium import spaces

from app.domain.game import create_game, player_for
from app.rl.goal_context import GOAL_CONTEXT_SIZE, encode_goal_context
from app.rl.observation import encode_observation, get_observation
from app.rl.candidate_policy import (
    GoalGraphFamilyHierarchicalCandidateMaskablePolicy,
    GraphHierarchicalCandidateMaskablePolicy,
    initialize_goal_policy_from_graph,
)


class GoalContextTests(unittest.TestCase):
    def test_v3_appends_normalized_planner_context(self):
        game = create_game(42)
        encoded = encode_observation(get_observation(game, 1, version="v3"))

        self.assertEqual(encoded.shape, (1822,))
        self.assertEqual(encoded[1667:].shape, (GOAL_CONTEXT_SIZE,))
        self.assertTrue(np.isfinite(encoded).all())
        self.assertGreaterEqual(float(encoded.min()), 0.0)
        self.assertLessEqual(float(encoded.max()), 1.0)

    def test_goal_context_reacts_to_own_hand_without_mutating_game(self):
        game = create_game(43)
        player = player_for(game, 1)
        game.settlements[0] = 1
        player.settlements = 1
        before_revision = game.revision
        first = encode_goal_context(game, 1)
        player.resources.update(wheat=2, ore=3)
        second = encode_goal_context(game, 1)

        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(game.revision, before_revision)

    def test_goal_policy_starts_with_exact_source_logits(self):
        action_space = spaces.Discrete(377)
        source = GraphHierarchicalCandidateMaskablePolicy(
            spaces.Box(0, 1, shape=(1667,), dtype=np.float32), action_space,
            lambda _: 1e-4, net_arch=[], ortho_init=False,
        )
        target = GoalGraphFamilyHierarchicalCandidateMaskablePolicy(
            spaces.Box(0, 1, shape=(1822,), dtype=np.float32), action_space,
            lambda _: 1e-4, net_arch=[], ortho_init=False,
        )
        initialize_goal_policy_from_graph(target, source)
        old = torch.rand((2, 1667))
        goal = torch.rand((2, GOAL_CONTEXT_SIZE))
        with torch.no_grad():
            source_logits = source.mlp_extractor.forward_actor(old)
            target_logits = target.mlp_extractor.forward_actor(
                torch.cat((old, goal), dim=1)
            )
        torch.testing.assert_close(source_logits, target_logits, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
