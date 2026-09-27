import unittest

import numpy as np
import torch
from gymnasium import spaces

from app.domain.game import create_game
from app.rl.belief_context import BELIEF_CONTEXT_SIZE, encode_belief_context
from app.rl.candidate_policy import (
    BeliefGraphFamilyHierarchicalCandidateMaskablePolicy,
    ContextualBeliefGraphFamilyHierarchicalCandidateMaskablePolicy,
    GraphHierarchicalCandidateMaskablePolicy,
    RobberBeliefGraphFamilyHierarchicalCandidateMaskablePolicy,
    initialize_belief_policy_from_graph,
)
from app.rl.observation import encode_observation, get_observation
from app.rl.action_space import ACTION_SPACE_SIZE
from app.rl.train_strongest_league import robber_soft_targets


def _stolen_game(private_resource: str):
    game = create_game(20260928)
    game.resource_events = [
        {"deltas": {"2": {"wood": 1, "ore": 1}}},
        {"hidden_transfer": {
            "from_player_id": 2, "to_player_id": 1, "count": 1,
            "private_resource": private_resource, "known_to": [1, 2],
        }},
    ]
    return game


class BeliefContextTests(unittest.TestCase):
    def test_robber_soft_targets_preserve_scores_and_mask_illegal_actions(self):
        scores = np.full((2, ACTION_SPACE_SIZE), -np.inf, dtype=np.float32)
        scores[0, 10:13] = (0.0, 4.0, 8.0)
        scores[1, 20:22] = (3.0, 3.0)

        targets = robber_soft_targets(scores, temperature=4.0)

        np.testing.assert_allclose(targets.sum(axis=1), 1.0, atol=1e-7)
        self.assertEqual(float(targets[0, 0]), 0.0)
        self.assertGreater(targets[0, 12], targets[0, 11])
        self.assertGreater(targets[0, 11], targets[0, 10])
        self.assertAlmostEqual(float(targets[1, 20]), 0.5)
        self.assertAlmostEqual(float(targets[1, 21]), 0.5)

    def test_lower_soft_target_temperature_emphasizes_best_candidate(self):
        scores = np.full((1, ACTION_SPACE_SIZE), -np.inf, dtype=np.float32)
        scores[0, 10:13] = (0.0, 4.0, 8.0)

        warm = robber_soft_targets(scores, temperature=4.0)
        cold = robber_soft_targets(scores, temperature=1.0)

        self.assertGreater(cold[0, 12], warm[0, 12])

    def test_v4_appends_normalized_bayesian_context(self):
        encoded = encode_observation(
            get_observation(_stolen_game("wood"), 3, version="v4")
        )

        self.assertEqual(encoded.shape, (1718,))
        self.assertEqual(encoded[1667:].shape, (BELIEF_CONTEXT_SIZE,))
        self.assertTrue(np.isfinite(encoded).all())
        self.assertGreaterEqual(float(encoded.min()), 0.0)
        self.assertLessEqual(float(encoded.max()), 1.0)

    def test_v5_adds_absolute_self_seat_for_victim_action_mapping(self):
        encoded = encode_observation(
            get_observation(_stolen_game("wood"), 3, version="v5")
        )

        self.assertEqual(encoded.shape, (1722,))
        np.testing.assert_array_equal(
            encoded[-4:], np.asarray((0, 0, 1, 0), dtype=np.float32)
        )

    def test_third_party_context_does_not_leak_private_resource(self):
        wood = encode_belief_context(_stolen_game("wood"), 3)
        ore = encode_belief_context(_stolen_game("ore"), 3)

        np.testing.assert_array_equal(wood, ore)

    def test_robber_party_can_use_resource_it_legitimately_saw(self):
        wood = encode_belief_context(_stolen_game("wood"), 1)
        ore = encode_belief_context(_stolen_game("ore"), 1)

        self.assertFalse(np.array_equal(wood, ore))

    def test_belief_policy_starts_with_exact_source_logits(self):
        action_space = spaces.Discrete(377)
        source = GraphHierarchicalCandidateMaskablePolicy(
            spaces.Box(0, 1, shape=(1667,), dtype=np.float32), action_space,
            lambda _: 1e-4, net_arch=[], ortho_init=False,
        )
        target = BeliefGraphFamilyHierarchicalCandidateMaskablePolicy(
            spaces.Box(0, 1, shape=(1718,), dtype=np.float32), action_space,
            lambda _: 1e-4, net_arch=[], ortho_init=False,
        )
        initialize_belief_policy_from_graph(target, source)
        old = torch.rand((2, 1667))
        belief = torch.rand((2, BELIEF_CONTEXT_SIZE))
        with torch.no_grad():
            source_logits = source.mlp_extractor.forward_actor(old)
            target_logits = target.mlp_extractor.forward_actor(
                torch.cat((old, belief), dim=1)
            )
        torch.testing.assert_close(source_logits, target_logits, rtol=0, atol=1e-7)

    def test_contextual_belief_policy_starts_with_exact_source_logits(self):
        action_space = spaces.Discrete(377)
        source = GraphHierarchicalCandidateMaskablePolicy(
            spaces.Box(0, 1, shape=(1667,), dtype=np.float32), action_space,
            lambda _: 1e-4, net_arch=[], ortho_init=False,
        )
        target = ContextualBeliefGraphFamilyHierarchicalCandidateMaskablePolicy(
            spaces.Box(0, 1, shape=(1718,), dtype=np.float32), action_space,
            lambda _: 1e-4, net_arch=[], ortho_init=False,
        )
        initialize_belief_policy_from_graph(target, source)
        old = torch.rand((2, 1667))
        belief = torch.rand((2, BELIEF_CONTEXT_SIZE))
        with torch.no_grad():
            source_logits = source.mlp_extractor.forward_actor(old)
            target_logits = target.mlp_extractor.forward_actor(
                torch.cat((old, belief), dim=1)
            )
        torch.testing.assert_close(source_logits, target_logits, rtol=0, atol=1e-7)

    def test_robber_policy_changes_only_robber_and_victim_candidates(self):
        action_space = spaces.Discrete(377)
        source = GraphHierarchicalCandidateMaskablePolicy(
            spaces.Box(0, 1, shape=(1667,), dtype=np.float32), action_space,
            lambda _: 1e-4, net_arch=[], ortho_init=False,
        )
        target = RobberBeliefGraphFamilyHierarchicalCandidateMaskablePolicy(
            spaces.Box(0, 1, shape=(1722,), dtype=np.float32), action_space,
            lambda _: 1e-4, net_arch=[], ortho_init=False,
        )
        initialize_belief_policy_from_graph(target, source)
        old = torch.rand((2, 1667))
        belief = torch.rand((2, BELIEF_CONTEXT_SIZE))
        seats = torch.eye(4)[:2]
        extended = torch.cat((old, belief, seats), dim=1)
        with torch.no_grad():
            baseline = target.mlp_extractor.forward_actor(extended)
            target.mlp_extractor.robber_tile_head[-1].bias.fill_(1.0)
            target.mlp_extractor.robber_victim_head[-1].bias.fill_(1.0)
            changed = target.mlp_extractor.forward_actor(extended)
        difference = changed - baseline
        allowed = torch.zeros_like(difference, dtype=torch.bool)
        extractor = target.mlp_extractor
        allowed[:, extractor.ROBBER_START:extractor.ROBBER_START + extractor.ROBBER_COUNT] = True
        allowed[:, extractor.VICTIM_START:extractor.VICTIM_START + extractor.VICTIM_COUNT] = True
        torch.testing.assert_close(
            difference[allowed], torch.ones_like(difference[allowed]),
            rtol=0, atol=1e-6,
        )
        torch.testing.assert_close(
            difference[~allowed], torch.zeros_like(difference[~allowed]),
            rtol=0, atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
