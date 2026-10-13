"""Investment-only Actor masks, semi-MDP boundaries, and L1/L2 gradient tests."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from app.rl.hierarchical_l3_shadow_step3b5 import L3DecisionFacts, ROUTE_ORDER
from app.rl.hierarchical_ppo_step3b6 import (
    FAMILY_ORDER, HierarchicalInvestmentCritic, HierarchicalInvestmentPolicy,
    investment_training_mask, macro_gae, masked_probabilities,
)
from app.rl.hierarchical_shadow_controller_step3b4 import HierarchicalRoute


class HierarchicalPpoStep3B6Tests(unittest.TestCase):
    def test_training_mask_excludes_protected_and_noninvestment(self):
        diagnostic = (False, False, True, True, True, True, True)
        executable = (False, False, True, True, True, False, True)
        expected = (False, False, True, True, True, False, True)
        self.assertEqual(investment_training_mask(
            HierarchicalRoute.INVESTMENT, diagnostic, executable), expected)
        self.assertEqual(investment_training_mask(
            HierarchicalRoute.CITY_VALUE_COMPARISON, diagnostic, executable),
            (False,) * len(FAMILY_ORDER))

    def test_masked_distribution_strict_zeros_and_all_invalid(self):
        logits = torch.randn(2, 7)
        mask = torch.tensor([[False, False, True, True, True, False, True],
                             [False] * 7])
        _, probs, entropy, available = masked_probabilities(logits, mask)
        self.assertEqual(available.tolist(), [True, False])
        self.assertTrue(torch.all(probs[~mask] == 0))
        self.assertAlmostEqual(float(probs[0].sum()), 1., places=6)
        self.assertEqual(float(probs[1].sum()), 0.)
        self.assertEqual(float(entropy[1]), 0.)

    def test_variable_duration_gae_and_terminal_bootstrap(self):
        rows = [
            {"done": False, "truncated": False, "macro_duration": 2,
             "reward_accumulated": .1, "value": .2, "next_actual_delegated": True,
             "start_policy_step": 4, "close_policy_step": 6},
            {"done": True, "truncated": False, "macro_duration": 3,
             "reward_accumulated": 1., "value": .3, "next_actual_delegated": False,
             "start_policy_step": 6, "close_policy_step": 9},
        ]
        advantages, returns = macro_gae(rows)
        self.assertAlmostEqual(float(advantages[1]), .7, places=6)
        expected = .1 + .99 ** 2 * .3 - .2 + .99 ** 2 * .95 * .7
        self.assertAlmostEqual(float(advantages[0]), expected, places=6)
        self.assertAlmostEqual(float(returns[0]), expected + .2, places=6)
        with self.assertRaises(ValueError):
            macro_gae(rows[:-1])

    def test_actor_gradient_reaches_plan_priority_but_not_frozen_latent(self):
        torch.manual_seed(41)
        policy = HierarchicalInvestmentPolicy(8)
        latent = torch.randn(1, 8, requires_grad=True)
        runtime = torch.zeros(1, 79)
        global_features = torch.zeros(1, 30)
        candidates = torch.zeros(1, 36, 9)
        candidates[0, 0, 0] = .1
        candidates[0, 1, 0] = .3
        candidates[0, :2, 6] = 1.
        plan_mask = torch.zeros(1, 36, dtype=torch.bool)
        plan_mask[0, :2] = True
        family_mask = torch.tensor([[False, False, True, True, False, False, False]])
        route = torch.zeros(1, len(ROUTE_ORDER))
        route[0, ROUTE_ORDER.index(HierarchicalRoute.INVESTMENT)] = 1.
        facts = L3DecisionFacts(route, family_mask, family_mask,
                                torch.zeros(1, 1), torch.ones(1, dtype=torch.bool))
        # The default zero L3 final layer blocks upstream gradients on the first
        # minibatch. Inject a nonzero test-only path to prove learnability later.
        with torch.no_grad():
            policy.l3_actor.encoder[0].weight.zero_()
            policy.l3_actor.encoder[0].bias.zero_()
            soft_offset = 8 + 79 + 30
            policy.l3_actor.encoder[0].weight[0, soft_offset] = .5
            policy.l3_actor.encoder[0].weight[0, soft_offset + 6] = .5
            policy.l3_actor.family_head.weight.zero_()
            policy.l3_actor.family_head.bias.zero_()
            policy.l3_actor.family_head.weight[2, 0] = 1.
            policy.l3_actor.family_head.weight[3, 0] = -1.
        plan, bridge, l3 = policy(latent, runtime, global_features,
                                  candidates, plan_mask, facts)
        self.assertTrue(bool(plan.plan_available[0]))
        self.assertAlmostEqual(float(l3.probabilities.detach().sum()), 1., places=6)
        self.assertTrue(torch.all(l3.probabilities[~family_mask] == 0))
        (-l3.probabilities[0, 2].log()).backward()
        self.assertIsNone(latent.grad)
        self.assertGreater(float(policy.l3_actor.family_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(policy.plan_heads.priority_head.output.weight.grad.abs().sum()), 0)
        self.assertGreater(float(policy.plan_heads.plan_scorer.output.weight.grad.abs().sum()), 0)
        critic = HierarchicalInvestmentCritic(8)
        self.assertFalse({id(p) for p in critic.parameters()} &
                         {id(p) for p in policy.parameters()})


if __name__ == "__main__":
    unittest.main()
