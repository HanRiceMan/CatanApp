"""Multi-choice delegation and diagnostic-only L1/L2 ablations."""

from __future__ import annotations

import unittest

import torch

from app.rl.hierarchical_l3_shadow_step3b5 import L3DecisionFacts, ROUTE_ORDER
from app.rl.hierarchical_ppo_step3b6 import HierarchicalInvestmentPolicy
from app.rl.hierarchical_rollout_step3b6 import _delegation_candidate
from app.rl.hierarchical_shadow_controller_step3b4 import HierarchicalRoute
from app.rl.run_hierarchical_ppo_step3b7 import (
    EVAL_BASES, TRAIN_BASES, _l3_with_ablation,
)
from app.rl.run_hierarchical_ppo_step3b6 import (
    EVAL_BASES as PRIOR_EVAL_BASES, TRAIN_BASES as PRIOR_TRAIN_BASES,
)


class HierarchicalPpoStep3B7Tests(unittest.TestCase):
    def test_only_multi_choice_enters_new_actor_buffer(self):
        self.assertFalse(_delegation_candidate((False,) * 7, True))
        self.assertFalse(_delegation_candidate((False,) * 6 + (True,), True))
        self.assertTrue(_delegation_candidate((False, False, True, True,
                                               False, False, False), True))
        self.assertTrue(_delegation_candidate((False,) * 6 + (True,), False))

    def test_training_evaluation_and_prior_seeds_do_not_overlap(self):
        self.assertFalse(set(TRAIN_BASES.values()) & set(EVAL_BASES.values()))
        self.assertTrue(all(abs(a - b) >= 1000 for a in TRAIN_BASES.values()
                            for b in EVAL_BASES.values()))
        prior = list(PRIOR_TRAIN_BASES.values()) + list(PRIOR_EVAL_BASES.values())
        self.assertTrue(all(abs(a - b) >= 1000 for a in TRAIN_BASES.values()
                            for b in prior))
        self.assertTrue(all(abs(a - b) >= 1000 for a in EVAL_BASES.values()
                            for b in prior))

    def test_ablation_changes_only_shadow_representation(self):
        torch.manual_seed(77)
        policy = HierarchicalInvestmentPolicy(8)
        with torch.no_grad():
            policy.plan_heads.plan_scorer.output.bias.fill_(.1)
            policy.plan_heads.priority_head.output.bias.copy_(
                torch.tensor([.4, -.3, .2, -.1]))
            policy.l3_actor.family_head.weight.normal_(0., .01)
        mask = torch.zeros(2, 36, dtype=torch.bool)
        mask[:, :2] = True
        candidates = torch.zeros(2, 36, 9)
        candidates[:, 0, 0] = .1
        candidates[:, 1, 0] = .3
        candidates[:, :2, 6] = 1.
        family_mask = torch.tensor([[False, False, True, True, False, False, True]] * 2)
        route = torch.zeros(2, len(ROUTE_ORDER))
        route[:, ROUTE_ORDER.index(HierarchicalRoute.INVESTMENT)] = 1.
        facts = L3DecisionFacts(route, family_mask, family_mask,
                                torch.zeros(2, 1), torch.ones(2, dtype=torch.bool))
        batch = {"latent": torch.zeros(2, 8), "runtime": torch.zeros(2, 79),
                 "global": torch.zeros(2, 30), "candidates": candidates,
                 "plan_mask": mask, "facts": facts}
        before = [value.clone() for value in policy.state_dict().values()]
        with torch.no_grad():
            normal = _l3_with_ablation(policy, batch)
            plan_uniform = _l3_with_ablation(policy, batch, uniform_plan=True)
            neutral = _l3_with_ablation(policy, batch, neutral_priority=True)
            both = _l3_with_ablation(policy, batch, uniform_plan=True,
                                     neutral_priority=True)
        self.assertTrue(torch.allclose(normal[2].probabilities, plan_uniform[2].probabilities))
        self.assertFalse(torch.allclose(normal[1].expected_priorities,
                                        neutral[1].expected_priorities))
        self.assertTrue(torch.allclose(neutral[2].probabilities, both[2].probabilities))
        self.assertTrue(torch.all(normal[2].probabilities[~family_mask] == 0))
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(
            before, policy.state_dict().values(), strict=True)))


if __name__ == "__main__":
    unittest.main()
