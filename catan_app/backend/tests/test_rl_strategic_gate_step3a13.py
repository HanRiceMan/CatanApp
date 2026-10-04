"""Step 3A-13 zero-impact priors, executors, resolver, semi-MDP typing."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from app.domain.actions import BuildCityAction, EndTurnAction
from app.domain.game import create_game, player_for
from app.rl.action_families import ACTION_FAMILY_INDEX
from app.rl.action_space import ACTION_SPACE_SIZE, action_to_id, get_action_mask
from app.rl.hierarchical_distribution import FAMILY_COUNT
from app.rl.strategic_action_surface import FAMILIES
from app.rl.strategic_gate_step3a13 import (
    FrozenSnapshot, FrozenStrategicExecutors, StrategicActionGate,
    StrategicSurfaceCritic, StrategicTransition, family_priors,
    resolve_immediate_win,
)


def _snapshot() -> FrozenSnapshot:
    return FrozenSnapshot(torch.zeros(1, 8), torch.zeros(1, 1),
                          np.zeros(ACTION_SPACE_SIZE, dtype=np.float32),
                          np.zeros(FAMILY_COUNT, dtype=np.float32),
                          np.zeros(79, dtype=np.float32))


class StrategicGateTests(unittest.TestCase):
    def test_native_prior_has_no_mechanical_cardinality_bias(self):
        snapshot = _snapshot()
        snapshot.old_family_logits[ACTION_FAMILY_INDEX["road"]] = 2.0
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        from app.domain.actions import BuildRoadAction, BankTradeAction
        mask[action_to_id(BuildRoadAction(1))] = True
        mask[action_to_id(BuildRoadAction(2))] = True
        mask[action_to_id(BankTradeAction("wood", "ore"))] = True
        mask[action_to_id(EndTurnAction())] = True
        available = (False, False, True, False, True, True)
        prior = family_priors(snapshot, mask, available)
        self.assertEqual(prior["legal_action_counts"], [0, 0, 2, 0, 1, 1])
        native = prior["probabilities"]["native_family_logits"]
        self.assertGreater(native[2], native[4])
        self.assertAlmostEqual(sum(native), 1.0)
        self.assertIsNone(prior["raw_scores"]["max_legal_logit"][0])
        one_road = mask.copy()
        one_road[action_to_id(BuildRoadAction(2))] = False
        one_road_prior = family_priors(snapshot, one_road, available)
        self.assertEqual(native,
                         one_road_prior["probabilities"]["native_family_logits"])

    def test_zero_residual_gate_and_critic_preserve_frozen_outputs(self):
        gate = StrategicActionGate(8)
        critic = StrategicSurfaceCritic(8)
        latent = torch.randn(1, 8, requires_grad=True)
        base = torch.tensor([[0.3]], requires_grad=True)
        context = torch.randn(1, 79)
        mask = torch.tensor([[1, 0, 1, 0, 0, 1]], dtype=torch.bool)
        prior_logits = torch.tensor([[1., 5., 2., 4., 3., 0.]])
        result = gate(latent, context, mask, prior_logits)
        self.assertTrue(torch.equal(result[0, mask[0]], prior_logits[0, mask[0]]))
        self.assertEqual(result[0, 1].item(), -1e8)
        self.assertTrue(torch.equal(critic(base, latent, context, mask), base.detach()))
        (result[0, mask[0]].sum() + critic(base, latent, context, mask).sum()).backward()
        self.assertIsNone(latent.grad)
        self.assertIsNone(base.grad)

    def test_surface_transition_only_bootstraps_at_next_actual_delegation(self):
        mask = (True, False, False, False, False, True)
        transition = StrategicTransition(0, mask, -0.5, 0.2, 0.1, 3, False,
                                         True, next_candidate_mask=mask)
        self.assertAlmostEqual(transition.bootstrap_discount(0.99), 0.99 ** 3)
        with self.assertRaises(ValueError):
            StrategicTransition(0, mask, -0.5, 0.2, 0.1, 3, False, False,
                                next_candidate_mask=mask)
        with self.assertRaises(ValueError):
            StrategicTransition(0, mask, -0.5, 0.2, 0.1, 3, False, False)
        with self.assertRaises(ValueError):
            StrategicTransition(1, mask, -0.5, 0.2, 0.1, 3, False, True,
                                next_candidate_mask=mask)
        terminal = StrategicTransition(0, mask, -0.5, 0.2, 0.1, 3, True, False)
        self.assertEqual(terminal.bootstrap_discount(0.99), 0.0)

    def test_bank_executor_b_is_family_only_legal_and_nonmutating(self):
        game = create_game(20261310)
        game.phase = "action"
        game.current_player_id = 1
        player_for(game, 1).resources.update({"wood": 4, "brick": 0,
                                               "sheep": 0, "wheat": 0, "ore": 0})
        before = deepcopy(game)
        executor = FrozenStrategicExecutors(SimpleNamespace())
        mask = get_action_mask(game, 1)
        first = executor.execute(game, 1, "BANK_TRADE", mask=mask,
                                 snapshot=_snapshot())
        second = executor.execute(game, 1, "BANK_TRADE", mask=mask,
                                  snapshot=_snapshot())
        self.assertTrue(first.success, first.failure)
        self.assertEqual(first.action, second.action)
        self.assertEqual(game, before)
        unsafe = executor.execute(game, 1, "BANK_TRADE",
                                  method="A_PLANNER_EXTRACTION", mask=mask,
                                  snapshot=_snapshot())
        self.assertEqual(unsafe.failure, "UNSAFE_GOAL_BUDGET_DEPENDENCY")

    def test_immediate_win_resolver_does_not_use_champion_action_or_live_rng(self):
        game = create_game(20261311)
        game.phase = "action"
        game.current_player_id = 1
        game.settlements[0] = 1
        player = player_for(game, 1)
        player.settlements = 1
        player.development_cards.extend(["victory_point"] * 8)
        player.resources.update({"wood": 0, "brick": 0, "sheep": 0,
                                 "wheat": 2, "ore": 3})
        before = deepcopy(game)
        result = resolve_immediate_win(game, 1)
        self.assertIsInstance(result.action, BuildCityAction)
        self.assertEqual(result.action, BuildCityAction(0))
        self.assertEqual(game, before)


if __name__ == "__main__":
    unittest.main()
