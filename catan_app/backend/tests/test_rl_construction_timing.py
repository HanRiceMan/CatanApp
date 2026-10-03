"""Step 3A-8: no-update gate, counterfactual executors and surface-only buffer."""

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from app.domain.actions import (BuildCityAction, BuildSettlementAction,
                                EndTurnAction, ProposeTradeAction, TradeOffer)
from app.domain.game import create_game
from app.rl.action_space import action_to_id, get_action_mask, id_to_action
from app.rl.construction_shadow import ConstructionSurface
from app.rl.construction_timing import (
    BUILD_NOW, DEFER, ConstructionTimingDelegator, ConstructionTimingGate,
    FrozenConstructionExecutors, SurfaceCritic, TimingCandidates,
    SurfaceRolloutBuffer, SurfaceTransition, SurfaceTransitionBuilder,
    _nonconstruction_mask,
    choose_delegation, timing_surface,
)
from app.rl.expansion_planner import analyze_expansion_plan
from app.rl.env import CatanEnv, CatanEnvConfig
from app.rl.runtime_context import encode_runtime_context
from app.rl.reward import actual_score
from app.rl.structured_context import (
    build_structured_context_report,
    build_structured_context_report_with_plan,
)


class ConstructionTimingTests(unittest.TestCase):
    def test_only_single_build_surface_is_delegatable(self):
        game = create_game(20261201)
        dual = ConstructionSurface("SETTLEMENT_AND_CITY", None, 1, 1)
        with patch("app.rl.construction_timing.detect_construction_surface",
                   return_value=dual):
            self.assertEqual(timing_surface(game, 1).exclusion, "DUAL_BUILD_DEFERRED")

    def test_nonconstruction_mask_retains_legal_nonbuild_actions(self):
        game = create_game(20261202)
        game.phase = "action"
        game.current_player_id = 1
        game.players[0].resources.update({"wood": 4, "brick": 4, "sheep": 4,
                                          "wheat": 4, "ore": 4})
        original = get_action_mask(game, 1)
        masked = _nonconstruction_mask(game, 1)
        self.assertTrue(masked[action_to_id(EndTurnAction())])
        self.assertLessEqual(masked.sum(), original.sum())
        self.assertFalse(any(masked[action_to_id(action)] for action in (
            BuildSettlementAction(0), BuildCityAction(0))))

    def test_gate_and_surface_critic_do_not_train_frozen_latent(self):
        latent = torch.randn(3, 16, requires_grad=True)
        context = torch.randn(3, 79)
        subtype = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
        gate = ConstructionTimingGate(16)
        critic = SurfaceCritic(16)
        self.assertTrue(torch.equal(gate(latent, context, subtype), torch.zeros(3, 2)))
        base_value = torch.tensor([[0.2], [0.4], [0.6]], requires_grad=True)
        self.assertTrue(torch.equal(critic(base_value, latent, context, subtype),
                                    base_value.detach()))
        (gate(latent, context, subtype).sum()
         + critic(base_value, latent, context, subtype).sum()).backward()
        self.assertIsNone(latent.grad)
        self.assertIsNone(base_value.grad)

    def test_surface_buffer_excludes_planner_decisions(self):
        buffer = SurfaceRolloutBuffer(0.99)
        transition = SurfaceTransition(
            np.zeros(1667), np.zeros(79), "CITY_ONLY", DEFER, -0.69,
            0.1 + 0.99 * 0.2, "NEXT_SURFACE", False, 2, 0.0)
        buffer.append(transition)
        self.assertAlmostEqual(buffer.discounted_reward([0.1, 0.2]),
                               transition.reward_accumulated)
        self.assertAlmostEqual(buffer.bootstrap_discount(2), 0.99 ** 2)
        with self.assertRaises(ValueError):
            buffer.append(SurfaceTransition(
                np.zeros(1667), np.zeros(79), "CITY_ONLY", BUILD_NOW, -0.69,
                0.0, "NEXT_SURFACE", False, 1, 0.0, delegated=False))
        with self.assertRaises(ValueError):
            buffer.append(SurfaceTransition(
                np.zeros(1667), np.zeros(79), "SETTLEMENT_AND_CITY", BUILD_NOW,
                -0.69, 0.0, "NEXT_SURFACE", False, 1, 0.0))

    def test_macro_transition_counts_learner_steps_not_primitive_actions(self):
        builder = SurfaceTransitionBuilder(0.99)
        builder.start(surface_observation=np.zeros(1667),
                      runtime_context=np.zeros(79), surface_type="SETTLEMENT_ONLY",
                      gate_action=BUILD_NOW, gate_log_prob=-0.69, value=0.2)
        builder.add_step(0.1)  # delegated action plus auto opponent steps
        builder.add_step(0.2)  # a later fixed Planner learner action
        transition = builder.finish(next_surface_or_terminal="CITY_ONLY", done=False)
        self.assertEqual(transition.macro_duration, 2)
        self.assertAlmostEqual(transition.reward_accumulated, 0.1 + 0.99 * 0.2)
        self.assertIsNone(builder.pending)

    def test_delegation_uses_external_rng_only(self):
        self.assertFalse(choose_delegation(0, lambda: 0.5))
        self.assertTrue(choose_delegation(1, lambda: 0.5))

    def test_default_delegator_never_applies_gate_action(self):
        game = create_game(20261204)
        champion = Mock()
        champion.select_action.return_value = EndTurnAction()
        gate = ConstructionTimingGate(16)
        critic = SurfaceCritic(16)
        surface = ConstructionSurface("SETTLEMENT_ONLY", None, 1, 0)
        candidates = TimingCandidates(surface, BuildSettlementAction(0),
                                      EndTurnAction())
        with (patch("app.rl.construction_timing.FrozenConstructionExecutors")
              as executors, patch("app.rl.construction_timing.timing_surface",
                                  return_value=surface)):
            executors.return_value.candidates.return_value = candidates
            delegator = ConstructionTimingDelegator(champion, gate, critic)
            choice = delegator.select(game, 1, random_uniform=lambda: 0.5)
        self.assertFalse(choice.delegated)
        self.assertEqual(choice.action, EndTurnAction())
        champion.select_action.assert_called_once_with(game, 1)

    def test_shared_context_plan_preserves_79_vector(self):
        game = create_game(20261203)
        game.phase = "action"
        game.current_player_id = 1
        before = deepcopy(game)
        report, plan = build_structured_context_report_with_plan(game, 1)
        self.assertEqual(report, build_structured_context_report(game, 1))
        self.assertEqual(plan, analyze_expansion_plan(deepcopy(game), 1,
                                                       max_additional_roads=3))
        self.assertEqual(game, before)
        context = encode_runtime_context(game, 1)
        self.assertEqual(context.shape, (79,))
        self.assertTrue(np.isfinite(context).all())

    def test_action_object_step_matches_legacy_id_step(self):
        first = CatanEnv(CatanEnvConfig())
        second = CatanEnv(CatanEnvConfig())
        first.reset(seed=20261205)
        second.reset(seed=20261205)
        action_id = int(np.flatnonzero(first.action_masks())[0])
        old = first.step(action_id)
        new = second.step_action(id_to_action(action_id))
        self.assertTrue(np.array_equal(old[0], new[0]))
        self.assertEqual(old[1:4], new[1:4])
        self.assertEqual(first.game.random_source.counter,
                         second.game.random_source.counter)
        first.close()
        second.close()

    def test_action_object_step_supports_player_trade_without_catalog_change(self):
        env = CatanEnv(CatanEnvConfig(allow_player_trades=True))
        env.reset(seed=20261206)
        game = env.game
        game.phase = "action"
        game.seat_order = [1, 2, 3, 4]
        game.current_player_id = 1
        game.players[0].resources["wheat"] = 1
        env._learning_score = actual_score(game, 1)
        action = ProposeTradeAction(2, (TradeOffer({"wheat": 1}, {"ore": 1}),))
        _, reward, _, _, info = env.step_action(action)
        self.assertEqual(env.policy_steps, 1)
        self.assertEqual(info["policy_steps"], 1)
        self.assertTrue(np.isfinite(reward))
        env.close()


if __name__ == "__main__":
    unittest.main()
