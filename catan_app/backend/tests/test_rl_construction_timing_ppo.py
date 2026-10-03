"""Variable-duration Construction Timing advantage and boundary checks."""

import unittest
from dataclasses import replace

import numpy as np

from app.rl.construction_timing import (BUILD_NOW, DEFER, SurfaceTransition,
                                        SurfaceTransitionBuilder)
from app.rl.construction_timing_ppo import macro_gae


def transition(subtype, action, reward, duration, value, next_subtype=None,
               done=False, truncated=False):
    return SurfaceTransition(
        np.zeros(1667), np.zeros(79), subtype, action, -0.69314718056,
        reward, "TERMINAL" if done else next_subtype or "TRUNCATED",
        done, duration, value, next_surface_type=next_subtype,
        truncated=truncated)


class MacroGAETests(unittest.TestCase):
    def test_variable_gamma_duration_and_next_actual_delegation(self):
        first = transition("CITY_ONLY", DEFER, 0.1 + 0.99 * 0.2, 2, 0.3,
                           next_subtype="SETTLEMENT_ONLY")
        last = transition("SETTLEMENT_ONLY", BUILD_NOW, 1.0, 1, 0.4, done=True)
        advantage, returns = macro_gae((first, last))
        last_delta = 1.0 - 0.4
        first_delta = 0.1 + 0.99 * 0.2 + 0.99 ** 2 * 0.4 - 0.3
        self.assertAlmostEqual(float(advantage[1]), last_delta, places=6)
        self.assertAlmostEqual(float(advantage[0]),
                               first_delta + 0.99 ** 2 * 0.95 * last_delta,
                               places=6)
        self.assertAlmostEqual(float(returns[0]), float(advantage[0]) + 0.3,
                               places=6)

    def test_non_delegated_surface_does_not_close_transition(self):
        builder = SurfaceTransitionBuilder(0.99)
        builder.start(surface_observation=np.zeros(1667),
                      runtime_context=np.zeros(79), surface_type="CITY_ONLY",
                      gate_action=DEFER, gate_log_prob=-0.693, value=0.1)
        builder.add_step(0.1)  # delegated DEFER
        builder.record_nondelegated_surface()
        builder.add_step(0.2)  # non-delegated eligible Surface -> Champion
        builder.add_step(0.3)  # fixed Planner decision
        item = builder.finish(next_surface_or_terminal="SETTLEMENT_ONLY",
                              done=False, next_surface_type="SETTLEMENT_ONLY")
        self.assertEqual(item.macro_duration, 3)
        self.assertEqual(item.nondelegated_surfaces_crossed, 1)
        self.assertAlmostEqual(item.reward_accumulated,
                               0.1 + 0.99 * 0.2 + 0.99 ** 2 * 0.3)

    def test_truncation_is_never_mislabelled_terminal(self):
        censored = transition("CITY_ONLY", DEFER, 0.0, 2, 0.2,
                              truncated=True)
        with self.assertRaises(ValueError):
            macro_gae((censored,))
        builder = SurfaceTransitionBuilder(0.99)
        builder.start(surface_observation=np.zeros(1667),
                      runtime_context=np.zeros(79), surface_type="CITY_ONLY",
                      gate_action=DEFER, gate_log_prob=-0.693, value=0.1)
        builder.add_step(0.0)
        with self.assertRaises(ValueError):
            builder.finish(next_surface_or_terminal="TRUNCATED", done=True,
                           truncated=True)

    def test_bootstrap_mismatch_is_rejected(self):
        first = transition("CITY_ONLY", DEFER, 0.0, 2, 0.3,
                           next_subtype="CITY_ONLY")
        second = transition("SETTLEMENT_ONLY", BUILD_NOW, 1.0, 1, 0.4,
                            done=True)
        with self.assertRaises(ValueError):
            macro_gae((first, second))

    def test_bootstrap_observation_mismatch_is_rejected(self):
        first = transition("CITY_ONLY", DEFER, 0.0, 2, 0.3,
                           next_subtype="SETTLEMENT_ONLY")
        first = replace(first, next_surface_observation=np.ones(1667),
                        next_runtime_context=np.zeros(79))
        second = transition("SETTLEMENT_ONLY", BUILD_NOW, 1.0, 1, 0.4,
                            done=True)
        with self.assertRaisesRegex(ValueError, "Bootstrap observation"):
            macro_gae((first, second))


if __name__ == "__main__":
    unittest.main()
