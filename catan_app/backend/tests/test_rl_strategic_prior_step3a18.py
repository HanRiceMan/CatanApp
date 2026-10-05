"""Step 3A-18 prior transforms remain legal, symmetric, and untrained."""

import unittest

import numpy as np

from app.rl.strategic_prior_step3a18 import prior_probabilities, prior_scores
from app.rl.run_strategic_prior_step3a18 import summarize_policy_pairs


class StrategicPriorTests(unittest.TestCase):
    def setUp(self):
        self.logits = [1.0, -2.0, 3.0, 0.0, 2.0, -5.0]
        self.mask = (True, False, True, True, True, False)

    def test_all_transforms_are_masked_distributions_and_retain_top(self):
        for spec in ("native_t2", "rank_b0.5", "rank_b1.0", "rank_b1.5",
                     "gap_c1.5", "gap_c2.0", "gap_c3.0", "mix_e0.05", "mix_e0.10"):
            with self.subTest(spec=spec):
                probabilities = prior_probabilities(self.logits, self.mask, spec)
                self.assertAlmostEqual(float(probabilities.sum()), 1.0)
                self.assertEqual(float(probabilities[1]), 0.0)
                self.assertEqual(float(probabilities[5]), 0.0)
                self.assertEqual(int(np.argmax(probabilities)), 2)

    def test_rank_ignores_logit_magnitude_and_uses_fixed_tie_break(self):
        self.assertTrue(np.array_equal(
            prior_probabilities(self.logits, self.mask, "rank_b1.0"),
            prior_probabilities([100, -50, 300, 10, 200, -70], self.mask, "rank_b1.0")))
        ties = prior_scores([0, 0, 5, 0, 5, 0], self.mask, "rank_b1.0")
        self.assertGreater(ties[2], ties[4])

    def test_gap_clip_is_family_symmetric(self):
        scores = prior_scores(self.logits, self.mask, "gap_c2.0")
        self.assertEqual(scores[2], 0.0)
        self.assertEqual(scores[3], -2.0)
        permuted = [self.logits[4], self.logits[1], self.logits[0], self.logits[3],
                    self.logits[2], self.logits[5]]
        expected = prior_probabilities(self.logits, self.mask, "gap_c2.0")
        actual = prior_probabilities(permuted, self.mask, "gap_c2.0")
        self.assertTrue(np.allclose(actual, expected[[4, 1, 0, 3, 2, 5]]))

    def test_mixture_is_exact_single_categorical(self):
        base = prior_probabilities(self.logits, self.mask, "native_t2")
        mixed = prior_probabilities(self.logits, self.mask, "mix_e0.10")
        uniform = np.array([.25, 0, .25, .25, .25, 0])
        self.assertTrue(np.allclose(mixed, .9 * base + .1 * uniform))
        self.assertTrue(np.isfinite(prior_scores(self.logits, self.mask, "mix_e0.10")).all())

    def test_invalid_input_fails(self):
        with self.assertRaises(ValueError):
            prior_probabilities(self.logits, (False,) * 6, "native_t2")
        with self.assertRaises(ValueError):
            prior_probabilities([float("nan")] * 6, self.mask, "native_t2")

    def test_policy_pair_summary_preserves_state_and_future_units(self):
        def pair(state, delta, safety_won=False, full_won=False):
            return {"state_id": state, "first_pair": "BUILD_ROAD -> BUY_DEVELOPMENT",
                    "delta_return": delta, "delta_vp": 1, "delta_rank": -1,
                    "safety": {"winner": 1 if safety_won else 2, "won": safety_won,
                               "policy_steps": 50},
                    "full": {"winner": 1 if full_won else 2, "won": full_won,
                             "policy_steps": 55}}
        summary = summarize_policy_pairs([
            pair("rule-1", .5, full_won=True), pair("rule-1", .3),
            pair("mixed-2", -.4, safety_won=True), pair("mixed-2", -.2)])
        self.assertEqual(summary["states"], 2)
        self.assertEqual(summary["paired_futures"], 4)
        self.assertEqual(summary["profile_states"], {"rule": 1, "mixed": 1})
        self.assertEqual(summary["full_mean_better"], 1)
        self.assertEqual(summary["safety_mean_better"], 1)
        self.assertEqual(summary["winner_changed"], 2)


if __name__ == "__main__":
    unittest.main()
