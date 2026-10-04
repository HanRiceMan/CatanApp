"""Temperature calibration leaves masking, ranking and on-policy probabilities intact."""

import math
import unittest

import torch

from app.rl.run_strategic_prior_step3a15 import (
    _exploration, _validate_on_policy_records,
)
from app.rl.strategic_action_surface import FAMILIES
from app.rl.strategic_gate_step3a13 import StrategicActionGate
from app.rl.runtime_context import RUNTIME_CONTEXT_SIZE
from app.rl.strategic_prior_calibration_step3a15 import (
    audit_records, masked_mixture_probabilities,
    masked_temperature_probabilities,
)


class PriorCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.scores = (4.0, 3.0, 2.0, 1.0, 0.0, -1.0)
        self.mask = (True, True, False, True, False, True)

    def test_temperature_normalization_mask_and_rank(self):
        native = masked_temperature_probabilities(self.scores, self.mask, 1.0)
        softened = masked_temperature_probabilities(self.scores, self.mask, 2.0)
        self.assertAlmostEqual(sum(softened), 1.0)
        self.assertEqual(softened[2], 0.0)
        self.assertEqual(softened[4], 0.0)
        self.assertEqual(native.index(max(native)), softened.index(max(softened)))
        self.assertLess(max(softened), max(native))
        with self.assertRaises(ValueError):
            masked_temperature_probabilities(self.scores, self.mask, 0.0)
        with self.assertRaises(ValueError):
            masked_temperature_probabilities(self.scores, self.mask, float("nan"))

    def test_mixture_is_one_distribution_not_posthoc_random(self):
        native = masked_temperature_probabilities(self.scores, self.mask, 1.0)
        mixed = masked_mixture_probabilities(self.scores, self.mask, .1)
        for index, enabled in enumerate(self.mask):
            expected = .9 * native[index] + (.1 / 4 if enabled else 0)
            self.assertAlmostEqual(mixed[index], expected)
        self.assertAlmostEqual(sum(mixed), 1.0)

    def test_gate_uses_calibrated_prior_and_zero_residual(self):
        gate = StrategicActionGate(frozen_actor_latent_dim=3)
        latent = torch.zeros(1, 3)
        context = torch.zeros(1, RUNTIME_CONTEXT_SIZE)
        mask = torch.tensor([self.mask])
        logits = torch.tensor([self.scores])
        native = gate(latent, context, mask, logits)
        warm = gate(latent, context, mask, logits, temperature=2.0)
        self.assertTrue(torch.equal(native[0, mask[0]], logits[0, mask[0]]))
        self.assertTrue(torch.equal(warm[0, mask[0]], logits[0, mask[0]] / 2))
        self.assertTrue((warm[0, ~mask[0]] == -1e8).all())
        with self.assertRaises(ValueError):
            gate(latent, context, mask, logits, temperature=float("nan"))

    def test_offline_audit_retains_native_top(self):
        rows = [{"candidate_mask": self.mask,
                 "native_family_logits": self.scores}]
        audit = audit_records(rows)
        self.assertEqual(audit["T=3"]["top1_retention"], 1.0)
        self.assertGreater(audit["T=3"]["alternative_mass"]["mean"],
                           audit["T=1"]["alternative_mass"]["mean"])
        self.assertEqual(audit["T=2"]["family_probability_when_available"][
            FAMILIES[2]]["n"], 0)

    def test_legacy_t1_exploration_record_without_flag(self):
        decisions = [
            {"selected_family": FAMILIES[0], "selected_probability": .7,
             "masked_probability": [.7, .3, 0, 0, 0, 0]},
            {"selected_family": FAMILIES[1], "selected_probability": .3,
             "masked_probability": [.7, .3, 0, 0, 0, 0]},
        ]
        result = _exploration([{"decisions": decisions}])
        self.assertEqual(result["non_top_selected"], 1)
        self.assertEqual(result["non_top_by_family"][FAMILIES[1]], 1)

    def test_logged_distribution_is_actual_sample_policy(self):
        decision = {"candidate_mask": [True, True, False, False, False, False],
                    "masked_probability": [.7, .3, 0, 0, 0, 0],
                    "selected_probability": .3, "gate_log_prob": math.log(.3)}
        game = {"decisions": [decision],
                "transitions": [{"decision_index": 0, "gate_log_prob": math.log(.3)}]}
        self.assertEqual(_validate_on_policy_records([game]), 1)
        game["decisions"][0]["gate_log_prob"] = math.log(.7)
        with self.assertRaises(AssertionError):
            _validate_on_policy_records([game])


if __name__ == "__main__":
    unittest.main()
