"""Step 3A-16 semi-MDP boundaries, artifact isolation and temperature contract."""

from pathlib import Path
import tempfile
import unittest

import torch

from app.rl.strategic_action_surface import FAMILIES
from app.rl.strategic_gate_ppo_step3a16 import (
    load_strategic_artifact, save_strategic_artifact, strategic_macro_gae,
)
from app.rl.strategic_gate_step3a13 import StrategicActionGate, StrategicSurfaceCritic


class StrategicPPOTests(unittest.TestCase):
    def test_variable_duration_gae_and_terminal_bootstrap(self):
        mask = [True, False, False, False, False, True]
        transitions = [
            {"truncated": False, "done": False, "macro_duration": 2,
             "reward_accumulated": .2, "value": .1,
             "next_actual_delegated": True, "next_candidate_mask": mask,
             "candidate_mask": mask, "close_policy_step": 5,
             "start_policy_step": 3},
            {"truncated": False, "done": True, "macro_duration": 1,
             "reward_accumulated": 1.0, "value": .3,
             "next_actual_delegated": False, "next_candidate_mask": None,
             "candidate_mask": mask, "close_policy_step": 6,
             "start_policy_step": 5},
        ]
        advantage, returns = strategic_macro_gae(transitions)
        second_delta = 1 - .3
        first_delta = .2 + .99**2 * .3 - .1
        self.assertAlmostEqual(float(advantage[1]), second_delta)
        self.assertAlmostEqual(float(advantage[0]),
                               first_delta + .99**2 * .95 * second_delta, places=6)
        self.assertAlmostEqual(float(returns[0]), float(advantage[0]) + .1)
        transitions[0]["next_candidate_mask"] = [False] * 6
        with self.assertRaises(ValueError):
            strategic_macro_gae(transitions)

    def test_truncation_is_not_terminal_training_data(self):
        with self.assertRaises(ValueError):
            strategic_macro_gae([{"truncated": True, "done": False}])

    def test_independent_artifact_roundtrip_and_family_order(self):
        torch.manual_seed(9)
        gate = StrategicActionGate(8)
        critic = StrategicSurfaceCritic(8)
        manifest = {"temperature": 2.0, "strategic_family_order": list(FAMILIES)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pre"
            save_strategic_artifact(path, gate, critic, manifest)
            loaded_gate, loaded_critic, loaded_manifest = load_strategic_artifact(path, 8)
            self.assertEqual(loaded_manifest, manifest)
            for original, reloaded in ((gate, loaded_gate), (critic, loaded_critic)):
                for key, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value, reloaded.state_dict()[key]))
            with self.assertRaises(FileExistsError):
                save_strategic_artifact(path, gate, critic, manifest)


if __name__ == "__main__":
    unittest.main()
