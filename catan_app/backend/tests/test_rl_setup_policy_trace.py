import unittest

from app.rl.setup_policy_trace import summarize


class SetupPolicyTraceTests(unittest.TestCase):
    def test_summary_separates_first_and_second_settlements(self):
        rows = [
            {"placement_number": placement, "selected_vertex_id": vertex,
             "selected_probability": probability, "max_probability": probability,
             "normalized_entropy": 0.1, "probability_mass_on_best_pips": 0.2,
             "policy_expected_pips": 8.0, "best_legal_pips": 11, "selected_pips": 8}
            for placement, vertex, probability in ((1, 41, 0.9), (2, 31, 0.8),
                                                    (1, 41, 1.0), (2, 32, 0.7))
        ]
        summary = summarize(rows)
        self.assertEqual(summary["1"]["selected_vertex_unique"], 1)
        self.assertEqual(summary["1"]["selected_vertex_most_common"], {"vertex_id": 41, "share": 1.0})
        self.assertEqual(summary["1"]["mean_selected_probability"], 0.95)
        self.assertEqual(summary["2"]["selected_vertex_unique"], 2)
        self.assertEqual(summary["2"]["mean_selected_probability"], 0.75)


if __name__ == "__main__":
    unittest.main()
