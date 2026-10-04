"""Step 3A-11 repeated future-seed selection and statistics."""

import unittest
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from app.rl.construction_city_multifuture import (
    CONTEXT_FIELDS, _corr, choose_phase2, diagnostic_ci, future_seed,
    select_states, state_stats, summarize)
from app.rl.run_construction_city_counterfactual_step3a10 import read_records
from app.rl.run_construction_city_multifuture_step3a11 import read_continuations


class CityMultiFutureTests(unittest.TestCase):
    def test_future_seeds_are_stable_and_indexed(self):
        self.assertEqual(future_seed("state-a", 3), future_seed("state-a", 3))
        self.assertNotEqual(future_seed("state-a", 3), future_seed("state-a", 4))
        self.assertNotEqual(future_seed("state-a", 3), future_seed("state-b", 3))
        with self.assertRaises(ValueError):
            future_seed("state-a", 16)

    def test_selected_audit_cohort_covers_every_champion_defer(self):
        dataset = (Path(__file__).resolve().parents[1] / "experiments"
                   / "construction_city_step3a10_20261004" / "pairs.jsonl")
        records = read_records(dataset)
        selection = select_states(records)
        self.assertEqual(selection["selected_count"], 52)
        self.assertEqual(Counter(selection["state_categories"].values())
                         ["CHAMPION_DEFER"], 31)
        defer_ids = {r["state_id"] for r in records if not r["champion_build"]}
        self.assertTrue(defer_ids <= set(selection["state_categories"]))

    def test_continuation_reader_allows_same_state_different_future_seed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            first = {"state_id": "a", "future_seed_index": 0}
            second = {"state_id": "a", "future_seed_index": 1}
            path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n",
                            encoding="utf-8")
            self.assertEqual(len(read_continuations(path)), 2)
            path.write_text(json.dumps(first) + "\n" + json.dumps(first) + "\n",
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                read_continuations(path)

    def test_state_stats_near_ties_and_bootstrap_are_deterministic(self):
        rows = [{"delta_q": value, "build_win": value > 0,
                 "defer_win": value < 0, "build_final_vp": 10,
                 "defer_final_vp": 9, "delta_terminal": value / 2,
                 "delta_vp": value / 2}
                for value in (0.0, 0.001, 0.05, -0.06, 0.07, -0.08, 0.03, -0.04)]
        stats = state_stats(rows, "state-a")
        self.assertEqual(stats["n"], 8)
        self.assertEqual((stats["build_better"], stats["defer_better"],
                          stats["near_tie"]), (3, 3, 2))
        self.assertEqual(stats["sign_consistency_non_ties"], 0.5)
        self.assertEqual(stats["final_vp_delta_mean"], 1)
        self.assertEqual(diagnostic_ci([r["delta_q"] for r in rows], "state-a"),
                         (stats["ci95_bootstrap_low"], stats["ci95_bootstrap_high"]))

    def test_phase2_requires_uncertainty_and_caps_selection(self):
        source = {f"s{i}": {"champion_build": i % 2 == 0,
                            "defer_action_family": "EndTurnAction" if i % 2
                            else "BankTradeAction"}
                  for i in range(20)}
        stats = {sid: {"n": 8, "se_delta_q": 0.1,
                       "ci95_crosses_zero": True,
                       "sign_consistency_non_ties": 0.5,
                       "mean_delta_q": 0.0}
                 for sid in source}
        stats["s0"]["se_delta_q"] = 0.001
        picked = choose_phase2(stats, source, limit=6)
        self.assertEqual(len(picked), 6)
        self.assertNotIn("s0", picked)
        self.assertEqual(len(set(picked)), len(picked))
        self.assertEqual({source[sid]["defer_action_family"] for sid in picked},
                         {"EndTurnAction", "BankTradeAction"})

    def test_correlations_preserve_rank(self):
        pearson, spearman = _corr([1, 2, 3, 4], [10, 20, 30, 100])
        self.assertLess(pearson, 1)
        self.assertAlmostEqual(spearman, 1)

    def test_balanced_phase1_variance_decomposition(self):
        source = [{"state_id": sid, "profile": "rule", "seat": 1,
                   "champion_build": sid == "a",
                   "defer_action_family": "EndTurnAction",
                   "delta": {"total": 0.5 if sid == "a" else -0.5},
                   "context": {field: 1 for field in CONTEXT_FIELDS}}
                  for sid in ("a", "b")]
        rows = []
        for sid, delta in (("a", 0.2), ("b", -0.2)):
            for index in range(8):
                rows.append({"state_id": sid, "future_seed_index": index,
                             "delta_q": delta, "delta_terminal": delta,
                             "delta_vp": 0, "build_win": sid == "a",
                             "defer_win": sid == "b",
                             "build_final_vp": 10, "defer_final_vp": 9})
        result = summarize(rows, source, {"a": "CONTROL", "b": "DEFER"})
        self.assertEqual(result["future_seed_count_distribution"], {8: 2})
        self.assertAlmostEqual(result["phase1_variance"]["within_mean_sample_variance"], 0)
        self.assertAlmostEqual(result["phase1_variance"]["icc_like"], 1)
        self.assertEqual(result["state_mean_distribution"]["near_zero"], 0)


if __name__ == "__main__":
    unittest.main()
