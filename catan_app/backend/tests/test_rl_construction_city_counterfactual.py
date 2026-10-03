"""Step 3A-10 counterfactual boundary and paired-difference tests."""

import unittest

from app.agents.heuristic import HeuristicAgent
from app.rl.construction_city_counterfactual import (
    capture_boundary, difference_stats, fork_boundary, summarize_city_pairs)
from app.rl.env import CatanEnv, CatanEnvConfig


class CityCounterfactualTests(unittest.TestCase):
    def test_boundary_fork_copies_game_rng_reward_and_time(self):
        source = CatanEnv(CatanEnvConfig(learning_player_id=1))
        source.reset(seed=2631001)
        boundary = capture_boundary(source)
        fork = fork_boundary(source, boundary)
        self.assertEqual(source.game, fork.game)
        self.assertIsNot(source.game, fork.game)
        self.assertEqual(source.game.random_source.counter,
                         fork.game.random_source.counter)
        self.assertEqual(source.policy_steps, fork.policy_steps)
        self.assertEqual(source._learning_score, fork._learning_score)
        action = HeuristicAgent().select_action(source.game, 1)
        _, reward_a, done_a, truncated_a, _ = source.step_action(action)
        _, reward_b, done_b, truncated_b, _ = fork.step_action(action)
        self.assertEqual((reward_a, done_a, truncated_a),
                         (reward_b, done_b, truncated_b))
        self.assertEqual(source.game, fork.game)
        source.close()
        fork.close()

    def test_future_rng_override_is_opt_in_and_does_not_change_source(self):
        source = CatanEnv(CatanEnvConfig(learning_player_id=1))
        source.reset(seed=2631002)
        boundary = capture_boundary(source)
        fork = fork_boundary(source, boundary, continuation_seed=991)
        self.assertEqual(fork.game.random_source.seed, 991)
        self.assertEqual(fork.game.random_source.counter, 0)
        self.assertEqual(source.game, boundary.game)
        source.close()
        fork.close()

    def test_paired_signs_and_ties(self):
        stats = difference_stats([2.0, -1.0, 0.0, 3.0])
        self.assertEqual((stats["build_better"], stats["defer_better"],
                          stats["tie"]), (2, 1, 1))
        self.assertEqual(stats["median"], 1.0)

    def test_summary_reward_alignment(self):
        def record(state_id, total, terminal, champion_build):
            return {
                "state_id": state_id, "profile": "rule", "seat": 1,
                "champion_build": champion_build,
                "defer_action_family": "EndTurnAction",
                "context": {"risk.hand_size": 8, "score": 6},
                "build": {"truncated": False,
                          "discounted_environment_return": total,
                          "discounted_vp_component": total - terminal,
                          "discounted_terminal_component": terminal,
                          "win": True, "final_vp": 10, "final_rank": 1,
                          "policy_steps_total": 100, "city_count": 2,
                          "settlement_count": 3, "development_count": 1,
                          "discard_actions_after_branch": 0,
                          "longest_road_title": False,
                          "largest_army_title": False},
                "defer": {"truncated": False,
                          "discounted_environment_return": 0,
                          "discounted_vp_component": 0,
                          "discounted_terminal_component": 0,
                          "win": False, "final_vp": 9, "final_rank": 2,
                          "policy_steps_total": 103, "city_count": 1,
                          "settlement_count": 3, "development_count": 1,
                          "discard_actions_after_branch": 0,
                          "longest_road_title": False,
                          "largest_army_title": False},
                "delta": {"total": total, "vp": total - terminal,
                          "terminal": terminal, "terminal_outcome": 2,
                          "final_vp": 1, "final_rank": -1,
                          "policy_steps_total": -3},
            }
        summary = summarize_city_pairs([
            record("a", 1, 1, True), record("b", -1, -1, False),
            record("c", 1, -1, False), record("d", -1, 1, True)])
        self.assertEqual(summary["pairs"], 4)
        self.assertEqual(summary["reward_alignment"], {
            "A_BOTH_BUILD": 1, "B_BOTH_DEFER": 1,
            "C_REWARD_BUILD_TERMINAL_DEFER": 1,
            "D_REWARD_DEFER_TERMINAL_BUILD": 1})
        self.assertEqual(summary["reward_vs_undiscounted_win_alignment"], {
            "A_BOTH_BUILD": 2, "D_REWARD_DEFER_WIN_BUILD": 2})
        self.assertEqual(summary["game_cluster"]["games"], 4)
        self.assertEqual(summary["paired_delta"]["total"]["tie"], 0)


if __name__ == "__main__":
    unittest.main()
