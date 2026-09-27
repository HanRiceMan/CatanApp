import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from app.agents.heuristic import HeuristicAgent
from app.domain.actions import EndTurnAction
from app.domain.game import create_game, player_for
from app.rl.action_space import ACTION_SPACE_SIZE
from app.rl.macro_teacher_dataset import (TeacherDataset,
                                          collect_teacher_dataset,
                                          observe_teacher_decision,
                                          save_teacher_dataset,
                                          summarize_taxonomy,
                                          verify_action_parity)
from app.rl.observation import OBSERVATION_VECTOR_SIZES
from app.rl.settlement_planning_agent import SettlementPlanningAgent


class Manifest:
    observation_version = "v2"
    observation_size = OBSERVATION_VECTOR_SIZES["v2"]


class StubPolicy:
    manifest = Manifest()
    initial_setup_policy = None

    def __init__(self, action=None):
        self.action = action
        self.heuristic = HeuristicAgent()

    def select_action(self, game, player_id):
        return self.action or self.heuristic.select_action(game, player_id)


class UniformPlanningAgent(SettlementPlanningAgent):
    def _probabilities(self, game, player_id, mask):
        return np.ones(ACTION_SPACE_SIZE, dtype=np.float32)


def scenario():
    game = create_game(20261004)
    game.phase = "action"
    game.current_player_id = 1
    game.last_roll = None
    game.roads[30] = 1
    player_for(game, 1).settlements = 2
    return game


def record(goal, execution, *, reason="GOAL_ACTION_AMBIGUOUS", won=False,
           action_type="BuildRoadAction", turn=10, value=1.5):
    return {
        "objective_goal": goal, "execution_mode": execution,
        "reason_codes": [reason], "selected_action": {"type": action_type},
        "phase": "action", "turn": turn,
        "turn_band": "early" if turn <= 25 else "mid", "won": won,
        "goal_scores": {"SETTLEMENT": value},
        "goal_score_semantics": {"SETTLEMENT": "expansion_plan.plan_value"},
    }


class MacroTeacherDatasetTests(unittest.TestCase):
    def test_observation_and_report_do_not_change_state_or_rng(self):
        game = scenario()
        agent = UniformPlanningAgent(StubPolicy(EndTurnAction()))
        before = deepcopy(game)
        counter = game.random_source.counter

        action, report, observation = observe_teacher_decision(agent, game, 1)
        second_action, second_report, second_observation = observe_teacher_decision(
            agent, game, 1,
        )

        self.assertEqual(game, before)
        self.assertEqual(game.random_source.counter, counter)
        self.assertEqual(action, second_action)
        self.assertEqual(report, second_report)
        np.testing.assert_array_equal(observation, second_observation)
        self.assertTrue(report.is_macro_decision)
        self.assertEqual(observation.shape, (OBSERVATION_VECTOR_SIZES["v2"],))

    def test_taxonomy_keeps_none_and_cross_counts(self):
        rows = [
            record("SETTLEMENT", "BUILD_ROAD", reason="SETTLEMENT_ROAD_STEP"),
            record(None, "BUILD_ROAD"),
            record("CITY", "TRADE_BANK", reason="BANK_TRADE_REACHABLE", won=True,
                   action_type="BankTradeAction", value=2.5),
        ]

        summary = summarize_taxonomy(rows)

        self.assertEqual(
            summary["objective_goal_distribution"]["None"]["count"], 1
        )
        self.assertEqual(
            summary["objective_goal_by_execution_mode"]["SETTLEMENT"]
                   ["BUILD_ROAD"]["count"], 1
        )
        self.assertEqual(summary["none_analysis"]["build_road"]["none_rate"], 0.5)
        stats = summary["goal_score_semantics"]["expansion_plan.plan_value"]
        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["mean"], 11 / 6)

    def test_dataset_files_link_observations_and_game_results(self):
        observation = np.zeros((2, OBSERVATION_VECTOR_SIZES["v2"]),
                               dtype=np.float32)
        rows = []
        for index, goal in enumerate((None, "HOLD")):
            row = record(goal, "END_TURN", action_type="EndTurnAction")
            row.update({
                "observation_index": index, "winner": 1,
                "final_vp": 10, "final_rank": 1.0,
            })
            rows.append(row)
        dataset = TeacherDataset(tuple(rows), observation, ({"game_seed": 1},))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            result = save_teacher_dataset(output, dataset, definition={"test": True})
            saved = np.load(output / "observations.npz")["observations"]
            lines = (output / "decisions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()

            self.assertEqual(saved.shape, observation.shape)
            self.assertEqual(len(lines), 2)
            self.assertIsNone(json.loads(lines[0])["objective_goal"])
            self.assertEqual(json.loads(lines[1])["final_vp"], 10)
            self.assertEqual(result["manifest"]["decision_count"], 2)

    def test_episode_collection_links_result_and_observation(self):
        agent = UniformPlanningAgent(StubPolicy())
        dataset = collect_teacher_dataset(agent, [20261005])

        self.assertEqual(len(dataset.games), 1)
        self.assertGreater(len(dataset.records), 0)
        self.assertEqual(len(dataset.records), len(dataset.observations))
        game = dataset.games[0]
        for row in dataset.records:
            self.assertEqual(row["game_id"], game["game_id"])
            self.assertEqual(row["winner"], game["winner"])
            self.assertEqual(row["final_vp"], game["final_vp"])
            self.assertEqual(row["final_rank"], game["final_rank"])

    def test_fixed_seed_parity_uses_full_action_trace_and_final_state(self):
        plain = UniformPlanningAgent(StubPolicy())
        diagnostic = UniformPlanningAgent(StubPolicy())

        result = verify_action_parity(plain, diagnostic, [20261006])

        self.assertEqual(result["games"], 1)
        self.assertTrue(result["action_trace_equal"])
        self.assertTrue(result["final_game_state_equal"])


if __name__ == "__main__":
    unittest.main()
