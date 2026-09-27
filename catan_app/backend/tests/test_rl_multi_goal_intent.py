import json
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from app.domain.actions import BuildRoadAction, BuyDevelopmentAction, EndTurnAction
from app.domain.game import create_game, player_for
from app.rl.action_space import ACTION_SPACE_SIZE
from app.rl.audit_multi_goal_intents import (backfill_multi_goal_signals,
                                              summarize_multi_goal_backfill)
from app.rl.expansion_planner import ExpansionPlan, SettlementTarget
from app.rl.macro_goal import ObjectiveGoal
from app.rl.multi_goal_intent import (IntentStatus,
                                      diagnose_multi_goal_intents)
from app.rl.settlement_planning_agent import SettlementPlanningAgent


class StubPolicy:
    def select_action(self, game, player_id):
        return EndTurnAction()


class UniformPlanningAgent(SettlementPlanningAgent):
    def _probabilities(self, game, player_id, mask):
        return np.ones(ACTION_SPACE_SIZE, dtype=np.float32)


def _target():
    return SettlementTarget(
        vertex_id=12, additional_roads=1, first_road_ids=(31,),
        production_pips=(1, 2, 3, 2, 1), total_pips=9,
        new_resource_kinds=1, port_resource=None, port_ratio=None,
        site_value=12.0, plan_value=8.5,
        required_resources=(2, 2, 1, 1, 0), wait_rounds=2.0,
    )


def _plan():
    target = _target()
    return ExpansionPlan(
        targets=(target,), selected_target=target, best_connected_target=None,
        recommended_road_ids=target.first_road_ids,
        should_wait_for_settlement=False, settlement_wait_rounds=2.0,
        city_wait_rounds=3.0, build_wait_rounds=2.0,
    )


def _record(goal, execution, index=0):
    return {
        "game_id": "g1", "player_id": 1, "observation_index": index,
        "opponent_profile": "rule", "objective_goal": goal,
        "execution_mode": execution,
        "available_goal_mask": {
            "SETTLEMENT": True, "CITY": True, "DEVELOPMENT": True,
            "ROAD_TITLE": True, "KNIGHT_TITLE": True,
        },
        "estimated_turns_to_goal": {
            "SETTLEMENT": 2.0, "CITY": 3.0,
        },
    }


def _observation(*, score=6, settlements=3, cities=0, roads=4):
    vector = np.zeros(1667, dtype=np.float32)
    vector[16] = score / 10
    vector[17] = settlements / 5
    vector[18] = cities / 4
    vector[19] = roads / 15
    return vector


class MultiGoalIntentTests(unittest.TestCase):
    def test_source_diagnostic_allows_multiple_active_goals_without_mutation(self):
        game = create_game(20260928)
        game.phase = "action"
        game.current_player_id = 1
        player_for(game, 1).settlements = 3
        player_for(game, 1).cities = 1
        agent = UniformPlanningAgent(
            StubPolicy(), target_sites=3, prioritize_immediate_titles=True,
            title_min_score=0, adaptive_development_purchase=True,
            endgame_point_reserve_score=5,
        )
        assessment = {
            "eligible": True, "reasons": ["construction_stall", "largest_army_race"],
            "build_wait_before": 4.0, "affordable": True,
            "largest_army_gap": 1,
        }
        before = deepcopy(game)
        rng_counter = game.random_source.counter
        with (patch("app.rl.multi_goal_intent.analyze_expansion_plan",
                    return_value=_plan()),
              patch("app.rl.multi_goal_intent.construction_trade_goal",
                    return_value=("city", {})),
              patch("app.rl.multi_goal_intent.estimated_build_wait", return_value=3.0),
              patch("app.rl.multi_goal_intent.assess_development_purchase",
                    return_value=assessment),
              patch.object(agent, "_best_city", return_value=None),
              patch.object(agent, "_longest_road_progress",
                           return_value=BuildRoadAction(31)),
              patch.object(agent, "_road_wins_now", return_value=False),
              patch.object(agent, "_largest_army_progress",
                           return_value=BuyDevelopmentAction())):
            report = diagnose_multi_goal_intents(agent, game, 1)

        self.assertEqual(game, before)
        self.assertEqual(game.random_source.counter, rng_counter)
        self.assertEqual(report.signals[ObjectiveGoal.SETTLEMENT].status,
                         IntentStatus.ACTIVE)
        self.assertEqual(report.signals[ObjectiveGoal.CITY].status,
                         IntentStatus.ACTIVE)
        self.assertEqual(report.signals[ObjectiveGoal.DEVELOPMENT].status,
                         IntentStatus.ACTIVE)
        self.assertEqual(report.signals[ObjectiveGoal.ROAD_TITLE].status,
                         IntentStatus.ACTIVE)
        self.assertEqual(report.signals[ObjectiveGoal.KNIGHT_TITLE].status,
                         IntentStatus.ACTIVE)
        json.dumps(report.to_dict())

    def test_backfill_keeps_knight_and_development_simultaneously_active(self):
        signals = backfill_multi_goal_signals(
            _record("KNIGHT_TITLE", "BUY_DEVELOPMENT"),
            _observation(),
            {"target_sites": 5, "max_wait_rounds": 6,
             "max_city_wait_rounds": 9},
        )

        self.assertEqual(signals["DEVELOPMENT"], "active")
        self.assertEqual(signals["KNIGHT_TITLE"], "active")
        self.assertEqual(signals["ROAD_TITLE"], "unknown")

    def test_backfill_does_not_turn_missing_evidence_into_inactive(self):
        signals = backfill_multi_goal_signals(
            _record(None, "END_TURN"), _observation(settlements=2),
            {"target_sites": 5, "max_wait_rounds": 1,
             "max_city_wait_rounds": 1},
        )

        self.assertEqual(signals["SETTLEMENT"], "unknown")
        # 開拓計画が現在の閾値外なら、construction_trade_goalの既存規則で
        # 都市が次の建設目標になるため、これは根拠のあるactive。
        self.assertEqual(signals["CITY"], "active")
        self.assertEqual(signals["DEVELOPMENT"], "unknown")

    def test_summary_reports_cooccurrence_and_active_persistence(self):
        records = [
            _record("KNIGHT_TITLE", "BUY_DEVELOPMENT", 0),
            _record("KNIGHT_TITLE", "BUY_DEVELOPMENT", 1),
            _record("ROAD_TITLE", "BUILD_ROAD", 2),
        ]
        observations = np.stack([_observation()] * 3)
        result = summarize_multi_goal_backfill(
            records, observations,
            {"target_sites": 5, "max_wait_rounds": 6,
             "max_city_wait_rounds": 9},
        )

        self.assertEqual(
            result["active_cooccurrence_counts"]["DEVELOPMENT"]["KNIGHT_TITLE"],
            2,
        )
        self.assertEqual(result["persistence"]["KNIGHT_TITLE"]["max"], 2)
        self.assertEqual(result["decision_count"], 3)


if __name__ == "__main__":
    unittest.main()
