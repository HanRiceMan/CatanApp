import json
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from app.domain.actions import (BankTradeAction, BuildCityAction,
                                BuildRoadAction, BuildSettlementAction,
                                EndTurnAction, RollTurnDiceAction)
from app.domain.game import RESOURCES, create_game, player_for
from app.rl.action_space import ACTION_SPACE_SIZE
from app.rl.expansion_planner import ExpansionPlan, SettlementTarget
from app.rl.macro_goal import (ExecutionMode, ObjectiveGoal,
                               PlannerReasonCode,
                               build_planner_decision_report)
from app.rl.settlement_planning_agent import SettlementPlanningAgent


class StubPolicy:
    def __init__(self, action):
        self.action = action

    def select_action(self, game, player_id):
        return self.action


class UniformPlanningAgent(SettlementPlanningAgent):
    def _probabilities(self, game, player_id, mask):
        return np.ones(ACTION_SPACE_SIZE, dtype=np.float32)


def scenario():
    game = create_game(20260928)
    game.phase = "action"
    game.current_player_id = 1
    game.last_roll = (3, 4)
    game.roads[30] = 1
    player = player_for(game, 1)
    player.settlements = 2
    return game


def target(*, vertex=12, roads=(31,), additional=1, wait=2.0):
    return SettlementTarget(
        vertex_id=vertex, additional_roads=additional,
        first_road_ids=roads, production_pips=(1, 2, 3, 2, 1),
        total_pips=9, new_resource_kinds=1, port_resource=None,
        port_ratio=None, site_value=12.0, plan_value=8.5,
        required_resources=(2, 2, 1, 1, 0), wait_rounds=wait,
    )


def plan(selected=None):
    targets = (selected,) if selected is not None else ()
    return ExpansionPlan(
        targets=targets, selected_target=selected, best_connected_target=None,
        recommended_road_ids=selected.first_road_ids if selected else (),
        should_wait_for_settlement=False,
        settlement_wait_rounds=selected.wait_rounds if selected else 99.0,
        city_wait_rounds=3.0, build_wait_rounds=2.0,
    )


class MacroGoalDiagnosticsTests(unittest.TestCase):
    def test_direct_builds_separate_objective_from_execution(self):
        game = scenario()
        agent = UniformPlanningAgent(StubPolicy(EndTurnAction()))

        settlement = build_planner_decision_report(
            agent, game, 1, BuildSettlementAction(12)
        )
        city = build_planner_decision_report(agent, game, 1, BuildCityAction(12))

        self.assertEqual(settlement.objective_goal, ObjectiveGoal.SETTLEMENT)
        self.assertEqual(settlement.execution_mode, ExecutionMode.BUILD_SETTLEMENT)
        self.assertEqual(city.objective_goal, ObjectiveGoal.CITY)
        self.assertEqual(city.execution_mode, ExecutionMode.BUILD_CITY)

    def test_expansion_road_keeps_settlement_as_objective(self):
        game = scenario()
        expansion = target(roads=(31,))
        agent = UniformPlanningAgent(StubPolicy(EndTurnAction()), target_sites=4)

        with patch("app.rl.macro_goal.analyze_expansion_plan",
                   return_value=plan(expansion)):
            report = build_planner_decision_report(
                agent, game, 1, BuildRoadAction(31)
            )

        self.assertEqual(report.objective_goal, ObjectiveGoal.SETTLEMENT)
        self.assertEqual(report.execution_mode, ExecutionMode.BUILD_ROAD)
        self.assertIn(PlannerReasonCode.SETTLEMENT_ROAD_STEP, report.reason_codes)

    def test_unrelated_road_is_not_forced_into_a_fake_objective(self):
        game = scenario()
        agent = UniformPlanningAgent(StubPolicy(EndTurnAction()))

        with (patch("app.rl.macro_goal.analyze_expansion_plan",
                    return_value=plan(None)),
              patch.object(agent, "_road_wins_now", return_value=False),
              patch.object(agent, "_road_increases_longest", return_value=False)):
            report = build_planner_decision_report(
                agent, game, 1, BuildRoadAction(31)
            )

        self.assertIsNone(report.objective_goal)
        self.assertIn(PlannerReasonCode.GOAL_ACTION_AMBIGUOUS,
                      report.reason_codes)

    def test_bank_trade_uses_construction_goal_but_not_trade_objective(self):
        game = scenario()
        player_for(game, 1).resources = dict.fromkeys(RESOURCES, 0)
        agent = UniformPlanningAgent(StubPolicy(EndTurnAction()))

        with patch("app.rl.trade_strategy.construction_trade_goal",
                   return_value=("city", {**dict.fromkeys(RESOURCES, 0), "ore": 3})):
            report = build_planner_decision_report(
                agent, game, 1, BankTradeAction("sheep", "ore")
            )

        self.assertEqual(report.objective_goal, ObjectiveGoal.CITY)
        self.assertEqual(report.execution_mode, ExecutionMode.TRADE_BANK)
        self.assertNotIn("TRADE", report.available_goal_mask)

    def test_non_action_phase_is_excluded_from_macro_dataset(self):
        game = scenario()
        game.phase = "turn_pre_roll"
        agent = UniformPlanningAgent(StubPolicy(RollTurnDiceAction()))
        report = build_planner_decision_report(
            agent, game, 1, RollTurnDiceAction()
        )

        self.assertFalse(report.is_macro_decision)
        self.assertIsNone(report.objective_goal)
        self.assertEqual(report.reason_codes,
                         (PlannerReasonCode.NON_MACRO_PHASE,))

    def test_report_is_json_serializable_and_does_not_invent_margin(self):
        game = scenario()
        agent = UniformPlanningAgent(StubPolicy(EndTurnAction()))
        report = build_planner_decision_report(agent, game, 1, EndTurnAction())

        json.dumps(report.to_dict(), ensure_ascii=False)
        self.assertFalse(report.scores_comparable)
        self.assertIsNone(report.margin)
        self.assertIsNone(report.second_goal)

    def test_select_action_with_report_has_exact_action_and_state_parity(self):
        game_plain = scenario()
        game_reported = deepcopy(game_plain)
        agent = UniformPlanningAgent(StubPolicy(EndTurnAction()))

        plain_action = agent.select_action(game_plain, 1)
        before_report = deepcopy(game_reported)
        reported_action, report = agent.select_action_with_report(game_reported, 1)

        self.assertEqual(reported_action, plain_action)
        self.assertEqual(game_reported, before_report)
        self.assertEqual(report.selected_action["type"], type(plain_action).__name__)


if __name__ == "__main__":
    unittest.main()
