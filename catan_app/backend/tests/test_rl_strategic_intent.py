import unittest
from unittest.mock import patch

import numpy as np

from app.domain.actions import (BuyDevelopmentAction, EndTurnAction,
                                UseDevelopmentAction)
from app.domain.game import create_game, player_for
from app.rl.action_space import ACTION_SPACE_SIZE
from app.rl.macro_teacher_dataset import (_normalized_final_state,
                                          summarize_strategic_intents)
from app.rl.settlement_planning_agent import SettlementPlanningAgent
from app.rl.strategic_intent import (ActionSelectionTrace, DiagnosticExecution,
                                     ExecutionSource, IntentState, StrategicIntent,
                                     StrategicIntentReport, StrategicIntentSignal,
                                     STRATEGIC_INTENTS,
                                     build_strategic_intent_report,
                                     diagnostic_execution)


class StubPolicy:
    initial_setup_policy = None

    def select_action(self, game, player_id):
        return EndTurnAction()


class UniformPlanningAgent(SettlementPlanningAgent):
    def _probabilities(self, game, player_id, mask):
        return np.ones(ACTION_SPACE_SIZE, dtype=np.float32)


class StrategicIntentTests(unittest.TestCase):
    def test_source_contribution_is_an_active_intent_even_without_generic_gate(self):
        game = create_game(43)
        game.phase = "action"
        game.current_player_id = 1
        agent = UniformPlanningAgent(StubPolicy())
        for intent, source in (
            (StrategicIntent.SETTLEMENT, ExecutionSource.SETTLEMENT_PLANNER),
            (StrategicIntent.ROAD_TITLE, ExecutionSource.TITLE_PLANNER),
        ):
            with self.subTest(intent=intent):
                report = build_strategic_intent_report(
                    agent, game, 1, EndTurnAction(),
                    ActionSelectionTrace(source, (intent,)),
                )
                self.assertEqual(report.signals[intent].state, IntentState.ACTIVE)
                self.assertIs(report.reason_contributed_to_selection[intent], True)

    def test_full_state_parity_ignores_only_random_identifiers(self):
        left = create_game(42)
        right = create_game(42)
        self.assertEqual(_normalized_final_state(left), _normalized_final_state(right))
        right.bank["ore"] -= 1
        self.assertNotEqual(_normalized_final_state(left), _normalized_final_state(right))

    def test_unknown_is_masked_not_inactive(self):
        states = (IntentState.ACTIVE, IntentState.UNKNOWN, IntentState.INACTIVE,
                  IntentState.UNKNOWN, IntentState.ACTIVE)
        signals = {
            intent: StrategicIntentSignal(state, (), {})
            for intent, state in zip(STRATEGIC_INTENTS, states, strict=True)
        }
        report = StrategicIntentReport(
            signals=signals,
            selected_execution=DiagnosticExecution.END_TURN,
            execution_source=ExecutionSource.BASE_PPO,
            reason_contributed_to_selection={intent: None for intent in STRATEGIC_INTENTS},
            selection_reason_codes=(),
            development_eligible=None,
            new_development_cards_blocked=None,
            new_development_cards_blocked_purchase=None,
            new_knight_blocked_use=None,
        )
        encoded = report.to_dict()
        self.assertEqual(encoded["strategic_intent_targets"], [1, 0, 0, 0, 1])
        self.assertEqual(encoded["strategic_intent_known_mask"], [1, 0, 1, 0, 1])

    def test_card_specific_execution_mapping(self):
        self.assertEqual(
            diagnostic_execution(UseDevelopmentAction("knight")),
            DiagnosticExecution.PLAY_KNIGHT,
        )
        self.assertEqual(
            diagnostic_execution(UseDevelopmentAction("road_building")),
            DiagnosticExecution.PLAY_ROAD_BUILDING,
        )
        self.assertEqual(
            diagnostic_execution(UseDevelopmentAction("year_of_plenty")),
            DiagnosticExecution.PLAY_YEAR_OF_PLENTY,
        )
        self.assertEqual(
            diagnostic_execution(UseDevelopmentAction("monopoly")),
            DiagnosticExecution.PLAY_MONOPOLY,
        )

    def test_adaptive_development_records_causal_intents(self):
        game = create_game(20260929)
        game.phase = "action"
        game.current_player_id = 1
        agent = UniformPlanningAgent(
            StubPolicy(), adaptive_development_purchase=True,
        )

        def planned(_game, _player_id, trace):
            trace.update(execution_source="BASE_PPO",
                         contributed_intents=(), contributed_reasons=())
            return EndTurnAction()

        assessment = {
            "eligible": True,
            "reasons": ["largest_army_race", "robber_relief"],
            "already_bought_this_turn": False,
        }
        with (patch.object(agent, "_select_planned_action", side_effect=planned),
              patch("app.rl.development_strategy.assess_development_purchase",
                    return_value=assessment)):
            action, _, trace = agent._select_action_with_diagnostics(game, 1)

        self.assertIsInstance(action, BuyDevelopmentAction)
        self.assertEqual(trace.execution_source,
                         ExecutionSource.PLANNER_ADAPTIVE_DEVELOPMENT)
        self.assertCountEqual(
            trace.contributed_intents,
            (StrategicIntent.KNIGHT_TITLE, StrategicIntent.ROBBER_RELIEF),
        )

    def test_base_ppo_buy_is_not_falsely_attributed(self):
        game = create_game(20260930)
        game.phase = "action"
        game.current_player_id = 1
        agent = UniformPlanningAgent(StubPolicy())

        def planned(_game, _player_id, trace):
            trace.update(execution_source="BASE_PPO",
                         contributed_intents=(), contributed_reasons=())
            return BuyDevelopmentAction()

        with patch.object(agent, "_select_planned_action", side_effect=planned):
            action, _, trace = agent._select_action_with_diagnostics(game, 1)

        self.assertIsInstance(action, BuyDevelopmentAction)
        self.assertEqual(trace.execution_source, ExecutionSource.BASE_PPO)
        self.assertEqual(trace.contributed_intents, ())

    def test_summary_separates_active_state_from_action_contribution(self):
        states = {
            "SETTLEMENT": "active", "CITY": "unknown",
            "ROAD_TITLE": "unknown", "KNIGHT_TITLE": "active",
            "ROBBER_RELIEF": "active",
        }
        contribution = {
            "SETTLEMENT": False, "CITY": None, "ROAD_TITLE": None,
            "KNIGHT_TITLE": True, "ROBBER_RELIEF": False,
        }
        row = {
            "game_id": "g", "player_id": 1, "observation_index": 0,
            "opponent_profile": "rule", "strategic_intent_states": states,
            "reason_contributed_to_selection": contribution,
            "selected_execution": "BUY_DEVELOPMENT",
            "execution_source": "PLANNER_ADAPTIVE_DEVELOPMENT",
            "new_development_cards_blocked": False,
        }

        summary = summarize_strategic_intents([row])

        self.assertEqual(summary["intent_coverage"]["ROBBER_RELIEF"]["active"], 1)
        self.assertEqual(summary["buy_development_attribution"]["KNIGHT_TITLE"], 1)
        self.assertEqual(summary["active_but_not_contributed"]["ROBBER_RELIEF"], 1)


if __name__ == "__main__":
    unittest.main()
