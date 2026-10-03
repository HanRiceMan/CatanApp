"""Step 3A-5の非介入Contextとv5 Teacher schema。"""

import unittest
from copy import deepcopy

import numpy as np

from app.agents.heuristic import HeuristicAgent
from app.domain.game import BUILD_COSTS, create_game, player_for
from app.rl.action_space import ACTION_SPACE_SIZE
from app.rl.macro_teacher_dataset import (
    _attribution_fields, collect_teacher_dataset, summarize_structured_context,
)
from app.rl.observation import OBSERVATION_VECTOR_SIZES
from app.rl.settlement_planning_agent import SettlementPlanningAgent
from app.rl.structured_context import build_structured_context_report


class _Manifest:
    observation_version = "v2"
    observation_size = OBSERVATION_VECTOR_SIZES["v2"]


class _Policy:
    manifest = _Manifest()
    initial_setup_policy = None

    def select_action(self, game, player_id):
        return HeuristicAgent().select_action(game, player_id)


class _Agent(SettlementPlanningAgent):
    def _probabilities(self, game, player_id, mask):
        return np.ones(ACTION_SPACE_SIZE, dtype=np.float32)


class StructuredContextTests(unittest.TestCase):
    def test_context_is_pre_action_and_does_not_change_state_or_rng(self):
        game = create_game(20261013)
        game.phase = "action"
        game.current_player_id = 1
        before = deepcopy(game)
        first = build_structured_context_report(game, 1)
        second = build_structured_context_report(game, 1)
        self.assertEqual(game, before)
        self.assertEqual(first, second)
        self.assertEqual(game.random_source.counter, before.random_source.counter)
        self.assertEqual(first.sections["resource_risk"]["hand_size"].value, 0)
        self.assertFalse(first.sections["city"]["estimated_wait_rounds"].valid)
        self.assertIsNone(first.sections["city"]["estimated_wait_rounds"].value)

    def test_future_city_and_shortage_are_not_immediate_city(self):
        game = create_game(20261014)
        game.phase = "action"
        game.current_player_id = 1
        game.settlements[0] = 1
        player_for(game, 1).settlements = 1
        report = build_structured_context_report(game, 1)
        city = report.sections["city"]
        self.assertEqual(city["upgradeable_count"].value, 1)
        self.assertEqual(city["best_future_vertex"].value, 0)
        self.assertFalse(city["immediate_build_possible"].value)
        self.assertEqual(city["resource_shortage"].value,
                         {name: BUILD_COSTS["city"].get(name, 0)
                          for name in ("wood", "brick", "sheep", "wheat", "ore")})

    def test_new_knight_has_missing_v2_provenance_and_is_not_usable(self):
        game = create_game(20261015)
        game.phase = "turn_pre_roll"
        game.current_player_id = 1
        player = player_for(game, 1)
        player.development_cards.append("knight")
        player.new_development_cards.append("knight")
        report = build_structured_context_report(game, 1)
        knight = report.sections["knight_title"]
        self.assertEqual(knight["usable_knights"].value, 0)
        self.assertEqual(knight["usable_knights"].provenance, "MISSING_FROM_V2")
        self.assertTrue(report.observability_gaps["new_development_cards_effect"])

    def test_structural_road_options_exist_even_before_roll(self):
        game = create_game(20261018)
        game.phase = "turn_pre_roll"
        game.current_player_id = 1
        game.roads[0] = 1
        before = deepcopy(game)
        report = build_structured_context_report(game, 1)
        self.assertGreater(
            report.sections["road_title"]["structurally_extendable_edge_count"].value, 0)
        self.assertEqual(game, before)

    def test_v5_collects_tactical_phase_and_observation_reference(self):
        dataset = collect_teacher_dataset(_Agent(_Policy()), [20261016])
        self.assertTrue(any(row["phase"] == "turn_pre_roll" for row in dataset.records))
        self.assertTrue(any(row["phase"] == "action" for row in dataset.records))
        self.assertEqual(len(dataset.records), len(dataset.observations))
        for row in dataset.records:
            self.assertEqual(row["observation_index"], row["observation_reference"])
            self.assertIn("strategic_context_valid_mask", row)
            self.assertIn("source_confirmed_reasons", row)
        summary = summarize_structured_context(dataset.records)
        self.assertEqual(summary["decision_count"], len(dataset.records))
        self.assertFalse(summary["sentinel_or_invalid_value_count"])

    def test_requested_seat_is_actual_seat_across_four_games(self):
        dataset = collect_teacher_dataset(
            _Agent(_Policy()), range(20261020, 20261024),
        )
        self.assertEqual([game["seat"] for game in dataset.games], [1, 2, 3, 4])

    def test_base_ppo_reason_is_unobserved_not_inferred(self):
        fields = _attribution_fields({
            "execution_source": "BASE_PPO",
            "reason_contributed_to_selection": {"ROBBER_RELIEF": None},
            "selection_reason_codes": ["ROBBER_BLOCKING_IMPORTANT_TILE"],
        })
        self.assertEqual(fields["attribution_status"], "UNOBSERVED_BASE_PPO")
        self.assertEqual(fields["source_confirmed_reasons"], [])


if __name__ == "__main__":
    unittest.main()
