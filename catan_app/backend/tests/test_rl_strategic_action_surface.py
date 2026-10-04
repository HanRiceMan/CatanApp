"""Step 3A-12 state-based candidate mask and non-intervention checks."""

from copy import deepcopy
import unittest

import numpy as np

from app.domain.actions import (BankTradeAction, BuildCityAction, BuildRoadAction,
                                BuildSettlementAction, BuyDevelopmentAction,
                                EndTurnAction, ProposeTradeAction, TradeOffer,
                                UseDevelopmentAction)
from app.domain.game import create_game, player_for
from app.rl.action_space import ACTION_SPACE_SIZE, action_to_id
from app.rl.strategic_action_surface import (
    FAMILIES, detect_strategic_action_surface, family_of,
    summarize_strategic_surface,
)


class StrategicActionSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.game = create_game(20261301)
        self.game.phase = "action"
        self.game.current_player_id = 1
        self.mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        self.mask[action_to_id(EndTurnAction())] = True

    def test_candidate_mask_uses_only_state_and_legal_actions(self):
        self.mask[action_to_id(BuildCityAction(3))] = True
        self.mask[action_to_id(BuyDevelopmentAction())] = True
        self.mask[action_to_id(BankTradeAction("wood", "ore"))] = True
        before = deepcopy(self.game)
        surface = detect_strategic_action_surface(self.game, 1, self.mask)
        self.assertEqual(surface.family_mask, (False, True, False, True, True, True))
        self.assertEqual(surface.candidate_count, 4)
        self.assertTrue(surface.eligible)
        self.assertEqual(self.game, before)

    def test_one_candidate_is_forced_not_gate_decision(self):
        surface = detect_strategic_action_surface(self.game, 1, self.mask)
        self.assertEqual(surface.candidate_count, 1)
        self.assertFalse(surface.eligible)
        self.assertEqual(surface.hard_exclusion, "FORCED_SINGLE_ACTION")

    def test_hard_exclusions_do_not_require_champion_action(self):
        self.mask[action_to_id(BuildSettlementAction(0))] = True
        player_for(self.game, 1).development_cards.extend(["victory_point"] * 9)
        surface = detect_strategic_action_surface(self.game, 1, self.mask)
        self.assertEqual(surface.hard_exclusion, "IMMEDIATE_CONSTRUCTION_WIN")
        self.assertFalse(surface.eligible)
        self.game.phase = "robber_move"
        self.assertEqual(detect_strategic_action_surface(
            self.game, 1, self.mask).hard_exclusion, "OUTSIDE_ACTION_PHASE")

    def test_road_diagnostics_preserve_state_and_are_not_gate_subfamilies(self):
        self.mask[action_to_id(BuildRoadAction(0))] = True
        before = deepcopy(self.game)
        surface = detect_strategic_action_surface(self.game, 1, self.mask)
        self.assertEqual(FAMILIES[2], "BUILD_ROAD")
        self.assertTrue(surface.family_mask[2])
        purposes = surface.road_purposes
        self.assertEqual(set(purposes), {"expansion_edges", "longest_progress_edges",
                                         "both_edges", "other_edges"})
        self.assertEqual(self.game, before)

    def test_protected_actions_remain_outside_mask(self):
        self.assertEqual(family_of(ProposeTradeAction(
            2, (TradeOffer({"wood": 1}, {"ore": 1}),))),
            "PROTECTED_PLAYER_TRADE")
        self.assertEqual(family_of(UseDevelopmentAction("knight")),
                         "PROTECTED_PLAY_DEVELOPMENT")
        self.assertNotIn("PROTECTED_PLAYER_TRADE", FAMILIES)

    def test_summary_counts_unrepresented_action_separately(self):
        self.mask[action_to_id(BuildCityAction(3))] = True
        surface = detect_strategic_action_surface(self.game, 1, self.mask)
        base = surface.to_dict()
        records = [dict(base, profile="rule", seat=1,
                        champion_family="BUILD_CITY",
                        champion_in_candidate_set=True,
                        champion_execution_source="CITY_PLANNER",
                        champion_action_fields={"vertex_id": 3}),
                   dict(base, profile="rule", seat=1,
                        champion_family="PROTECTED_PLAYER_TRADE",
                        champion_in_candidate_set=False,
                        champion_execution_source="TRADE_PLANNER",
                        champion_action_fields={})]
        summary = summarize_strategic_surface(records, [{"parity": True}])
        self.assertEqual(summary["gate_eligible_decisions"], 2)
        self.assertEqual(summary["champion_family_represented_eligible"], 1)
        self.assertEqual(summary["champion_family_unrepresented_eligible"],
                         {"PROTECTED_PLAYER_TRADE": 1})


if __name__ == "__main__":
    unittest.main()
