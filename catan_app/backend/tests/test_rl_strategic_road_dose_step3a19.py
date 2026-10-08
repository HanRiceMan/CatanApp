"""Read-only Road intervention and dose-response accounting guards."""

import pytest
import numpy as np
from copy import deepcopy

from app.domain.game import create_game, legal_road_ids, player_for
from app.rl.action_space import ACTION_SPACE_SIZE

from app.rl.run_strategic_road_dose_step3a19 import (
    RATES, _edge, edge_diagnostics, family_of_repr, summarize_dose, summarize_road,
)


def test_road_action_parser_is_exact():
    assert _edge("BuildRoadAction(edge_id=7)") == 7
    with pytest.raises(ValueError):
        _edge("BuildCityAction(vertex_id=7)")
    assert family_of_repr("BuildRoadAction(edge_id=7)") == "BUILD_ROAD"
    assert family_of_repr("ProposeTradeAction(target_player_id=2)") == "PROTECTED_PLAYER_TRADE"


def test_edge_diagnostics_does_not_change_game_or_rng():
    game = create_game(20261901)
    game.phase = "action"
    game.current_player_id = 1
    game.settlements[0] = 1
    player = player_for(game, 1)
    player.settlements = 1
    player.resources.update({"wood": 1, "brick": 1, "sheep": 0, "wheat": 0, "ore": 0})
    edge = legal_road_ids(game, 1, initial=False)[0]
    before = deepcopy(game)
    row = edge_diagnostics(game, 1, edge, np.zeros(ACTION_SPACE_SIZE))
    assert row["legal"] and row["candidate_logit_rank"] >= 1
    assert row["longest_road_after"] >= row["longest_road_before"]
    assert game == before


def test_road_summary_keeps_state_and_future_units_separate():
    edge = {"structural_class": "expansion_only", "candidate_logit_rank": 1,
            "frozen_candidate_logit": 0.2,
            "longest_road_before": 2, "longest_road_after": 2,
            "newly_reachable_empty_vertices_within_three_roads": [3],
            "legal_settlement_count_after": 1,
            "legal_road_count_after": 2,
            "minimum_additional_roads_before": 2,
            "minimum_additional_roads_after": 1}
    row = {"state_id": "rule-1-5", "profile": "rule", "seat": 1,
           "same_edge": False, "b_edge": edge, "champion_edge": edge}
    outcome = {"policy_steps": 50, "won": True,
               "window30": {"action_counts": {"BUILD_CITY": 1},
                            "settlement_access": [
                                {"legal_settlement_count": 0,
                                 "minimum_additional_roads": 2,
                                 "reachable_empty_vertices_within_three_roads": 3},
                                {"legal_settlement_count": 1,
                                 "minimum_additional_roads": 1,
                                 "reachable_empty_vertices_within_three_roads": 4}]}}
    pairs = [
        {"state_id": "rule-1-5", "branches": {"safety": {
            "delta_return": value, "delta_vp": 1, "delta_rank": -1,
            "winner_flip": True, "b": outcome, "champion": outcome}}}
        for value in (.4, -.2)
    ]
    summary = summarize_road([row], pairs)
    assert summary["both_road"] == summary["different_edge"] == 1
    assert summary["safety"]["pairs"] == 2
    assert summary["safety"]["b_favored_states"] == 1
    assert summary["safety"]["unstable_2_each"] == 0
    assert summary["safety"]["winner_flips"] == 2


def test_dose_summary_uses_actual_delegation_denominator():
    def record(rate):
        delegated = int(rate > 0)
        return {"profile": "rule", "seat": 1, "actual_delegated_count": delegated,
                "learner_action_families": {"BUILD_SETTLEMENT": 2 - delegated,
                                            "BUY_DEVELOPMENT": delegated},
                "won": False, "final_vp": 7 - delegated, "final_rank": 2,
                "policy_steps": 50 + delegated, "terminal": True, "truncated": False,
                "gate_surface_count": 3, "protected_preemption_count": 1,
                "protected_trade_available_count": 1,
                "delegation_lottery_count": 2, "first_divergence": (
                    {"safety_family": "BUILD_ROAD", "gate_family": "BUILD_ROAD"}
                    if delegated else None)}
    summary = summarize_dose({f"{rate:g}": [record(rate)] for rate in RATES})
    assert summary["0.1"]["per_delegation_reference"]["construction"] == -1
    assert summary["0.1"]["per_delegation_reference"]["development"] == 1
    assert summary["0.1"]["first_road_road"] == 1
    assert summary["0"]["per_delegation_reference"] is None
