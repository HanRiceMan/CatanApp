"""Turn-level construction follow-through is diagnostic, not an Action rule."""

from app.rl.analyze_construction_followthrough_step3b3a import (
    analyze_episodes, follow_one,
)


def _row(step, family, *, kind="CITY_ONLY", turn=7, score=5,
         city=True, settlement=False, resources=None):
    return {
        "profile": "rule", "seed": 34000000, "seat": 1,
        "turn": turn, "revision": step + 10, "policy_step": step,
        "phase": "action", "immediate_win": False,
        "build_class": kind, "action_family": family,
        "action": ("BankTradeAction(give_resource='wood', receive_resource='ore')"
                   if family == "BANK_TRADE" else family),
        "settlement_vertices": [1] if settlement else [],
        "city_vertices": [2] if city else [],
        "current_score": score,
        "resource_vector": resources or {
            "wood": 0, "brick": 0, "sheep": 0, "wheat": 2, "ore": 3},
    }


_EPISODE = {"terminal": True, "final_vp": 6, "profile": "rule", "seat": 1}


def test_road_title_before_city_still_counts_same_turn_target():
    rows = [_row(1, "BUILD_ROAD"), _row(2, "BUILD_CITY", score=7)]
    result = follow_one(rows, 0, _EPISODE)
    assert result["outcome"] == "BUILT_TARGET_LATER_SAME_TURN"
    assert result["other_score_before_close"]
    assert result["actions_before_build"] == ["BUILD_ROAD"]


def test_temporary_loss_then_recovery_and_city_build():
    rows = [
        _row(1, "BANK_TRADE"),
        _row(2, "PLAYER_TRADE", city=False),
        _row(3, "BUILD_CITY"),
    ]
    result = follow_one(rows, 0, _EPISODE)
    assert result["outcome"] == "BUILT_TARGET_LATER_SAME_TURN"
    assert result["buildability_lost_before_close"]
    assert result["actions_before_build_count"] == 2
    assert result["trade_events_before_close"][1]["returned_to_action_same_turn"]


def test_direct_end_turn_and_no_target_are_distinct():
    result = follow_one([_row(1, "END_TURN")], 0, _EPISODE)
    assert result["outcome"] == "ENDED_TURN_WITHOUT_BUILD"
    assert result["ended_turn_without_target"]
    rows = [_row(1, "BANK_TRADE"), _row(2, "END_TURN", city=False)]
    result = follow_one(rows, 0, _EPISODE)
    assert result["outcome"] == "BUILDABILITY_LOST_WITHOUT_BUILD"
    assert result["ended_turn_without_target"]


def test_same_class_in_turn_deduplicates_opportunity_but_keeps_states():
    rows = [_row(1, "BUILD_ROAD"), _row(2, "BANK_TRADE"),
            _row(3, "BUILD_CITY")]
    states, opportunities, summary = analyze_episodes([(_EPISODE, rows)])
    assert len(states) == 3
    assert len(opportunities) == 1
    assert summary["class_summary"]["CITY_ONLY"]["same_turn_target_build"] == 1


def test_both_records_actual_build_family():
    rows = [_row(1, "BUILD_SETTLEMENT", kind="BOTH", settlement=True)]
    result = follow_one(rows, 0, _EPISODE)
    assert result["outcome"] == "BUILT_TARGET_IMMEDIATELY"
    assert result["built_family"] == "BUILD_SETTLEMENT"
