"""Stage 0 qualification accounting; never updates Policy parameters."""

from collections import Counter

import pytest

from app.rl.run_strategic_curriculum_step3a20 import (
    PROFILES, RATES, curriculum_manifest, paired_rows, seed_rows,
    stratified_paired_bootstrap, validate_episode,
)


def test_qualification_seeds_are_new_balanced_and_paired():
    rows = list(seed_rows())
    assert len(rows) == 72
    assert len({seed for _, seed, _ in rows}) == 72
    assert min(seed for _, seed, _ in rows) == 32000000
    assert Counter(profile for profile, _, _ in rows) == dict.fromkeys(PROFILES, 24)
    assert all(sum(profile == p and seat == s for profile, _, seat in rows) == 6
               for p in PROFILES for s in range(1, 5))
    assert RATES == (0.0, 0.1, 0.2)


def test_curriculum_is_preupdate_and_not_automatically_promoted():
    manifest = curriculum_manifest()
    stage0 = manifest["stage0"]
    assert stage0["parent_artifact"] == "Strategic-T2-preupdate"
    assert stage0["delegation_probability"] == .1
    assert stage0["training_transition_target"] == [256, 320]
    assert manifest["promotion_criteria"]["automatic_promotion"] is False
    assert manifest["stage1_candidate"]["promotion_status"] == "LOCKED"


def test_stratified_paired_bootstrap_preserves_profile_balance():
    rows = []
    for profile in PROFILES:
        for seed in range(24):
            rows.append({"profile": profile, "delta_vp": 1.0,
                         "delta_rank": -1.0, "delta_policy_steps": 2.0,
                         "delta_construction": 3.0, "delta_development": -2.0})
    result = stratified_paired_bootstrap(rows, resamples=2000, seed=11)
    assert result["vp"] == {"mean": 1.0, "ci95_low": 1.0, "ci95_high": 1.0}
    assert result["construction"]["ci95_low"] == 3.0
    with pytest.raises(ValueError):
        stratified_paired_bootstrap(rows, resamples=1999)


def test_paired_differences_match_game_seed_not_row_order():
    def row(profile, seed, vp, settlement, dev):
        return {"profile": profile, "seed": seed, "seat": 1,
                "final_vp": vp, "final_rank": 2, "policy_steps": 50,
                "learner_action_families": {"BUILD_SETTLEMENT": settlement,
                                            "BUY_DEVELOPMENT": dev}}
    safety = [row("rule", 1, 7, 2, 1), row("champion", 2, 6, 3, 2)]
    test = [row("champion", 2, 8, 1, 3), row("rule", 1, 9, 4, 1)]
    paired = paired_rows(test, safety)
    assert [(item["delta_vp"], item["delta_construction"],
             item["delta_development"]) for item in paired] == [(2, -2, 1), (2, 2, 0)]


def test_validate_episode_rejects_off_policy_and_failed_probe():
    base = {"seed": 42, "learner_id": 1,
            "mode": "safety", "delegation_probability": 0.0,
            "delegation_seed": None, "gate_sampling_seed": None,
            "terminal": True, "truncated": False, "final_vp": 8,
            "final_rank": 1, "policy_steps": 100,
            "actual_delegated_count": 0, "decisions": [],
            "transitions": [], "road_diagnostics": []}
    validate_episode(base, 0.0)
    decision = {"forced_integration": False, "concrete_action_family": "BUILD_ROAD",
                "selected_family": "BUILD_ROAD", "protected_player_trade_selected": False,
                "application_probe": {"game_rng_before": 2, "probe_rng_after": 2,
                                      "road_owner_after": 1,
                                      "resource_before": {"wood": 2, "brick": 2},
                                      "resource_after": {"wood": 1, "brick": 1}}}
    row = {**base, "mode": "strategic", "delegation_probability": .1,
           "delegation_seed": 42 + 12345, "gate_sampling_seed": 42 + 54321,
           "actual_delegated_count": 1, "decisions": [decision],
           "transitions": [{}], "road_diagnostics": [{"b_legal": True}]}
    validate_episode(row, .1)
    with pytest.raises(AssertionError):
        validate_episode(row, 0.0)
    row["decisions"][0]["protected_player_trade_selected"] = True
    with pytest.raises(AssertionError):
        validate_episode(row, .1)
