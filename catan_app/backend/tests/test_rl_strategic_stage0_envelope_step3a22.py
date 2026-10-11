"""Deterministic seed strata, paired bootstrap and divergence boundaries."""

from collections import Counter

import pytest

from app.rl.run_strategic_stage0_envelope_step3a22 import (
    SET_START, _first_divergence, seed_rows, set_profile_bootstrap,
)


def test_envelope_seed_sets_are_balanced_and_disjoint():
    all_seeds = []
    for set_id in "ABCD":
        rows = list(seed_rows(set_id))
        assert len(rows) == 72
        assert Counter((profile, seat) for profile, _, seat in rows) == {
            (profile, seat): 6 for profile in ("rule", "champion", "mixed")
            for seat in range(1, 5)}
        assert min(seed for _, seed, _ in rows) == SET_START[set_id]
        all_seeds.extend(seed for _, seed, _ in rows)
    assert len(all_seeds) == len(set(all_seeds)) == 288


def test_pooled_bootstrap_preserves_set_and_profile_strata():
    rows = [{"set_id": set_id, "profile": profile,
             **{f"delta_{field}": float(index) for field in
                ("vp", "rank", "policy_steps", "construction", "development")}}
            for set_id in "ABCD" for profile in ("rule", "champion", "mixed")
            for index in range(24)]
    result = set_profile_bootstrap(rows, seed=123)
    assert result["vp"]["mean"] == pytest.approx(11.5)
    assert result["vp"]["ci95_low"] < 11.5 < result["vp"]["ci95_high"]
    with pytest.raises(AssertionError, match="12 set/profile strata"):
        set_profile_bootstrap(rows[:-1], seed=123)


def test_first_concrete_divergence_not_just_family_divergence():
    safety = {"winner": 1, "won": True, "final_vp": 10, "final_rank": 1,
              "policy_steps": 30, "rng_counter": 7,
              "learner_action_families": {"BUILD_ROAD": 1}}
    row = {**safety, "decisions": [{
        "pre_policy_step": 8, "selected_family": "BUILD_ROAD",
        "champion_counterfactual_family": "BUILD_ROAD",
        "concrete_action": "BuildRoadAction(edge_id=2)",
        "champion_counterfactual_action": "BuildRoadAction(edge_id=1)"}]}
    first = _first_divergence(row, safety)
    assert first["pair"] == "BUILD_ROAD -> BUILD_ROAD edge difference"
    assert first["policy_step"] == 8
    row["decisions"][0]["concrete_action"] = "BuildRoadAction(edge_id=1)"
    assert _first_divergence(row, safety) is None
