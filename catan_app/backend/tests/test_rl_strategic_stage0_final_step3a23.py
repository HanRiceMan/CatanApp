"""Fixed six-set Stage 0 qualification boundaries and paired inference."""

from collections import Counter

import pytest

from app.rl.run_strategic_stage0_final_step3a23 import (
    ALL_SETS, NEW_SETS, SET_START, seed_rows, stratified_bootstrap,
)


def test_final_seed_sets_are_preregistered_balanced_and_disjoint():
    assert NEW_SETS == "EF"
    assert SET_START["E"] == 32500000
    assert SET_START["F"] == 32600000
    seeds = []
    for set_id in ALL_SETS:
        rows = list(seed_rows(set_id))
        assert len(rows) == 72
        assert Counter((p, seat) for p, _, seat in rows) == {
            (p, seat): 6 for p in ("rule", "champion", "mixed")
            for seat in range(1, 5)}
        seeds.extend(seed for _, seed, _ in rows)
    assert len(seeds) == len(set(seeds)) == 432


def test_final_bootstrap_retains_eighteen_set_profile_strata():
    rows = [{"set_id": sid, "profile": p,
             **{f"delta_{field}": float(index) for field in
                ("vp", "rank", "policy_steps", "construction", "development")}}
            for sid in ALL_SETS for p in ("rule", "champion", "mixed")
            for index in range(24)]
    result = stratified_bootstrap(rows, seed=42)
    assert result["vp"]["mean"] == pytest.approx(11.5)
    assert result["vp"]["ci95_low"] < 11.5 < result["vp"]["ci95_high"]
    with pytest.raises(AssertionError, match="18 set/profile strata"):
        stratified_bootstrap(rows[:-1], seed=42)
    leave_e = stratified_bootstrap(rows, seed=42, set_ids=ALL_SETS.replace("E", ""))
    assert leave_e["vp"]["mean"] == pytest.approx(11.5)
