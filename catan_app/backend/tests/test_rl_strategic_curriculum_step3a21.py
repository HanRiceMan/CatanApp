"""Stage 0 balanced scheduling and guarded-update boundaries."""

from collections import Counter

import pytest

from app.rl.run_strategic_curriculum_step3a21 import (
    DELEGATION, EVAL_SEED_START, MINIBATCH_SEED, PARENT,
    TEMPERATURE, evaluation_rows, training_block, update_once,
)


def test_stage0_training_blocks_are_balanced_and_disjoint_from_heldout():
    first = list(training_block(0))
    second = list(training_block(1))
    assert len(first) == len(second) == 12
    assert Counter((profile, seat) for profile, _, seat in first) == {
        (profile, seat): 1 for profile in ("rule", "champion", "mixed")
        for seat in range(1, 5)}
    assert not {seed for _, seed, _ in first} & {seed for _, seed, _ in second}
    heldout = list(evaluation_rows())
    assert len(heldout) == 72
    assert not {seed for _, seed, _ in first + second} & {
        seed for _, seed, _ in heldout}
    assert min(seed for _, seed, _ in heldout) == EVAL_SEED_START
    assert DELEGATION == .10 and TEMPERATURE == 2.0
    assert PARENT.name == "Strategic-T2-preupdate"
    assert MINIBATCH_SEED not in {seed for _, seed, _ in heldout}


def test_update_cannot_start_without_preflight_or_after_attempt(tmp_path):
    with pytest.raises(FileNotFoundError):
        update_once(tmp_path)
    (tmp_path / "update_attempt_started.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already attempted"):
        update_once(tmp_path)
