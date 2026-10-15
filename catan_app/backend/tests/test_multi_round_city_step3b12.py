"""The two City rounds must use distinct data and preserve terminal semantics."""

import pytest
from types import SimpleNamespace

from app.rl.run_multi_round_city_step3b12 import (
    EVAL_BASES, EVAL_PER_PROFILE, TARGET, TRAIN_BASES, TRAIN_RATE,
    VALIDATION_BASES, _realized_returns,
    _raw_safety_parity,
)


def test_fixed_seed_ranges_disjoint_and_balanced():
    ranges = []
    for bases, count in ((TRAIN_BASES[1], 600), (TRAIN_BASES[2], 600),
                         (VALIDATION_BASES, 8), (EVAL_BASES, EVAL_PER_PROFILE)):
        for base in bases.values():
            ranges.append(set(range(base, base + count)))
    assert all(not left & right for i, left in enumerate(ranges)
               for right in ranges[i + 1:])
    assert EVAL_PER_PROFILE % 4 == 0
    assert TARGET >= 256 and TRAIN_RATE == .5


def test_realized_semi_mdp_return_uses_macro_discount_and_terminal_zero():
    transitions = [
        {"reward_accumulated": .2, "discount": .99 ** 3,
         "truncated": False},
        {"reward_accumulated": 1.1, "discount": 0.,
         "truncated": False},
    ]
    actual = _realized_returns(transitions)
    assert actual == pytest.approx([.2 + .99 ** 3 * 1.1, 1.1])


def test_truncation_never_becomes_terminal_return():
    with pytest.raises(ValueError, match="Truncated"):
        _realized_returns([{"reward_accumulated": 0., "discount": .99,
                            "truncated": True}])


def test_zero_percent_parity_compares_typed_actions_before_serialization():
    action = object()
    state = object()
    episode = {"action_trace": [(1, action)], "final_state": state,
               "rng_counter": 8, "winner": 1}
    reference = SimpleNamespace(action_trace=[(1, action)], final_state=state,
                                rng_counter=8, winner=1)
    assert _raw_safety_parity(episode, reference)
    serialized = {**episode, "action_trace": [[1, repr(action)]]}
    assert not _raw_safety_parity(serialized, reference)
