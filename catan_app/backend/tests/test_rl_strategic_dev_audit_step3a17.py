"""Distribution and intervention-selection guards for the read-only audit."""

import numpy as np
import pytest
from collections import Counter
from types import SimpleNamespace

from app.rl.action_families import ACTION_FAMILIES
from app.rl.action_space import ACTION_SPACE_SIZE
from app.rl.hierarchical_distribution import ACTION_FAMILY_IDS
from app.rl.strategic_dev_audit_step3a17 import (
    diagnostic_distribution, select_representative_states)


def _legal_mask(*family_names):
    selected = {ACTION_FAMILIES.index(name) for name in family_names}
    ids = np.asarray(ACTION_FAMILY_IDS)
    assert ids.shape == (ACTION_SPACE_SIZE,)
    return np.isin(ids, list(selected))


def test_temperature_and_candidate_mask_effects_are_separate():
    native = np.zeros(15)
    native[ACTION_FAMILIES.index("development_purchase")] = 2.0
    native[ACTION_FAMILIES.index("development_use")] = 4.0
    legal = _legal_mask("development_purchase", "development_use", "end_turn")
    result = diagnostic_distribution(native, legal,
                                     (False, False, False, True, False, True))
    assert result["native15_t1_dev_probability"] < result["native15_t2_dev_probability"]
    assert result["strategic6_t2_dev_probability"] > result["native15_t2_dev_probability"]
    assert result["removed_family_mass_t2"]["development_use"] > .5
    assert result["mask_inflation_same_t2"] > 0


def test_no_removed_family_means_no_mask_inflation():
    native = np.zeros(15)
    legal = _legal_mask("development_purchase", "end_turn")
    result = diagnostic_distribution(native, legal,
                                     (False, False, False, True, False, True))
    assert result["removed_mass_t2"] == pytest.approx(0)
    assert result["mask_inflation_same_t2"] == pytest.approx(0)
    assert result["native15_t2_dev_probability"] == pytest.approx(.5)


def test_mismatched_candidate_and_377_masks_rejected():
    native = np.zeros(15)
    legal = _legal_mask("development_purchase", "end_turn")
    with pytest.raises(AssertionError, match="masks disagree"):
        diagnostic_distribution(native, legal,
                                (False, False, False, False, False, True))


def test_representative_selection_balances_profiles():
    states = []
    for profile in ("rule", "champion", "mixed"):
        for index in range(12):
            states.append(SimpleNamespace(row={
                "profile": profile, "seed": index,
                "pre_policy_step": index,
                "selected_dev_probability": .91 if index % 2 else .81,
                "construction_before": {"settlement": index == 0,
                                        "city": False},
                "champion_counterfactual_family": "BUILD_ROAD" if index % 2
                                                   else "END_TURN"}))
    selected = select_representative_states(states, 24)
    assert Counter(item.row["profile"] for item in selected) == {
        "rule": 8, "champion": 8, "mixed": 8}
    assert len({(item.row["profile"], item.row["seed"])
                for item in selected}) == 24
