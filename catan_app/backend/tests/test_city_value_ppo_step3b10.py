"""City Value route, training mask, and independent Critic boundaries."""

from types import SimpleNamespace

import pytest
import torch

from app.rl.city_value_ppo_step3b10 import (
    CITY_FAMILIES, CityValueCritic, city_value_training_mask,
)
from app.rl.hierarchical_l3_shadow_step3b5 import ROUTE_ORDER
from app.rl.hierarchical_ppo_step3b6 import _batch
from app.rl.hierarchical_rollout_step3b6 import _finish_city_opportunity
from app.rl.hierarchical_shadow_controller_step3b4 import FAMILY_ORDER, HierarchicalRoute
from app.domain.actions import BuildCityAction, BuildRoadAction, EndTurnAction


def test_city_training_mask_legal_executor_and_protected_trade():
    diagnostic = (False, True, True, True, True, True, True)
    executable = (False, True, True, True, False, False, True)
    mask = city_value_training_mask(HierarchicalRoute.CITY_VALUE_COMPARISON,
                                    diagnostic, executable)
    assert tuple(FAMILY_ORDER[i] for i, selected in enumerate(mask) if selected) == (
        "BUILD_CITY", "BUILD_ROAD", "BUY_DEVELOPMENT", "END_TURN")
    assert all(not value for value in city_value_training_mask(
        HierarchicalRoute.INVESTMENT, diagnostic, executable))
    assert "PLAYER_TRADE" not in CITY_FAMILIES
    with pytest.raises(ValueError):
        city_value_training_mask(HierarchicalRoute.CITY_VALUE_COMPARISON,
                                 diagnostic[:-1], executable)


def test_city_batch_route_is_not_relabelled_investment():
    decision = {
        "route": HierarchicalRoute.CITY_VALUE_COMPARISON.value,
        "family_mask": [False, True, False, False, False, False, True],
        "diagnostic_family_mask": [False, True, False, False, False, True, True],
        "actor_latent": [0.] * 16,
        "runtime_context": [0.] * 79,
        "global_plan_features": [0.] * 30,
        "candidate_plan_features": [[0.] * 9 for _ in range(36)],
        "candidate_plan_mask": [False] * 36,
        "selected_index": 1, "old_log_prob": -0.69,
        "old_probability": [0., .5, 0., 0., 0., 0., .5], "old_value": 0.,
    }
    batch = _batch([decision])
    assert batch["facts"].route_one_hot[0, ROUTE_ORDER.index(
        HierarchicalRoute.CITY_VALUE_COMPARISON)] == 1
    assert batch["facts"].route_one_hot[0, ROUTE_ORDER.index(
        HierarchicalRoute.INVESTMENT)] == 0
    assert not batch["family_mask"][0, FAMILY_ORDER.index("PLAYER_TRADE")]


def test_city_critic_is_independent_and_requires_city_route():
    critic = CityValueCritic(16)
    route = torch.zeros(2, len(ROUTE_ORDER))
    route[:, ROUTE_ORDER.index(HierarchicalRoute.CITY_VALUE_COMPARISON)] = 1
    bridge = SimpleNamespace(weighted_features=torch.zeros(2, 6),
                             expected_priorities=torch.full((2, 4), .5),
                             available=torch.ones(2, dtype=torch.bool))
    values = critic(torch.zeros(2, 16), torch.zeros(2, 79), torch.zeros(2, 30),
                    bridge, torch.ones(2, 7, dtype=torch.bool), route_one_hot=route)
    assert values.shape == (2,)
    with pytest.raises(ValueError):
        critic(torch.zeros(2, 16), torch.zeros(2, 79), torch.zeros(2, 30),
               bridge, torch.ones(2, 7, dtype=torch.bool))
    route[:, :] = 0
    route[:, ROUTE_ORDER.index(HierarchicalRoute.INVESTMENT)] = 1
    with pytest.raises(ValueError):
        critic(torch.zeros(2, 16), torch.zeros(2, 79), torch.zeros(2, 30),
               bridge, torch.ones(2, 7, dtype=torch.bool), route_one_hot=route)


def test_city_opportunity_records_end_turn_even_when_observer_turn_advanced():
    active = {"turn": 5, "learner_id": 2, "trace_index": 0,
              "first_action_family": "END_TURN", "first_action_delegated": False}
    result = _finish_city_opportunity(active, [(6, 2, EndTurnAction())])
    assert result["action_families"] == ["END_TURN"]
    assert not result["city_same_turn"]


def test_city_opportunity_follows_through_after_other_learner_action():
    active = {"turn": 8, "learner_id": 2, "trace_index": 0,
              "first_action_family": "BUILD_ROAD", "first_action_delegated": False}
    result = _finish_city_opportunity(active, [
        (8, 2, BuildRoadAction(edge_id=0)), (8, 3, EndTurnAction()),
        (8, 2, BuildCityAction(vertex_id=0)),
    ])
    assert result["city_same_turn"]
    assert not result["city_immediately"]
    assert result["actions_before_city"] == 1
