"""Step 3B-4 routing is factual and cannot change a concrete Action."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from app.domain.actions import (
    BuildCityAction, BuildRoadAction, BuildSettlementAction, EndTurnAction,
    RollTurnDiceAction, UseDevelopmentAction,
)
from app.domain.game import create_game
from app.rl.hierarchical_shadow_controller_step3b4 import (
    FAMILY_ORDER, HierarchicalRoute, classify_hierarchical_route,
    soft_plan_representation, with_soft_plan,
)
from app.rl.runtime_context import RUNTIME_CONTEXT_SIZE
from app.rl.victory_plan_context import PlanFeasibility, build_victory_plan_context
from app.rl.victory_plan_policy import VictoryPlanShadowHeads, encode_victory_plan_features


def _game(phase="action", **changes):
    player = SimpleNamespace(id=1, resources={"wood": 1})
    opponent = SimpleNamespace(id=2, resources={"wood": 0})
    values = dict(phase=phase, current_player_id=1, free_road_remaining=0,
                  pending_trade=None, trade_count=0, players=[player, opponent])
    values.update(changes)
    return SimpleNamespace(**values)


def test_precedence_and_forced_boundaries():
    for phase in ("rolling_order", "setup_ready", "discarding", "robber_move",
                  "robber_steal", "trade_response", "trade_counter_offer"):
        game = _game(phase, pending_trade={"recipient_id": 1}
                     if phase.startswith("trade_") else None)
        assert classify_hierarchical_route(
            game, 1, legal=[],
            immediate_win_available=False).route is HierarchicalRoute.FORCED_PHASE
    game = _game("action", free_road_remaining=1)
    assert classify_hierarchical_route(
        game, 1, legal=[BuildRoadAction(0)],
        immediate_win_available=False).forced_reason == "FREE_ROAD"
    assert classify_hierarchical_route(
        game, 1, legal=[BuildRoadAction(0)],
        immediate_win_available=True).route is HierarchicalRoute.IMMEDIATE_WIN


def test_normal_routes_and_masks_do_not_force_next_action():
    game = _game()
    settlement = classify_hierarchical_route(
        game, 1, legal=[BuildSettlementAction(0), EndTurnAction()],
        immediate_win_available=False)
    assert settlement.route is HierarchicalRoute.CONSTRUCTION_COMMIT
    assert settlement.construction_commit_active
    assert settlement.legal_family_mask[FAMILY_ORDER.index("END_TURN")]
    assert settlement.comparison_family_mask[FAMILY_ORDER.index("END_TURN")]
    assert settlement.player_trade_rule_available
    city = classify_hierarchical_route(
        game, 1, legal=[BuildCityAction(0), EndTurnAction()],
        immediate_win_available=False)
    assert city.route is HierarchicalRoute.CITY_VALUE_COMPARISON
    assert not city.construction_commit_active
    both = classify_hierarchical_route(
        game, 1, legal=[BuildSettlementAction(0), BuildCityAction(1),
                         BuildRoadAction(2), EndTurnAction()],
        immediate_win_available=False)
    assert both.route is HierarchicalRoute.DUAL_CONSTRUCTION_COMPARISON
    assert both.comparison_family_mask[:2] == (True, True)
    assert not any(both.comparison_family_mask[2:])
    assert both.legal_family_mask[FAMILY_ORDER.index("BUILD_ROAD")]
    none = classify_hierarchical_route(
        game, 1, legal=[BuildRoadAction(2), EndTurnAction()],
        immediate_win_available=False)
    assert none.route is HierarchicalRoute.INVESTMENT
    assert not any(none.comparison_family_mask[:2])


def test_development_usage_has_no_use_and_fallback():
    pre = classify_hierarchical_route(
        _game("turn_pre_roll"), 1,
        legal=[RollTurnDiceAction(), UseDevelopmentAction("knight")],
        immediate_win_available=False)
    assert pre.route is HierarchicalRoute.PRE_ROLL_DEV_USAGE
    assert pre.development_usage_mask == (True, True, False, False, False)
    regular = classify_hierarchical_route(
        _game(), 1,
        legal=[BuildCityAction(0), EndTurnAction(),
               UseDevelopmentAction("monopoly", ("ore",))],
        immediate_win_available=False)
    assert regular.route is HierarchicalRoute.ACTION_DEV_USAGE
    assert regular.fallback_route is HierarchicalRoute.CITY_VALUE_COMPARISON
    assert regular.development_usage_mask == (True, False, True, False, False)
    assert classify_hierarchical_route(
        _game("turn_pre_roll"), 1, legal=[RollTurnDiceAction()],
        immediate_win_available=False).route is HierarchicalRoute.FORCED_PHASE


def test_player_trade_is_rule_fact_not_a_protected_controller_choice():
    legal = [BuildRoadAction(2), EndTurnAction()]
    possible = classify_hierarchical_route(
        _game(), 1, legal=legal, immediate_win_available=False)
    exhausted = classify_hierarchical_route(
        _game(trade_count=5), 1, legal=legal, immediate_win_available=False)
    assert possible.player_trade_rule_available
    assert not exhausted.player_trade_rule_available
    assert possible.route is exhausted.route is HierarchicalRoute.INVESTMENT
    assert possible.comparison_family_mask[FAMILY_ORDER.index("PLAYER_TRADE")]
    assert not exhausted.comparison_family_mask[FAMILY_ORDER.index("PLAYER_TRADE")]


def _shadow(plan_context):
    encoded = encode_victory_plan_features(plan_context)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3302)
        heads = VictoryPlanShadowHeads(8).eval()
    with torch.no_grad():
        output = heads(
            torch.zeros(1, 8), torch.zeros(1, RUNTIME_CONTEXT_SIZE),
            torch.tensor(encoded.global_features.copy()).unsqueeze(0),
            torch.tensor(encoded.candidate_features.copy()).unsqueeze(0),
            torch.tensor(encoded.candidate_mask.copy()).unsqueeze(0),
        )
    return soft_plan_representation(plan_context, output)


def test_soft_plan_uniform_neutral_and_permutation_invariant():
    context = build_victory_plan_context(create_game(33000001), 1)
    soft = _shadow(context)
    assert soft.available
    assert np.isclose(sum(soft.candidate_probabilities), 1.0)
    assert all(probability == 0 for probability, valid in
               zip(soft.candidate_probabilities, soft.candidate_mask, strict=True)
               if not valid)
    assert np.allclose(soft.expected_priorities, (0.5, 0.5, 0.5, 0.5), atol=1e-6)
    reordered = _shadow(replace(context, candidates=tuple(reversed(context.candidates))))
    assert np.allclose(soft.weighted_features, reordered.weighted_features)
    assert np.allclose(soft.expected_priorities, reordered.expected_priorities)
    decision = classify_hierarchical_route(
        _game(), 1, legal=[EndTurnAction()], immediate_win_available=False)
    attached = with_soft_plan(decision, soft)
    assert attached.route is decision.route
    assert attached.comparison_family_mask == decision.comparison_family_mask


def test_unavailable_plan_has_no_fake_zero_plan():
    context = build_victory_plan_context(create_game(33000002), 1)
    unavailable = replace(context, current_score=10, remaining_points=0,
                          candidates=(), unresolved=False)
    soft = _shadow(unavailable)
    assert not soft.available
    assert soft.weighted_features is None
    assert soft.expected_priorities is None
    assert soft.candidate_probabilities == ()
    unresolved = replace(
        context,
        candidates=tuple(replace(candidate, feasibility=PlanFeasibility.IMPOSSIBLE,
                                 valid=False) for candidate in context.candidates),
        unresolved=True,
    )
    unresolved_soft = _shadow(unresolved)
    assert not unresolved_soft.available
    assert unresolved_soft.weighted_features is None
    assert unresolved_soft.expected_priorities is None
    assert all(probability == 0 for probability in unresolved_soft.candidate_probabilities)


def test_soft_plan_uses_actual_probabilities_not_first_candidate():
    original = build_victory_plan_context(create_game(33000003), 1)
    candidates = tuple(candidate for candidate in original.candidates if candidate.valid)[:2]
    assert len(candidates) == 2
    context = replace(original, candidates=candidates)
    output = SimpleNamespace(
        probabilities=torch.tensor([[0.25, 0.75]]),
        priorities=torch.tensor([[[0.2, 0.9, 0.3, 0.8],
                                  [0.8, 0.1, 0.7, 0.4]]]),
        candidate_mask=torch.tensor([[True, True]]),
        plan_available=torch.tensor([True]),
    )
    soft = soft_plan_representation(context, output)
    assert np.isclose(soft.weighted_features[0],
                      .25 * candidates[0].settlement_gain_count
                      + .75 * candidates[1].settlement_gain_count)
    assert np.allclose(soft.expected_priorities, (.65, .30, .60, .50), atol=1e-6)
    reverse = replace(context, candidates=tuple(reversed(candidates)))
    reversed_output = SimpleNamespace(
        probabilities=output.probabilities.flip(1),
        priorities=output.priorities.flip(1),
        candidate_mask=output.candidate_mask.flip(1),
        plan_available=output.plan_available,
    )
    reversed_soft = soft_plan_representation(reverse, reversed_output)
    assert np.allclose(soft.weighted_features, reversed_soft.weighted_features)
    assert np.allclose(soft.expected_priorities, reversed_soft.expected_priorities)
