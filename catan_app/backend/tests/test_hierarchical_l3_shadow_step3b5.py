"""Step 3B-5: differentiable bridge, route masks, and zero-impact L3 head."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.domain.actions import (
    BuildCityAction, BuildRoadAction, BuildSettlementAction, EndTurnAction,
    RollTurnDiceAction, UseDevelopmentAction,
)
from app.domain.game import create_game
from app.rl.hierarchical_l3_shadow_step3b5 import (
    FAMILY_ORDER, L3ShadowActor, differentiable_soft_plan,
    encode_l3_decision_facts,
)
from app.rl.hierarchical_shadow_controller_step3b4 import (
    HierarchicalRoute, classify_hierarchical_route, soft_plan_representation,
)
from app.rl.runtime_context import RUNTIME_CONTEXT_SIZE
from app.rl.victory_plan_context import PlanFeasibility, build_victory_plan_context
from app.rl.victory_plan_policy import VictoryPlanShadowHeads, encode_victory_plan_features


def _game(phase="action"):
    return SimpleNamespace(
        phase=phase, current_player_id=1, free_road_remaining=0,
        pending_trade=None, trade_count=0,
        players=[SimpleNamespace(id=1, resources={"wood": 1}),
                 SimpleNamespace(id=2, resources={"wood": 0})],
    )


def _decision(phase="action", legal=()):
    return classify_hierarchical_route(
        _game(phase), 1, legal=list(legal), immediate_win_available=False)


def _plan_forward(context, heads, latent):
    encoded = encode_victory_plan_features(context)
    global_features = torch.tensor(encoded.global_features.copy()).unsqueeze(0)
    candidate_features = torch.tensor(encoded.candidate_features.copy()).unsqueeze(0)
    output = heads(
        latent, torch.zeros(1, RUNTIME_CONTEXT_SIZE), global_features,
        candidate_features, torch.tensor(encoded.candidate_mask.copy()).unsqueeze(0),
    )
    return output, candidate_features, global_features


def test_bridge_matches_diagnostic_and_is_permutation_invariant():
    context = build_victory_plan_context(create_game(33000001), 1)
    torch.manual_seed(35)
    heads = VictoryPlanShadowHeads(8)
    with torch.no_grad():
        heads.plan_scorer.output.weight.fill_(0.08)
        heads.priority_head.output.weight.fill_(0.03)
    latent = torch.zeros(1, 8)
    output, features, global_features = _plan_forward(context, heads, latent)
    bridge = differentiable_soft_plan(output, features)
    diagnostic = soft_plan_representation(context, output)
    assert bridge.available.tolist() == [diagnostic.available]
    assert np.allclose(bridge.weighted_features.detach().numpy()[0],
                       diagnostic.weighted_features, atol=2e-6)
    assert np.allclose(bridge.expected_priorities.detach().numpy()[0],
                       diagnostic.expected_priorities, atol=2e-6)
    assert torch.all(output.probabilities[~output.candidate_mask] == 0)
    shuffled = replace(context, candidates=tuple(reversed(context.candidates)))
    other_output, other_features, other_global = _plan_forward(shuffled, heads, latent)
    other_bridge = differentiable_soft_plan(other_output, other_features)
    assert torch.allclose(bridge.weighted_features, other_bridge.weighted_features, atol=2e-6)
    assert torch.allclose(bridge.expected_priorities, other_bridge.expected_priorities, atol=2e-6)
    actor = L3ShadowActor(8)
    facts = encode_l3_decision_facts([_decision(legal=(BuildRoadAction(0), EndTurnAction()))])
    old_l3 = actor(latent, torch.zeros(1, RUNTIME_CONTEXT_SIZE),
                   global_features, bridge, facts)
    new_l3 = actor(latent, torch.zeros(1, RUNTIME_CONTEXT_SIZE),
                   other_global, other_bridge, facts)
    assert torch.allclose(old_l3.raw_logits, new_l3.raw_logits)
    assert torch.allclose(old_l3.probabilities, new_l3.probabilities)


def test_bridge_batch_padding_impossible_and_unavailable():
    original = build_victory_plan_context(create_game(33000002), 1)
    impossible = replace(
        original, candidates=tuple(replace(candidate, valid=False,
                                           feasibility=PlanFeasibility.IMPOSSIBLE)
                                   for candidate in original.candidates), unresolved=True,
    )
    heads = VictoryPlanShadowHeads(8)
    first = encode_victory_plan_features(original)
    second = encode_victory_plan_features(impossible)
    candidates = torch.tensor(np.stack([first.candidate_features,
                                        second.candidate_features]).copy())
    mask = torch.tensor(np.stack([first.candidate_mask, second.candidate_mask]).copy())
    output = heads(torch.zeros(2, 8), torch.zeros(2, RUNTIME_CONTEXT_SIZE),
                   torch.tensor(np.stack([first.global_features,
                                          second.global_features]).copy()),
                   candidates, mask)
    bridge = differentiable_soft_plan(output, candidates)
    assert bridge.available.tolist() == [True, False]
    assert torch.all(bridge.weighted_features[1] == 0)
    assert torch.all(bridge.expected_priorities[1] == 0)
    assert torch.all(output.probabilities[1] == 0)
    assert torch.all(output.probabilities[0, ~mask[0]] == 0)


def test_route_masks_and_no_use_fallback():
    decisions = [
        _decision(legal=(BuildRoadAction(0), EndTurnAction())),
        _decision(legal=(BuildCityAction(0), BuildRoadAction(0), EndTurnAction())),
        _decision(legal=(BuildSettlementAction(0), BuildCityAction(1), EndTurnAction())),
        _decision(legal=(BuildSettlementAction(0), EndTurnAction())),
        _decision(legal=(BuildCityAction(0), EndTurnAction(),
                         UseDevelopmentAction("knight"))),
        _decision("turn_pre_roll", (RollTurnDiceAction(),
                                     UseDevelopmentAction("knight"))),
        classify_hierarchical_route(_game(), 1, legal=[EndTurnAction()],
                                    immediate_win_available=True),
        _decision("robber_move"),
    ]
    assert [decision.route for decision in decisions] == [
        HierarchicalRoute.INVESTMENT, HierarchicalRoute.CITY_VALUE_COMPARISON,
        HierarchicalRoute.DUAL_CONSTRUCTION_COMPARISON,
        HierarchicalRoute.CONSTRUCTION_COMMIT, HierarchicalRoute.ACTION_DEV_USAGE,
        HierarchicalRoute.PRE_ROLL_DEV_USAGE, HierarchicalRoute.IMMEDIATE_WIN,
        HierarchicalRoute.FORCED_PHASE,
    ]
    facts = encode_l3_decision_facts(decisions)
    fallback = encode_l3_decision_facts(decisions, allow_no_use_fallback=True)
    actor = L3ShadowActor(8)
    output = actor(torch.zeros(8, 8), torch.zeros(8, RUNTIME_CONTEXT_SIZE),
                   torch.zeros(8, 30),
                   SimpleNamespace(weighted_features=torch.zeros(8, 6),
                                   expected_priorities=torch.full((8, 4), .5),
                                   available=torch.ones(8, dtype=torch.bool)), facts)
    assert output.available.tolist() == [True, True, True, True, False, False, False, False]
    assert torch.all(output.raw_logits == 0)
    assert torch.all(output.probabilities[4:] == 0)
    assert torch.allclose(output.probabilities[:4].sum(dim=-1), torch.ones(4))
    assert torch.all(output.probabilities[2, 2:] == 0)  # Dual: only S / C.
    assert output.probabilities[1, FAMILY_ORDER.index("BUILD_SETTLEMENT")] == 0
    assert torch.all(output.probabilities[0, :2] == 0)  # Investment: no S / C.
    assert output.probabilities[3, FAMILY_ORDER.index("END_TURN")] > 0
    assert not facts.l3_route_available[4]
    assert fallback.l3_route_available[4]
    assert decisions[4].fallback_route is HierarchicalRoute.CITY_VALUE_COMPARISON
    assert fallback.route_one_hot[4, list(HierarchicalRoute).index(
        HierarchicalRoute.CITY_VALUE_COMPARISON)] == 1
    after_no_use = actor(torch.zeros(8, 8), torch.zeros(8, RUNTIME_CONTEXT_SIZE),
                         torch.zeros(8, 30),
                         SimpleNamespace(weighted_features=torch.zeros(8, 6),
                                         expected_priorities=torch.full((8, 4), .5),
                                         available=torch.ones(8, dtype=torch.bool)), fallback)
    assert after_no_use.available[4]
    assert after_no_use.probabilities[4, FAMILY_ORDER.index("BUILD_CITY")] > 0
    assert facts.legal_family_mask[2, FAMILY_ORDER.index("END_TURN")]
    assert not facts.comparison_family_mask[2, FAMILY_ORDER.index("END_TURN")]


def test_no_comparison_candidate_never_fabricates_end_turn():
    with pytest.raises(ValueError, match="cannot be empty"):
        encode_l3_decision_facts([])
    game = _game()
    game.trade_count = 5
    decision = classify_hierarchical_route(
        game, 1, legal=[], immediate_win_available=False)
    facts = encode_l3_decision_facts([decision])
    actor = L3ShadowActor(8)
    soft = SimpleNamespace(weighted_features=torch.zeros(1, 6),
                           expected_priorities=torch.zeros(1, 4),
                           available=torch.zeros(1, dtype=torch.bool))
    output = actor(torch.zeros(1, 8), torch.zeros(1, RUNTIME_CONTEXT_SIZE),
                   torch.zeros(1, 30), soft, facts)
    assert not output.available.any()
    assert torch.all(output.probabilities == 0)
    assert torch.all(output.entropy == 0)


def test_l3_gradient_reaches_plan_and_priority_but_not_frozen_latent():
    context = build_victory_plan_context(create_game(33000003), 1)
    torch.manual_seed(36)
    heads = VictoryPlanShadowHeads(8)
    actor = L3ShadowActor(8)
    latent = torch.randn(1, 8, requires_grad=True)
    output, candidates, global_features = _plan_forward(context, heads, latent)
    bridge = differentiable_soft_plan(output, candidates)
    decision = _decision(legal=(BuildRoadAction(0), EndTurnAction()))
    facts = encode_l3_decision_facts([decision])
    # Final layer is zero-initialized by design. Set a *temporary test-only*
    # nonzero route to both bridge arms to prove potential gradient flow.
    with torch.no_grad():
        actor.encoder[0].weight.zero_()
        actor.encoder[0].bias.zero_()
        bridge_start = 8 + RUNTIME_CONTEXT_SIZE + 30
        actor.encoder[0].weight[0, bridge_start] = .4  # Plan S gain.
        actor.encoder[0].weight[0, bridge_start + 6] = .7  # Priority S.
        actor.family_head.weight[FAMILY_ORDER.index("BUILD_ROAD"), 0] = 1.0
        actor.family_head.weight[FAMILY_ORDER.index("END_TURN"), 0] = -1.0
    result = actor(latent, torch.zeros(1, RUNTIME_CONTEXT_SIZE),
                   global_features, bridge, facts)
    loss = -result.probabilities[0, FAMILY_ORDER.index("BUILD_ROAD")].log()
    loss.backward()
    assert heads.plan_scorer.output.weight.grad is not None
    assert heads.plan_scorer.output.weight.grad.abs().sum() > 0
    assert heads.priority_head.output.weight.grad is not None
    assert heads.priority_head.output.weight.grad.abs().sum() > 0
    assert latent.grad is None
    # No optimizer is created; all weights remain unchanged by backward().


def test_priorities_are_independent_and_shift_l3_input():
    context = build_victory_plan_context(create_game(33000004), 1)
    heads = VictoryPlanShadowHeads(8)
    actor = L3ShadowActor(8)
    latent = torch.zeros(1, 8)
    first, candidates, global_features = _plan_forward(context, heads, latent)
    first_bridge = differentiable_soft_plan(first, candidates)
    with torch.no_grad():
        heads.priority_head.output.bias.copy_(torch.tensor([3., 3., -3., -3.]))
        heads.plan_scorer.output.weight.fill_(.1)
    second, _, _ = _plan_forward(context, heads, latent)
    second_bridge = differentiable_soft_plan(second, candidates)
    assert torch.all(second_bridge.expected_priorities[0, :2] > .9)
    assert torch.all(second_bridge.expected_priorities[0, 2:] < .1)
    assert not torch.allclose(first_bridge.expected_priorities,
                              second_bridge.expected_priorities)
    assert not torch.allclose(first_bridge.weighted_features,
                              second_bridge.weighted_features)
    # The L3 input contains both arms and can react once its final layer learns.
    with torch.no_grad():
        actor.encoder[0].weight.zero_()
        actor.encoder[0].bias.zero_()
        actor.encoder[0].weight[0, 8 + RUNTIME_CONTEXT_SIZE + 30] = 1
        actor.encoder[0].weight[0, 8 + RUNTIME_CONTEXT_SIZE + 30 + 6] = 1
        actor.family_head.weight[FAMILY_ORDER.index("BUILD_ROAD"), 0] = 1
    facts = encode_l3_decision_facts([_decision(legal=(BuildRoadAction(0), EndTurnAction()))])
    old = actor(latent, torch.zeros(1, RUNTIME_CONTEXT_SIZE), global_features,
                first_bridge, facts)
    new = actor(latent, torch.zeros(1, RUNTIME_CONTEXT_SIZE), global_features,
                second_bridge, facts)
    assert not torch.allclose(old.raw_logits, new.raw_logits)
