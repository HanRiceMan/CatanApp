"""Plan-conditioned mixture invariants; no optimizer is constructed here."""

from dataclasses import replace

import torch

from app.rl.hierarchical_l3_shadow_step3b5 import L3DecisionFacts, ROUTE_ORDER
from app.rl.hierarchical_shadow_controller_step3b4 import FAMILY_ORDER, HierarchicalRoute
from app.rl.plan_conditioned_policy_step3b8 import PlanConditionedInvestmentPolicy
from app.rl.victory_plan_policy import MAX_PLAN_CANDIDATES


def _inputs(batch=1, *, plan_available=True, action_available=True):
    torch.manual_seed(3808)
    latent = torch.randn(batch, 16, requires_grad=True)
    runtime = torch.randn(batch, 79)
    global_features = torch.randn(batch, 30)
    candidates = torch.zeros(batch, MAX_PLAN_CANDIDATES, 9)
    candidates[:, 0, :6] = torch.tensor([.1, .3, 0., 1., .27, 0.])
    candidates[:, 1, :6] = torch.tensor([.4, 0., 1., 0., .55, 0.])
    plan_mask = torch.zeros(batch, MAX_PLAN_CANDIDATES, dtype=torch.bool)
    if plan_available:
        plan_mask[:, :2] = True
    route = torch.zeros(batch, len(ROUTE_ORDER))
    route[:, ROUTE_ORDER.index(HierarchicalRoute.INVESTMENT)] = 1
    family = torch.zeros(batch, len(FAMILY_ORDER), dtype=torch.bool)
    if action_available:
        family[:, [2, 3, 4, 6]] = True
    facts = L3DecisionFacts(route, family, family, torch.zeros(batch, 1),
                            torch.ones(batch, dtype=torch.bool))
    return latent, runtime, global_features, candidates, plan_mask, facts


def test_mixture_mask_permutation_and_interventions():
    policy = PlanConditionedInvestmentPolicy(16)
    args = _inputs()
    plan, _, output = policy(*args)
    assert output.available.all()
    assert torch.allclose(output.probabilities.sum(-1), torch.ones(1))
    assert torch.count_nonzero(output.probabilities[~args[-1].comparison_family_mask]) == 0
    assert torch.count_nonzero(output.per_plan_probabilities[:, 2:]) == 0
    assert torch.allclose(output.probabilities,
                          (plan.probabilities.unsqueeze(-1) *
                           output.per_plan_probabilities).sum(1), atol=1e-6)
    one = torch.zeros_like(plan.probabilities)
    two = torch.zeros_like(plan.probabilities)
    one[:, 0], two[:, 1] = 1, 1
    a = policy.l3_actor(*args[:4], plan, args[-1], plan_probabilities=one)
    b = policy.l3_actor(*args[:4], plan, args[-1], plan_probabilities=two)
    assert not torch.allclose(a.probabilities, b.probabilities, atol=1e-8)
    neutral = policy.l3_actor(*args[:4], plan, args[-1],
                               priorities=torch.full_like(plan.priorities, .5))
    raised = policy.l3_actor(*args[:4], plan, args[-1],
                              priorities=torch.ones_like(plan.priorities))
    assert not torch.allclose(neutral.probabilities, raised.probabilities, atol=1e-8)
    swapped = args[3].clone()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]
    perm = policy(*args[:3], swapped, *args[4:])[-1]
    assert torch.allclose(output.probabilities, perm.probabilities, atol=1e-7)


def test_upstream_gradient_and_frozen_latent():
    policy = PlanConditionedInvestmentPolicy(16)
    args = _inputs()
    output = policy(*args)[-1]
    (-output.probabilities[:, 2].clamp_min(1e-30).log()).backward()
    assert args[0].grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in policy.plan_heads.plan_scorer.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in policy.plan_heads.priority_head.parameters())


def test_unavailable_plan_and_all_invalid_action():
    policy = PlanConditionedInvestmentPolicy(16)
    no_plan = policy(*_inputs(plan_available=False))[-1]
    assert no_plan.available.item()
    assert no_plan.probabilities.sum().item() == 1
    assert torch.count_nonzero(no_plan.per_plan_probabilities) == 0
    no_action = policy(*_inputs(action_available=False))[-1]
    assert not no_action.available.item()
    assert torch.count_nonzero(no_action.probabilities) == 0
