"""City adapter starts from the frozen base and cannot alter Investment."""

import torch

from app.rl.hierarchical_l3_shadow_step3b5 import L3DecisionFacts, ROUTE_ORDER
from app.rl.hierarchical_shadow_controller_step3b4 import FAMILY_ORDER, HierarchicalRoute
from app.rl.plan_action_interaction_step3b9 import PlanActionInvestmentPolicy
from app.rl.route_isolated_city_step3b11 import RouteIsolatedCityPolicy
from app.rl.victory_plan_policy import MAX_PLAN_CANDIDATES


def _state(route: HierarchicalRoute):
    torch.manual_seed(3111)
    latent = torch.randn(2, 16, requires_grad=True)
    runtime = torch.randn(2, 79)
    global_features = torch.randn(2, 30)
    candidates = torch.zeros(2, MAX_PLAN_CANDIDATES, 9)
    candidates[:, 0, :6] = torch.tensor([0., 0., 1., 1., .4, 0.])
    candidates[:, 1, :6] = torch.tensor([.3, .2, 0., 1., .6, 0.])
    plan_mask = torch.zeros(2, MAX_PLAN_CANDIDATES, dtype=torch.bool)
    plan_mask[:, :2] = True
    one_hot = torch.zeros(2, len(ROUTE_ORDER))
    one_hot[:, ROUTE_ORDER.index(route)] = 1
    family = torch.zeros(2, len(FAMILY_ORDER), dtype=torch.bool)
    family[:, ([1, 2, 3, 4, 6] if route is HierarchicalRoute.CITY_VALUE_COMPARISON
                else [2, 3, 4, 6])] = True
    facts = L3DecisionFacts(one_hot, family, family, torch.zeros(2, 1),
                            torch.ones(2, dtype=torch.bool))
    return latent, runtime, global_features, candidates, plan_mask, facts


def test_zero_adapter_matches_parent_city_distribution_and_mask():
    parent = PlanActionInvestmentPolicy(16)
    policy = RouteIsolatedCityPolicy(16)
    policy.load_base(parent)
    state = _state(HierarchicalRoute.CITY_VALUE_COMPARISON)
    baseline = parent(*state)[-1]
    initial = policy(*state)[-1]
    assert torch.equal(initial.probabilities, baseline.probabilities)
    assert torch.equal(initial.per_plan_probabilities, baseline.per_plan_probabilities)
    assert torch.count_nonzero(initial.probabilities[:, [0, 5]]) == 0
    assert torch.equal(initial.probabilities.sum(-1), torch.ones(2))
    assert all(not parameter.requires_grad for parameter in policy.plan_heads.parameters())
    assert all(not parameter.requires_grad for parameter in policy.l3_actor.parameters())


def test_city_gradient_only_adapter_and_investment_exact_after_step():
    parent = PlanActionInvestmentPolicy(16)
    policy = RouteIsolatedCityPolicy(16)
    policy.load_base(parent)
    investment = _state(HierarchicalRoute.INVESTMENT)
    before = policy(*investment)[-1].probabilities.clone()
    city = _state(HierarchicalRoute.CITY_VALUE_COMPARISON)
    optimizer = torch.optim.Adam(policy.city_adapter.parameters(), lr=1e-3)
    loss = -policy(*city)[-1].probabilities[:, 1].log().mean()
    loss.backward()
    assert city[0].grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in policy.city_adapter.parameters())
    assert all(p.grad is None for p in policy.plan_heads.parameters())
    assert all(p.grad is None for p in policy.l3_actor.parameters())
    optimizer.step()
    after = policy(*investment)[-1].probabilities
    assert torch.equal(before, after)
    assert torch.equal(after, parent(*investment)[-1].probabilities)


def test_city_plan_and_priority_diagnostic_interventions_preserve_mask():
    policy = RouteIsolatedCityPolicy(16)
    state = _state(HierarchicalRoute.CITY_VALUE_COMPARISON)
    with torch.no_grad():
        policy.city_adapter.output.weight.fill_(.03)
        plan, _, normal = policy(*state)
        uniform = plan.candidate_mask.float() / plan.candidate_mask.sum(-1, keepdim=True)
        _, _, alternate = policy(*state, plan_probabilities=uniform,
                                 priorities=torch.full_like(plan.priorities, .5))
    assert torch.allclose(alternate.probabilities.sum(-1), torch.ones(2))
    assert torch.count_nonzero(alternate.probabilities[:, [0, 5]]) == 0
    assert torch.count_nonzero(normal.per_plan_probabilities[:, 2:]) == 0
