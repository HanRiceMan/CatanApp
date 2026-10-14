"""Step 3B-9 bilinear Plan–Action invariants (no optimizer or game mutation)."""

import torch

from app.rl.hierarchical_l3_shadow_step3b5 import L3DecisionFacts, ROUTE_ORDER
from app.rl.hierarchical_shadow_controller_step3b4 import FAMILY_ORDER, HierarchicalRoute
from app.rl.plan_action_interaction_step3b9 import PlanActionInvestmentPolicy
from app.rl.victory_plan_policy import MAX_PLAN_CANDIDATES


def _state(*, plans=True, actions=True):
    torch.manual_seed(3909)
    latent = torch.randn(1, 16, requires_grad=True)
    runtime = torch.randn(1, 79)
    global_features = torch.randn(1, 30)
    candidates = torch.zeros(1, MAX_PLAN_CANDIDATES, 9)
    candidates[0, 0, :6] = torch.tensor([0., 0., 1., 1., .36, 0.])
    candidates[0, 1, :6] = torch.tensor([.4, .3, 0., 0., .64, 0.])
    mask = torch.zeros(1, MAX_PLAN_CANDIDATES, dtype=torch.bool)
    if plans:
        mask[:, :2] = True
    route = torch.zeros(1, len(ROUTE_ORDER))
    route[:, ROUTE_ORDER.index(HierarchicalRoute.INVESTMENT)] = 1
    family = torch.zeros(1, len(FAMILY_ORDER), dtype=torch.bool)
    if actions:
        family[:, [2, 3, 4, 6]] = True
    facts = L3DecisionFacts(route, family, family, torch.zeros(1, 1),
                            torch.ones(1, dtype=torch.bool))
    return latent, runtime, global_features, candidates, mask, facts


def test_plan_intervention_mixture_mask_and_permutation():
    policy = PlanActionInvestmentPolicy(16)
    args = _state()
    plan, _, normal = policy(*args)
    assert normal.available.item()
    assert torch.allclose(normal.probabilities.sum(-1), torch.ones(1))
    assert torch.count_nonzero(normal.probabilities[~args[-1].comparison_family_mask]) == 0
    assert torch.count_nonzero(normal.per_plan_probabilities[:, 2:]) == 0
    assert torch.allclose(normal.probabilities,
                          (plan.probabilities.unsqueeze(-1) *
                           normal.per_plan_probabilities).sum(1), atol=1e-6)
    a_weights = torch.zeros_like(plan.probabilities)
    b_weights = torch.zeros_like(plan.probabilities)
    a_weights[:, 0], b_weights[:, 1] = 1, 1
    actor_args = (*args[:4], plan, args[-1])
    a = policy.l3_actor(*actor_args, plan_probabilities=a_weights)
    b = policy.l3_actor(*actor_args, plan_probabilities=b_weights)
    tv = .5 * (a.probabilities - b.probabilities).abs().sum()
    assert tv > 1e-5
    changed_priority = policy.l3_actor(
        *actor_args, priorities=torch.ones_like(plan.priorities))
    assert .5 * (normal.probabilities - changed_priority.probabilities).abs().sum() > 1e-5
    permuted = args[3].clone()
    permuted[:, [0, 1]] = permuted[:, [1, 0]]
    perm_output = policy(*args[:3], permuted, *args[4:])[-1]
    assert torch.allclose(normal.probabilities, perm_output.probabilities, atol=1e-7)


def test_marginal_action_gradient_reaches_l1_l2_l3_but_not_champion():
    policy = PlanActionInvestmentPolicy(16)
    args = _state()
    _, _, output = policy(*args)
    (-output.probabilities[:, 2].clamp_min(1e-30).log()).backward()
    # The expression above exercises log P(a|s), not a per-Plan log probability.
    assert args[0].grad is None
    for module in (policy.l3_actor.state_encoder, policy.l3_actor.plan_encoder,
                   policy.l3_actor.action_query, policy.plan_heads.priority_head,
                   policy.plan_heads.plan_scorer):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in module.parameters())


def test_plan_unavailable_and_all_invalid_action_fallback():
    policy = PlanActionInvestmentPolicy(16)
    fallback = policy(*_state(plans=False))[-1]
    assert fallback.available.item()
    assert fallback.probabilities.sum().item() == 1
    assert torch.count_nonzero(fallback.per_plan_probabilities) == 0
    empty = policy(*_state(actions=False))[-1]
    assert not empty.available.item()
    assert torch.count_nonzero(empty.probabilities) == 0
