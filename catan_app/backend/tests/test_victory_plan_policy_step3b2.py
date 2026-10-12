"""Step 3B-2: untrained Plan/Priority heads never select an Action."""

from dataclasses import replace
from math import log

import numpy as np
import torch

from app.domain.game import create_game
from app.rl.runtime_context import RUNTIME_CONTEXT_SIZE
from app.rl.victory_plan_context import PlanFeasibility, build_victory_plan_context
from app.rl.victory_plan_policy import (
    CANDIDATE_FEATURE_DIM, GLOBAL_FEATURE_DIM, MAX_PLAN_CANDIDATES,
    VictoryPlanShadowHeads, encode_victory_plan_features,
)


def _context():
    game = create_game(33000001)
    return build_victory_plan_context(game, 1)


def _forward(heads, encoded, *, latent_dim=8, latent=None):
    if latent is None:
        latent = torch.randn(1, latent_dim)
    return heads(
        latent, torch.zeros(1, RUNTIME_CONTEXT_SIZE),
        torch.tensor(encoded.global_features).unsqueeze(0),
        torch.tensor(encoded.candidate_features).unsqueeze(0),
        torch.tensor(encoded.candidate_mask).unsqueeze(0),
    )


def test_zero_initialized_plan_distribution_and_priority_mask():
    encoded = encode_victory_plan_features(_context())
    assert encoded.global_features.shape == (GLOBAL_FEATURE_DIM,)
    assert encoded.candidate_features.shape == (MAX_PLAN_CANDIDATES, CANDIDATE_FEATURE_DIM)
    assert len(encoded.candidate_ids) <= MAX_PLAN_CANDIDATES
    assert not encoded.global_features.flags.writeable
    assert np.isfinite(encoded.global_features).all()
    assert np.isfinite(encoded.candidate_features).all()
    heads = VictoryPlanShadowHeads(8)
    output = _forward(heads, encoded)
    mask = output.candidate_mask[0]
    valid_count = int(mask.sum())
    assert output.plan_available.item()
    assert torch.equal(output.raw_logits, torch.zeros_like(output.raw_logits))
    assert torch.equal(output.probabilities[0, ~mask], torch.zeros_like(output.probabilities[0, ~mask]))
    assert torch.allclose(output.probabilities[0, mask], torch.full((valid_count,), 1/valid_count))
    assert output.probabilities.sum().item() == 1.0
    assert abs(output.entropy.item() - log(valid_count)) < 1e-6
    assert torch.equal(output.priorities, torch.full_like(output.priorities, 0.5))


def test_unresolved_and_terminal_have_no_plan_distribution():
    original = _context()
    invalid = tuple(replace(candidate, feasibility=PlanFeasibility.IMPOSSIBLE,
                            valid=False) for candidate in original.candidates)
    for context in (replace(original, candidates=invalid, unresolved=True),
                    replace(original, current_score=10, remaining_points=0,
                            candidates=(), unresolved=False)):
        encoded = encode_victory_plan_features(context)
        output = _forward(VictoryPlanShadowHeads(8), encoded)
        assert not encoded.plan_available
        assert not output.plan_available.item()
        assert torch.equal(output.probabilities, torch.zeros_like(output.probabilities))
        assert output.entropy.item() == 0


def test_candidate_permutation_preserves_id_scores_and_priorities():
    original = _context()
    shuffled = replace(original, candidates=tuple(reversed(original.candidates)))
    heads = VictoryPlanShadowHeads(8)
    with torch.no_grad():
        heads.plan_scorer.output.weight.fill_(0.07)
        heads.priority_head.output.weight.fill_(0.03)
    latent = torch.randn(1, 8)
    first_features = encode_victory_plan_features(original)
    second_features = encode_victory_plan_features(shuffled)
    first = _forward(heads, first_features, latent=latent)
    second = _forward(heads, second_features, latent=latent)
    for field in ("raw_logits", "probabilities", "priorities"):
        left = getattr(first, field)[0, :len(original.candidates)]
        right = getattr(second, field)[0, :len(shuffled.candidates)]
        by_id_left = dict(zip(first_features.candidate_ids, left, strict=True))
        by_id_right = dict(zip(second_features.candidate_ids, right, strict=True))
        assert by_id_left.keys() == by_id_right.keys()
        assert all(torch.allclose(by_id_left[key], by_id_right[key], atol=1e-7)
                   for key in by_id_left)


def test_two_states_with_different_candidate_counts_share_padded_batch():
    first_context = _context()
    second_context = replace(first_context, candidates=tuple(
        candidate for candidate in first_context.candidates if candidate.valid))
    first = encode_victory_plan_features(first_context)
    second = encode_victory_plan_features(second_context)
    assert len(first.candidate_ids) != len(second.candidate_ids)
    heads = VictoryPlanShadowHeads(8)
    output = heads(
        torch.zeros(2, 8), torch.zeros(2, RUNTIME_CONTEXT_SIZE),
        torch.tensor(np.stack([first.global_features, second.global_features])),
        torch.tensor(np.stack([first.candidate_features, second.candidate_features])),
        torch.tensor(np.stack([first.candidate_mask, second.candidate_mask])),
    )
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(2))
    assert torch.equal(output.probabilities[~output.candidate_mask],
                       torch.zeros_like(output.probabilities[~output.candidate_mask]))


def test_independent_priorities_can_be_high_together_and_latent_is_detached():
    heads = VictoryPlanShadowHeads(8)
    with torch.no_grad():
        heads.priority_head.output.bias.copy_(torch.tensor([3.0, 3.0, -3.0, -3.0]))
        heads.plan_scorer.output.weight.fill_(0.1)
    latent = torch.randn(1, 8, requires_grad=True)
    output = _forward(heads, encode_victory_plan_features(_context()), latent=latent)
    priorities = output.priorities[0, 0]
    assert priorities[0] > 0.95 and priorities[1] > 0.95
    assert priorities[2] < 0.05 and priorities[3] < 0.05
    assert priorities.sum() > 1.5  # no four-way normalization
    output.raw_logits.sum().backward()
    assert latent.grad is None
