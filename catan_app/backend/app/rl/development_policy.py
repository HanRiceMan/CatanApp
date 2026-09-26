"""道路を含む既存Actorを固定し、発展購入のFamily logitだけを補正する。"""

import torch
from torch import nn

from .action_families import ACTION_FAMILY_INDEX
from .action_space import ACTION_SPACE_SIZE
from .candidate_policy import (ExpansionGraphHierarchicalCandidateActorCritic,
                               ExpansionGraphHierarchicalCandidateMaskablePolicy)
from .development_budget import DevelopmentBudgetFeatures, FEATURE_SIZE


DEVELOPMENT_LOGIT = ACTION_SPACE_SIZE + ACTION_FAMILY_INDEX["development_purchase"]


class DevelopmentBudgetActorCritic(ExpansionGraphHierarchicalCandidateActorCritic):
    def __init__(self, observation_size):
        super().__init__(observation_size)
        self.development_features = DevelopmentBudgetFeatures()
        self.development_head = nn.Sequential(
            nn.Linear(FEATURE_SIZE, 64), nn.Tanh(), nn.Linear(64, 1),
        )
        nn.init.zeros_(self.development_head[-1].weight)
        nn.init.zeros_(self.development_head[-1].bias)
        self.register_buffer("development_strength", torch.tensor(1.0))

    def development_correction(self, observations):
        assessment = self.development_features(observations)
        raw = self.development_head(assessment["features"]).squeeze(1)
        # 例外外で、購入が近い建設を遅らせる時だけ補正。最大でも約1/55のodds。
        return (raw.clamp(-4.0, 0.0) * assessment["reserve"].float()
                * self.development_strength)

    def forward_actor(self, observations):
        logits = super().forward_actor(observations)
        correction = torch.zeros_like(logits)
        correction[:, DEVELOPMENT_LOGIT] = self.development_correction(observations)
        return logits + correction


class DevelopmentBudgetMaskablePolicy(ExpansionGraphHierarchicalCandidateMaskablePolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = DevelopmentBudgetActorCritic(self.features_dim)


def initialize_development_policy(policy, source):
    incompatible = policy.load_state_dict(source.state_dict(), strict=False)
    allowed = ("mlp_extractor.development_features.", "mlp_extractor.development_head.",
               "mlp_extractor.development_strength")
    if incompatible.unexpected_keys or any(
        not key.startswith(allowed) for key in incompatible.missing_keys
    ):
        raise AssertionError(f"発展専用方策への移植に失敗: {incompatible}")
    return list(incompatible.missing_keys)


def configure_development_only(policy):
    """critic、道路head、共有GNNを含め既存学習済みparameterをすべて固定する。"""
    for parameter in policy.parameters():
        parameter.requires_grad = False
    for parameter in policy.mlp_extractor.development_head.parameters():
        parameter.requires_grad = True
    return sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
