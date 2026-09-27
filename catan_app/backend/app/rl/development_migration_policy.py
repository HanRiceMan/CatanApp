"""盗賊PPOを固定し、発展カード購入Familyだけを計画層から移管する。"""

import torch
from torch import nn

from .action_families import ACTION_FAMILY_INDEX
from .hierarchical_distribution import FAMILY_COUNT
from .action_space import ACTION_SPACE_SIZE
from .candidate_policy import (
    RobberBeliefGraphFamilyHierarchicalCandidateActorCritic,
    RobberBeliefGraphFamilyHierarchicalCandidateMaskablePolicy,
)
from .development_context import DEVELOPMENT_CONTEXT_SIZE
from .observation import OBSERVATION_VECTOR_SIZES


DEVELOPMENT_FAMILY_LOGIT = (
    ACTION_SPACE_SIZE + ACTION_FAMILY_INDEX["development_purchase"]
)


class RobberDevelopmentMigrationActorCritic(
    RobberBeliefGraphFamilyHierarchicalCandidateActorCritic
):
    """v5盗賊方策へ、発展購入だけを変えられる小さな残差Headを加える。"""

    def __init__(self, observation_size: int):
        if observation_size != OBSERVATION_VECTOR_SIZES["v6"]:
            raise ValueError("発展移管方策はObservation v6専用です。")
        # 盗賊PPO本体にはv5までだけを見せ、既存重みを完全に維持する。
        super().__init__(OBSERVATION_VECTOR_SIZES["v5"])
        self.development_head = nn.Sequential(
            nn.Linear(DEVELOPMENT_CONTEXT_SIZE + FAMILY_COUNT, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.development_head[-1].weight)
        nn.init.zeros_(self.development_head[-1].bias)
        self.register_buffer("development_strength", torch.tensor(4.0))

    def development_correction(
        self, observations: torch.Tensor, base_family_logits: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat((
            observations[:, -DEVELOPMENT_CONTEXT_SIZE:],
            base_family_logits.detach(),
        ), dim=1)
        # 抑制専用にはしない。停滞・盗賊解除・騎士力争いでは正方向、
        # 近い建設を遅らせる局面では負方向を教師・勝敗から学べる。
        return torch.tanh(self.development_head(features).squeeze(1)) * (
            self.development_strength
        )

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        logits = super().forward_actor(observations)
        correction = torch.zeros_like(logits)
        correction[:, DEVELOPMENT_FAMILY_LOGIT] = self.development_correction(
            observations, logits[:, ACTION_SPACE_SIZE:]
        )
        return logits + correction


class RobberDevelopmentMigrationMaskablePolicy(
    RobberBeliefGraphFamilyHierarchicalCandidateMaskablePolicy
):
    """盗賊PPOを土台に発展購入判断だけを追加学習するPolicy。"""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = RobberDevelopmentMigrationActorCritic(
            self.features_dim
        )


def initialize_development_migration_policy(policy, source) -> list[str]:
    """盗賊PPOを完全移植し、発展Headだけをゼロ初期化のまま残す。"""
    incompatible = policy.load_state_dict(source.state_dict(), strict=False)
    allowed = (
        "mlp_extractor.development_head.",
        "mlp_extractor.development_strength",
    )
    if incompatible.unexpected_keys or any(
        not key.startswith(allowed) for key in incompatible.missing_keys
    ):
        raise AssertionError(f"発展移管方策への移植に失敗しました: {incompatible}")
    return list(incompatible.missing_keys)
