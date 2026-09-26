"""行動種別と種別内候補を分解する、MaskablePPO用の離散分布。"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from sb3_contrib.common.maskable.distributions import (MaskableCategorical,
                                                       MaskableDistribution,
                                                       MaybeMasks)

from .action_families import ACTION_FAMILIES, ACTION_FAMILY_INDEX, action_family
from .action_space import ACTION_CATALOG, ACTION_SPACE_SIZE


FAMILY_COUNT = len(ACTION_FAMILIES)
ACTION_FAMILY_IDS = tuple(ACTION_FAMILY_INDEX[action_family(item.action)]
                          for item in ACTION_CATALOG)


class HierarchicalMaskableCategoricalDistribution(MaskableDistribution):
    """P(family) × P(action | family) を377 Actionの分布として公開する。"""

    def __init__(self) -> None:
        super().__init__()
        self.action_dim = ACTION_SPACE_SIZE
        self.family_count = FAMILY_COUNT
        self.distribution: MaskableCategorical
        self._candidate_logits: torch.Tensor | None = None
        self._family_logits: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None
        self._family_ids: torch.Tensor | None = None
        self._conditional_log_probs: torch.Tensor | None = None
        self._family_log_probs: torch.Tensor | None = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, self.action_dim + self.family_count)

    def proba_distribution(self, action_logits: torch.Tensor):
        logits = action_logits.view(-1, self.action_dim + self.family_count)
        self._candidate_logits = logits[:, :self.action_dim]
        self._family_logits = logits[:, self.action_dim:]
        self.apply_masking(None)
        return self

    def apply_masking(self, masks: MaybeMasks) -> None:
        if self._candidate_logits is None or self._family_logits is None:
            raise RuntimeError("先にproba_distribution()を呼んでください。")
        device = self._candidate_logits.device
        batch = self._candidate_logits.shape[0]
        mask = (torch.ones((batch, self.action_dim), dtype=torch.bool, device=device)
                if masks is None else torch.as_tensor(masks, dtype=torch.bool, device=device)
                .reshape(batch, self.action_dim))
        if not mask.any(dim=1).all():
            raise ValueError("各状態に最低1つの合法Actionが必要です。")
        family_ids = torch.as_tensor(ACTION_FAMILY_IDS, dtype=torch.long, device=device)
        family_legal = torch.stack([mask[:, family_ids == family].any(dim=1)
                                    for family in range(self.family_count)], dim=1)
        negative = torch.tensor(-1e8, dtype=self._candidate_logits.dtype, device=device)
        family_logits = torch.where(family_legal, self._family_logits, negative)
        family_log_probs = torch.log_softmax(family_logits, dim=1)
        conditional = torch.zeros_like(self._candidate_logits)
        for family in range(self.family_count):
            indices = family_ids == family
            legal = mask[:, indices]
            logits = torch.where(legal, self._candidate_logits[:, indices], negative)
            normalized = torch.log_softmax(logits, dim=1)
            conditional[:, indices] = torch.where(legal, normalized, negative)
        flat_logits = family_log_probs[:, family_ids] + conditional
        self._mask = mask
        self._family_ids = family_ids
        self._conditional_log_probs = conditional
        self._family_log_probs = family_log_probs
        self.distribution = MaskableCategorical(logits=flat_logits, masks=mask)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        if any(value is None for value in
               (self._mask, self._family_ids, self._conditional_log_probs, self._family_log_probs)):
            raise RuntimeError("分布が初期化されていません。")
        assert self._mask is not None and self._family_ids is not None
        assert self._conditional_log_probs is not None and self._family_log_probs is not None
        family = torch.argmax(self._family_log_probs, dim=1)
        chosen = []
        negative = torch.tensor(-1e8, dtype=self._conditional_log_probs.dtype,
                                device=self._conditional_log_probs.device)
        for row, selected_family in enumerate(family):
            candidates = self._family_ids == selected_family
            scores = torch.where(candidates & self._mask[row],
                                 self._conditional_log_probs[row], negative)
            chosen.append(torch.argmax(scores))
        return torch.stack(chosen)

    def actions_from_params(self, action_logits: torch.Tensor,
                            deterministic: bool = False) -> torch.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)
