"""保存済みPPO方策の評価専用Action選択方式。学習・通常UIには接続しない。"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import torch

from app.agents.ppo import PPOAgent
from app.domain.actions import Action
from app.domain.game import GameActionError, GameState

from .action_space import ACTION_CATALOG, get_action_mask, id_to_action
from .action_families import action_family
from .observation import encode_observation, get_observation


SelectionMethod = Literal["global_argmax", "family_mass", "family_l2"]


def choose_action_id(probabilities: np.ndarray, mask: np.ndarray,
                     method: SelectionMethod) -> int:
    """合法候補だけから選ぶ。同点ならAction IDが小さい候補を優先する。"""
    if method not in {"global_argmax", "family_mass", "family_l2"}:
        raise ValueError(f"未対応の選択方式です: {method}")
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    legal_mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if values.shape != legal_mask.shape or values.size != len(ACTION_CATALOG):
        raise ValueError("Action確率とMaskの長さが一致しません。")
    if not np.isfinite(values).all() or np.any(values[legal_mask] < 0):
        raise ValueError("Action確率に非有限値または負値があります。")
    legal = np.flatnonzero(legal_mask)
    if not len(legal):
        raise ValueError("合法Actionがありません。")
    if method == "global_argmax":
        return int(max(legal, key=lambda index: values[index]))

    groups: dict[str, list[int]] = defaultdict(list)
    for index in legal:
        groups[action_family(ACTION_CATALOG[int(index)].action)].append(int(index))
    if method == "family_mass":
        score = lambda ids: float(values[ids].sum())
    else:
        score = lambda ids: float(np.linalg.norm(values[ids], ord=2))
    family = max(groups, key=lambda name: score(groups[name]))
    return max(groups[family], key=lambda index: values[index])


class FamilySelectionAgent:
    """通常ターンのみ行動種別を先に選び、種別内は最大確率の候補にする。"""

    def __init__(self, policy: PPOAgent, method: SelectionMethod):
        if method not in {"family_mass", "family_l2"}:
            raise ValueError("FamilySelectionAgentには行動種別選択方式を指定してください。")
        self.policy = policy
        self.method = method

    def select_action(self, game: GameState, player_id: int) -> Action:
        if game.phase != "action":
            return self.policy.select_action(game, player_id)
        observation = encode_observation(get_observation(
            game, player_id, version=self.policy.manifest.observation_version))
        mask = get_action_mask(game, player_id)
        tensor, _ = self.policy.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.policy.model.policy.get_distribution(tensor, action_masks=mask)
            probabilities = distribution.distribution.probs.cpu().numpy().reshape(-1)
        action_id = choose_action_id(probabilities, mask, self.method)
        if not mask[action_id]:
            raise GameActionError("選択方式が非法Actionを返しました。")
        return id_to_action(action_id)
