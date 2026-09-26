"""共同ベイズ手札推定をPPOへ渡す固定長の公開情報Context。"""

from __future__ import annotations

from math import log2

import numpy as np

from app.domain.game import GameState, RESOURCES

from .hand_belief import BayesianHandEstimator, MAX_JOINT_HYPOTHESES


BELIEF_FEATURES_PER_OPPONENT = len(RESOURCES) * 3 + 2
BELIEF_CONTEXT_SIZE = BELIEF_FEATURES_PER_OPPONENT * 3
MAX_RESOURCE_CARDS = 19
MAX_BELIEF_ENTROPY = log2(MAX_JOINT_HYPOTHESES)


def encode_belief_context(game: GameState, viewer_id: int) -> np.ndarray:
    """時計回りの相手3人について、手札事後分布の要約を返す。

    各相手は、資源別の期待枚数・1枚以上持つ確率・無作為に盗んだ時に
    その資源になる確率と、分布のentropy・候補数の17特徴を持つ。
    推定器へ渡すviewer_idにより、本人が盗賊の当事者として知った資源だけは
    利用するが、第三者だけが知るprivate_resourceは利用しない。
    """
    if viewer_id not in game.seat_order:
        raise ValueError(f"座席に存在しないプレイヤーです: {viewer_id}")
    beliefs = BayesianHandEstimator().estimate(game, viewer_id)
    self_index = game.seat_order.index(viewer_id)
    values: list[float] = []
    for offset in range(1, 4):
        opponent_id = game.seat_order[(self_index + offset) % 4]
        belief = beliefs[opponent_id]
        values.extend(
            min(belief.expected(resource), MAX_RESOURCE_CARDS) / MAX_RESOURCE_CARDS
            for resource in RESOURCES
        )
        values.extend(belief.probability_has(resource) for resource in RESOURCES)
        values.extend(
            belief.random_steal_probability(resource) for resource in RESOURCES
        )
        entropy = -sum(
            probability * log2(probability)
            for probability in belief.distribution.values() if probability
        )
        values.extend((
            min(entropy, MAX_BELIEF_ENTROPY) / MAX_BELIEF_ENTROPY,
            min(len(belief.distribution), MAX_JOINT_HYPOTHESES)
            / MAX_JOINT_HYPOTHESES,
        ))
    encoded = np.asarray(values, dtype=np.float32)
    if encoded.shape != (BELIEF_CONTEXT_SIZE,):
        raise AssertionError(f"手札推定Contextサイズが不正です: {encoded.shape}")
    return encoded
