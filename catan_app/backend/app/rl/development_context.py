"""発展カード購入判断をPPOへ渡す公開・本人情報だけのv6文脈。"""

from math import isfinite

import numpy as np

from .development_strategy import assess_development_purchase


DEVELOPMENT_CONTEXT_SIZE = 25
_CARDS = ("knight", "road_building", "year_of_plenty", "monopoly", "victory_point")


def _cap(value: float, maximum: float) -> float:
    if not isfinite(value):
        return 1.0
    return min(max(float(value), 0.0), maximum) / maximum


def encode_development_context(game, player_id: int) -> np.ndarray:
    """現行計画層の入力要因を固定長化する。最終eligible判定自体は渡さない。"""
    assessment = assess_development_purchase(
        game, player_id,
        max_plan_roads=3,
        stall_wait_rounds=4.0,
        max_build_delay=1.0,
        tactical_score=8,
        title_horizon=2,
    )
    reasons = set(assessment["reasons"])
    probabilities = assessment["unseen_card_probabilities"]
    values = assessment["card_values"]
    vector = np.asarray([
        float(assessment["affordable"]),
        float(assessment["immediate_construction"]),
        float(assessment["already_bought_this_turn"]),
        _cap(assessment["site_count"], 9),
        _cap(assessment["score"], 10),
        _cap(assessment["build_wait_before"], 12),
        _cap(assessment["build_wait_after"], 12),
        _cap(assessment["build_delay"], 6),
        _cap(assessment["allowed_build_delay"], 6),
        _cap(assessment["blocked_production_pips"], 30),
        _cap(assessment["largest_army_gap"], 5),
        float("construction_stall" in reasons),
        float("robber_relief" in reasons),
        float("largest_army_race" in reasons),
        *(float(probabilities.get(card, 0.0)) for card in _CARDS),
        *(_cap(values.get(card, 0.0), 3) for card in _CARDS),
        _cap(assessment["expected_card_value"], 3),
    ], dtype=np.float32)
    if vector.shape != (DEVELOPMENT_CONTEXT_SIZE,):
        raise AssertionError(f"発展購入Contextサイズが不正です: {vector.shape}")
    return vector
