"""既存の初期配置方策へ、開拓資源の最低限の補完条件を加える。"""

from __future__ import annotations

import numpy as np

from app.domain.actions import PlaceInitialSettlementAction
from app.domain.game import GameState, player_for

from .action_space import ACTION_CATALOG
from .expansion_planner import player_production, vertex_production


def expansion_coverage_mask(game: GameState, player_id: int,
                            action_mask: np.ndarray, *, min_pips: int = 1,
                            min_wheat_pips: int = 0,
                            min_ore_pips: int = 0,
                            fallback_to_best_coverage: bool = False) -> np.ndarray:
    """2個目の初期開拓地で、可能なら開拓・都市資源を確保する。"""
    result = np.asarray(action_mask, dtype=np.bool_).copy()
    player = player_for(game, player_id)
    if (game.phase != "setup_ready" or game.pending_settlement_id is not None
            or player.settlements != 1):
        return result
    current = player_production(game, player_id)
    expansion_ids: set[int] = set()
    wheat_ids: set[int] = set()
    coverage_by_id: dict[int, int] = {}
    settlement_definitions = [
        item for item in ACTION_CATALOG
        if result[item.id] and isinstance(item.action, PlaceInitialSettlementAction)
    ]
    for item in settlement_definitions:
        candidate = vertex_production(game, item.action.vertex_id)
        coverage_by_id[item.id] = min(
            current["wood"] + candidate["wood"],
            current["brick"] + candidate["brick"],
        )
        if (current["wood"] + candidate["wood"] >= min_pips
                and current["brick"] + candidate["brick"] >= min_pips):
            expansion_ids.add(item.id)
            if current["wheat"] + candidate["wheat"] >= min_wheat_pips:
                wheat_ids.add(item.id)
    if not expansion_ids:
        if not fallback_to_best_coverage or not coverage_by_id:
            return result
        best_coverage = max(coverage_by_id.values())
        expansion_ids = {
            item_id for item_id, coverage in coverage_by_id.items()
            if coverage == best_coverage
        }
    eligible_ids = wheat_ids or expansion_ids
    if min_ore_pips:
        ore_ids = {
            item.id for item in settlement_definitions
            if (item.id in eligible_ids
                and current["ore"] + vertex_production(
                    game, item.action.vertex_id,
                )["ore"] >= min_ore_pips)
        }
        eligible_ids = ore_ids or eligible_ids
    for item in settlement_definitions:
        if item.id not in eligible_ids:
            result[item.id] = False
    return result
