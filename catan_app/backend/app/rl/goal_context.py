"""公開盤面と自分の手札から作る、高レベル戦略目標の固定長特徴。"""

from __future__ import annotations

import numpy as np

from app.domain.game import BUILD_COSTS, RESOURCES, GameState, player_for

from .expansion_planner import analyze_expansion_plan, estimated_build_wait


GOAL_TYPES = ("none", "settlement", "city", "development", "longest_road", "largest_army")
GOAL_CONTEXT_SIZE = 155
MAX_WAIT = 12.0


def _unit(value: float, maximum: float) -> float:
    return min(max(float(value), 0.0), maximum) / maximum


def encode_goal_context(game: GameState, player_id: int, *,
                        max_plan_roads: int = 3) -> np.ndarray:
    """計画器の判断材料を、PPOが直接読める155次元へ変換する。

    相手の非公開手札は使わない。目標は各意思決定時に再計算されるため、盤面や
    手札が変われば開拓地から都市などへ自然に切り替わる。
    """
    player = player_for(game, player_id)
    plan = analyze_expansion_plan(
        game, player_id, max_additional_roads=max_plan_roads,
    )
    target = plan.selected_target
    settlement_wait = target.wait_rounds if target is not None and player.settlements < 5 else 99.0
    city_wait = (estimated_build_wait(game, player_id, BUILD_COSTS["city"])
                 if player.settlements > 0 and player.cities < 4 else 99.0)

    # これは命令ではなく、現時点で最も近い建設方針を示す教師信号。
    if target is not None and settlement_wait < city_wait:
        goal = "settlement"
        required = dict(zip(RESOURCES, target.required_resources))
    elif city_wait < 99.0:
        goal = "city"
        required = {**dict.fromkeys(RESOURCES, 0), **BUILD_COSTS["city"]}
    elif target is not None:
        goal = "settlement"
        required = dict(zip(RESOURCES, target.required_resources))
    else:
        goal = "none"
        required = dict.fromkeys(RESOURCES, 0)

    # 建設が停滞し、購入による建設遅延も許容範囲なら、現在の戦術目標を
    # 発展カードへ切り替える。これは拠点数ではなく毎回の盤面状態で決める。
    if game.phase == "action":
        from .development_strategy import assess_development_purchase
        development = assess_development_purchase(
            game, player_id, max_plan_roads=max_plan_roads,
            stall_wait_rounds=4.0, max_build_delay=1.0,
            tactical_score=8, title_horizon=2,
        )
        if development["eligible"]:
            goal = "development"
            required = {**dict.fromkeys(RESOURCES, 0), **BUILD_COSTS["development"]}

    values: list[float] = [float(name == goal) for name in GOAL_TYPES]
    values.extend(float(target is not None and vertex_id == target.vertex_id)
                  for vertex_id in range(54))
    recommended = set(target.first_road_ids if target is not None else ())
    values.extend(float(edge_id in recommended) for edge_id in range(72))
    values.extend(_unit(required[resource], 5) for resource in RESOURCES)
    values.extend(_unit(max(0, required[resource] - player.resources[resource]), 5)
                  for resource in RESOURCES)
    values.extend((
        _unit(settlement_wait, MAX_WAIT),
        _unit(city_wait, MAX_WAIT),
        _unit(min(settlement_wait, city_wait), MAX_WAIT),
        _unit(target.additional_roads if target is not None else 0, 5),
        _unit(target.total_pips if target is not None else 0, 15),
        _unit(target.new_resource_kinds if target is not None else 0, 5),
        _unit(target.site_value if target is not None else 0, 30),
        _unit(target.plan_value if target is not None else 0, 30),
        float(target is not None and target.additional_roads == 0),
        float(plan.should_wait_for_settlement),
        float(min(settlement_wait, city_wait) >= 4.0),
        float(sum(player.resources.values()) >= 8),
        float(any(
            game.robber_tile_id in game.board.vertices[vertex_id].hex_ids
            for buildings in (game.settlements, game.cities)
            for vertex_id, owner_id in buildings.items() if owner_id == player_id
        )),
    ))
    encoded = np.asarray(values, dtype=np.float32)
    if encoded.shape != (GOAL_CONTEXT_SIZE,):
        raise AssertionError(f"目標Contextサイズが不正です: {encoded.shape}")
    return encoded
