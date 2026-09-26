"""拠点数ではなく、建設停滞と戦術価値から発展カード購入を判断する。"""

from __future__ import annotations

from collections import Counter
from math import isfinite

from app.domain.actions import BuildCityAction, BuildSettlementAction, BuyDevelopmentAction
from app.domain.game import (BUILD_COSTS, DEVELOPMENT_DECK, GameState,
                             player_for)

from .action_space import ACTION_CATALOG, action_to_id, get_action_mask
from .expansion_planner import NUMBER_PIPS, analyze_expansion_plan
from .reward import actual_score


def _blocked_production_pips(game: GameState, player_id: int) -> int:
    tile = game.board.tiles[game.robber_tile_id]
    pips = NUMBER_PIPS.get(tile.number, 0)
    if not pips:
        return 0
    settlements = sum(
        owner == player_id and game.robber_tile_id in game.board.vertices[vertex].hex_ids
        for vertex, owner in game.settlements.items()
    )
    cities = sum(
        owner == player_id and game.robber_tile_id in game.board.vertices[vertex].hex_ids
        for vertex, owner in game.cities.items()
    )
    return pips * (settlements + 2 * cities)


def _unseen_card_probabilities(game: GameState, player_id: int) -> dict[str, float]:
    """公開済み使用カードと自分の手札だけを除いた、次カードの事前確率。"""
    unseen = Counter(DEVELOPMENT_DECK)
    for player in game.players:
        unseen.subtract(player.used_development_cards)
    unseen.subtract(player_for(game, player_id).development_cards)
    unseen = Counter({card: max(0, count) for card, count in unseen.items()})
    total = sum(unseen.values())
    return {card: count / total for card, count in unseen.items()} if total else {}


def _wait_after_purchase(game: GameState, player_id: int,
                         max_plan_roads: int) -> float:
    player = player_for(game, player_id)
    for resource, count in BUILD_COSTS["development"].items():
        player.resources[resource] -= count
    try:
        return analyze_expansion_plan(
            game, player_id, max_additional_roads=max_plan_roads,
        ).build_wait_rounds
    finally:
        for resource, count in BUILD_COSTS["development"].items():
            player.resources[resource] += count


def assess_development_purchase(
    game: GameState,
    player_id: int,
    *,
    max_plan_roads: int = 3,
    stall_wait_rounds: float = 2.5,
    max_build_delay: float = 2.0,
    tactical_score: int = 8,
    robber_blocked_pips: int = 4,
    title_horizon: int = 2,
) -> dict:
    """発展カードを今買う理由と、建設計画への影響を公開診断可能にする。"""
    mask = get_action_mask(game, player_id)
    buy = BuyDevelopmentAction()
    affordable = bool(mask[action_to_id(buy)])
    immediate_construction = any(
        mask[item.id] and isinstance(item.action, (BuildSettlementAction, BuildCityAction))
        for item in ACTION_CATALOG
    )
    player = player_for(game, player_id)
    plan = analyze_expansion_plan(
        game, player_id, max_additional_roads=max_plan_roads,
    )
    wait_before = plan.build_wait_rounds
    wait_after = (_wait_after_purchase(game, player_id, max_plan_roads)
                  if affordable else wait_before)
    if isfinite(wait_before) and isfinite(wait_after):
        build_delay = max(0.0, wait_after - wait_before)
    else:
        # 建設候補そのものがない99/inf相当の局面では、購入で悪化したとは扱わない。
        build_delay = 0.0 if not isfinite(wait_before) else 99.0

    blocked_pips = _blocked_production_pips(game, player_id)
    score = actual_score(game, player_id)
    rival_knights = max(other.played_knights for other in game.players
                        if other.id != player_id)
    own_known_knights = player.played_knights + player.development_cards.count("knight")
    army_target = max(3, rival_knights + 1)
    army_gap = max(0, army_target - own_known_knights)
    reasons = []
    if wait_before >= stall_wait_rounds:
        reasons.append("construction_stall")
    if blocked_pips >= robber_blocked_pips:
        reasons.append("robber_relief")
    if 0 < army_gap <= title_horizon:
        reasons.append("largest_army_race")

    probabilities = _unseen_card_probabilities(game, player_id)
    road_target = plan.selected_target
    card_values = {
        "knight": (1.0
                   + (1.4 if "robber_relief" in reasons else 0.0)
                   + (1.6 if "largest_army_race" in reasons else 0.0)),
        "road_building": (1.5 if road_target is not None
                          and road_target.additional_roads else 0.6),
        "year_of_plenty": 1.2 if isfinite(wait_before) else 0.8,
        "monopoly": 0.9,
        # 勝利点カードは引ければ強いが、種類を選べないため購入理由にはしない。
        # 特に9点での「勝利点カード待ち」は、確実な建設計画を壊しやすい。
        "victory_point": 0.8,
    }
    expected_card_value = sum(
        probabilities.get(card, 0.0) * value for card, value in card_values.items()
    )

    # 通常の停滞時は建設を2ラウンド超遅らせない。終盤・盗賊・称号競争は
    # 戦術的な見返りがあるため、さらに1ラウンドだけ許容する。
    knight_reasons = {"robber_relief", "largest_army_race"}
    tactical = any(reason in knight_reasons for reason in reasons)
    allowed_delay = max_build_delay + (1.0 if tactical else 0.0)
    eligible = bool(
        affordable
        and not immediate_construction
        and not player.new_development_cards
        and reasons
        and build_delay <= allowed_delay
        and expected_card_value >= 0.9
    )
    return {
        "eligible": eligible,
        "affordable": affordable,
        "immediate_construction": immediate_construction,
        "already_bought_this_turn": bool(player.new_development_cards),
        "site_count": player.settlements + player.cities,
        "score": score,
        "build_wait_before": wait_before,
        "build_wait_after": wait_after,
        "build_delay": build_delay,
        "allowed_build_delay": allowed_delay,
        "blocked_production_pips": blocked_pips,
        "largest_army_gap": army_gap,
        "reasons": reasons,
        "unseen_card_probabilities": probabilities,
        "card_values": card_values,
        "expected_card_value": expected_card_value,
    }
