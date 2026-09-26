"""公開盤面だけから道路・盗賊のAI判断を事後検証できる形にする。"""

from __future__ import annotations

from app.domain.actions import (Action, BuildRoadAction, MoveRobberAction,
                                PlaceInitialRoadAction, ProposeTradeAction)
from app.domain.game import (GameState, _building_owner, _longest_road_length,
                             legal_road_ids)

from .expansion_planner import analyze_expansion_plan
from .initial_road_planner import assess_initial_road
from .victory_race import (belief_robber_tile_value, desired_resource_weights,
                           public_score, robber_tile_value)


def _road_snapshot(game: GameState, player_id: int, action: BuildRoadAction) -> dict:
    before = _longest_road_length(game, player_id)
    held_before = game.longest_road_holder_id == player_id
    edge = game.board.edges[action.edge_id]
    blocked = [
        {"vertex_id": vertex_id, "owner_id": owner}
        for vertex_id in edge.vertex_ids
        if (owner := _building_owner(game, vertex_id)) not in {None, player_id}
    ]
    contested = [
        {
            "vertex_id": vertex_id,
            "opponent_road_owner_ids": sorted({
                owner for edge_id in game.board.vertices[vertex_id].edge_ids
                if (owner := game.roads.get(edge_id)) not in {None, player_id}
            }),
        }
        for vertex_id in edge.vertex_ids
        if _building_owner(game, vertex_id) is None
        and any(game.roads.get(edge_id) not in {None, player_id}
                for edge_id in game.board.vertices[vertex_id].edge_ids)
    ]
    plan = analyze_expansion_plan(game, player_id)
    target = plan.selected_target
    recommended = bool(target and action.edge_id in target.first_road_ids)
    game.roads[action.edge_id] = player_id
    try:
        after = _longest_road_length(game, player_id)
        continuations = legal_road_ids(game, player_id, initial=False)
        next_lengths = []
        for edge_id in continuations:
            game.roads[edge_id] = player_id
            try:
                next_lengths.append(_longest_road_length(game, player_id))
            finally:
                del game.roads[edge_id]
    finally:
        del game.roads[action.edge_id]
    rival = max(_longest_road_length(game, other.id)
                for other in game.players if other.id != player_id)
    holds_after = after >= 5 and after > rival
    return {
        "kind": "road",
        "edge_id": action.edge_id,
        "edge_vertices": list(edge.vertex_ids),
        "blocked_endpoints": blocked,
        "contested_endpoints": contested,
        "longest_before": before,
        "longest_after": after,
        "longest_increased": after > before,
        "rival_longest": rival,
        "held_longest_road_before": held_before,
        "holds_longest_road_after": holds_after,
        "takes_longest_road_now": not held_before and holds_after,
        "legal_continuation_count": len(continuations),
        "best_longest_after_one_more": max(next_lengths, default=after),
        "expansion_recommended": recommended,
        "expansion_target_vertex": target.vertex_id if target else None,
        "expansion_target_distance": target.additional_roads if target else None,
    }


def _initial_road_snapshot(game: GameState, player_id: int,
                           action: PlaceInitialRoadAction, agent: object) -> dict:
    assessments = [
        assess_initial_road(game, player_id, edge_id)
        for edge_id in legal_road_ids(game, player_id)
    ]
    ranked = sorted(
        assessments,
        key=lambda item: (-item["score"], -item["best_target_value"], item["edge_id"]),
    )
    chosen_rank = next(
        index for index, item in enumerate(ranked, 1)
        if item["edge_id"] == action.edge_id
    )
    return {
        "kind": "initial_road",
        "edge_id": action.edge_id,
        "chosen_rank": chosen_rank,
        "collision_aware_enabled": bool(
            getattr(agent, "collision_aware_initial_roads", False)
        ),
        "candidates": ranked,
    }


def _robber_snapshot(game: GameState, player_id: int, action: MoveRobberAction,
                     agent: object) -> dict:
    belief_enabled = bool(getattr(agent, "belief_aware_robber", False))
    beliefs = None
    weights = None
    if belief_enabled:
        from .hand_belief import BayesianHandEstimator
        # 診断ログは全画面へ出るため、盗賊当事者だけが知る資源も使わない。
        beliefs = BayesianHandEstimator().estimate(game, None)
        weights = desired_resource_weights(game, player_id)
    candidates = []
    for tile in game.board.tiles:
        if tile.id == game.robber_tile_id:
            continue
        adjacent = sorted({
            owner for vertex in game.board.vertices
            if tile.id in vertex.hex_ids
            if (owner := _building_owner(game, vertex.id)) not in {None, player_id}
        })
        candidate = {
            "tile_id": tile.id,
            "terrain": tile.terrain,
            "number": tile.number,
            "threat_value": robber_tile_value(game, player_id, tile.id),
            "adjacent_opponents": adjacent,
            "adjacent_public_scores": {
                str(owner): public_score(game, owner) for owner in adjacent
            },
        }
        if beliefs is not None:
            candidate["belief_value"] = belief_robber_tile_value(
                game, player_id, tile.id, beliefs
            )
        candidates.append(candidate)
    value_key = "belief_value" if belief_enabled else "threat_value"
    ranked = sorted(candidates, key=lambda item: (-item[value_key], item["tile_id"]))
    chosen_rank = next(index for index, item in enumerate(ranked, 1)
                       if item["tile_id"] == action.tile_id)
    chosen = next(item for item in candidates if item["tile_id"] == action.tile_id)
    return {
        "kind": "robber",
        "tile_id": action.tile_id,
        "terrain": chosen["terrain"],
        "number": chosen["number"],
        "public_threat_rank": chosen_rank,
        "opponent_aware_enabled": bool(getattr(agent, "opponent_aware_robber", False)),
        "belief_aware_enabled": belief_enabled,
        "desired_resource_weights": weights,
        "opponent_hand_beliefs": (
            {str(owner): belief.to_dict() for owner, belief in beliefs.items()
             if owner != player_id} if beliefs is not None else None
        ),
        "candidates": ranked,
    }


def _trade_snapshot(game: GameState, player_id: int,
                    action: ProposeTradeAction) -> dict:
    from .hand_belief import BayesianHandEstimator
    # 実際のAI判断は当事者知識を使えるが、公開診断は第三者視点に落とす。
    belief = BayesianHandEstimator().estimate(game, None)[action.target_id]
    wanted = sorted({resource for offer in action.offers
                     for resource, count in offer.want.items() if count})
    return {
        "kind": "player_trade",
        "target_id": action.target_id,
        "target_public_score": public_score(game, action.target_id),
        "offers": [
            {"give": dict(offer.give), "want": dict(offer.want)}
            for offer in action.offers
        ],
        "wanted_probability_has": {
            resource: belief.probability_has(resource) for resource in wanted
        },
        "target_expected_hand_size": belief.expected_total,
    }


def diagnose_action(game: GameState, player_id: int, action: Action,
                    agent: object) -> dict | None:
    """状態を変更せず、公開情報だけの診断レコードを返す。"""
    if isinstance(action, PlaceInitialRoadAction):
        detail = _initial_road_snapshot(game, player_id, action, agent)
    elif isinstance(action, BuildRoadAction) and not game.free_road_remaining:
        detail = _road_snapshot(game, player_id, action)
    elif isinstance(action, MoveRobberAction):
        detail = _robber_snapshot(game, player_id, action, agent)
    elif isinstance(action, ProposeTradeAction):
        detail = _trade_snapshot(game, player_id, action)
    else:
        return None
    return {
        "sequence_before": game.revision,
        "turn_number": game.turn_number,
        "player_id": player_id,
        **detail,
    }
