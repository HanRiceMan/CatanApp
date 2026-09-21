"""単純なルールベースAI。学習・自動デバッグ用で、強さは目的にしない。"""

from collections import Counter
from random import Random

from .game import (BUILD_COSTS, RESOURCES, GameActionError, GameState, _can_afford,
                   _port_ratio, bank_trade, build_piece, discard_for_seven, end_turn,
                   legal_city_ids, legal_road_ids, legal_settlement_ids, move_robber,
                   pending_ai_player_id, place_initial_piece, player_for, propose_counter_trade, propose_trade,
                   respond_to_trade, roll_for_order, roll_turn_dice, setup_player_id,
                   start_normal_game, steal_with_robber, use_development)


def _request_id(game: GameState) -> str:
    return f"ai-{game.ai_action_number:08d}"


def _rng(game: GameState) -> Random:
    return Random((game.seed << 20) ^ game.ai_action_number ^ (game.revision << 5))


def _record(game: GameState, message: str) -> None:
    game.ai_action_number += 1
    game.ai_log.append(message)
    if len(game.ai_log) > 500:
        del game.ai_log[:-500]


def _vertex_value(game: GameState, vertex_id: int, player_id: int) -> int:
    pips = 0
    terrains = set()
    for tile_id in game.board.vertices[vertex_id].hex_ids:
        tile = game.board.tiles[tile_id]
        if tile.number:
            pips += 6 - abs(7 - tile.number)
            terrains.add(tile.terrain)
    owned_terrains = {game.board.tiles[tile_id].terrain
                      for vertex, owner in game.settlements.items() if owner == player_id
                      for tile_id in game.board.vertices[vertex].hex_ids}
    return pips * 10 + len(terrains) * 4 + len(terrains - owned_terrains) * 3


def _best_vertex(game: GameState, candidates: list[int], player_id: int) -> int:
    rng = _rng(game)
    return max(candidates, key=lambda vertex: (_vertex_value(game, vertex, player_id), rng.random()))


def _best_initial_road(game: GameState, candidates: list[int], settlement_id: int) -> int:
    rng = _rng(game)
    def value(edge_id: int) -> tuple[int, float]:
        a, b = game.board.edges[edge_id].vertex_ids
        far = b if a == settlement_id else a
        onward = [other for other in game.board.vertices[far].edge_ids if other != edge_id]
        reachable = [vertex for other in onward for vertex in game.board.edges[other].vertex_ids if vertex != far]
        return (max((_vertex_value(game, vertex, setup_player_id(game) or 1) for vertex in reachable), default=0), rng.random())
    return max(candidates, key=value)


def _missing_cost(player, cost: dict[str, int]) -> list[str]:
    return [resource for resource in RESOURCES if player.resources[resource] < cost.get(resource, 0)]


def _goal_cost(game: GameState, player_id: int) -> dict[str, int]:
    if legal_city_ids(game, player_id) and player_for(game, player_id).cities < 4:
        return BUILD_COSTS["city"]
    if legal_settlement_ids(game, player_id, initial=False):
        return BUILD_COSTS["settlement"]
    return BUILD_COSTS["road"]


def _try_bank_trade(game: GameState, player_id: int, request_id: str) -> bool:
    player = player_for(game, player_id)
    goal = _goal_cost(game, player_id)
    missing = _missing_cost(player, goal)
    for wanted in missing:
        if game.bank[wanted] < 1:
            continue
        for offered in RESOURCES:
            ratio = _port_ratio(game, player_id, offered)
            reserve = goal.get(offered, 0)
            if offered != wanted and player.resources[offered] - reserve >= ratio:
                bank_trade(game, player_id, offered, wanted, game.revision, request_id)
                return True
    return False


def _make_trade_offers(game: GameState, player_id: int) -> list[dict]:
    player = player_for(game, player_id)
    goal = _goal_cost(game, player_id)
    wanted = _missing_cost(player, goal)
    give = [resource for resource in RESOURCES if player.resources[resource] > goal.get(resource, 0)]
    offers = []
    for index, resource in enumerate(wanted[:3]):
        available = [candidate for candidate in give if candidate != resource]
        if not available:
            break
        offered = available[index % len(available)]
        offers.append({"give": {offered: 1}, "want": {resource: 1}})
    return offers


def _respond_trade(game: GameState, player_id: int, request_id: str) -> str:
    trade = game.pending_trade
    if trade is None:
        raise GameActionError("AIが応答する交渉がありません。")
    player = player_for(game, player_id)
    for index, offer in enumerate(trade["offers"]):
        if _can_afford(player, offer["want"]) and sum(offer["give"].values()) >= sum(offer["want"].values()):
            respond_to_trade(game, player_id, "accept", index, game.revision, request_id)
            return f"P{player_id}: 交渉候補{index + 1}を承諾"
    if not trade["is_counter"] and sum(player.resources.values()) > 0 and _rng(game).random() < .45:
        respond_to_trade(game, player_id, "counter", None, game.revision, request_id)
        return f"P{player_id}: 逆交渉を選択"
    respond_to_trade(game, player_id, "decline", None, game.revision, request_id)
    return f"P{player_id}: 交渉を拒否"


def _counter_offer(game: GameState, player_id: int, request_id: str) -> str:
    player = player_for(game, player_id)
    give = [resource for resource in RESOURCES if player.resources[resource] > 0]
    if not give:
        # AIは資源0枚なら逆交渉を選ばないため、通常ここには来ない。
        raise GameActionError("逆交渉に出せる資源がありません。")
    offered = _rng(game).choice(give)
    wanted = _rng(game).choice([resource for resource in RESOURCES if resource != offered])
    propose_counter_trade(game, player_id, [{"give": {offered: 1}, "want": {wanted: 1}}], game.revision, request_id)
    return f"P{player_id}: {offered}と{wanted}の逆交渉"


def _choose_robber_tile(game: GameState, player_id: int) -> int:
    rng = _rng(game)
    def score(tile_id: int) -> tuple[int, float]:
        value = 0
        for vertex in game.board.vertices:
            # 下の条件ですぐ絞る。54交点でもデバッグ用には十分軽い。
            if tile_id not in vertex.hex_ids:
                continue
            owner = game.settlements.get(vertex.id, game.cities.get(vertex.id))
            if owner == player_id:
                value -= 4
            elif owner:
                value += 2 if vertex.id in game.cities else 1
        return value, rng.random()
    return max((tile.id for tile in game.board.tiles if tile.id != game.robber_tile_id), key=score)


def ai_actor(game: GameState) -> int | None:
    return pending_ai_player_id(game)


def play_ai_step(game: GameState) -> str:
    player_id = ai_actor(game)
    if player_id is None:
        raise GameActionError("現在はAIが操作する場面ではありません。")
    request_id, rng = _request_id(game), _rng(game)
    phase = game.phase
    if phase == "rolling_order":
        dice = (rng.randint(1, 6), rng.randint(1, 6))
        roll_for_order(game, player_id, game.revision, request_id, dice)
        message = f"P{player_id}: 順番決め {sum(dice)}"
    elif phase == "setup_ready":
        if game.pending_settlement_id is None:
            target = _best_vertex(game, legal_settlement_ids(game), player_id)
            place_initial_piece(game, player_id, "settlement", target, game.revision, request_id)
            message = f"P{player_id}: 初期開拓地 V{target}"
        else:
            target = _best_initial_road(game, legal_road_ids(game), game.pending_settlement_id)
            place_initial_piece(game, player_id, "road", target, game.revision, request_id)
            message = f"P{player_id}: 初期道路 E{target}"
    elif phase == "ready_to_play":
        start_normal_game(game, player_id, game.revision, request_id)
        message = f"P{player_id}: ゲーム開始"
    elif phase == "turn_pre_roll":
        player = player_for(game, player_id)
        if (not game.used_development_this_turn
                and player.development_cards.count("knight") > player.new_development_cards.count("knight")
                and rng.random() < .1):
            use_development(game, player_id, "knight", game.revision, request_id)
            message = f"P{player_id}: 騎士を使用"
        else:
            dice = (rng.randint(1, 6), rng.randint(1, 6))
            roll_turn_dice(game, player_id, game.revision, request_id, dice)
            message = f"P{player_id}: 通常ダイス {sum(dice)}"
    elif phase == "discarding":
        player = player_for(game, player_id); need = sum(player.resources.values()) // 2; cards = Counter()
        pool = [resource for resource in RESOURCES for _ in range(player.resources[resource])]; rng.shuffle(pool)
        cards.update(pool[:need]); discard_for_seven(game, player_id, dict(cards), game.revision, request_id)
        message = f"P{player_id}: 7で{need}枚破棄"
    elif phase == "robber_move":
        target = _choose_robber_tile(game, player_id); move_robber(game, player_id, target, game.revision, request_id)
        message = f"P{player_id}: 盗賊をH{target}へ"
    elif phase == "robber_steal":
        victim = max(game.robber_victim_ids, key=lambda pid: sum(player_for(game, pid).resources.values()))
        steal_with_robber(game, player_id, victim, game.revision, request_id)
        message = f"P{player_id}: P{victim}から1枚略奪"
    elif phase == "trade_response":
        message = _respond_trade(game, player_id, request_id)
    elif phase == "trade_counter_offer":
        message = _counter_offer(game, player_id, request_id)
    elif phase == "action":
        player = player_for(game, player_id)
        roads = legal_road_ids(game, player_id, initial=False)
        settlements = legal_settlement_ids(game, player_id, initial=False)
        cities = legal_city_ids(game, player_id)
        if game.free_road_remaining and roads and sum(owner == player_id for owner in game.roads.values()) < 15:
            target = rng.choice(roads); build_piece(game, player_id, "road", target, game.revision, request_id); message = f"P{player_id}: 無料道路 E{target}"
        elif cities and player.cities < 4 and _can_afford(player, BUILD_COSTS["city"]):
            target = rng.choice(cities); build_piece(game, player_id, "city", target, game.revision, request_id); message = f"P{player_id}: 都市 V{target}"
        elif settlements and player.settlements < 5 and _can_afford(player, BUILD_COSTS["settlement"]):
            target = _best_vertex(game, settlements, player_id); build_piece(game, player_id, "settlement", target, game.revision, request_id); message = f"P{player_id}: 開拓地 V{target}"
        elif roads and sum(owner == player_id for owner in game.roads.values()) < 15 and _can_afford(player, BUILD_COSTS["road"]):
            target = rng.choice(roads); build_piece(game, player_id, "road", target, game.revision, request_id); message = f"P{player_id}: 道路 E{target}"
        elif game.development_deck and _can_afford(player, BUILD_COSTS["development"]) and rng.random() < .45:
            build_piece(game, player_id, "development", None, game.revision, request_id); message = f"P{player_id}: 発展カード購入"
        elif _try_bank_trade(game, player_id, request_id):
            message = f"P{player_id}: 銀行・港交易"
        elif game.trade_count < 2 and (offers := _make_trade_offers(game, player_id)):
            target = rng.choice([other.id for other in game.players if other.id != player_id])
            propose_trade(game, player_id, target, offers, game.revision, request_id); message = f"P{player_id}: P{target}へ{len(offers)}候補で交渉"
        else:
            end_turn(game, player_id, game.revision, request_id); message = f"P{player_id}: 手番終了"
    else:
        raise GameActionError(f"AIが未対応のフェーズです: {phase}")
    _record(game, message)
    return message


def run_ai_until_pause(game: GameState, max_steps: int = 100_000) -> dict:
    start_actions = game.ai_action_number
    while game.phase != "game_over" and ai_actor(game) is not None:
        if game.ai_action_number - start_actions >= max_steps:
            return {"status": "step_limit", "steps": max_steps, "phase": game.phase, "winner_id": game.winner_id}
        play_ai_step(game)
    return {"status": "game_over" if game.phase == "game_over" else "waiting_for_human",
            "steps": game.ai_action_number - start_actions, "phase": game.phase, "winner_id": game.winner_id}
