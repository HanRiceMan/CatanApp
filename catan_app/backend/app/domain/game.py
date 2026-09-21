"""対局状態とカタン基本ルール。HTTPや描画には依存しない。"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from random import Random
from secrets import randbelow, token_urlsafe
from uuid import uuid4

from .board import Board, BoardRules, create_board

RESOURCES = ("wood", "brick", "sheep", "wheat", "ore")
RESOURCE_NAMES = {"wood": "木材", "brick": "レンガ", "sheep": "羊毛", "wheat": "小麦", "ore": "鉱石"}
DEVELOPMENT_NAMES = {"knight": "騎士", "road_building": "街道建設", "year_of_plenty": "豊作", "monopoly": "独占"}
BUILD_COSTS = {"road": {"wood": 1, "brick": 1}, "settlement": {"wood": 1, "brick": 1, "sheep": 1, "wheat": 1}, "city": {"wheat": 2, "ore": 3}, "development": {"sheep": 1, "wheat": 1, "ore": 1}}
DEVELOPMENT_DECK = ("knight",) * 14 + ("road_building",) * 2 + ("year_of_plenty",) * 2 + ("monopoly",) * 2 + ("victory_point",) * 5


class GameActionError(ValueError):
    pass


@dataclass
class Player:
    id: int
    name: str
    color: str
    is_ai: bool = False
    resources: dict[str, int] = field(default_factory=lambda: dict.fromkeys(RESOURCES, 0))
    development_cards: list[str] = field(default_factory=list)
    new_development_cards: list[str] = field(default_factory=list)
    used_development_cards: list[str] = field(default_factory=list)
    settlements: int = 0
    cities: int = 0
    played_knights: int = 0


@dataclass
class GameState:
    id: str
    seed: int
    board: Board
    rules: BoardRules
    players: list[Player]
    player_tokens: dict[int, str]
    phase: str = "rolling_order"
    revision: int = 0
    seat_order: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    setup_order: list[int] = field(default_factory=list)
    ranking_groups: list[list[int]] = field(default_factory=lambda: [[1, 2, 3, 4]])
    pending_rollers: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    round_number: int = 1
    round_totals: dict[int, int] = field(default_factory=dict)
    rolls: list[dict] = field(default_factory=list)
    longest_road_holder_id: int | None = None
    largest_army_holder_id: int | None = None
    accepted_requests: dict[tuple[int, str], tuple[object, ...]] = field(default_factory=dict)
    settlements: dict[int, int] = field(default_factory=dict)
    cities: dict[int, int] = field(default_factory=dict)
    roads: dict[int, int] = field(default_factory=dict)
    setup_index: int = 0
    pending_settlement_id: int | None = None
    bank: dict[str, int] = field(default_factory=lambda: dict.fromkeys(RESOURCES, 19))
    development_deck: list[str] = field(default_factory=list)
    robber_tile_id: int = 0
    current_player_id: int | None = None
    turn_number: int = 0
    last_roll: dict | None = None
    dice_history: list[dict] = field(default_factory=list)
    latest_dice_event: dict | None = None
    activity_log: list[dict] = field(default_factory=list)
    pending_discards: list[int] = field(default_factory=list)
    robber_victim_ids: list[int] = field(default_factory=list)
    robber_reason: str | None = None
    trade_count: int = 0
    pending_trade: dict | None = None
    used_development_this_turn: bool = False
    free_road_remaining: int = 0
    winner_id: int | None = None
    ai_action_number: int = 0
    ai_log: list[str] = field(default_factory=list)
    ai_watch_mode: bool = False


def create_game(seed: int, rules: BoardRules | None = None, ai_player_ids: tuple[int, ...] = (),
                ai_watch_mode: bool = False) -> GameState:
    rules = rules or BoardRules()
    if any(type(player_id) is not int or player_id not in range(1, 5) for player_id in ai_player_ids):
        raise ValueError("AIプレイヤーIDは1〜4で指定してください。")
    players = [Player(i, f"プレイヤー {i}", color, i in ai_player_ids)
               for i, color in enumerate(("red", "cyan", "purple", "yellow"), 1)]
    deck = list(DEVELOPMENT_DECK)
    Random(seed ^ 0xCA7A).shuffle(deck)
    board = create_board(seed, rules)
    return GameState(str(uuid4()), seed, board, rules, players, {p.id: token_urlsafe(32) for p in players},
                     development_deck=deck, robber_tile_id=board.robber_tile_id,
                     ai_watch_mode=ai_watch_mode)


def player_for(game: GameState, player_id: int) -> Player:
    try:
        return next(player for player in game.players if player.id == player_id)
    except StopIteration as error:
        raise GameActionError("プレイヤーIDが不正です。") from error


def _request_is_repeat(game: GameState, player_id: int, request_id: str, fingerprint: tuple[object, ...], expected_revision: int) -> bool:
    key = (player_id, request_id)
    if key in game.accepted_requests:
        if game.accepted_requests[key] != fingerprint:
            raise GameActionError("同じリクエストIDを別の操作には使えません。")
        return True
    if expected_revision != game.revision:
        raise GameActionError("対局が更新されています。画面の更新を待って再度お試しください。")
    return False


def _finish(game: GameState, player_id: int, request_id: str, fingerprint: tuple[object, ...]) -> None:
    game.accepted_requests[(player_id, request_id)] = fingerprint
    game.revision += 1


def _record_activity(game: GameState, player_id: int, kind: str, text: str, *, card: str | None = None,
                     trade: dict | None = None, bank: dict | None = None) -> None:
    """全員に見せてよい行動だけを、確定済みのリビジョンに紐づける。"""
    entry = {"sequence": game.revision, "turn_number": game.turn_number,
             "player_id": player_id, "kind": kind, "text": text}
    if card is not None: entry["card"] = card
    if trade is not None: entry["trade"] = trade
    if bank is not None: entry["bank"] = bank
    game.activity_log.append(entry)


def _resource_summary(cards: dict[str, int]) -> str:
    return "・".join(f"{RESOURCE_NAMES[resource]}{cards[resource]}枚" for resource in RESOURCES if cards.get(resource, 0))


def _resource_map(values: dict[str, int], *, allow_empty: bool = False) -> dict[str, int]:
    if not isinstance(values, dict) or any(key not in RESOURCES or type(value) is not int or value < 0 for key, value in values.items()):
        raise GameActionError("資源の指定が不正です。")
    result = {resource: values.get(resource, 0) for resource in RESOURCES}
    if not allow_empty and not any(result.values()):
        raise GameActionError("少なくとも1枚の資源を選んでください。")
    return result


def _can_afford(player: Player, cards: dict[str, int]) -> bool:
    return all(player.resources[resource] >= count for resource, count in cards.items())


def _move_to_bank(game: GameState, player: Player, cards: dict[str, int]) -> None:
    if not _can_afford(player, cards):
        raise GameActionError("必要な資源カードを持っていません。")
    for resource, count in cards.items():
        player.resources[resource] -= count
        game.bank[resource] += count


def _take_from_bank(game: GameState, player: Player, cards: dict[str, int]) -> None:
    if any(game.bank[resource] < count for resource, count in cards.items()):
        raise GameActionError("銀行の資源が不足しています。")
    for resource, count in cards.items():
        player.resources[resource] += count
        game.bank[resource] -= count


def roll_for_order(game: GameState, player_id: int, expected_revision: int, request_id: str, dice: tuple[int, int] | None = None) -> None:
    fingerprint = ("order_roll", expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    if game.phase != "rolling_order": raise GameActionError("順番は確定済みです。席は変更できません。")
    if not game.pending_rollers or player_id != game.pending_rollers[0]: raise GameActionError("まだあなたがサイコロを振る順番ではありません。")
    dice = dice or (randbelow(6) + 1, randbelow(6) + 1)
    if len(dice) != 2 or any(type(value) is not int or not 1 <= value <= 6 for value in dice): raise GameActionError("サイコロの値が不正です。")
    game.round_totals[player_id] = sum(dice)
    game.rolls.append({"player_id": player_id, "round": game.round_number, "dice": list(dice), "total": sum(dice)})
    game.latest_dice_event = dict(game.rolls[-1], sequence=game.revision + 1, kind="order")
    game.pending_rollers.pop(0)
    _finish(game, player_id, request_id, fingerprint)
    _record_activity(game, player_id, "order_roll", f"順番決めのサイコロで{sum(dice)}を振りました。")
    if game.pending_rollers: return
    ranked: list[list[int]] = []
    for group in game.ranking_groups:
        if len(group) == 1: ranked.append(group); continue
        totals = sorted({game.round_totals[p] for p in group}, reverse=True)
        ranked.extend([[p for p in group if game.round_totals[p] == total] for total in totals])
    game.ranking_groups = ranked
    if all(len(group) == 1 for group in ranked):
        game.seat_order = [group[0] for group in ranked]
        game.setup_order = game.seat_order + list(reversed(game.seat_order))
        game.phase = "setup_ready"
    else:
        game.round_number += 1; game.round_totals = {}
        game.pending_rollers = [p for group in ranked if len(group) > 1 for p in group]


def setup_player_id(game: GameState) -> int | None:
    return game.setup_order[game.setup_index] if game.phase == "setup_ready" and game.setup_index < len(game.setup_order) else None


def _building_owner(game: GameState, vertex_id: int) -> int | None:
    return game.settlements.get(vertex_id, game.cities.get(vertex_id))


def _all_building_vertices(game: GameState) -> set[int]: return set(game.settlements) | set(game.cities)


def legal_settlement_ids(game: GameState, player_id: int | None = None, *, initial: bool = True) -> list[int]:
    if initial:
        if setup_player_id(game) is None or game.pending_settlement_id is not None: return []
    elif game.phase != "action" or player_id != game.current_player_id: return []
    blocked = _all_building_vertices(game)
    for vertex_id in list(blocked):
        for edge_id in game.board.vertices[vertex_id].edge_ids: blocked.update(game.board.edges[edge_id].vertex_ids)
    candidates = [vertex.id for vertex in game.board.vertices if vertex.id not in blocked]
    if initial: return [vertex_id for vertex_id in candidates if any(edge_id not in game.roads for edge_id in game.board.vertices[vertex_id].edge_ids)]
    return [vertex_id for vertex_id in candidates if any(game.roads.get(edge_id) == player_id for edge_id in game.board.vertices[vertex_id].edge_ids)]


def legal_road_ids(game: GameState, player_id: int | None = None, *, initial: bool = True) -> list[int]:
    if initial:
        if setup_player_id(game) is None or game.pending_settlement_id is None: return []
        return [edge_id for edge_id in game.board.vertices[game.pending_settlement_id].edge_ids if edge_id not in game.roads]
    if game.phase != "action" or player_id != game.current_player_id: return []
    result = []
    for edge in game.board.edges:
        if edge.id in game.roads: continue
        for vertex_id in edge.vertex_ids:
            owner = _building_owner(game, vertex_id)
            if owner == player_id or (owner is None and any(game.roads.get(other) == player_id for other in game.board.vertices[vertex_id].edge_ids)):
                result.append(edge.id); break
    return result


def legal_city_ids(game: GameState, player_id: int | None) -> list[int]:
    return [vertex for vertex, owner in game.settlements.items() if owner == player_id] if game.phase == "action" and player_id == game.current_player_id else []


def place_initial_piece(game: GameState, player_id: int, kind: str, target_id: int, expected_revision: int, request_id: str) -> None:
    fingerprint = ("initial", kind, target_id, expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    if game.phase != "setup_ready": raise GameActionError("現在は初期配置の時間ではありません。")
    if player_id != setup_player_id(game): raise GameActionError("まだあなたが配置する順番ではありません。")
    if type(target_id) is not int: raise GameActionError("配置場所のIDが不正です。")
    player = player_for(game, player_id)
    if kind == "settlement":
        if target_id not in legal_settlement_ids(game): raise GameActionError("その交差点には置けません。建物のある場所と、その隣は選べません。")
        grant = Counter()
        if player.settlements == 1: grant.update(game.board.tiles[i].terrain for i in game.board.vertices[target_id].hex_ids if game.board.tiles[i].terrain != "desert")
        _take_from_bank(game, player, grant)
        game.settlements[target_id] = player_id; player.settlements += 1; game.pending_settlement_id = target_id
    elif kind == "road":
        if target_id not in legal_road_ids(game): raise GameActionError("直前に置いた開拓地につながる、空いている辺を選んでください。")
        game.roads[target_id] = player_id; game.pending_settlement_id = None; game.setup_index += 1
        if game.setup_index == len(game.setup_order): game.phase = "ready_to_play"
    else: raise GameActionError("置けるコマは開拓地か道です。")
    _finish(game, player_id, request_id, fingerprint)
    _record_activity(game, player_id, "setup", f"初期配置で{'開拓地' if kind == 'settlement' else '道'}を置きました。")


def start_normal_game(game: GameState, player_id: int, expected_revision: int, request_id: str) -> None:
    fingerprint = ("start_normal", expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    if game.phase != "ready_to_play" or player_id != game.seat_order[0]: raise GameActionError("初期配置を終えた先手プレイヤーだけがゲームを開始できます。")
    game.phase = "turn_pre_roll"; game.current_player_id = player_id; game.turn_number = 1
    _finish(game, player_id, request_id, fingerprint)
    _record_activity(game, player_id, "start", "ゲームを開始しました。")


def _produce_resources(game: GameState, number: int) -> None:
    grants: dict[int, Counter] = defaultdict(Counter)
    for vertex_id, owner in {**game.settlements, **game.cities}.items():
        multiplier = 2 if vertex_id in game.cities else 1
        for tile_id in game.board.vertices[vertex_id].hex_ids:
            tile = game.board.tiles[tile_id]
            if tile.id != game.robber_tile_id and tile.number == number and tile.terrain != "desert": grants[owner][tile.terrain] += multiplier
    requested = Counter(); [requested.update(cards) for cards in grants.values()]
    for resource, count in requested.items():
        if game.bank[resource] < count:
            for cards in grants.values(): cards[resource] = 0
    for owner, cards in grants.items(): _take_from_bank(game, player_for(game, owner), cards)


def roll_turn_dice(game: GameState, player_id: int, expected_revision: int, request_id: str, dice: tuple[int, int] | None = None) -> None:
    fingerprint = ("turn_roll", expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    if game.phase != "turn_pre_roll" or game.current_player_id != player_id: raise GameActionError("今はあなたが通常手番のサイコロを振る時間ではありません。")
    dice = dice or (randbelow(6) + 1, randbelow(6) + 1)
    if len(dice) != 2 or any(type(value) is not int or not 1 <= value <= 6 for value in dice): raise GameActionError("サイコロの値が不正です。")
    total = sum(dice); game.last_roll = {"player_id": player_id, "dice": list(dice), "total": total, "turn_number": game.turn_number}
    game.dice_history.append(dict(game.last_roll, dice=list(game.last_roll["dice"])))
    game.latest_dice_event = dict(game.last_roll, sequence=game.revision + 1, kind="turn")
    if total == 7:
        game.pending_discards = [p.id for p in game.players if sum(p.resources.values()) > 7]
        game.phase = "discarding" if game.pending_discards else "robber_move"; game.robber_reason = "seven"
    else: _produce_resources(game, total); game.phase = "action"
    _finish(game, player_id, request_id, fingerprint)
    _record_activity(game, player_id, "roll", f"サイコロで{total}を振りました。")


def discard_for_seven(game: GameState, player_id: int, cards: dict[str, int], expected_revision: int, request_id: str) -> None:
    normalized = _resource_map(cards); fingerprint = ("discard", tuple(normalized.items()), expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    if game.phase != "discarding" or player_id not in game.pending_discards: raise GameActionError("あなたは現在、資源を捨てる対象ではありません。")
    player = player_for(game, player_id)
    if sum(normalized.values()) != sum(player.resources.values()) // 2: raise GameActionError("手札が8枚以上の場合、ちょうど半分を捨ててください。")
    _move_to_bank(game, player, normalized); game.pending_discards.remove(player_id)
    if not game.pending_discards: game.phase = "robber_move"
    _finish(game, player_id, request_id, fingerprint)
    _record_activity(game, player_id, "discard", f"資源カードを{sum(normalized.values())}枚捨てました。")


def move_robber(game: GameState, player_id: int, tile_id: int, expected_revision: int, request_id: str) -> None:
    fingerprint = ("move_robber", tile_id, expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    if game.phase != "robber_move" or player_id != game.current_player_id: raise GameActionError("今は盗賊を動かす時間ではありません。")
    if type(tile_id) is not int or tile_id not in range(len(game.board.tiles)) or tile_id == game.robber_tile_id: raise GameActionError("盗賊は現在とは異なる地形タイルへ動かしてください。")
    game.robber_tile_id = tile_id
    # タイルIDと交差点IDは別体系。移動先タイルに実際に接する全交差点から所有者を探す。
    owners = {_building_owner(game, vertex.id) for vertex in game.board.vertices if tile_id in vertex.hex_ids}
    game.robber_victim_ids = sorted(owner for owner in owners if owner and owner != player_id and sum(player_for(game, owner).resources.values()) > 0)
    if game.robber_victim_ids:
        game.phase = "robber_steal"
    else:
        # サイコロ前の騎士なら、盗賊処理後にも通常のサイコロを振る必要がある。
        game.phase = "turn_pre_roll" if game.robber_reason == "knight" and game.last_roll is None else "action"
        game.robber_reason = None
    _finish(game, player_id, request_id, fingerprint)
    tile = game.board.tiles[tile_id]
    _record_activity(game, player_id, "robber", f"盗賊を{RESOURCE_NAMES.get(tile.terrain, '砂漠')}のタイルへ移動しました。")


def steal_with_robber(game: GameState, player_id: int, victim_id: int, expected_revision: int, request_id: str) -> None:
    fingerprint = ("steal", victim_id, expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    if game.phase != "robber_steal" or player_id != game.current_player_id or victim_id not in game.robber_victim_ids: raise GameActionError("そのプレイヤーからは資源を奪えません。")
    victim, player = player_for(game, victim_id), player_for(game, player_id)
    cards = [resource for resource in RESOURCES for _ in range(victim.resources[resource])]
    if not cards: raise GameActionError("相手の資源は既にありません。")
    resource = cards[randbelow(len(cards))]; victim.resources[resource] -= 1; player.resources[resource] += 1
    next_phase = "turn_pre_roll" if game.robber_reason == "knight" and game.last_roll is None else "action"
    game.robber_victim_ids = []; game.robber_reason = None; game.phase = next_phase
    _finish(game, player_id, request_id, fingerprint)
    _record_activity(game, player_id, "steal", f"プレイヤー {victim_id} から資源カードを1枚奪いました。")


def _pay_build_cost(game: GameState, player: Player, kind: str) -> None:
    if kind not in BUILD_COSTS: raise GameActionError("建設の種類が不正です。")
    _move_to_bank(game, player, BUILD_COSTS[kind])


def _road_count(game: GameState, player_id: int) -> int: return sum(owner == player_id for owner in game.roads.values())


def _longest_road_length(game: GameState, owner_id: int) -> int:
    owned = {edge_id for edge_id, owner in game.roads.items() if owner == owner_id}
    if not owned: return 0
    adjacent: dict[int, list[int]] = defaultdict(list)
    for edge_id in owned:
        for vertex_id in game.board.edges[edge_id].vertex_ids: adjacent[vertex_id].append(edge_id)
    def visit(vertex_id: int, used: frozenset[int], entered: bool) -> int:
        if entered and (owner := _building_owner(game, vertex_id)) is not None and owner != owner_id: return 0
        best = 0
        for edge_id in adjacent[vertex_id]:
            if edge_id in used: continue
            a, b = game.board.edges[edge_id].vertex_ids; next_vertex = b if a == vertex_id else a
            best = max(best, 1 + visit(next_vertex, used | {edge_id}, True))
        return best
    return max(visit(vertex_id, frozenset(), False) for vertex_id in adjacent)


def _update_longest_road(game: GameState) -> None:
    lengths = {player.id: _longest_road_length(game, player.id) for player in game.players}; maximum = max(lengths.values(), default=0)
    if maximum < 5: game.longest_road_holder_id = None; return
    leaders = [pid for pid, length in lengths.items() if length == maximum]
    if game.longest_road_holder_id not in leaders: game.longest_road_holder_id = leaders[0] if len(leaders) == 1 else None


def _check_winner(game: GameState, player_id: int) -> None:
    player = player_for(game, player_id)
    score = player.settlements + 2 * player.cities + 2 * (game.longest_road_holder_id == player_id) + 2 * (game.largest_army_holder_id == player_id) + player.development_cards.count("victory_point")
    if score >= 10: game.winner_id = player_id; game.phase = "game_over"


def build_piece(game: GameState, player_id: int, kind: str, target_id: int | None, expected_revision: int, request_id: str) -> None:
    fingerprint = ("build", kind, target_id, expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    if game.phase != "action" or game.current_player_id != player_id: raise GameActionError("建設できるのは、自分の通常手番の行動フェーズだけです。")
    if game.free_road_remaining and kind != "road": raise GameActionError("街道建設で残っている無料の道を先に配置してください。")
    player = player_for(game, player_id)
    if kind == "road":
        if target_id not in legal_road_ids(game, player_id, initial=False): raise GameActionError("その辺には、あなたの道をつなげられません。")
        if _road_count(game, player_id) >= 15: raise GameActionError("道のコマを使い切っています。")
        if game.free_road_remaining: game.free_road_remaining -= 1
        else: _pay_build_cost(game, player, kind)
        game.roads[target_id] = player_id; _update_longest_road(game)
        if game.free_road_remaining and (_road_count(game, player_id) >= 15 or not legal_road_ids(game, player_id, initial=False)):
            game.free_road_remaining = 0
        if not game.free_road_remaining and game.last_roll is None:
            game.phase = "turn_pre_roll"
    elif kind == "settlement":
        if target_id not in legal_settlement_ids(game, player_id, initial=False): raise GameActionError("その交差点には、あなたの開拓地を置けません。")
        if player.settlements >= 5: raise GameActionError("開拓地のコマを使い切っています。")
        _pay_build_cost(game, player, kind); game.settlements[target_id] = player_id; player.settlements += 1; _update_longest_road(game)
    elif kind == "city":
        if target_id not in legal_city_ids(game, player_id): raise GameActionError("自分の開拓地だけを都市へアップグレードできます。")
        if player.cities >= 4: raise GameActionError("都市のコマを使い切っています。")
        _pay_build_cost(game, player, kind); del game.settlements[target_id]; game.cities[target_id] = player_id; player.settlements -= 1; player.cities += 1; _update_longest_road(game)
    elif kind == "development":
        if not game.development_deck: raise GameActionError("発展カードの山札が空です。")
        _pay_build_cost(game, player, kind); card = game.development_deck.pop(); player.development_cards.append(card); player.new_development_cards.append(card)
    else: raise GameActionError("建設の種類が不正です。")
    _check_winner(game, player_id); _finish(game, player_id, request_id, fingerprint)
    building = {"road": "道を建設", "settlement": "開拓地を建設", "city": "開拓地を都市に発展",
                "development": "発展カードを1枚購入"}[kind]
    _record_activity(game, player_id, "build", f"{building}しました。")


def _port_ratio(game: GameState, player_id: int, resource: str) -> int:
    owned = {vertex for vertex in _all_building_vertices(game) if _building_owner(game, vertex) == player_id}; ratios = [4]
    for port in game.board.ports:
        if owned.intersection(game.board.edges[port.edge_id].vertex_ids) and (port.resource is None or port.resource == resource): ratios.append(port.ratio)
    return min(ratios)


def bank_trade(game: GameState, player_id: int, give_resource: str, receive_resource: str, expected_revision: int, request_id: str) -> None:
    fingerprint = ("bank_trade", give_resource, receive_resource, expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    if game.phase != "action" or game.current_player_id != player_id: raise GameActionError("銀行との交易は自分の行動フェーズだけです。")
    if game.free_road_remaining: raise GameActionError("街道建設で残っている無料の道を先に配置してください。")
    if give_resource not in RESOURCES or receive_resource not in RESOURCES or give_resource == receive_resource: raise GameActionError("異なる2種類の資源を選んでください。")
    player = player_for(game, player_id); ratio = _port_ratio(game, player_id, give_resource)
    _move_to_bank(game, player, {give_resource: ratio}); _take_from_bank(game, player, {receive_resource: 1})
    _finish(game, player_id, request_id, fingerprint)
    _record_activity(game, player_id, "bank_trade",
                     f"銀行に{RESOURCE_NAMES[give_resource]}{ratio}枚を渡し、{RESOURCE_NAMES[receive_resource]}1枚を受け取りました。",
                     bank={"give": {give_resource: ratio}, "want": {receive_resource: 1}})


def _normalize_offers(offers: list[dict]) -> list[dict]:
    if not isinstance(offers, list) or not 1 <= len(offers) <= 3: raise GameActionError("交渉候補は1〜3個にしてください。")
    normalized = []
    for offer in offers:
        if not isinstance(offer, dict): raise GameActionError("交渉候補の形式が不正です。")
        give, want = _resource_map(offer.get("give", {})), _resource_map(offer.get("want", {}))
        if any(give[resource] and want[resource] for resource in RESOURCES): raise GameActionError("同じ資源を、渡す側と受け取る側の両方には選べません。")
        normalized.append({"give": give, "want": want})
    return normalized


def _trade_event(trade: dict, outcome: str, selected_offer_index: int | None = None) -> dict:
    """卓上で提示された候補と返答を、全画面の演出用に公開する。手札全体は含めない。"""
    event = {"proposer_id": trade["proposer_id"], "recipient_id": trade["recipient_id"],
             "offers": [{"give": dict(offer["give"]), "want": dict(offer["want"])}
                        for offer in trade["offers"]],
             "is_counter": trade["is_counter"], "outcome": outcome}
    if selected_offer_index is not None: event["selected_offer_index"] = selected_offer_index
    return event


def propose_trade(game: GameState, player_id: int, target_id: int, offers: list[dict], expected_revision: int, request_id: str) -> None:
    normalized = _normalize_offers(offers); fingerprint = ("trade", target_id, tuple((tuple(o["give"].items()), tuple(o["want"].items())) for o in normalized), expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    if game.phase != "action" or game.current_player_id != player_id or game.pending_trade: raise GameActionError("今は新しい交渉を提案できません。")
    if game.free_road_remaining: raise GameActionError("街道建設で残っている無料の道を先に配置してください。")
    if game.trade_count >= 5: raise GameActionError("この手番の交渉は5回までです。")
    if target_id == player_id: raise GameActionError("自分自身とは交渉できません。")
    proposer = player_for(game, player_id); player_for(game, target_id)
    if any(not _can_afford(proposer, offer["give"]) for offer in normalized): raise GameActionError("自分が持っていない資源を渡す提案にはできません。")
    game.trade_count += 1; game.pending_trade = {"proposer_id": player_id, "recipient_id": target_id, "offers": normalized, "is_counter": False, "awaiting_counter": False}; game.phase = "trade_response"
    _finish(game, player_id, request_id, fingerprint)
    _record_activity(game, player_id, "trade", f"プレイヤー {target_id} に交渉を提案しました。",
                     trade=_trade_event(game.pending_trade, "proposed"))


def respond_to_trade(game: GameState, player_id: int, decision: str, offer_index: int | None, expected_revision: int, request_id: str) -> None:
    fingerprint = ("trade_response", decision, offer_index, expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    trade = game.pending_trade
    if game.phase != "trade_response" or not trade or player_id != trade["recipient_id"]: raise GameActionError("あなたは現在、この交渉に応答できません。")
    if decision == "accept":
        if type(offer_index) is not int or offer_index not in range(len(trade["offers"])): raise GameActionError("受け入れる候補を選んでください。")
        offer = trade["offers"][offer_index]; sender, recipient = player_for(game, trade["proposer_id"]), player_for(game, trade["recipient_id"])
        if not _can_afford(sender, offer["give"]): raise GameActionError("提案者の資源が変化したため、この候補は成立しません。")
        if not _can_afford(recipient, offer["want"]): raise GameActionError("要求されている資源カードが不足しています。")
        for resource in RESOURCES:
            sender.resources[resource] -= offer["give"][resource]; recipient.resources[resource] += offer["give"][resource]
            recipient.resources[resource] -= offer["want"][resource]; sender.resources[resource] += offer["want"][resource]
        trade_result = (f"プレイヤー {sender.id} とプレイヤー {recipient.id} の交渉が成立："
                        f"プレイヤー {sender.id} が{_resource_summary(offer['give'])}、"
                        f"プレイヤー {recipient.id} が{_resource_summary(offer['want'])}を渡しました。")
        game.pending_trade = None; game.phase = "action"
    elif decision == "counter":
        if trade["is_counter"]: raise GameActionError("逆交渉にさらに逆交渉はできません。")
        trade["awaiting_counter"] = True; game.phase = "trade_counter_offer"
    elif decision == "decline": game.pending_trade = None; game.phase = "action"
    else: raise GameActionError("交渉への応答が不正です。")
    _finish(game, player_id, request_id, fingerprint)
    if decision == "accept":
        _record_activity(game, player_id, "trade_accepted", trade_result,
                         trade=_trade_event(trade, "accepted", offer_index))
    elif decision == "counter":
        _record_activity(game, player_id, "trade_counter", "逆交渉を選びました。",
                         trade=_trade_event(trade, "counter_requested"))
    else:
        _record_activity(game, player_id, "trade_declined", f"プレイヤー {trade['proposer_id']} との交渉を断りました。",
                         trade=_trade_event(trade, "declined"))


def propose_counter_trade(game: GameState, player_id: int, offers: list[dict], expected_revision: int, request_id: str) -> None:
    normalized = _normalize_offers(offers); fingerprint = ("counter_trade", tuple((tuple(o["give"].items()), tuple(o["want"].items())) for o in normalized), expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    trade = game.pending_trade
    if game.phase != "trade_counter_offer" or not trade or player_id != trade["recipient_id"]: raise GameActionError("あなたは現在、逆交渉を提案できません。")
    sender = player_for(game, player_id)
    if any(not _can_afford(sender, offer["give"]) for offer in normalized): raise GameActionError("自分が持っていない資源を渡す提案にはできません。")
    game.pending_trade = {"proposer_id": player_id, "recipient_id": trade["proposer_id"], "offers": normalized, "is_counter": True, "awaiting_counter": False}; game.phase = "trade_response"
    _finish(game, player_id, request_id, fingerprint)
    _record_activity(game, player_id, "trade", f"プレイヤー {trade['proposer_id']} に逆交渉を提案しました。",
                     trade=_trade_event(game.pending_trade, "proposed"))


def use_development(game: GameState, player_id: int, card: str, expected_revision: int, request_id: str, *, resources: list[str] | None = None) -> None:
    fingerprint = ("development", card, tuple(resources or ()), expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    if game.phase not in {"turn_pre_roll", "action"} or game.current_player_id != player_id: raise GameActionError("発展カードは自分の通常手番だけに使えます。")
    player = player_for(game, player_id)
    if player.development_cards.count(card) <= player.new_development_cards.count(card): raise GameActionError("その発展カードは持っていないか、今買ったばかりで使えません。")
    if card == "victory_point": raise GameActionError("勝利点カードは使わず、勝利時に公開されます。")
    if game.used_development_this_turn: raise GameActionError("発展カードは1手番に1枚までです。")
    if game.free_road_remaining: raise GameActionError("街道建設で残っている無料の道を先に配置してください。")
    # 失敗時にカードだけ消えないよう、選択内容と銀行在庫を先に検証する。
    if card == "year_of_plenty":
        if not isinstance(resources, list) or len(resources) != 2 or any(resource not in RESOURCES for resource in resources): raise GameActionError("豊作では受け取る資源を2枚選んでください。")
        requested = Counter(resources)
        if any(game.bank[resource] < count for resource, count in requested.items()): raise GameActionError("銀行の資源が不足しています。")
    elif card == "monopoly":
        if not isinstance(resources, list) or len(resources) != 1 or resources[0] not in RESOURCES: raise GameActionError("独占では資源を1種類選んでください。")
    player.development_cards.remove(card); game.used_development_this_turn = True
    if card == "knight":
        player.played_knights += 1; highest = max(p.played_knights for p in game.players); leaders = [p.id for p in game.players if p.played_knights == highest]
        if highest >= 3 and (game.largest_army_holder_id in leaders or len(leaders) == 1): game.largest_army_holder_id = game.largest_army_holder_id if game.largest_army_holder_id in leaders else leaders[0]
        game.robber_reason = "knight"; game.phase = "robber_move"
    elif card == "road_building":
        game.phase = "action"
        game.free_road_remaining = min(2, 15 - _road_count(game, player_id)) if legal_road_ids(game, player_id, initial=False) else 0
        if not game.free_road_remaining and game.last_roll is None: game.phase = "turn_pre_roll"
    elif card == "year_of_plenty":
        _take_from_bank(game, player, dict(Counter(resources)))
    elif card == "monopoly":
        resource = resources[0]
        for other in game.players:
            if other.id != player_id: player.resources[resource] += other.resources[resource]; other.resources[resource] = 0
    else: raise GameActionError("発展カードの種類が不正です。")
    player.used_development_cards.append(card)
    _check_winner(game, player_id); _finish(game, player_id, request_id, fingerprint)
    _record_activity(game, player_id, "development", f"{DEVELOPMENT_NAMES[card]}を使いました。", card=card)


def end_turn(game: GameState, player_id: int, expected_revision: int, request_id: str) -> None:
    fingerprint = ("end_turn", expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    if game.phase != "action" or game.current_player_id != player_id or game.pending_trade: raise GameActionError("行動フェーズの手番プレイヤーだけが手番を終了できます。")
    if game.last_roll is None: raise GameActionError("手番を終える前に、通常手番のサイコロを振ってください。")
    if game.free_road_remaining: raise GameActionError("街道建設で残っている無料の道を先に配置してください。")
    index = game.seat_order.index(player_id); game.current_player_id = game.seat_order[(index + 1) % len(game.seat_order)]
    game.turn_number += 1; game.phase = "turn_pre_roll"; game.last_roll = None; game.trade_count = 0; game.used_development_this_turn = False; game.free_road_remaining = 0
    for player in game.players: player.new_development_cards.clear()
    _finish(game, player_id, request_id, fingerprint)
    _record_activity(game, player_id, "turn_end", "手番を終了しました。")


def set_player_controller(game: GameState, player_id: int, is_ai: bool,
                          expected_revision: int, request_id: str) -> None:
    """本人用トークンを持つ画面から、操作担当を人間・AIで切り替える。"""
    fingerprint = ("set_controller", is_ai, expected_revision)
    if _request_is_repeat(game, player_id, request_id, fingerprint, expected_revision): return
    if type(is_ai) is not bool: raise GameActionError("操作担当の指定が不正です。")
    player_for(game, player_id).is_ai = is_ai
    _finish(game, player_id, request_id, fingerprint)


def _private_trade_view(game: GameState, viewer_id: int | None) -> dict | None:
    trade = game.pending_trade
    if not trade: return None
    public = {"active": True, "proposer_id": trade["proposer_id"], "recipient_id": trade["recipient_id"], "is_counter": trade["is_counter"], "awaiting_counter": trade["awaiting_counter"]}
    if viewer_id in {trade["proposer_id"], trade["recipient_id"]}: return {**public, "offers": [{"give": dict(o["give"]), "want": dict(o["want"])} for o in trade["offers"]]}
    return public


def pending_ai_player_id(game: GameState) -> int | None:
    """現在1操作できるAI。観戦モードと通常の自動進行で共通利用する。"""
    def is_ai(player_id: int | None) -> bool:
        return player_id is not None and player_for(game, player_id).is_ai
    if game.phase == "rolling_order":
        player_id = game.pending_rollers[0] if game.pending_rollers else None
    elif game.phase == "setup_ready":
        player_id = setup_player_id(game)
    elif game.phase == "ready_to_play":
        player_id = game.seat_order[0]
    elif game.phase == "discarding":
        player_id = next((pid for pid in game.pending_discards if is_ai(pid)), None)
    elif game.phase in {"trade_response", "trade_counter_offer"}:
        player_id = game.pending_trade["recipient_id"] if game.pending_trade else None
    elif game.phase in {"turn_pre_roll", "robber_move", "robber_steal", "action"}:
        player_id = game.current_player_id
    else:
        player_id = None
    return player_id if is_ai(player_id) else None


def pending_player_actions(game: GameState) -> dict[int, str]:
    """全員に公開する操作待ち。手番以外の返答・破棄も含める。"""
    if game.phase == "discarding":
        return {pid: "資源を捨てる" for pid in game.pending_discards}
    if game.phase in {"trade_response", "trade_counter_offer"}:
        trade = game.pending_trade
        return {trade["recipient_id"]: "逆提案を作る" if game.phase == "trade_counter_offer" else "交渉に返答"} if trade else {}
    if game.phase == "rolling_order":
        return {game.pending_rollers[0]: "順番決めのサイコロ"} if game.pending_rollers else {}
    if game.phase == "setup_ready":
        pid = setup_player_id(game)
        return {pid: "道を置く" if game.pending_settlement_id is not None else "開拓地を置く"} if pid is not None else {}
    if game.phase == "ready_to_play":
        return {game.seat_order[0]: "ゲームを開始"}
    reason = {"turn_pre_roll": "サイコロ・発展カード", "robber_move": "盗賊を移動", "robber_steal": "カードを奪う", "action": "手番の操作"}.get(game.phase)
    return {game.current_player_id: reason} if reason and game.current_player_id is not None else {}


def game_view(game: GameState, viewer_id: int | None = None) -> dict:
    """観戦では手札内訳を公開するが、未公開の勝利点込み得点は本人だけに返す。"""
    players = []
    waiting_actions = pending_player_actions(game)
    reveal_all_hands = game.ai_watch_mode
    for player in game.players:
        revealed_victory_points = (player.development_cards.count("victory_point")
                                   if game.phase == "game_over" and game.winner_id == player.id else 0)
        public_points = (player.settlements + 2 * player.cities
                         + 2 * (game.longest_road_holder_id == player.id)
                         + 2 * (game.largest_army_holder_id == player.id)
                         + revealed_victory_points)
        view = {"id": player.id, "name": player.name, "color": player.color, "is_ai": player.is_ai, "resource_count": sum(player.resources.values()), "development_count": len(player.development_cards), "public_score": public_points, "played_knights": player.played_knights, "settlement_count": player.settlements, "city_count": player.cities, "road_count": _road_count(game, player.id), "has_longest_road": game.longest_road_holder_id == player.id, "has_largest_army": game.largest_army_holder_id == player.id}
        if player.id == viewer_id or reveal_all_hands:
            view.update(resources=dict(player.resources), development_cards=list(player.development_cards),
                        new_development_cards=list(player.new_development_cards))
        if player.id == viewer_id:
            hidden_victory_points = player.development_cards.count("victory_point") - revealed_victory_points
            view["score"] = public_points + hidden_victory_points
        view["used_development_cards"] = list(player.used_development_cards)
        view["pending_action"] = waiting_actions.get(player.id)
        players.append(view)
    current_can_see = viewer_id == game.current_player_id
    action_legal = {"road_ids": legal_road_ids(game, viewer_id, initial=False), "settlement_ids": legal_settlement_ids(game, viewer_id, initial=False), "city_ids": legal_city_ids(game, viewer_id)} if current_can_see and game.phase == "action" else {"road_ids": [], "settlement_ids": [], "city_ids": []}
    return {"id": game.id, "seed": game.seed, "phase": game.phase, "revision": game.revision, "board": asdict(game.board), "rules": asdict(game.rules), "players": players, "viewer_id": viewer_id, "seat_order": list(game.seat_order), "setup_order": list(game.setup_order), "round_number": game.round_number, "ranking_groups": [list(group) for group in game.ranking_groups], "next_roller_id": game.pending_rollers[0] if game.pending_rollers else None, "rolls": [dict(roll, dice=list(roll["dice"])) for roll in game.rolls], "longest_road_holder_id": game.longest_road_holder_id, "largest_army_holder_id": game.largest_army_holder_id, "settlements": [{"vertex_id": vertex, "player_id": owner} for vertex, owner in sorted(game.settlements.items())], "cities": [{"vertex_id": vertex, "player_id": owner} for vertex, owner in sorted(game.cities.items())], "roads": [{"edge_id": edge, "player_id": owner} for edge, owner in sorted(game.roads.items())], "bank": dict(game.bank), "robber_tile_id": game.robber_tile_id, "first_player_id": game.seat_order[0] if game.phase == "ready_to_play" else None, "current_player_id": game.current_player_id, "turn_number": game.turn_number, "last_roll": game.last_roll, "dice_history": [dict(roll, dice=list(roll["dice"])) for roll in game.dice_history], "latest_dice_event": game.latest_dice_event, "activity_log": [dict(entry) for entry in game.activity_log], "winner_id": game.winner_id, "trade_count": game.trade_count, "trade_limit": 5, "pending_trade": _private_trade_view(game, viewer_id), "pending_discard_count": len(game.pending_discards), "your_discard_count": sum(player_for(game, viewer_id).resources.values()) // 2 if viewer_id in game.pending_discards else 0, "robber": {"legal_tile_ids": [tile.id for tile in game.board.tiles if tile.id != game.robber_tile_id] if viewer_id == game.current_player_id and game.phase == "robber_move" else [], "victim_ids": list(game.robber_victim_ids) if viewer_id == game.current_player_id and game.phase == "robber_steal" else []}, "action": {"legal": action_legal, "free_road_remaining": game.free_road_remaining, "development_remaining": len(game.development_deck), "used_development_this_turn": game.used_development_this_turn, "port_ratios": {resource: _port_ratio(game, viewer_id, resource) for resource in RESOURCES} if current_can_see else {}}, "ai": {"player_ids": [player.id for player in game.players if player.is_ai], "actions": game.ai_action_number, "recent_log": list(game.ai_log[-30:]), "watch_mode": game.ai_watch_mode, "actor_id": pending_ai_player_id(game), "can_step": pending_ai_player_id(game) is not None}, "initial_placement": {"completed_pairs": game.setup_index, "total_pairs": 8, "player_id": setup_player_id(game), "piece": ("road" if game.pending_settlement_id is not None else "settlement") if setup_player_id(game) is not None else None, "pending_vertex_id": game.pending_settlement_id, "legal_vertex_ids": legal_settlement_ids(game) if viewer_id == setup_player_id(game) else [], "legal_edge_ids": legal_road_ids(game) if viewer_id == setup_player_id(game) else []}}
