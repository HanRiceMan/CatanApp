"""プレイヤーから見える情報だけを表す、版管理付き強化学習Observation。"""

from collections import Counter
from dataclasses import dataclass
from typing import Final

import numpy as np

from app.domain.game import DEVELOPMENT_DECK, RESOURCES, GameState, _road_count, _port_ratio, player_for

OBSERVATION_VERSION: Final[str] = "v1"
OBSERVATION_VECTOR_SIZE: Final[int] = 1343
OBSERVATION_VECTOR_SIZES: Final[dict[str, int]] = {
    "v1": 1343, "v2": 1667, "v3": 1822, "v4": 1718, "v5": 1722,
    "v6": 1747,
}
NUMBER_PIPS: Final[dict[int, int]] = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
                                      8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
MAX_VERTEX_PIPS: Final[int] = 15
RESOURCE_ORDER: Final[tuple[str, ...]] = RESOURCES
DEVELOPMENT_CARD_ORDER: Final[tuple[str, ...]] = (
    "knight", "road_building", "year_of_plenty", "monopoly", "victory_point",
)
TERRAIN_ORDER: Final[tuple[str, ...]] = ("wood", "brick", "sheep", "wheat", "ore", "desert")
NUMBER_ORDER: Final[tuple[int, ...]] = (0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
PHASE_ORDER: Final[tuple[str, ...]] = (
    "rolling_order", "setup_ready", "ready_to_play", "turn_pre_roll", "discarding",
    "robber_move", "robber_steal", "trade_response", "trade_counter_offer", "action", "game_over",
)
DEVELOPMENT_CARD_MAX: Final[dict[str, int]] = dict(Counter(DEVELOPMENT_DECK))
MAX_PLAYER_RESOURCES: Final[int] = 19 * len(RESOURCES)
MAX_TURNS_FOR_NORMALIZATION: Final[int] = 200


@dataclass(frozen=True)
class ObservedPlayer:
    """自分は内訳、相手は公開済み集計だけを持つ。"""

    id: int
    seat_index: int
    is_self: bool
    resource_count: int
    development_count: int
    public_score: int
    actual_score: int | None
    resources: tuple[int, ...] | None
    development_cards: tuple[str, ...] | None
    new_development_cards: tuple[str, ...] | None
    used_development_cards: tuple[str, ...]
    settlement_count: int
    city_count: int
    road_count: int
    played_knights: int
    has_longest_road: bool
    has_largest_army: bool


@dataclass(frozen=True)
class TileObservation:
    id: int
    terrain: str
    number: int | None
    has_robber: bool


@dataclass(frozen=True)
class VertexObservation:
    id: int
    owner_id: int | None
    building: str  # empty / settlement / city
    production_pips: tuple[int, ...] | None = None  # v2: RESOURCE_ORDER順の公開盤面情報


@dataclass(frozen=True)
class EdgeObservation:
    id: int
    owner_id: int | None


@dataclass(frozen=True)
class PortObservation:
    id: int
    edge_id: int
    resource: str | None
    ratio: int
    is_owned_by_self: bool


@dataclass(frozen=True)
class LastRollObservation:
    player_id: int
    dice: tuple[int, int]
    total: int


@dataclass(frozen=True)
class PendingTradeObservation:
    proposer_id: int
    recipient_id: int
    is_counter: bool
    awaiting_counter: bool


@dataclass(frozen=True)
class Observation:
    """完全なGameStateへの参照を一切持たない、immutableな観測値。"""

    version: str
    player_id: int
    players: tuple[ObservedPlayer, ...]
    tiles: tuple[TileObservation, ...]
    vertices: tuple[VertexObservation, ...]
    edges: tuple[EdgeObservation, ...]
    ports: tuple[PortObservation, ...]
    phase: str
    current_player_id: int | None
    seat_order: tuple[int, ...]
    turn_number: int
    round_number: int
    setup_index: int
    first_player_id: int | None
    last_roll: LastRollObservation | None
    pending_discards: tuple[int, ...]
    robber_victim_ids: tuple[int, ...]
    pending_trade: PendingTradeObservation | None
    bank_resources: tuple[int, ...]
    development_deck_count: int
    free_road_remaining: int
    used_development_this_turn: bool
    trade_count: int
    winner_id: int | None
    strategic_goal: tuple[float, ...] | None = None
    opponent_hand_beliefs: tuple[float, ...] | None = None
    development_context: tuple[float, ...] | None = None


def _public_score(game: GameState, player_id: int) -> int:
    player = player_for(game, player_id)
    revealed_victory_points = (player.development_cards.count("victory_point")
                               if game.phase == "game_over" and game.winner_id == player_id else 0)
    return (player.settlements + 2 * player.cities
            + 2 * (game.longest_road_holder_id == player_id)
            + 2 * (game.largest_army_holder_id == player_id)
            + revealed_victory_points)


def _actual_score(game: GameState, player_id: int) -> int:
    player = player_for(game, player_id)
    # 終局時は公開得点にも勝利点カードが含まれるため、そこへ再加算しない。
    return (player.settlements + 2 * player.cities
            + 2 * (game.longest_road_holder_id == player_id)
            + 2 * (game.largest_army_holder_id == player_id)
            + player.development_cards.count("victory_point"))


def _port_is_owned_by(game: GameState, edge_id: int, player_id: int) -> bool:
    return any((game.settlements.get(vertex_id) == player_id or game.cities.get(vertex_id) == player_id)
               for vertex_id in game.board.edges[edge_id].vertex_ids)


def _vertex_production_pips(game: GameState, vertex_id: int) -> tuple[int, ...]:
    values = dict.fromkeys(RESOURCE_ORDER, 0)
    for tile_id in game.board.vertices[vertex_id].hex_ids:
        tile = game.board.tiles[tile_id]
        if tile.terrain in values:
            values[tile.terrain] += NUMBER_PIPS.get(tile.number, 0)
    return tuple(values[resource] for resource in RESOURCE_ORDER)


def get_observation(game: GameState, player_id: int, *, version: str = OBSERVATION_VERSION) -> Observation:
    """``player_id`` が実際に知り得る情報だけを切り出す。

    相手の資源・未使用発展カードの内訳と、非公開勝利点を渡さない。
    銀行・盤面・使用済み発展カードは全員が参照できる公開情報として扱う。
    """
    if version not in OBSERVATION_VECTOR_SIZES:
        raise ValueError(f"未対応のObservation版です: {version}")
    player_for(game, player_id)  # IDを既存ルールと同じ例外で検証する。
    seat_index = {pid: index for index, pid in enumerate(game.seat_order)}
    observed_players = []
    for player in sorted(game.players, key=lambda item: item.id):
        is_self = player.id == player_id
        observed_players.append(ObservedPlayer(
            id=player.id,
            seat_index=seat_index.get(player.id, player.id - 1),
            is_self=is_self,
            resource_count=sum(player.resources.values()),
            development_count=len(player.development_cards),
            public_score=_public_score(game, player.id),
            actual_score=_actual_score(game, player.id) if is_self else None,
            resources=tuple(player.resources[resource] for resource in RESOURCE_ORDER) if is_self else None,
            development_cards=tuple(player.development_cards) if is_self else None,
            new_development_cards=tuple(player.new_development_cards) if is_self else None,
            used_development_cards=tuple(player.used_development_cards),
            settlement_count=player.settlements,
            city_count=player.cities,
            road_count=_road_count(game, player.id),
            played_knights=player.played_knights,
            has_longest_road=game.longest_road_holder_id == player.id,
            has_largest_army=game.largest_army_holder_id == player.id,
        ))
    pending_trade = None
    if game.pending_trade is not None:
        pending_trade = PendingTradeObservation(
            proposer_id=game.pending_trade["proposer_id"],
            recipient_id=game.pending_trade["recipient_id"],
            is_counter=game.pending_trade["is_counter"],
            awaiting_counter=game.pending_trade["awaiting_counter"],
        )
    strategic_goal = None
    if version == "v3":
        from .goal_context import encode_goal_context
        strategic_goal = tuple(encode_goal_context(game, player_id))
    opponent_hand_beliefs = None
    if version in {"v4", "v5", "v6"}:
        from .belief_context import encode_belief_context
        opponent_hand_beliefs = tuple(encode_belief_context(game, player_id))
    development_context = None
    if version == "v6":
        from .development_context import encode_development_context
        development_context = tuple(encode_development_context(game, player_id))
    return Observation(
        version=version,
        player_id=player_id,
        players=tuple(observed_players),
        tiles=tuple(TileObservation(tile.id, tile.terrain, tile.number, tile.id == game.robber_tile_id)
                    for tile in game.board.tiles),
        vertices=tuple(VertexObservation(vertex.id, game.settlements.get(vertex.id), "settlement",
                                         _vertex_production_pips(game, vertex.id) if version in {"v2", "v3", "v4", "v5", "v6"} else None)
                       if vertex.id in game.settlements
                       else VertexObservation(vertex.id, game.cities.get(vertex.id), "city",
                                              _vertex_production_pips(game, vertex.id) if version in {"v2", "v3", "v4", "v5", "v6"} else None)
                       if vertex.id in game.cities
                       else VertexObservation(vertex.id, None, "empty",
                                              _vertex_production_pips(game, vertex.id) if version in {"v2", "v3", "v4", "v5", "v6"} else None)
                       for vertex in game.board.vertices),
        edges=tuple(EdgeObservation(edge.id, game.roads.get(edge.id)) for edge in game.board.edges),
        ports=tuple(PortObservation(port.id, port.edge_id, port.resource, port.ratio,
                                    _port_is_owned_by(game, port.edge_id, player_id))
                    for port in game.board.ports),
        phase=game.phase,
        current_player_id=game.current_player_id,
        seat_order=tuple(game.seat_order),
        turn_number=game.turn_number,
        round_number=game.round_number,
        setup_index=game.setup_index,
        first_player_id=game.seat_order[0] if game.seat_order else None,
        last_roll=(LastRollObservation(game.last_roll["player_id"], tuple(game.last_roll["dice"]), game.last_roll["total"])
                   if game.last_roll is not None else None),
        pending_discards=tuple(game.pending_discards),
        robber_victim_ids=tuple(game.robber_victim_ids),
        pending_trade=pending_trade,
        bank_resources=tuple(game.bank[resource] for resource in RESOURCE_ORDER),
        development_deck_count=len(game.development_deck),
        free_road_remaining=game.free_road_remaining,
        used_development_this_turn=game.used_development_this_turn,
        trade_count=game.trade_count,
        winner_id=game.winner_id,
        strategic_goal=strategic_goal,
        opponent_hand_beliefs=opponent_hand_beliefs,
        development_context=development_context,
    )


def _one_hot(size: int, index: int | None) -> list[float]:
    result = [0.0] * size
    if index is not None and 0 <= index < size:
        result[index] = 1.0
    return result


def _relative_player_index(observation: Observation, player_id: int | None) -> int:
    """0=なし、1=自分、2〜4=時計回りの相対座席。"""
    if player_id is None or player_id not in observation.seat_order:
        return 0
    self_index = observation.seat_order.index(observation.player_id)
    target_index = observation.seat_order.index(player_id)
    return ((target_index - self_index) % len(observation.seat_order)) + 1


def _normalized(value: int, maximum: int) -> float:
    return min(max(value, 0), maximum) / maximum


def _card_counts(cards: tuple[str, ...] | None) -> Counter[str]:
    return Counter(cards or ())


def encode_observation(observation: Observation) -> np.ndarray:
    """指定版のObservationを固定長 ``float32`` ベクトルに変換する。"""
    if observation.version not in OBSERVATION_VECTOR_SIZES:
        raise ValueError(f"未対応のObservation版です: {observation.version}")
    players_by_id = {player.id: player for player in observation.players}
    self_player = players_by_id[observation.player_id]
    vector: list[float] = []

    resources = self_player.resources or (0,) * len(RESOURCE_ORDER)
    vector.extend(_normalized(value, 19) for value in resources)
    development = _card_counts(self_player.development_cards)
    used_development = _card_counts(self_player.used_development_cards)
    vector.extend(_normalized(development[card], DEVELOPMENT_CARD_MAX[card]) for card in DEVELOPMENT_CARD_ORDER)
    vector.extend(_normalized(used_development[card], DEVELOPMENT_CARD_MAX[card]) for card in DEVELOPMENT_CARD_ORDER)
    vector.extend((
        _normalized(self_player.public_score, 10), _normalized(self_player.actual_score or 0, 10),
        _normalized(self_player.settlement_count, 5), _normalized(self_player.city_count, 4),
        _normalized(self_player.road_count, 15), _normalized(self_player.played_knights, 14),
        float(self_player.has_longest_road), float(self_player.has_largest_army),
    ))
    for resource in (None, *RESOURCE_ORDER):
        vector.append(float(any(port.resource == resource and port.is_owned_by_self for port in observation.ports)))

    seat_players = [players_by_id[player_id] for player_id in observation.seat_order]
    self_seat_index = observation.seat_order.index(observation.player_id)
    for offset in range(1, len(seat_players)):
        player = seat_players[(self_seat_index + offset) % len(seat_players)]
        used = _card_counts(player.used_development_cards)
        vector.extend((
            _normalized(player.resource_count, MAX_PLAYER_RESOURCES), _normalized(player.development_count, len(DEVELOPMENT_DECK)),
            _normalized(player.public_score, 10), _normalized(player.settlement_count, 5),
            _normalized(player.city_count, 4), _normalized(player.road_count, 15),
            _normalized(player.played_knights, 14), float(player.has_longest_road), float(player.has_largest_army),
        ))
        vector.extend(_normalized(used[card], DEVELOPMENT_CARD_MAX[card]) for card in DEVELOPMENT_CARD_ORDER)

    for tile in observation.tiles:
        vector.extend(_one_hot(len(TERRAIN_ORDER), TERRAIN_ORDER.index(tile.terrain)))
        number_index = NUMBER_ORDER.index(tile.number or 0)
        vector.extend(_one_hot(len(NUMBER_ORDER), number_index))
        vector.append(float(tile.has_robber))
    for vertex in observation.vertices:
        vector.extend(_one_hot(5, _relative_player_index(observation, vertex.owner_id)))
        vector.extend(_one_hot(3, ("empty", "settlement", "city").index(vertex.building)))
        if observation.version in {"v2", "v3", "v4", "v5", "v6"}:
            if vertex.production_pips is None or len(vertex.production_pips) != len(RESOURCE_ORDER):
                raise ValueError("Observation v2の交差点生産情報が不足しています。")
            vector.append(_normalized(sum(vertex.production_pips), MAX_VERTEX_PIPS))
            vector.extend(_normalized(value, MAX_VERTEX_PIPS) for value in vertex.production_pips)
    for edge in observation.edges:
        vector.extend(_one_hot(5, _relative_player_index(observation, edge.owner_id)))
    for port in observation.ports:
        resource_index = (None, *RESOURCE_ORDER).index(port.resource)
        vector.extend(_one_hot(6, resource_index))
        vector.extend((_normalized(port.ratio, 4), float(port.is_owned_by_self)))

    vector.extend(_one_hot(len(PHASE_ORDER), PHASE_ORDER.index(observation.phase)))
    vector.extend(_one_hot(5, _relative_player_index(observation, observation.current_player_id)))
    vector.extend(_one_hot(5, _relative_player_index(observation, observation.first_player_id)))
    vector.extend((
        _normalized(observation.turn_number, MAX_TURNS_FOR_NORMALIZATION),
        _normalized(observation.round_number, 10), _normalized(observation.setup_index, 8),
    ))
    if observation.last_roll is None:
        vector.extend((0.0, 0.0, 0.0, 0.0))
        vector.extend(_one_hot(5, 0))
    else:
        vector.extend((1.0, _normalized(observation.last_roll.dice[0], 6),
                       _normalized(observation.last_roll.dice[1], 6), _normalized(observation.last_roll.total, 12)))
        vector.extend(_one_hot(5, _relative_player_index(observation, observation.last_roll.player_id)))
    vector.extend(_normalized(value, 19) for value in observation.bank_resources)
    vector.extend((
        _normalized(observation.development_deck_count, len(DEVELOPMENT_DECK)),
        _normalized(observation.free_road_remaining, 2), float(observation.used_development_this_turn),
        _normalized(observation.trade_count, 5),
    ))
    for player_id in observation.seat_order:
        vector.append(float(player_id in observation.pending_discards))
    for player_id in observation.seat_order:
        vector.append(float(player_id in observation.robber_victim_ids))
    if observation.pending_trade is None:
        vector.extend(_one_hot(5, 0)); vector.extend(_one_hot(5, 0)); vector.append(0.0)
    else:
        vector.extend(_one_hot(5, _relative_player_index(observation, observation.pending_trade.proposer_id)))
        vector.extend(_one_hot(5, _relative_player_index(observation, observation.pending_trade.recipient_id)))
        vector.append(float(observation.pending_trade.is_counter))
    vector.extend(_one_hot(5, _relative_player_index(observation, observation.winner_id)))
    if observation.version == "v3":
        from .goal_context import GOAL_CONTEXT_SIZE
        if observation.strategic_goal is None or len(observation.strategic_goal) != GOAL_CONTEXT_SIZE:
            raise ValueError("Observation v3の戦略目標情報が不足しています。")
        vector.extend(observation.strategic_goal)
    if observation.version in {"v4", "v5", "v6"}:
        from .belief_context import BELIEF_CONTEXT_SIZE
        if (observation.opponent_hand_beliefs is None
                or len(observation.opponent_hand_beliefs) != BELIEF_CONTEXT_SIZE):
            raise ValueError("Observation v4の相手手札推定情報が不足しています。")
        vector.extend(observation.opponent_hand_beliefs)
        if observation.version in {"v5", "v6"}:
            vector.extend(_one_hot(4, observation.player_id - 1))
        if observation.version == "v6":
            from .development_context import DEVELOPMENT_CONTEXT_SIZE
            if (observation.development_context is None
                    or len(observation.development_context) != DEVELOPMENT_CONTEXT_SIZE):
                raise ValueError("Observation v6の発展購入情報が不足しています。")
            vector.extend(observation.development_context)

    encoded = np.asarray(vector, dtype=np.float32)
    if encoded.shape != (OBSERVATION_VECTOR_SIZES[observation.version],):
        raise AssertionError(f"Observationサイズが不正です: {encoded.shape}")
    return encoded
