"""既存のルールベース判断を、状態を変更しない Agent として実装する。"""

from collections import Counter
from random import Random

from app.domain.actions import (Action, BankTradeAction, BuildCityAction, BuildRoadAction,
                                BuildSettlementAction, BuyDevelopmentAction, DiscardResourcesAction,
                                EndTurnAction, MoveRobberAction, PlaceInitialRoadAction,
                                PlaceInitialSettlementAction, ProposeCounterTradeAction,
                                ProposeTradeAction, RespondToTradeAction, RollOrderAction,
                                RollTurnDiceAction, StartGameAction, StealResourceAction, TradeOffer,
                                UseDevelopmentAction)
from app.domain.game import (BUILD_COSTS, RESOURCES, GameActionError, GameState, _can_afford,
                             _port_ratio, legal_city_ids, legal_road_ids, legal_settlement_ids,
                             player_for, setup_player_id)


def _rng(game: GameState) -> Random:
    """判断の同点解消用。ゲーム状態を変えず、同じ状態なら同じ選択を返す。"""
    return Random((game.seed << 20) ^ game.ai_action_number ^ (game.revision << 5))


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


def _best_initial_road(game: GameState, candidates: list[int], settlement_id: int, player_id: int) -> int:
    rng = _rng(game)

    def value(edge_id: int) -> tuple[int, float]:
        a, b = game.board.edges[edge_id].vertex_ids
        far = b if a == settlement_id else a
        onward = [other for other in game.board.vertices[far].edge_ids if other != edge_id]
        reachable = [vertex for other in onward for vertex in game.board.edges[other].vertex_ids if vertex != far]
        return max((_vertex_value(game, vertex, player_id) for vertex in reachable), default=0), rng.random()

    return max(candidates, key=value)


def _missing_cost(player, cost: dict[str, int]) -> list[str]:
    return [resource for resource in RESOURCES if player.resources[resource] < cost.get(resource, 0)]


def _goal_cost(game: GameState, player_id: int) -> dict[str, int]:
    if legal_city_ids(game, player_id) and player_for(game, player_id).cities < 4:
        return BUILD_COSTS["city"]
    if legal_settlement_ids(game, player_id, initial=False):
        return BUILD_COSTS["settlement"]
    return BUILD_COSTS["road"]


def _try_bank_trade(game: GameState, player_id: int) -> BankTradeAction | None:
    player = player_for(game, player_id)
    goal = _goal_cost(game, player_id)
    for wanted in _missing_cost(player, goal):
        if game.bank[wanted] < 1:
            continue
        for offered in RESOURCES:
            ratio = _port_ratio(game, player_id, offered)
            reserve = goal.get(offered, 0)
            if offered != wanted and player.resources[offered] - reserve >= ratio:
                return BankTradeAction(offered, wanted)
    return None


def _make_trade_offers(game: GameState, player_id: int) -> tuple[TradeOffer, ...]:
    player = player_for(game, player_id)
    goal = _goal_cost(game, player_id)
    wanted = _missing_cost(player, goal)
    give = [resource for resource in RESOURCES if player.resources[resource] > goal.get(resource, 0)]
    offers: list[TradeOffer] = []
    for index, resource in enumerate(wanted[:3]):
        available = [candidate for candidate in give if candidate != resource]
        if not available:
            break
        offered = available[index % len(available)]
        offers.append(TradeOffer({offered: 1}, {resource: 1}))
    return tuple(offers)


def _respond_trade(game: GameState, player_id: int) -> RespondToTradeAction:
    trade = game.pending_trade
    if trade is None:
        raise GameActionError("AIが応答する交渉がありません。")
    player = player_for(game, player_id)
    for index, offer in enumerate(trade["offers"]):
        if _can_afford(player, offer["want"]) and sum(offer["give"].values()) >= sum(offer["want"].values()):
            return RespondToTradeAction("accept", index)
    if not trade["is_counter"] and sum(player.resources.values()) > 0 and _rng(game).random() < .45:
        return RespondToTradeAction("counter")
    return RespondToTradeAction("decline")


def _counter_offer(game: GameState, player_id: int) -> ProposeCounterTradeAction:
    player = player_for(game, player_id)
    give = [resource for resource in RESOURCES if player.resources[resource] > 0]
    if not give:
        raise GameActionError("逆交渉に出せる資源がありません。")
    offered = _rng(game).choice(give)
    wanted = _rng(game).choice([resource for resource in RESOURCES if resource != offered])
    return ProposeCounterTradeAction((TradeOffer({offered: 1}, {wanted: 1}),))


def _choose_robber_tile(game: GameState, player_id: int) -> int:
    rng = _rng(game)

    def score(tile_id: int) -> tuple[int, float]:
        value = 0
        for vertex in game.board.vertices:
            if tile_id not in vertex.hex_ids:
                continue
            owner = game.settlements.get(vertex.id, game.cities.get(vertex.id))
            if owner == player_id:
                value -= 4
            elif owner:
                value += 2 if vertex.id in game.cities else 1
        return value, rng.random()

    return max((tile.id for tile in game.board.tiles if tile.id != game.robber_tile_id), key=score)


class HeuristicAgent:
    """既存の自動対戦と同じ優先順位で、1手分の Action だけを選ぶ。"""

    def select_action(self, game: GameState, player_id: int) -> Action:
        rng = _rng(game)
        phase = game.phase
        if phase == "rolling_order":
            return RollOrderAction((rng.randint(1, 6), rng.randint(1, 6)))
        if phase == "setup_ready":
            if game.pending_settlement_id is None:
                return PlaceInitialSettlementAction(_best_vertex(game, legal_settlement_ids(game), player_id))
            return PlaceInitialRoadAction(_best_initial_road(game, legal_road_ids(game), game.pending_settlement_id, player_id))
        if phase == "ready_to_play":
            return StartGameAction()
        if phase == "turn_pre_roll":
            player = player_for(game, player_id)
            if (not game.used_development_this_turn
                    and player.development_cards.count("knight") > player.new_development_cards.count("knight")
                    and rng.random() < .1):
                return UseDevelopmentAction("knight")
            return RollTurnDiceAction((rng.randint(1, 6), rng.randint(1, 6)))
        if phase == "discarding":
            player = player_for(game, player_id)
            need = sum(player.resources.values()) // 2
            pool = [resource for resource in RESOURCES for _ in range(player.resources[resource])]
            rng.shuffle(pool)
            cards = Counter(pool[:need])
            return DiscardResourcesAction(dict(cards))
        if phase == "robber_move":
            return MoveRobberAction(_choose_robber_tile(game, player_id))
        if phase == "robber_steal":
            victim = max(game.robber_victim_ids, key=lambda pid: sum(player_for(game, pid).resources.values()))
            return StealResourceAction(victim)
        if phase == "trade_response":
            return _respond_trade(game, player_id)
        if phase == "trade_counter_offer":
            return _counter_offer(game, player_id)
        if phase != "action":
            raise GameActionError(f"AIが未対応のフェーズです: {phase}")

        player = player_for(game, player_id)
        roads = legal_road_ids(game, player_id, initial=False)
        settlements = legal_settlement_ids(game, player_id, initial=False)
        cities = legal_city_ids(game, player_id)
        if game.free_road_remaining and roads and sum(owner == player_id for owner in game.roads.values()) < 15:
            return BuildRoadAction(rng.choice(roads))
        if cities and player.cities < 4 and _can_afford(player, BUILD_COSTS["city"]):
            return BuildCityAction(rng.choice(cities))
        if settlements and player.settlements < 5 and _can_afford(player, BUILD_COSTS["settlement"]):
            return BuildSettlementAction(_best_vertex(game, settlements, player_id))
        if roads and sum(owner == player_id for owner in game.roads.values()) < 15 and _can_afford(player, BUILD_COSTS["road"]):
            return BuildRoadAction(rng.choice(roads))
        if game.development_deck and _can_afford(player, BUILD_COSTS["development"]) and rng.random() < .45:
            return BuyDevelopmentAction()
        if action := _try_bank_trade(game, player_id):
            return action
        if game.trade_count < 2 and (offers := _make_trade_offers(game, player_id)):
            target = rng.choice([other.id for other in game.players if other.id != player_id])
            return ProposeTradeAction(target, offers)
        return EndTurnAction()
