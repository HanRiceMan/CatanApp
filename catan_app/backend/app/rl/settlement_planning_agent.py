"""現行PPOの上に、目標開拓地と資源予約を扱う高レベル計画層を重ねる。"""

from __future__ import annotations

import numpy as np
import torch

from app.agents.ppo import PPOAgent
from app.domain.actions import (Action, BankTradeAction, BuildCityAction,
                                BuildRoadAction, BuildSettlementAction,
                                BuyDevelopmentAction, EndTurnAction, MoveRobberAction,
                                PlaceInitialRoadAction, ProposeCounterTradeAction, ProposeTradeAction,
                                RespondToTradeAction, StealResourceAction,
                                UseDevelopmentAction)
from app.domain.game import (BUILD_COSTS, RESOURCES, GameState, _longest_road_length,
                             _port_ratio, legal_road_ids, player_for)

from .action_space import ACTION_CATALOG, action_to_id, get_action_mask, id_to_action
from .expansion_planner import (SettlementTarget, analyze_expansion_plan,
                                estimated_build_wait, settlement_target_cost,
                                player_production, vertex_production)
from .observation import encode_observation, get_observation
from .reward import actual_score
from .setup_planner import expansion_coverage_mask
from .victory_race import (best_belief_robber_tile, best_belief_robber_victim,
                           best_robber_tile, best_robber_victim, public_score)


class SettlementPlanningAgent:
    """4拠点目まで、道路・交換・開拓地を一つの予算として実行する候補Agent。"""

    def __init__(self, policy: PPOAgent, *, target_sites: int = 4,
                 max_wait_rounds: float = 6.0, target_cities: int = 0,
                 max_city_wait_rounds: float = 6.0,
                 max_plan_roads: int = 3,
                 first_city_max_deficit: int = 0,
                 second_city_max_deficit: int = 0,
                 second_city_min_score: int = 0,
                 third_city_max_deficit: int = 0,
                 third_city_min_score: int = 0,
                 post_target_connected_reserve_max_wait: float = 0.0,
                 post_target_connected_reserve_min_score: int = 0,
                 post_target_connected_reserve_max_score: int = 9,
                 expansion_aware_setup: bool = False,
                 expansion_setup_min_pips: int = 1,
                 setup_min_wheat_pips: int = 0,
                 setup_min_ore_pips: int = 0,
                 setup_best_coverage_fallback: bool = False,
                 collision_aware_initial_roads: bool = False,
                 city_growth_value_weight: float = 0.0,
                 prioritize_affordable_city: bool = False,
                 early_affordable_city_min_sites: int = 0,
                 three_site_city_value_ratio: float = 0.0,
                 prioritize_immediate_win: bool = False,
                 endgame_point_reserve_score: int = 0,
                 prioritize_immediate_titles: bool = False,
                 construction_before_titles: bool = False,
                 title_road_build_wait_threshold: float = 0.0,
                 title_horizon: int = 1,
                 title_min_score: int = 6,
                 longest_road_min_score: int = 0,
                 extended_road_horizon_score: int = 0,
                 longest_road_lookahead: bool = False,
                 buy_for_largest_army: bool = False,
                 opponent_aware_robber: bool = False,
                 belief_aware_robber: bool = False,
                 proactive_player_trade: bool = False,
                 proactive_trade_max_scarcity_deficit: int = 1,
                 strategic_surplus_trade: bool = False,
                 purposeful_paid_roads: bool = False,
                 protect_city_bank_trade_reserve: bool = False,
                 city_reservation_wait_margin: float = 1.0,
                 city_reservation_value_ratio: float = 0.9,
                 proactive_trade_opponent_score_limit: int = 8,
                 strategic_trade_response: bool = False,
                 strategic_trade_opponent_score_limit: int = 10,
                 adaptive_development_purchase: bool = False,
                 development_stall_wait_rounds: float = 2.5,
                 development_max_build_delay: float = 2.0,
                 development_tactical_score: int = 8,
                 endgame_development_fallback: bool = False,
                 defend_titles: bool = False,
                 defend_titles_min_score: int = 0):
        if (target_sites not in range(3, 6) or max_wait_rounds <= 0
                or target_cities not in range(5) or max_city_wait_rounds <= 0
                or max_plan_roads not in range(1, 6)
                or first_city_max_deficit not in range(0, 6)
                or second_city_max_deficit not in range(0, 6)
                or second_city_min_score not in range(0, 10)
                or third_city_max_deficit not in range(0, 6)
                or third_city_min_score not in range(0, 10)
                or not 0 <= post_target_connected_reserve_max_wait <= 6
                or post_target_connected_reserve_min_score not in range(0, 10)
                or post_target_connected_reserve_max_score not in range(0, 10)
                or (post_target_connected_reserve_min_score
                    > post_target_connected_reserve_max_score)
                or expansion_setup_min_pips not in range(1, 6)
                or setup_min_wheat_pips not in range(0, 6)
                or setup_min_ore_pips not in range(0, 6)
                or not 0 <= city_growth_value_weight <= 2.0
                or early_affordable_city_min_sites not in (0, 3, 4)
                or (three_site_city_value_ratio != 0
                    and not 0.5 <= three_site_city_value_ratio <= 2.0)
                or endgame_point_reserve_score not in range(0, 10)
                or not 0 <= title_road_build_wait_threshold <= 6
                or title_horizon not in range(1, 4)
                or title_min_score not in range(0, 10)
                or longest_road_min_score not in range(0, 10)
                or extended_road_horizon_score not in range(0, 10)
                or type(longest_road_lookahead) is not bool
                or type(collision_aware_initial_roads) is not bool
                or type(belief_aware_robber) is not bool
                or type(proactive_player_trade) is not bool
                or type(strategic_surplus_trade) is not bool
                or type(purposeful_paid_roads) is not bool
                or type(protect_city_bank_trade_reserve) is not bool
                or not 0 <= city_reservation_wait_margin <= 4
                or not 0.5 <= city_reservation_value_ratio <= 1.5
                or proactive_trade_max_scarcity_deficit not in range(1, 4)
                or proactive_trade_opponent_score_limit not in range(6, 10)
                or type(strategic_trade_response) is not bool
                or strategic_trade_opponent_score_limit not in range(6, 11)
                or type(adaptive_development_purchase) is not bool
                or not 1 <= development_stall_wait_rounds <= 12
                or not 0 <= development_max_build_delay <= 6
                or development_tactical_score not in range(6, 10)
                or type(endgame_development_fallback) is not bool
                or defend_titles_min_score not in range(0, 10)):
            raise ValueError("拠点・都市目標と待ち時間の指定が不正です。")
        self.policy = policy
        self.target_sites = target_sites
        self.max_wait_rounds = max_wait_rounds
        self.target_cities = target_cities
        self.max_city_wait_rounds = max_city_wait_rounds
        self.max_plan_roads = max_plan_roads
        self.first_city_max_deficit = first_city_max_deficit
        self.second_city_max_deficit = second_city_max_deficit
        self.second_city_min_score = second_city_min_score
        self.third_city_max_deficit = third_city_max_deficit
        self.third_city_min_score = third_city_min_score
        self.post_target_connected_reserve_max_wait = (
            post_target_connected_reserve_max_wait
        )
        self.post_target_connected_reserve_min_score = (
            post_target_connected_reserve_min_score
        )
        self.post_target_connected_reserve_max_score = (
            post_target_connected_reserve_max_score
        )
        self.expansion_aware_setup = expansion_aware_setup
        self.expansion_setup_min_pips = expansion_setup_min_pips
        self.setup_min_wheat_pips = setup_min_wheat_pips
        self.setup_min_ore_pips = setup_min_ore_pips
        self.setup_best_coverage_fallback = setup_best_coverage_fallback
        self.collision_aware_initial_roads = collision_aware_initial_roads
        self.city_growth_value_weight = city_growth_value_weight
        self.prioritize_affordable_city = prioritize_affordable_city
        self.early_affordable_city_min_sites = early_affordable_city_min_sites
        self.three_site_city_value_ratio = three_site_city_value_ratio
        self.prioritize_immediate_win = prioritize_immediate_win
        self.endgame_point_reserve_score = endgame_point_reserve_score
        self.prioritize_immediate_titles = prioritize_immediate_titles
        self.construction_before_titles = construction_before_titles
        self.title_road_build_wait_threshold = title_road_build_wait_threshold
        self.title_horizon = title_horizon
        self.title_min_score = title_min_score
        self.longest_road_min_score = longest_road_min_score
        self.extended_road_horizon_score = extended_road_horizon_score
        self.longest_road_lookahead = longest_road_lookahead
        self.buy_for_largest_army = buy_for_largest_army
        self.opponent_aware_robber = opponent_aware_robber
        self.belief_aware_robber = belief_aware_robber
        self.proactive_player_trade = proactive_player_trade
        self.proactive_trade_max_scarcity_deficit = (
            proactive_trade_max_scarcity_deficit
        )
        self.strategic_surplus_trade = strategic_surplus_trade
        self.purposeful_paid_roads = purposeful_paid_roads
        self.protect_city_bank_trade_reserve = protect_city_bank_trade_reserve
        self.city_reservation_wait_margin = city_reservation_wait_margin
        self.city_reservation_value_ratio = city_reservation_value_ratio
        self.proactive_trade_opponent_score_limit = (
            proactive_trade_opponent_score_limit
        )
        self.strategic_trade_response = strategic_trade_response
        self.strategic_trade_opponent_score_limit = (
            strategic_trade_opponent_score_limit
        )
        self.adaptive_development_purchase = adaptive_development_purchase
        self.development_stall_wait_rounds = development_stall_wait_rounds
        self.development_max_build_delay = development_max_build_delay
        self.development_tactical_score = development_tactical_score
        self.endgame_development_fallback = endgame_development_fallback
        self.defend_titles = defend_titles
        self.defend_titles_min_score = defend_titles_min_score

    def _probabilities(self, game: GameState, player_id: int, mask: np.ndarray) -> np.ndarray:
        observation = encode_observation(get_observation(
            game, player_id, version=self.policy.manifest.observation_version
        ))
        tensor, _ = self.policy.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.policy.model.policy.get_distribution(tensor, action_masks=mask)
        return distribution.distribution.probs.cpu().numpy().reshape(-1)

    @staticmethod
    def _best(actions: list[Action], probabilities: np.ndarray) -> Action:
        return max(actions, key=lambda action: probabilities[action_to_id(action)])

    @staticmethod
    def _resource_budget(target: SettlementTarget, game: GameState) -> dict[str, int]:
        return settlement_target_cost(target, free_roads=game.free_road_remaining)

    @staticmethod
    def _helpful_trade_for_budget(game: GameState, player_id: int,
                                  required: dict[str, int], legal: list[Action],
                                  probabilities: np.ndarray) -> Action | None:
        player = player_for(game, player_id)
        trades: list[BankTradeAction] = []
        for action in legal:
            if not isinstance(action, BankTradeAction):
                continue
            ratio = _port_ratio(game, player_id, action.give_resource)
            receive_missing = player.resources[action.receive_resource] < required[action.receive_resource]
            reserved = min(player.resources[action.give_resource], required[action.give_resource])
            gives_only_surplus = player.resources[action.give_resource] - ratio >= reserved
            if receive_missing and gives_only_surplus:
                trades.append(action)
        return (SettlementPlanningAgent._best(trades, probabilities) if trades else None)

    @staticmethod
    def _helpful_trade(game: GameState, player_id: int, target: SettlementTarget,
                       legal: list[Action], probabilities: np.ndarray) -> Action | None:
        return SettlementPlanningAgent._helpful_trade_for_budget(
            game, player_id, SettlementPlanningAgent._resource_budget(target, game),
            legal, probabilities
        )

    @staticmethod
    def _helpful_year_of_plenty_for_budget(game: GameState, player_id: int,
                                           required: dict[str, int], legal: list[Action],
                                           probabilities: np.ndarray) -> Action | None:
        player = player_for(game, player_id)
        deficits = {resource: max(0, required[resource] - player.resources[resource])
                    for resource in RESOURCES}
        candidates: list[tuple[int, UseDevelopmentAction]] = []
        for action in legal:
            if not isinstance(action, UseDevelopmentAction) or action.card != "year_of_plenty":
                continue
            received = {resource: action.resources.count(resource) for resource in RESOURCES}
            useful = sum(min(deficits[resource], received[resource]) for resource in RESOURCES)
            if useful:
                candidates.append((useful, action))
        if not candidates:
            return None
        highest = max(useful for useful, _ in candidates)
        return SettlementPlanningAgent._best(
            [action for useful, action in candidates if useful == highest], probabilities
        )

    @staticmethod
    def _helpful_year_of_plenty(game: GameState, player_id: int, target: SettlementTarget,
                                legal: list[Action], probabilities: np.ndarray) -> Action | None:
        return SettlementPlanningAgent._helpful_year_of_plenty_for_budget(
            game, player_id, SettlementPlanningAgent._resource_budget(target, game),
            legal, probabilities
        )

    @staticmethod
    def _spends_budget(game: GameState, player_id: int,
                       required: dict[str, int], action: Action) -> bool:
        costs = {
            BuildCityAction: {"wheat": 2, "ore": 3},
            BuyDevelopmentAction: {"sheep": 1, "wheat": 1, "ore": 1},
        }
        cost = next((value for kind, value in costs.items() if isinstance(action, kind)), None)
        if cost is None:
            return False
        stock = player_for(game, player_id).resources
        return any(stock[resource] - amount < min(stock[resource], required[resource])
                   for resource, amount in cost.items())

    @staticmethod
    def _spends_reserved_resources(game: GameState, player_id: int,
                                   target: SettlementTarget, action: Action) -> bool:
        return SettlementPlanningAgent._spends_budget(
            game, player_id, SettlementPlanningAgent._resource_budget(target, game), action
        )

    def _city_action(self, game: GameState, player_id: int, base_action: Action) -> Action:
        required = dict.fromkeys(RESOURCES, 0)
        required.update(BUILD_COSTS["city"])
        if estimated_build_wait(game, player_id, required) > self.max_city_wait_rounds:
            return base_action
        mask = get_action_mask(game, player_id)
        legal = [item.action for item in ACTION_CATALOG if mask[item.id]]
        probabilities = self._probabilities(game, player_id, mask)
        if isinstance(base_action, UseDevelopmentAction):
            return base_action
        cities = [action for action in legal if isinstance(action, BuildCityAction)]
        if cities:
            return max(cities, key=lambda action: (
                self._city_vertex_value(game, player_id, action.vertex_id),
                probabilities[action_to_id(action)],
            ))
        plenty = self._helpful_year_of_plenty_for_budget(
            game, player_id, required, legal, probabilities
        )
        if plenty is not None:
            return plenty
        trade = self._helpful_trade_for_budget(
            game, player_id, required, legal, probabilities
        )
        if trade is not None:
            return trade
        if (not isinstance(base_action, (BuildRoadAction, BankTradeAction))
                and not self._spends_budget(game, player_id, required, base_action)):
            return base_action
        end_turn = next((action for action in legal if isinstance(action, EndTurnAction)), None)
        return end_turn or base_action

    def _best_city(self, game: GameState, player_id: int) -> Action | None:
        mask = get_action_mask(game, player_id)
        cities = [item.action for item in ACTION_CATALOG
                  if mask[item.id] and isinstance(item.action, BuildCityAction)]
        if not cities:
            return None
        probabilities = self._probabilities(game, player_id, mask)
        return max(cities, key=lambda action: (
            self._city_vertex_value(game, player_id, action.vertex_id),
            probabilities[action_to_id(action)],
        ))

    def _city_vertex_value(self, game: GameState, player_id: int,
                           vertex_id: int) -> float:
        """総生産に、次の都市資源を補う価値を任意で加える。"""
        production = vertex_production(game, vertex_id)
        total = sum(production.values())
        if not self.city_growth_value_weight:
            return float(total)
        current = player_production(game, player_id)
        ore = production["ore"] / (1.0 + current["ore"] / 5.0)
        wheat = production["wheat"] / (1.0 + current["wheat"] / 5.0)
        return total + self.city_growth_value_weight * (ore + 0.8 * wheat)

    def _best_settlement(self, game: GameState, player_id: int) -> Action | None:
        mask = get_action_mask(game, player_id)
        settlements = [item.action for item in ACTION_CATALOG
                       if mask[item.id] and isinstance(item.action, BuildSettlementAction)]
        if not settlements:
            return None
        probabilities = self._probabilities(game, player_id, mask)
        return max(settlements, key=lambda action: (
            sum(vertex_production(game, action.vertex_id).values()),
            probabilities[action_to_id(action)],
        ))

    @staticmethod
    def _city_upgrade_value(game: GameState, player_id: int) -> float:
        """都市化で追加される1拠点分の生産価値。現在少ない資源を少し重く見る。"""
        current = player_production(game, player_id)
        values = []
        for vertex_id, owner_id in game.settlements.items():
            if owner_id != player_id:
                continue
            production = vertex_production(game, vertex_id)
            balance = sum(
                pips / (1.0 + current[resource] / 5.0)
                for resource, pips in production.items()
            )
            values.append(sum(production.values()) + 0.55 * balance)
        return max(values, default=0.0)

    def _prefer_city_over_connected_settlement(
        self, game: GameState, player_id: int, target: SettlementTarget,
    ) -> bool:
        """3拠点時、同じ1点を得る都市と接続済み開拓地の生産効率を比較する。"""
        if (not self.three_site_city_value_ratio or target.additional_roads
                or player_for(game, player_id).cities >= 4):
            return False
        city_wait = estimated_build_wait(game, player_id, BUILD_COSTS["city"])
        if city_wait > self.max_city_wait_rounds:
            return False
        city_rate = self._city_upgrade_value(game, player_id) / (1.0 + city_wait)
        settlement_rate = target.site_value / (1.0 + target.wait_rounds)
        return city_rate >= self.three_site_city_value_ratio * settlement_rate

    def _longest_road_progress(self, game: GameState, player_id: int) -> Action | None:
        current = _longest_road_length(game, player_id)
        score = public_score(game, player_id)
        horizon = (3 if self.extended_road_horizon_score
                   and score >= self.extended_road_horizon_score
                   else self.title_horizon)
        rival_length = max(_longest_road_length(game, other.id)
                           for other in game.players if other.id != player_id)
        holding = game.longest_road_holder_id == player_id
        if holding and not self.defend_titles:
            return None
        if not holding and score < self.longest_road_min_score:
            return None
        if holding and score < self.defend_titles_min_score:
            return None
        # 同点なら現保持者が維持する。相手が1〜2本差まで来た時だけ1本先へ逃げる。
        target_length = current + 1 if holding else max(5, rival_length + 1)
        if holding and current - rival_length > horizon:
            return None
        if target_length - current > horizon:
            return None
        mask = get_action_mask(game, player_id)
        roads = [item.action for item in ACTION_CATALOG
                 if mask[item.id] and isinstance(item.action, BuildRoadAction)]
        if not roads:
            return None
        progress: list[tuple[int, BuildRoadAction, bool, bool, int, int]] = []
        for action in roads:
            vertices = game.board.edges[action.edge_id].vertex_ids
            blocked = any(
                (owner := (game.settlements.get(vertex_id)
                           or game.cities.get(vertex_id))) not in {None, player_id}
                for vertex_id in vertices
            )
            contested = any(
                _building_owner is None and any(
                    (road_owner := game.roads.get(edge_id)) not in {None, player_id}
                    for edge_id in game.board.vertices[vertex_id].edge_ids
                )
                for vertex_id in vertices
                if (_building_owner := (game.settlements.get(vertex_id)
                                        or game.cities.get(vertex_id))) is None
            )
            game.roads[action.edge_id] = player_id
            try:
                length = _longest_road_length(game, player_id)
                continuations = []
                next_lengths = []
                if self.longest_road_lookahead:
                    continuations = [edge_id for edge_id in legal_road_ids(
                        game, player_id, initial=False
                    ) if edge_id not in game.roads]
                    for edge_id in continuations:
                        game.roads[edge_id] = player_id
                        try:
                            next_lengths.append(_longest_road_length(game, player_id))
                        finally:
                            del game.roads[edge_id]
            finally:
                del game.roads[action.edge_id]
            if length > current:
                progress.append((length, action, blocked, contested, len(continuations),
                                 max(next_lengths, default=length)))
        if not progress:
            return None
        probabilities = self._probabilities(game, player_id, mask)
        if not self.longest_road_lookahead:
            best_length = max(length for length, _, _, _, _, _ in progress)
            return self._best(
                [action for length, action, _, _, _, _ in progress
                 if length == best_length],
                probabilities,
            )

        # 相手建物で終端になる道と、相手道路と衝突する空き交差点へ向かう道は、
        # その1本で称号を取れる時だけ候補に残す。
        safe = [item for item in progress
                if (not item[2] and not item[3])
                or (item[0] >= 5 and item[0] > rival_length)]
        if safe:
            progress = safe
        return max(progress, key=lambda item: (
            item[0] >= target_length,
            item[5] >= target_length,
            item[5],                 # さらに1本置いた時の最大道路長
            item[0],                 # 今の1本で得る道路長
            item[4],                 # 次に置ける道路候補数
            not item[2],
            not item[3],
            probabilities[action_to_id(item[1])],
        ))[1]

    def _largest_army_progress(self, game: GameState, player_id: int) -> Action | None:
        holding = game.largest_army_holder_id == player_id
        if game.used_development_this_turn or (holding and not self.defend_titles):
            return None
        if holding and public_score(game, player_id) < self.defend_titles_min_score:
            return None
        player = player_for(game, player_id)
        usable = (player.development_cards.count("knight")
                  > player.new_development_cards.count("knight"))
        rival = max(other.played_knights for other in game.players if other.id != player_id)
        target = player.played_knights + 1 if holding else max(3, rival + 1)
        if holding and player.played_knights - rival > self.title_horizon:
            return None
        if target - player.played_knights > self.title_horizon:
            return None
        mask = get_action_mask(game, player_id)
        if usable:
            action = UseDevelopmentAction("knight")
            return action if mask[action_to_id(action)] else None
        if (self.buy_for_largest_army and target - player.played_knights == 1
                and not player.development_cards.count("knight")):
            action = BuyDevelopmentAction()
            return action if mask[action_to_id(action)] else None
        return None

    @staticmethod
    def _road_wins_now(game: GameState, player_id: int, road: BuildRoadAction) -> bool:
        """この1本道で最長交易路を奪い、10点へ届くか。公開道路と自分の点のみ使用。"""
        if actual_score(game, player_id) < 8 or game.longest_road_holder_id == player_id:
            return False
        game.roads[road.edge_id] = player_id
        try:
            length = _longest_road_length(game, player_id)
            rival = max(_longest_road_length(game, other.id)
                        for other in game.players if other.id != player_id)
            return length >= 5 and length > rival
        finally:
            del game.roads[road.edge_id]

    @staticmethod
    def _road_increases_longest(game: GameState, player_id: int,
                                road: BuildRoadAction) -> bool:
        before = _longest_road_length(game, player_id)
        game.roads[road.edge_id] = player_id
        try:
            return _longest_road_length(game, player_id) > before
        finally:
            del game.roads[road.edge_id]

    @staticmethod
    def _road_is_expansion_step(plan, road: BuildRoadAction) -> bool:
        target = plan.selected_target
        return bool(target is not None and road.edge_id in target.first_road_ids)

    def _purposeful_road_candidates(self, game: GameState, player_id: int,
                                    plan, *, affordable_only: bool) -> list[BuildRoadAction]:
        if affordable_only:
            mask = get_action_mask(game, player_id)
            roads = [item.action for item in ACTION_CATALOG
                     if mask[item.id] and isinstance(item.action, BuildRoadAction)]
        else:
            roads = [BuildRoadAction(edge_id) for edge_id in legal_road_ids(
                game, player_id, initial=False
            )]
        return [road for road in roads
                if self._road_is_expansion_step(plan, road)
                or self._road_increases_longest(game, player_id, road)]

    @staticmethod
    def _trade_helps_budget(game: GameState, player_id: int,
                            action: BankTradeAction,
                            required: dict[str, int]) -> bool:
        player = player_for(game, player_id)
        ratio = _port_ratio(game, player_id, action.give_resource)
        receive_missing = (
            player.resources[action.receive_resource]
            < required[action.receive_resource]
        )
        reserved = min(
            player.resources[action.give_resource], required[action.give_resource]
        )
        gives_only_surplus = (
            player.resources[action.give_resource] - ratio >= reserved
        )
        return receive_missing and gives_only_surplus

    def _guard_paid_road_purpose(self, game: GameState, player_id: int,
                                 action: Action) -> Action:
        """拡張にも航路延長にもならない有料道路と、その準備交換を止める。"""
        if (not self.purposeful_paid_roads or game.phase != "action"
                or game.free_road_remaining):
            return action
        plan = analyze_expansion_plan(
            game, player_id, max_additional_roads=self.max_plan_roads,
        )
        if isinstance(action, BuildRoadAction):
            if (self._road_is_expansion_step(plan, action)
                    or self._road_increases_longest(game, player_id, action)):
                return action
            candidates = self._purposeful_road_candidates(
                game, player_id, plan, affordable_only=True,
            )
            if candidates:
                mask = get_action_mask(game, player_id)
                probabilities = self._probabilities(game, player_id, mask)
                before = _longest_road_length(game, player_id)

                def road_value(road: BuildRoadAction):
                    game.roads[road.edge_id] = player_id
                    try:
                        after = _longest_road_length(game, player_id)
                    finally:
                        del game.roads[road.edge_id]
                    return (
                        self._road_is_expansion_step(plan, road),
                        after - before,
                        after,
                        probabilities[action_to_id(road)],
                    )

                return max(candidates, key=road_value)
        elif (isinstance(action, BankTradeAction)
              and action.receive_resource in {"wood", "brick"}):
            target = plan.selected_target
            if target is not None and self._trade_helps_budget(
                    game, player_id, action, self._resource_budget(target, game)):
                return action
            roads = self._purposeful_road_candidates(
                game, player_id, plan, affordable_only=False,
            )
            if roads and self._trade_helps_budget(
                    game, player_id, action,
                    {**dict.fromkeys(RESOURCES, 0), **BUILD_COSTS["road"]}):
                return action
        else:
            return action

        mask = get_action_mask(game, player_id)
        return next(
            (item.action for item in ACTION_CATALOG
             if mask[item.id] and isinstance(item.action, EndTurnAction)),
            action,
        )

    def _near_city_reservation(self, game: GameState, player_id: int) -> dict | None:
        """都市が開拓地と比べても現実的に近い時だけ、現在ある都市資源を予約する。"""
        if not self.protect_city_bank_trade_reserve or game.phase != "action":
            return None
        player = player_for(game, player_id)
        if player.settlements <= 0 or player.cities >= 4:
            return None
        city_wait = estimated_build_wait(game, player_id, BUILD_COSTS["city"])
        if city_wait > self.max_city_wait_rounds:
            return None
        plan = analyze_expansion_plan(
            game, player_id, max_additional_roads=self.max_plan_roads,
        )
        target = plan.selected_target
        settlement_wait = target.wait_rounds if target is not None else 99.0
        city_rate = self._city_upgrade_value(game, player_id) / (1.0 + city_wait)
        settlement_rate = (
            target.site_value / (1.0 + settlement_wait) if target is not None else 0.0
        )
        competitive = bool(
            target is None
            or player.settlements >= 5
            or city_wait <= settlement_wait + self.city_reservation_wait_margin
            or (city_wait <= settlement_wait + 2.0
                and city_rate >= self.city_reservation_value_ratio * settlement_rate)
        )
        if not competitive:
            return None
        return {
            "city_wait": city_wait,
            "settlement_wait": settlement_wait,
            "city_rate": city_rate,
            "settlement_rate": settlement_rate,
            "required": {**dict.fromkeys(RESOURCES, 0), **BUILD_COSTS["city"]},
        }

    @staticmethod
    def _action_spends_city_reserve(game: GameState, player_id: int,
                                    action: Action, required: dict[str, int]) -> bool:
        """行動後に、現在確保済みの都市資源を割り込むかを判定する。"""
        player = player_for(game, player_id)
        costs = dict.fromkeys(RESOURCES, 0)
        if isinstance(action, BankTradeAction):
            costs[action.give_resource] = _port_ratio(
                game, player_id, action.give_resource
            )
        else:
            return False
        return any(
            player.resources[resource] - costs[resource]
            < min(player.resources[resource], required[resource])
            for resource in ("wheat", "ore")
        )

    def _guard_near_city_resources(self, game: GameState, player_id: int,
                                   action: Action) -> Action:
        reservation = self._near_city_reservation(game, player_id)
        if (reservation is None
                or not self._action_spends_city_reserve(
                    game, player_id, action, reservation["required"]
                )):
            return action
        mask = get_action_mask(game, player_id)
        return next(
            (item.action for item in ACTION_CATALOG
             if mask[item.id] and isinstance(item.action, EndTurnAction)),
            action,
        )

    def _select_planned_action(self, game: GameState, player_id: int) -> Action:
        if (self.expansion_aware_setup and game.phase == "setup_ready"
                and game.pending_settlement_id is None
                and player_for(game, player_id).settlements == 1
                and self.policy.initial_setup_policy is not None):
            observation = encode_observation(get_observation(
                game, player_id, version=self.policy.manifest.observation_version
            ))
            mask = expansion_coverage_mask(
                game, player_id, get_action_mask(game, player_id),
                min_pips=self.expansion_setup_min_pips,
                min_wheat_pips=self.setup_min_wheat_pips,
                min_ore_pips=self.setup_min_ore_pips,
                fallback_to_best_coverage=self.setup_best_coverage_fallback,
            )
            return id_to_action(
                self.policy.initial_setup_policy.select_action_id(observation, mask)
            )
        base_action = self.policy.select_action(game, player_id)
        if (self.collision_aware_initial_roads and game.phase == "setup_ready"
                and game.pending_settlement_id is not None
                and isinstance(base_action, PlaceInitialRoadAction)):
            from .initial_road_planner import choose_initial_road
            edge_id, _ = choose_initial_road(
                game, player_id, base_action.edge_id,
            )
            return PlaceInitialRoadAction(edge_id)
        if self.belief_aware_robber and game.phase == "robber_move":
            from .hand_belief import BayesianHandEstimator
            beliefs = BayesianHandEstimator().estimate(game, player_id)
            legal = [item.action for item in ACTION_CATALOG
                     if get_action_mask(game, player_id)[item.id]
                     and isinstance(item.action, MoveRobberAction)]
            return MoveRobberAction(best_belief_robber_tile(
                game, player_id, [action.tile_id for action in legal], beliefs
            ))
        if self.belief_aware_robber and game.phase == "robber_steal":
            from .hand_belief import BayesianHandEstimator
            beliefs = BayesianHandEstimator().estimate(game, player_id)
            return StealResourceAction(best_belief_robber_victim(
                game, player_id, game.robber_victim_ids, beliefs
            ))
        if self.opponent_aware_robber and game.phase == "robber_move":
            legal = [item.action for item in ACTION_CATALOG
                     if get_action_mask(game, player_id)[item.id]
                     and isinstance(item.action, MoveRobberAction)]
            return MoveRobberAction(best_robber_tile(
                game, player_id, [action.tile_id for action in legal]
            ))
        if self.opponent_aware_robber and game.phase == "robber_steal":
            return StealResourceAction(best_robber_victim(
                game, player_id, game.robber_victim_ids
            ))
        score = actual_score(game, player_id)
        opponent_near_win = max(public_score(game, other.id) for other in game.players
                                if other.id != player_id) >= 8
        if (self.prioritize_immediate_titles
                and (score >= self.title_min_score or opponent_near_win)
                and game.phase in {"turn_pre_roll", "action"}):
            knight = self._largest_army_progress(game, player_id)
            if knight is not None:
                return knight
        if game.phase != "action":
            return base_action
        player = player_for(game, player_id)
        if score >= 10:
            return base_action
        if self.prioritize_immediate_win and score >= 9:
            scoring_build = (self._best_city(game, player_id)
                             or self._best_settlement(game, player_id))
            if scoring_build is not None:
                return scoring_build
        site_count = player.settlements + player.cities
        fastest_win_plan = None
        if (self.prioritize_immediate_win and score >= 9
                and player.settlements < 5):
            candidate_plan = analyze_expansion_plan(
                game, player_id, max_additional_roads=self.max_plan_roads,
            )
            candidate_target = candidate_plan.selected_target
            city_wait = (estimated_build_wait(game, player_id, BUILD_COSTS["city"])
                         if player.settlements > 0 and player.cities < 4 else 99.0)
            if (candidate_target is not None
                    and candidate_target.wait_rounds <= self.max_wait_rounds
                    and candidate_target.wait_rounds < city_wait):
                fastest_win_plan = candidate_plan
            elif city_wait <= self.max_city_wait_rounds:
                # 同じ1点なら、同着時は生産増加と開拓地コマ回収のある都市。
                return self._city_action(game, player_id, base_action)
        three_site_plan = None
        if (fastest_win_plan is None
                and self.three_site_city_value_ratio and site_count == 3):
            three_site_plan = analyze_expansion_plan(
                game, player_id, max_additional_roads=self.max_plan_roads,
            )
            target = three_site_plan.selected_target
            if (target is not None
                    and self._prefer_city_over_connected_settlement(game, player_id, target)):
                return self._city_action(game, player_id, base_action)
        if (fastest_win_plan is None
                and self.early_affordable_city_min_sites
                and site_count >= self.early_affordable_city_min_sites
                and site_count < self.target_sites
                and self._best_settlement(game, player_id) is None):
            early_city = self._best_city(game, player_id)
            if early_city is not None:
                return early_city
        endgame_plan = fastest_win_plan
        deferred_title_for_build = False
        if site_count >= self.target_sites and fastest_win_plan is None:
            city = self._best_city(game, player_id) if self.prioritize_affordable_city else None
            if city is not None and actual_score(game, player_id) >= 9:
                return city
            connected_reserve_plan = None
            # 5拠点到達後でも、都市化によって開拓地コマが手元へ戻っており、
            # 道路不要の候補を短く待てる場合だけ追加開拓地を検討する。
            # settlements == 5 の時は盤面候補があってもコマ不足なので対象外。
            if (self.post_target_connected_reserve_max_wait
                    and player.settlements < 5 and player.cities > 0
                    and self.post_target_connected_reserve_min_score <= score
                    <= self.post_target_connected_reserve_max_score):
                candidate_plan = analyze_expansion_plan(
                    game, player_id, max_additional_roads=self.max_plan_roads,
                )
                candidate_target = candidate_plan.selected_target
                if (candidate_target is not None
                        and candidate_target.additional_roads == 0
                        and candidate_target.wait_rounds
                        <= self.post_target_connected_reserve_max_wait):
                    connected_reserve_plan = candidate_plan
            road = None
            if self.prioritize_immediate_titles:
                road = self._longest_road_progress(game, player_id)
            candidate_road = road or (
                base_action if isinstance(base_action, BuildRoadAction) else None
            )
            if (connected_reserve_plan is not None and candidate_road is not None
                    and not game.free_road_remaining
                    and not self._road_wins_now(game, player_id, candidate_road)):
                # 即時勝利にならない有料道路だけを止める。開拓地・有益な交換・
                # 資源待ちは、下の既存計画処理へそのまま委ねる。
                road = None
                endgame_plan = connected_reserve_plan
                deferred_title_for_build = True
            if (road is not None and self.title_road_build_wait_threshold
                    and not self._road_wins_now(game, player_id, road)):
                construction_plan = analyze_expansion_plan(
                    game, player_id, max_additional_roads=self.max_plan_roads,
                )
                settlement_wait = (construction_plan.selected_target.wait_rounds
                                   if construction_plan.selected_target is not None else 99.0)
                city_wait = (estimated_build_wait(game, player_id, BUILD_COSTS["city"])
                             if player.settlements > 0 and player.cities < 4 else 99.0)
                if min(settlement_wait, city_wait) <= self.title_road_build_wait_threshold:
                    if city_wait <= settlement_wait:
                        return self._city_action(game, player_id, base_action)
                    road = None
                    endgame_plan = construction_plan
                    deferred_title_for_build = True
            if self.construction_before_titles:
                # 今勝てる道路は例外。それ以外は購入可能な建設を先に実行する。
                if road is not None and self._road_wins_now(game, player_id, road):
                    return road
                settlement = self._best_settlement(game, player_id)
                if settlement is not None:
                    return settlement
                if city is not None:
                    return city
            if road is not None:
                return road
            if city is not None:
                return city
            city_deficit = sum(
                max(0, amount - player.resources[resource])
                for resource, amount in BUILD_COSTS["city"].items()
            )
            if (self.first_city_max_deficit and player.cities == 0
                    and city_deficit <= self.first_city_max_deficit):
                return self._city_action(game, player_id, base_action)
            if (self.second_city_max_deficit and player.cities == 1
                    and score >= self.second_city_min_score
                    and city_deficit <= self.second_city_max_deficit):
                return self._city_action(game, player_id, base_action)
            if (self.third_city_max_deficit and player.cities == 2
                    and score >= self.third_city_min_score
                    and city_deficit <= self.third_city_max_deficit):
                return self._city_action(game, player_id, base_action)
            if self.target_cities and player.cities < self.target_cities:
                return self._city_action(game, player_id, base_action)
            if self.endgame_point_reserve_score and score >= self.endgame_point_reserve_score:
                endgame_plan = endgame_plan or analyze_expansion_plan(
                    game, player_id, max_additional_roads=self.max_plan_roads,
                )
                settlement_wait = (endgame_plan.selected_target.wait_rounds
                                   if endgame_plan.selected_target is not None else 99.0)
                city_wait = (estimated_build_wait(game, player_id, BUILD_COSTS["city"])
                             if player.settlements > 0 and player.cities < 4 else 99.0)
                if city_wait <= settlement_wait and city_wait <= self.max_city_wait_rounds:
                    return self._city_action(game, player_id, base_action)
                if (endgame_plan.selected_target is None
                        or settlement_wait > self.max_wait_rounds):
                    return base_action
            elif not deferred_title_for_build:
                return base_action
        plan = endgame_plan or three_site_plan or analyze_expansion_plan(
            game, player_id, max_additional_roads=self.max_plan_roads,
        )
        target = plan.selected_target
        if target is None or target.wait_rounds > self.max_wait_rounds:
            return base_action

        mask = get_action_mask(game, player_id)
        legal = [item.action for item in ACTION_CATALOG if mask[item.id]]
        probabilities = self._probabilities(game, player_id, mask)

        # 現行PPOが既存カードを使おうとする判断は壊さない。
        if isinstance(base_action, UseDevelopmentAction):
            return base_action

        settlements = [action for action in legal if isinstance(action, BuildSettlementAction)]
        if settlements:
            exact = [action for action in settlements if action.vertex_id == target.vertex_id]
            return self._best(exact or settlements, probabilities)

        plenty = self._helpful_year_of_plenty(game, player_id, target, legal, probabilities)
        if plenty is not None:
            return plenty
        trade = self._helpful_trade(game, player_id, target, legal, probabilities)
        if trade is not None:
            return trade

        recommended = [action for action in legal if isinstance(action, BuildRoadAction)
                       and action.edge_id in target.first_road_ids]
        if recommended:
            return self._best(recommended, probabilities)

        # 予約分を使わない都市・発展購入なら、学習済み判断をそのまま許す。
        if (not isinstance(base_action, (BuildRoadAction, BankTradeAction))
                and not self._spends_reserved_resources(game, player_id, target, base_action)):
            return base_action

        end_turn = next((action for action in legal if isinstance(action, EndTurnAction)), None)
        return end_turn or base_action

    def select_action(self, game: GameState, player_id: int) -> Action:
        planned = self._select_planned_action(game, player_id)
        planned = self._guard_paid_road_purpose(game, player_id, planned)
        planned = self._guard_near_city_resources(game, player_id, planned)
        if not (self.proactive_player_trade or self.strategic_trade_response
                or self.adaptive_development_purchase
                or self.endgame_development_fallback):
            return planned
        from .trade_strategy import (choose_counter_trade, choose_proactive_trade,
                                     choose_trade_response)
        options = {
            "target_sites": self.target_sites,
            "max_plan_roads": self.max_plan_roads,
            "max_wait_rounds": self.max_wait_rounds,
            "max_city_wait_rounds": self.max_city_wait_rounds,
        }
        if self.strategic_trade_response and game.phase == "trade_response":
            return choose_trade_response(
                game, player_id,
                opponent_score_limit=self.strategic_trade_opponent_score_limit,
                **options,
            )
        if self.strategic_trade_response and game.phase == "trade_counter_offer":
            return choose_counter_trade(game, player_id, **options) or planned
        if (self.proactive_player_trade and game.phase == "action"
                and (isinstance(planned, EndTurnAction)
                     or self.proactive_trade_max_scarcity_deficit > 1)):
            trade = choose_proactive_trade(
                game, player_id,
                max_scarcity_deficit=self.proactive_trade_max_scarcity_deficit,
                strategic_surplus_trade=self.strategic_surplus_trade,
                opponent_score_limit=self.proactive_trade_opponent_score_limit,
                **options,
            )
            if trade is not None:
                return trade
        if (self.adaptive_development_purchase and game.phase == "action"
                and isinstance(planned, EndTurnAction)):
            from .development_strategy import assess_development_purchase
            assessment = assess_development_purchase(
                game, player_id,
                max_plan_roads=self.max_plan_roads,
                stall_wait_rounds=self.development_stall_wait_rounds,
                max_build_delay=self.development_max_build_delay,
                tactical_score=self.development_tactical_score,
                title_horizon=self.title_horizon,
            )
            if assessment["eligible"]:
                return self._guard_near_city_resources(
                    game, player_id, BuyDevelopmentAction()
                )
        if (self.endgame_development_fallback and game.phase == "action"
                and isinstance(planned, EndTurnAction)
                and actual_score(game, player_id) == 9):
            from .development_strategy import assess_development_purchase
            assessment = assess_development_purchase(
                game, player_id,
                max_plan_roads=self.max_plan_roads,
                stall_wait_rounds=self.development_stall_wait_rounds,
                max_build_delay=self.development_max_build_delay,
                tactical_score=self.development_tactical_score,
                title_horizon=self.title_horizon,
            )
            if assessment["eligible"]:
                return self._guard_near_city_resources(
                    game, player_id, BuyDevelopmentAction()
                )
        return planned
