import unittest
from unittest.mock import patch

import numpy as np

from app.domain.actions import (BankTradeAction, BuildCityAction,
                                BuildRoadAction, BuildSettlementAction,
                                BuyDevelopmentAction, EndTurnAction,
                                UseDevelopmentAction)
from app.domain.game import (RESOURCES, _longest_road_length, create_game,
                             legal_settlement_ids, player_for)
from app.rl.action_space import ACTION_SPACE_SIZE
from app.rl.expansion_planner import (ExpansionPlan, SettlementTarget,
                                      analyze_expansion_plan,
                                      settlement_target_cost)
from app.rl.settlement_planning_agent import SettlementPlanningAgent


def scenario():
    game = create_game(20260928)
    game.phase = "action"
    game.current_player_id = 1
    game.last_roll = (3, 4)
    game.roads[30] = 1
    player = player_for(game, 1)
    player.settlements = 2
    return game


class StubPolicy:
    def __init__(self, action):
        self.action = action

    def select_action(self, game, player_id):
        return self.action


class UniformPlanningAgent(SettlementPlanningAgent):
    def _probabilities(self, game, player_id, mask):
        return np.ones(ACTION_SPACE_SIZE, dtype=np.float32)


class SettlementPlanningTests(unittest.TestCase):
    def test_dead_end_paid_road_is_replaced_by_longest_increasing_road(self):
        game = scenario()
        game.roads.clear()
        game.settlements.clear()
        game.cities.clear()
        for edge_id in (10, 11, 12, 13, 14, 17, 18, 20, 22):
            game.roads[edge_id] = 1
        game.roads[1] = 3
        game.roads[2] = 3
        game.settlements[8] = 1
        game.settlements[9] = 3
        player = player_for(game, 1)
        player.settlements = 3
        player.cities = 1
        player.resources = dict.fromkeys(RESOURCES, 0)
        player.resources.update(wood=1, brick=1)
        before = _longest_road_length(game, 1)
        bad_road = BuildRoadAction(7)
        game.roads[7] = 1
        self.assertEqual(_longest_road_length(game, 1), before)
        del game.roads[7]
        agent = UniformPlanningAgent(
            StubPolicy(bad_road), purposeful_paid_roads=True,
        )

        with patch.object(agent, "_select_planned_action", return_value=bad_road):
            selected = agent.select_action(game, 1)

        self.assertIsInstance(selected, BuildRoadAction)
        self.assertNotEqual(selected.edge_id, 7)
        game.roads[selected.edge_id] = 1
        try:
            self.assertGreater(_longest_road_length(game, 1), before)
        finally:
            del game.roads[selected.edge_id]

    def test_bank_trade_for_wood_or_brick_is_stopped_without_road_purpose(self):
        game = scenario()
        player = player_for(game, 1)
        player.resources = dict.fromkeys(RESOURCES, 0)
        player.resources["sheep"] = 4
        trade = BankTradeAction("sheep", "brick")
        empty_plan = ExpansionPlan(
            targets=(), selected_target=None, best_connected_target=None,
            recommended_road_ids=(), should_wait_for_settlement=False,
            settlement_wait_rounds=99.0, city_wait_rounds=99.0,
            build_wait_rounds=99.0,
        )
        agent = UniformPlanningAgent(
            StubPolicy(trade), purposeful_paid_roads=True,
        )

        with (patch.object(agent, "_select_planned_action", return_value=trade),
              patch("app.rl.settlement_planning_agent.analyze_expansion_plan",
                    return_value=empty_plan),
              patch.object(agent, "_purposeful_road_candidates", return_value=[])):
            selected = agent.select_action(game, 1)

        self.assertIsInstance(selected, EndTurnAction)

    def test_near_city_reserves_last_three_ore_from_second_bank_trade(self):
        game = scenario()
        vertex_id = game.board.edges[30].vertex_ids[0]
        game.settlements[vertex_id] = 1
        player = player_for(game, 1)
        player.settlements = 4
        player.resources = dict.fromkeys(RESOURCES, 0)
        player.resources.update(ore=8, wheat=1)
        trade = BankTradeAction("ore", "wood")
        agent = UniformPlanningAgent(
            StubPolicy(trade), target_sites=4,
            protect_city_bank_trade_reserve=True,
        )

        with patch.object(agent, "_select_planned_action", return_value=trade):
            self.assertEqual(agent.select_action(game, 1), trade)
            player.resources["ore"] = 4
            self.assertIsInstance(agent.select_action(game, 1), EndTurnAction)

    def test_near_city_reservation_never_blocks_development_purchase(self):
        game = scenario()
        vertex_id = game.board.edges[30].vertex_ids[0]
        game.settlements[vertex_id] = 1
        player = player_for(game, 1)
        player.settlements = 4
        player.resources.update(sheep=1, wheat=2, ore=3)
        agent = UniformPlanningAgent(
            StubPolicy(BuyDevelopmentAction()), target_sites=4,
            protect_city_bank_trade_reserve=True,
        )

        with patch.object(agent, "_select_planned_action",
                          return_value=BuyDevelopmentAction()):
            self.assertIsInstance(agent.select_action(game, 1), BuyDevelopmentAction)

    def test_city_reservation_recalculates_after_hand_changes(self):
        game = scenario()
        vertex_id = game.board.edges[30].vertex_ids[0]
        game.settlements[vertex_id] = 1
        player = player_for(game, 1)
        player.settlements = 3
        player.resources = dict.fromkeys(RESOURCES, 0)
        agent = UniformPlanningAgent(
            StubPolicy(EndTurnAction()), target_sites=5,
            protect_city_bank_trade_reserve=True,
        )
        target = SettlementTarget(
            vertex_id=19, additional_roads=0, first_road_ids=(),
            production_pips=(0, 6, 0, 0, 0), total_pips=6,
            new_resource_kinds=1, port_resource=None, port_ratio=None,
            site_value=10.0, plan_value=10.0,
            required_resources=(1, 1, 1, 1, 0), wait_rounds=1.0,
        )
        plan = ExpansionPlan(
            targets=(target,), selected_target=target,
            best_connected_target=target, recommended_road_ids=(),
            should_wait_for_settlement=True, settlement_wait_rounds=1.0,
            city_wait_rounds=8.0, build_wait_rounds=1.0,
        )

        with patch("app.rl.settlement_planning_agent.analyze_expansion_plan",
                   return_value=plan):
            with patch("app.rl.settlement_planning_agent.estimated_build_wait",
                       return_value=8.0):
                self.assertIsNone(agent._near_city_reservation(game, 1))
            # 盗賊等で手札構成が変わり都市完成が近づけば、同じ局面でも即座に転換する。
            player.resources.update(wheat=2, ore=3)
            with patch("app.rl.settlement_planning_agent.estimated_build_wait",
                       return_value=1.0):
                self.assertIsNotNone(agent._near_city_reservation(game, 1))

    def test_city_reservation_yields_to_clearly_faster_settlement(self):
        game = scenario()
        vertex_id = game.board.edges[30].vertex_ids[0]
        game.settlements[vertex_id] = 1
        player = player_for(game, 1)
        player.settlements = 3
        player.resources.update(ore=4)
        trade = BankTradeAction("ore", "wood")
        target = SettlementTarget(
            vertex_id=19, additional_roads=0, first_road_ids=(),
            production_pips=(0, 6, 0, 0, 0), total_pips=6,
            new_resource_kinds=1, port_resource=None, port_ratio=None,
            site_value=10.0, plan_value=10.0,
            required_resources=(1, 1, 1, 1, 0), wait_rounds=0.5,
        )
        plan = ExpansionPlan(
            targets=(target,), selected_target=target,
            best_connected_target=target, recommended_road_ids=(),
            should_wait_for_settlement=True, settlement_wait_rounds=0.5,
            city_wait_rounds=8.0, build_wait_rounds=0.5,
        )
        agent = UniformPlanningAgent(
            StubPolicy(trade), target_sites=5,
            protect_city_bank_trade_reserve=True,
        )

        with (patch.object(agent, "_select_planned_action", return_value=trade),
              patch("app.rl.settlement_planning_agent.analyze_expansion_plan",
                    return_value=plan),
              patch("app.rl.settlement_planning_agent.estimated_build_wait",
                    return_value=8.0)):
            self.assertEqual(agent.select_action(game, 1), trade)

    def test_two_site_stall_can_buy_development(self):
        game = scenario()
        player = player_for(game, 1)
        self.assertEqual(player.settlements + player.cities, 2)
        player.resources.update(sheep=1, wheat=1, ore=1)
        agent = UniformPlanningAgent(
            StubPolicy(EndTurnAction()), adaptive_development_purchase=True,
        )

        with patch.object(agent, "_select_planned_action", return_value=EndTurnAction()):
            self.assertIsInstance(agent.select_action(game, 1), BuyDevelopmentAction)

    def test_immediate_construction_prevents_adaptive_development_purchase(self):
        game = scenario()
        player_for(game, 1).resources.update(
            wood=1, brick=1, sheep=1, wheat=1, ore=1,
        )
        self.assertTrue(legal_settlement_ids(game, 1, initial=False))
        agent = UniformPlanningAgent(
            StubPolicy(EndTurnAction()), adaptive_development_purchase=True,
        )

        with patch.object(agent, "_select_planned_action", return_value=EndTurnAction()):
            self.assertIsInstance(agent.select_action(game, 1), EndTurnAction)

    def test_adaptive_development_buys_at_most_once_per_turn(self):
        game = scenario()
        player = player_for(game, 1)
        player.resources.update(sheep=2, wheat=2, ore=2)
        player.new_development_cards.append("knight")
        agent = UniformPlanningAgent(
            StubPolicy(EndTurnAction()), adaptive_development_purchase=True,
        )

        with patch.object(agent, "_select_planned_action", return_value=EndTurnAction()):
            self.assertIsInstance(agent.select_action(game, 1), EndTurnAction)

    def test_nine_point_stall_may_buy_development_as_last_resort(self):
        game = scenario()
        player = player_for(game, 1)
        player.settlements = 1
        player.cities = 4
        player.resources.update(sheep=1, wheat=1, ore=1)
        agent = UniformPlanningAgent(
            StubPolicy(EndTurnAction()), endgame_development_fallback=True,
        )

        with patch.object(agent, "_select_planned_action", return_value=EndTurnAction()):
            self.assertIsInstance(agent.select_action(game, 1), BuyDevelopmentAction)

    def test_construction_precedes_title_road_except_when_road_wins(self):
        game = scenario()
        vertex_id = game.board.edges[30].vertex_ids[0]
        game.settlements[vertex_id] = 1
        player = player_for(game, 1)
        player.settlements = 4
        player.resources.update(wheat=2, ore=3)
        agent = UniformPlanningAgent(
            StubPolicy(EndTurnAction()), prioritize_affordable_city=True,
            prioritize_immediate_titles=True, construction_before_titles=True,
        )
        road = BuildRoadAction(31)
        with patch.object(agent, "_longest_road_progress", return_value=road):
            self.assertEqual(agent.select_action(game, 1), BuildCityAction(vertex_id))
            with patch.object(agent, "_road_wins_now", return_value=True):
                self.assertEqual(agent.select_action(game, 1), road)

    def test_target_budget_combines_remaining_roads_and_settlement(self):
        game = scenario()
        target = analyze_expansion_plan(game, 1).selected_target
        self.assertIsNotNone(target)
        cost = settlement_target_cost(target)
        self.assertEqual(tuple(cost), RESOURCES)
        self.assertEqual(cost["wood"], 1 + target.additional_roads)
        self.assertEqual(cost["brick"], 1 + target.additional_roads)
        self.assertEqual(cost["sheep"], 1)
        self.assertEqual(cost["wheat"], 1)
        self.assertEqual(cost["ore"], 0)

    def test_bank_trade_uses_only_surplus_to_fill_plan_deficit(self):
        game = scenario()
        target = analyze_expansion_plan(game, 1).selected_target
        self.assertIsNotNone(target)
        player_for(game, 1).resources["ore"] = 4
        trades = [BankTradeAction("ore", "wood"), BankTradeAction("ore", "sheep")]
        selected = SettlementPlanningAgent._helpful_trade(
            game, 1, target, trades, np.ones(ACTION_SPACE_SIZE)
        )
        self.assertIn(selected, trades)
        player_for(game, 1).resources["ore"] = 3
        self.assertIsNone(SettlementPlanningAgent._helpful_trade(
            game, 1, target, trades, np.ones(ACTION_SPACE_SIZE)
        ))

    def test_available_settlement_overrides_development_purchase(self):
        game = scenario()
        player_for(game, 1).resources.update(wood=1, brick=1, sheep=1, wheat=1, ore=1)
        self.assertTrue(legal_settlement_ids(game, 1, initial=False))
        agent = UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                     target_sites=4, max_wait_rounds=12)
        action = agent.select_action(game, 1)
        self.assertIsInstance(action, BuildSettlementAction)

    def test_city_may_spend_only_resources_above_settlement_reserve(self):
        game = scenario()
        target = analyze_expansion_plan(game, 1).selected_target
        self.assertIsNotNone(target)
        player = player_for(game, 1)
        player.resources.update(wheat=2, ore=3)
        self.assertTrue(SettlementPlanningAgent._spends_reserved_resources(
            game, 1, target, BuildCityAction(0)
        ))
        player.resources.update(wheat=3, ore=3)
        self.assertFalse(SettlementPlanningAgent._spends_reserved_resources(
            game, 1, target, BuildCityAction(0)
        ))

    def test_city_phase_upgrades_high_production_site(self):
        game = scenario()
        vertex_id = game.board.edges[30].vertex_ids[0]
        game.settlements[vertex_id] = 1
        player = player_for(game, 1)
        player.settlements = 4
        player.resources.update(sheep=1, wheat=2, ore=3)
        agent = UniformPlanningAgent(
            StubPolicy(BuyDevelopmentAction()), target_sites=4, max_wait_rounds=9,
            target_cities=2, max_city_wait_rounds=6,
        )
        action = agent.select_action(game, 1)
        self.assertEqual(action, BuildCityAction(vertex_id))

    def test_affordable_city_is_taken_at_three_sites_when_settlement_unavailable(self):
        game = scenario()
        vertex_id = game.board.edges[30].vertex_ids[0]
        game.settlements[vertex_id] = 1
        player = player_for(game, 1)
        player.settlements = 3
        player.resources.update(wheat=2, ore=3)
        agent = UniformPlanningAgent(
            StubPolicy(BuyDevelopmentAction()), target_sites=4,
            early_affordable_city_min_sites=3,
        )

        self.assertEqual(agent.select_action(game, 1), BuildCityAction(vertex_id))

    def test_one_road_to_longest_title_is_selected(self):
        game = scenario()
        # 分岐のない5辺の単純経路を1つ作る。
        def walk(vertex, edges, vertices):
            if len(edges) == 5:
                return edges
            for edge_id in game.board.vertices[vertex].edge_ids:
                if edge_id in edges:
                    continue
                a, b = game.board.edges[edge_id].vertex_ids
                other = b if a == vertex else a
                if other not in vertices:
                    found = walk(other, edges + [edge_id], vertices | {other})
                    if found:
                        return found
            return None

        path = walk(0, [], {0})
        self.assertIsNotNone(path)
        game.roads = {edge_id: 1 for edge_id in path[:4]}
        player_for(game, 1).resources.update(wood=1, brick=1)
        agent = UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()))
        action = agent._longest_road_progress(game, 1)
        self.assertIsInstance(action, BuildRoadAction)
        player_for(game, 1).settlements = 4
        player_for(game, 1).cities = 2
        before = dict(game.roads)
        self.assertTrue(agent._road_wins_now(game, 1, action))
        self.assertEqual(game.roads, before)
        game.roads[action.edge_id] = 1
        self.assertGreaterEqual(_longest_road_length(game, 1), 5)

    def test_one_knight_to_largest_army_is_selected(self):
        game = scenario()
        player = player_for(game, 1)
        player.played_knights = 2
        player.development_cards.append("knight")
        agent = UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()))
        self.assertEqual(agent._largest_army_progress(game, 1),
                         UseDevelopmentAction("knight"))

    def test_largest_army_is_defended_only_when_rival_is_close(self):
        game = scenario()
        player = player_for(game, 1)
        player.played_knights = 3
        player.development_cards.append("knight")
        game.largest_army_holder_id = 1
        agent = UniformPlanningAgent(
            StubPolicy(BuyDevelopmentAction()), defend_titles=True, title_horizon=2,
        )

        player_for(game, 2).played_knights = 2
        self.assertEqual(agent._largest_army_progress(game, 1),
                         UseDevelopmentAction("knight"))

        player_for(game, 2).played_knights = 0
        self.assertIsNone(agent._largest_army_progress(game, 1))

    def test_title_min_score_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()), title_min_score=10)

    def test_title_road_build_wait_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                 title_road_build_wait_threshold=7)

    def test_first_city_deficit_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                 first_city_max_deficit=6)

    def test_second_city_deficit_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                 second_city_max_deficit=6)

    def test_third_city_deficit_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                 third_city_max_deficit=6)

    def test_three_site_city_ratio_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                 three_site_city_value_ratio=0.4)

    def test_near_second_city_preserves_city_resources(self):
        game = scenario()
        vertex_id = game.board.edges[30].vertex_ids[0]
        game.settlements[vertex_id] = 1
        player = player_for(game, 1)
        player.settlements = 3
        player.cities = 1
        player.resources.update(sheep=1, wheat=2, ore=2)
        agent = UniformPlanningAgent(
            StubPolicy(BuyDevelopmentAction()), target_sites=4,
            second_city_max_deficit=1,
        )

        self.assertIsInstance(agent.select_action(game, 1), EndTurnAction)

    def test_max_plan_roads_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()), max_plan_roads=6)

    def test_title_defense_score_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                 defend_titles_min_score=10)

    def test_endgame_reserve_score_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                 endgame_point_reserve_score=10)

    def test_longest_road_score_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                 longest_road_min_score=10)

    def test_extended_road_horizon_score_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                 extended_road_horizon_score=10)

    def test_setup_min_pips_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                 expansion_setup_min_pips=0)

    def test_setup_min_wheat_pips_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                 setup_min_wheat_pips=6)

    def test_setup_min_ore_pips_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                 setup_min_ore_pips=6)

    def test_city_growth_value_weight_is_validated(self):
        with self.assertRaises(ValueError):
            UniformPlanningAgent(StubPolicy(BuyDevelopmentAction()),
                                 city_growth_value_weight=2.1)

    def test_nine_points_prioritizes_winning_settlement(self):
        game = scenario()
        game.longest_road_holder_id = 1
        game.largest_army_holder_id = 1
        player = player_for(game, 1)
        player.settlements = 3
        player.cities = 1
        player.resources.update(wood=1, brick=1, sheep=1, wheat=1)
        agent = UniformPlanningAgent(
            StubPolicy(BuyDevelopmentAction()), target_sites=4,
            prioritize_immediate_win=True,
        )

        action = agent.select_action(game, 1)
        self.assertIsInstance(action, BuildSettlementAction)

    def test_nine_points_chooses_faster_route_without_score_based_development_rule(self):
        game = scenario()
        player = player_for(game, 1)
        player.settlements = 3
        player.cities = 3
        player.resources.update(brick=1, sheep=1, wheat=1, ore=1)
        target = SettlementTarget(
            vertex_id=19, additional_roads=0, first_road_ids=(),
            production_pips=(0, 6, 0, 0, 0), total_pips=6,
            new_resource_kinds=1, port_resource=None, port_ratio=None,
            site_value=10.0, plan_value=10.0,
            required_resources=(1, 1, 1, 1, 0), wait_rounds=1.0,
        )
        plan = ExpansionPlan(
            targets=(target,), selected_target=target,
            best_connected_target=target, recommended_road_ids=(),
            should_wait_for_settlement=True, settlement_wait_rounds=1.0,
            city_wait_rounds=8.0, build_wait_rounds=1.0,
        )
        agent = UniformPlanningAgent(
            StubPolicy(BuyDevelopmentAction()), target_sites=5,
            prioritize_immediate_win=True, adaptive_development_purchase=True,
        )

        with (patch("app.rl.settlement_planning_agent.analyze_expansion_plan",
                    return_value=plan),
              patch("app.rl.settlement_planning_agent.estimated_build_wait",
                    return_value=8.0)):
            action = agent._select_planned_action(game, 1)

        self.assertIsInstance(action, EndTurnAction)

    def test_post_target_connected_reserve_stops_paid_road_when_piece_is_available(self):
        game = scenario()
        player = player_for(game, 1)
        player.settlements = 4
        player.cities = 1
        player.resources.update(wood=1, brick=1, sheep=1, wheat=1)
        plan = analyze_expansion_plan(game, 1)
        self.assertIsNotNone(plan.selected_target)
        self.assertEqual(plan.selected_target.additional_roads, 0)
        road = BuildRoadAction(31)
        agent = UniformPlanningAgent(
            StubPolicy(road), target_sites=5,
            post_target_connected_reserve_max_wait=2,
            post_target_connected_reserve_min_score=6,
            post_target_connected_reserve_max_score=7,
        )

        self.assertIsInstance(agent.select_action(game, 1), BuildSettlementAction)

    def test_post_target_connected_reserve_requires_unused_settlement_piece(self):
        game = scenario()
        player = player_for(game, 1)
        player.settlements = 5
        player.cities = 1
        player.resources.update(wood=1, brick=1)
        road = BuildRoadAction(31)
        agent = UniformPlanningAgent(
            StubPolicy(road), target_sites=5,
            post_target_connected_reserve_max_wait=2,
            post_target_connected_reserve_min_score=6,
            post_target_connected_reserve_max_score=7,
        )

        self.assertEqual(agent.select_action(game, 1), road)


if __name__ == "__main__":
    unittest.main()
