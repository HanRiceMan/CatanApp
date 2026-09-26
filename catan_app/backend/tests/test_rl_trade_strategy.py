import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.domain.actions import ProposeTradeAction
from app.domain.game import RESOURCES, create_game, player_for
from app.rl.trade_strategy import (choose_proactive_trade, construction_trade_goal,
                                   choose_trade_response)


def trade_game():
    game = create_game(20260930)
    game.phase = "action"
    game.current_player_id = 1
    game.turn_number = 12
    player = player_for(game, 1)
    player.settlements = 3
    player.cities = 2
    player.resources.update(wood=2, wheat=1, ore=3)
    return game


def options():
    return {
        "target_sites": 5,
        "max_plan_roads": 3,
        "max_wait_rounds": 6,
        "max_city_wait_rounds": 9,
    }


class ProactiveTradeTests(unittest.TestCase):
    def test_stalled_surplus_wheat_is_offered_for_scarce_ore(self):
        game = trade_game()
        player = player_for(game, 1)
        player.resources = dict.fromkeys(RESOURCES, 0)
        player.resources.update(sheep=1, wheat=4)
        game.resource_events = [{"deltas": {"2": {"ore": 1}}}]

        action = choose_proactive_trade(
            game, 1, strategic_surplus_trade=True, **options(),
        )

        self.assertIsInstance(action, ProposeTradeAction)
        self.assertEqual(action.target_id, 2)
        self.assertIn({"wheat": 1}, [offer.give for offer in action.offers])
        self.assertTrue(all(offer.want == {"ore": 1} for offer in action.offers))

    def test_nine_point_trade_goal_prefers_faster_winning_settlement(self):
        game = trade_game()
        player = player_for(game, 1)
        player.settlements = 3
        player.cities = 3
        target = SimpleNamespace(wait_rounds=1.0, additional_roads=0)
        plan = SimpleNamespace(selected_target=target)

        with (patch("app.rl.trade_strategy.analyze_expansion_plan", return_value=plan),
              patch("app.rl.trade_strategy.estimated_build_wait", return_value=8.0),
              patch("app.rl.trade_strategy.settlement_target_cost",
                    return_value={"wood": 1, "brick": 1, "sheep": 1,
                                  "wheat": 1, "ore": 0})):
            goal = construction_trade_goal(game, 1, **options())

        self.assertEqual(goal[0], "settlement")

    def test_burst_risk_escalates_rejected_one_for_one_to_two_for_one(self):
        game = trade_game()
        player = player_for(game, 1)
        player.resources = dict.fromkeys(RESOURCES, 0)
        player.resources.update(sheep=1, wheat=8)
        game.resource_events = [{"deltas": {"2": {"ore": 1}}}]
        game.activity_log.append({
            "kind": "trade", "turn_number": game.turn_number,
            "trade": {
                "proposer_id": 1, "recipient_id": 2, "is_counter": False,
                "offers": [{"give": {"wheat": 1}, "want": {"ore": 1}}],
            },
        })

        action = choose_proactive_trade(
            game, 1, strategic_surplus_trade=True, **options(),
        )

        self.assertIsInstance(action, ProposeTradeAction)
        self.assertEqual(action.target_id, 2)
        self.assertTrue(all(offer.give == {"wheat": 2} for offer in action.offers))
        self.assertTrue(all(offer.want == {"ore": 1} for offer in action.offers))

    def test_strategic_trade_does_not_spend_reserved_wheat(self):
        game = trade_game()
        player = player_for(game, 1)
        player.resources = dict.fromkeys(RESOURCES, 0)
        player.resources.update(wheat=2)
        game.resource_events = [{"deltas": {"2": {"ore": 1}}}]

        action = choose_proactive_trade(
            game, 1, strategic_surplus_trade=True, **options(),
        )

        self.assertIsNone(action)

    def test_strategic_trade_does_not_delay_immediate_construction(self):
        game = trade_game()
        game.roads[30] = 1
        player = player_for(game, 1)
        player.resources = dict.fromkeys(RESOURCES, 0)
        player.resources.update(wood=1, brick=1, sheep=1, wheat=4, ore=3)
        game.resource_events = [{"deltas": {"2": {"ore": 1}}}]

        action = choose_proactive_trade(
            game, 1, strategic_surplus_trade=True, **options(),
        )

        self.assertIsNone(action)

    def test_proposes_only_when_trade_completes_construction_budget(self):
        game = trade_game()
        game.resource_events = [{"deltas": {"2": {"wheat": 1}}}]

        action = choose_proactive_trade(game, 1, **options())

        self.assertIsInstance(action, ProposeTradeAction)
        self.assertEqual(action.target_id, 2)
        self.assertTrue(all(offer.want == {"wheat": 1} for offer in action.offers))
        self.assertTrue(all(offer.give == {"wood": 1} for offer in action.offers))

        player_for(game, 1).resources["ore"] = 2
        self.assertIsNone(choose_proactive_trade(game, 1, **options()))

    def test_avoids_eight_point_leader_when_another_holder_exists(self):
        game = trade_game()
        leader = player_for(game, 2)
        leader.settlements = 4
        leader.cities = 2
        game.resource_events = [{
            "deltas": {"2": {"wheat": 2}, "3": {"wheat": 1}},
        }]

        action = choose_proactive_trade(game, 1, **options())

        self.assertIsNotNone(action)
        self.assertEqual(action.target_id, 3)

    def test_scarce_resource_can_be_collected_before_last_missing_card(self):
        game = trade_game()
        player = player_for(game, 1)
        player.resources.update(wood=3, wheat=2, ore=0)
        game.resource_events = [{"deltas": {"2": {"ore": 1}}}]

        self.assertIsNone(choose_proactive_trade(game, 1, **options()))
        action = choose_proactive_trade(
            game, 1, max_scarcity_deficit=3, **options()
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.target_id, 2)
        self.assertTrue(all(offer.want == {"ore": 1} for offer in action.offers))

    def test_accepts_trade_that_reduces_own_construction_deficit(self):
        game = trade_game()
        game.phase = "trade_response"
        game.pending_trade = {
            "proposer_id": 2,
            "recipient_id": 1,
            "is_counter": False,
            "awaiting_counter": False,
            "offers": [{
                "give": {resource: int(resource == "wheat") for resource in RESOURCES},
                "want": {resource: int(resource == "wood") for resource in RESOURCES},
            }],
        }

        response = choose_trade_response(game, 1, **options())

        self.assertEqual(response.decision, "accept")
        self.assertEqual(response.offer_index, 0)

    def test_declines_helpful_trade_from_public_eight_point_opponent(self):
        game = trade_game()
        proposer = player_for(game, 2)
        proposer.settlements = 4
        proposer.cities = 2
        game.phase = "trade_response"
        game.pending_trade = {
            "proposer_id": 2,
            "recipient_id": 1,
            "is_counter": False,
            "awaiting_counter": False,
            "offers": [{
                "give": {resource: int(resource == "wheat") for resource in RESOURCES},
                "want": {resource: int(resource == "wood") for resource in RESOURCES},
            }],
        }

        response = choose_trade_response(
            game, 1, opponent_score_limit=8, **options()
        )

        self.assertEqual(response.decision, "decline")


if __name__ == "__main__":
    unittest.main()
