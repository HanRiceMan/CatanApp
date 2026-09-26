import unittest

from app.agents.heuristic import HeuristicAgent
from app.domain.game import create_game
from app.rl.train_strongest_league import (SeededMixedLeagueAgent,
                                           StrategicTradeAuxiliaryAgent,
                                           configure_league_family_finetune)


class StrongestLeagueTests(unittest.TestCase):
    def test_seeded_mix_rotates_two_strong_seats_and_keeps_heuristic_games(self):
        strongest = HeuristicAgent()
        league = SeededMixedLeagueAgent(strongest, learning_player_id=1)
        strong_sets = [league.strong_player_ids(seed) for seed in range(12)]

        self.assertTrue(any(not seats for seats in strong_sets))
        self.assertTrue(all(len(seats) in {0, 2} for seats in strong_sets))
        used = set().union(*(seats for seats in strong_sets if seats))
        self.assertEqual(used, {2, 3, 4})

    def test_seeded_mix_delegates_without_mutating_game(self):
        class CountingAgent:
            def __init__(self):
                self.calls = 0
                self.delegate = HeuristicAgent()

            def select_action(self, game, player_id):
                self.calls += 1
                return self.delegate.select_action(game, player_id)

        strongest = CountingAgent()
        league = SeededMixedLeagueAgent(strongest)
        game = create_game(0)
        before = game.revision
        strong_id = next(iter(league.strong_player_ids(game.seed)))

        league.select_action(game, strong_id)

        self.assertEqual(strongest.calls, 1)
        self.assertEqual(game.revision, before)

    def test_trade_auxiliary_does_not_call_planning_policy(self):
        class PlanningStub:
            target_sites = 5
            max_plan_roads = 3
            max_wait_rounds = 6.0
            max_city_wait_rounds = 9.0
            proactive_player_trade = False
            proactive_trade_max_scarcity_deficit = 3
            strategic_surplus_trade = True
            proactive_trade_opponent_score_limit = 8
            strategic_trade_response = True
            strategic_trade_opponent_score_limit = 10

        game = create_game(42)
        game.phase = "action"
        game.current_player_id = 1
        action = StrategicTradeAuxiliaryAgent(PlanningStub()).select_action(game, 1)
        self.assertEqual(type(action).__name__, "EndTurnAction")

    def test_family_finetune_function_rejects_wrong_policy_type(self):
        with self.assertRaises(TypeError):
            configure_league_family_finetune(object())


if __name__ == "__main__":
    unittest.main()
