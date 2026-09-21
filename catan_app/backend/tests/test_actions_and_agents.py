import unittest
from copy import deepcopy

from app.agents.heuristic import HeuristicAgent
from app.agents.random_agent import RandomAgent
from app.agents.runner import get_agent_for_player, run_ai_until_pause, set_agent_for_player
from app.domain.actions import EndTurnAction, RollOrderAction, RollTurnDiceAction, StartGameAction
from app.domain.game import apply_action, create_game, legal_actions


class ActionAndAgentTests(unittest.TestCase):
    def test_apply_action_dispatches_to_existing_rule_and_updates_state(self):
        game = create_game(42)

        apply_action(game, 1, RollOrderAction((3, 3)), game.revision, "action-order-roll")

        self.assertEqual(game.revision, 1)
        self.assertEqual(game.rolls[0]["player_id"], 1)
        self.assertEqual(game.rolls[0]["dice"], [3, 3])
        self.assertEqual(game.pending_rollers[0], 2)

    def test_heuristic_agent_selects_action_without_mutating_game_state(self):
        game = create_game(42, ai_player_ids=(1, 2, 3, 4))
        before = deepcopy(game)

        action = HeuristicAgent().select_action(game, 1)

        self.assertIsInstance(action, RollOrderAction)
        self.assertEqual(game, before)

    def test_random_agent_uses_legal_actions_and_can_be_selected_per_player(self):
        game = create_game(42, ai_player_ids=(1,))
        set_agent_for_player(game, 1, "random")

        action = get_agent_for_player(game, 1).select_action(game, 1)

        self.assertIsInstance(get_agent_for_player(game, 1), RandomAgent)
        self.assertIn(action, legal_actions(game, 1))

    def test_seeded_game_random_source_reproduces_engine_dice(self):
        first = create_game(99)
        second = create_game(99)

        apply_action(first, 1, RollOrderAction(), first.revision, "first-random-roll")
        apply_action(second, 1, RollOrderAction(), second.revision, "second-random-roll")

        self.assertEqual(first.rolls[0]["dice"], second.rolls[0]["dice"])
        self.assertEqual(first.random_source.counter, second.random_source.counter)

    def test_actions_cover_main_discrete_turn_choices(self):
        game = create_game(42)
        for player_id, dice in ((1, (3, 3)), (2, (6, 6)), (3, (4, 4)), (4, (1, 1))):
            apply_action(game, player_id, RollOrderAction(dice), game.revision, f"order-{player_id}")
        game.phase = "ready_to_play"
        first = game.seat_order[0]
        apply_action(game, first, StartGameAction(), game.revision, "start-game")
        apply_action(game, first, RollTurnDiceAction((2, 3)), game.revision, "turn-roll")

        actions = legal_actions(game, first)

        self.assertTrue(actions)
        self.assertTrue(any(isinstance(action, EndTurnAction) for action in actions))

    def test_four_heuristic_agents_finish_multiple_seeds(self):
        for seed in (20260914, 20260915):
            with self.subTest(seed=seed):
                game = create_game(seed, ai_player_ids=(1, 2, 3, 4))
                self.assertIsInstance(get_agent_for_player(game, 1), HeuristicAgent)
                result = run_ai_until_pause(game, 100_000)
                self.assertEqual(result["status"], "game_over")
                self.assertEqual(game.phase, "game_over")


if __name__ == "__main__":
    unittest.main()
