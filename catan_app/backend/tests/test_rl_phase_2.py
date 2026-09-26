import unittest

import numpy as np

from app.domain.actions import (BuildRoadAction, BuildSettlementAction, BuyDevelopmentAction,
                                PlaceInitialRoadAction,
                                PlaceInitialSettlementAction, RollOrderAction)
from app.agents.heuristic import HeuristicAgent
from app.domain.game import create_game, legal_settlement_ids, player_for
from app.rl.action_space import ACTION_CATALOG, action_to_id, get_action_mask
from app.rl.env import (CatanEnv, CatanEnvConfig, ExternalActionRequired,
                        IllegalPolicyAction, expansion_strategy_penalty,
                        expansion_target_weights)


class RlPhaseTwoTests(unittest.TestCase):
    def test_fixed_opponent_override_is_used(self):
        class CountingAgent:
            def __init__(self):
                self.calls = 0
                self.delegate = HeuristicAgent()

            def select_action(self, game, player_id):
                self.calls += 1
                return self.delegate.select_action(game, player_id)

        opponent = CountingAgent()
        env = CatanEnv(opponent_agents={2: opponent})
        _, info = env.reset(seed=42)
        for _ in range(4):
            if opponent.calls:
                break
            action_id = int(np.flatnonzero(info["action_mask"])[0])
            _, _, _, _, info = env.step(action_id)
        self.assertGreater(opponent.calls, 0)
        self.assertEqual(info["fixed_opponent_ids"], [2])
        env.close()

    def test_heuristic_setup_delegation_leaves_normal_turn_to_learner(self):
        observed = []
        env = CatanEnv(CatanEnvConfig(heuristic_initial_placement=True),
                       action_observer=lambda game, pid, action: observed.append((pid, action)))
        _, info = env.reset(seed=101)
        for _ in range(20):
            self.assertNotEqual(env.game.phase, "setup_ready")
            if env.game.phase not in {"rolling_order", "setup_ready"}:
                break
            action_id = int(np.flatnonzero(info["action_mask"])[0])
            _, _, terminated, truncated, info = env.step(action_id)
            self.assertFalse(terminated or truncated)
        self.assertEqual(env.game.players[0].settlements, 2)
        self.assertEqual(sum(owner == 1 for owner in env.game.roads.values()), 2)
        self.assertEqual(sum(pid == 1 and isinstance(action, PlaceInitialSettlementAction)
                             for pid, action in observed), 2)
        self.assertEqual(sum(pid == 1 and isinstance(action, PlaceInitialRoadAction)
                             for pid, action in observed), 2)
        env.close()

    def test_mixed_opponents_rotate_reproducibly_without_changing_default(self):
        from app.rl.env import CatanEnv, CatanEnvConfig

        default = CatanEnv()
        _, default_info = default.reset(seed=42)
        self.assertEqual([default_info["controllers"][pid] for pid in (2, 3, 4)],
                         ["heuristic"] * 3)
        default.close()

        env = CatanEnv(CatanEnvConfig(heuristic_opponents=1))
        assigned = []
        for seed in (42, 43, 44):
            _, info = env.reset(seed=seed)
            assigned.append(next(pid for pid in (2, 3, 4)
                                 if info["controllers"][pid] == "heuristic"))
            self.assertEqual(list(info["controllers"].values()).count("random"), 2)
        self.assertEqual(set(assigned), {2, 3, 4})
        _, repeated = env.reset(seed=42)
        self.assertEqual(next(pid for pid in (2, 3, 4)
                              if repeated["controllers"][pid] == "heuristic"), assigned[0])
        env.close()

    def test_reset_is_seeded_and_exposes_masked_gymnasium_values(self):
        first = CatanEnv()
        second = CatanEnv()
        first_observation, first_info = first.reset(seed=20260921)
        second_observation, second_info = second.reset(seed=20260921)

        self.assertTrue(np.array_equal(first_observation, second_observation))
        self.assertTrue(np.array_equal(first_info["action_mask"], second_info["action_mask"]))
        self.assertTrue(first.observation_space.contains(first_observation))
        self.assertTrue(first.action_space.contains(action_to_id(RollOrderAction())))
        self.assertEqual(first.episode_seed, 20260921)
        self.assertEqual(first_info["learning_player_id"], 1)

    def test_step_applies_masked_action_and_rejects_illegal_id(self):
        env = CatanEnv()
        _, info = env.reset(seed=42)
        revision_before = env.game.revision
        illegal_action = next(index for index, legal in enumerate(info["action_mask"]) if not legal)
        with self.assertRaises(IllegalPolicyAction):
            env.step(illegal_action)
        self.assertEqual(env.game.revision, revision_before)

        action_id = int(np.flatnonzero(info["action_mask"])[0])
        observation, reward, terminated, truncated, next_info = env.step(action_id)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("reward_components", next_info)
        self.assertGreater(env.game.revision, revision_before)

    def test_external_seat_pauses_then_accepts_external_action(self):
        env = CatanEnv(CatanEnvConfig(controller_by_player={4: "external"}))
        _, info = env.reset(seed=42)
        first_action = int(np.flatnonzero(info["action_mask"])[0])
        _, _, _, _, info = env.step(first_action)

        self.assertEqual(info["external_action_required"], 4)
        self.assertEqual(env.pending_external_player_id, 4)
        with self.assertRaises(ExternalActionRequired):
            env.step(first_action)

        _, next_info = env.step_external(4, action_to_id(RollOrderAction()))
        self.assertNotEqual(env.game.revision, 0)
        self.assertIn("external_action_required", next_info)

    def test_max_policy_steps_truncates_without_declaring_winner(self):
        env = CatanEnv(CatanEnvConfig(max_policy_steps=1))
        _, info = env.reset(seed=42)
        action_id = int(np.flatnonzero(info["action_mask"])[0])
        _, _, terminated, truncated, _ = env.step(action_id)

        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_expansion_curriculum_only_changes_training_mask(self):
        game = create_game(20260924)
        game.phase = "action"
        game.current_player_id = 1
        game.roads[30] = 1
        player = player_for(game, 1)
        player.resources.update(wood=4, brick=4, sheep=2, wheat=2, ore=2)
        self.assertTrue(legal_settlement_ids(game, 1, initial=False))
        normal_mask = get_action_mask(game, 1)
        self.assertTrue(normal_mask[action_to_id(BuyDevelopmentAction())])
        self.assertAlmostEqual(expansion_target_weights(
            game, 1, normal_mask)[action_to_id(BuyDevelopmentAction())], 0.35)
        self.assertTrue(any(normal_mask[item.id] for item in ACTION_CATALOG
                            if isinstance(item.action, BuildRoadAction)))

        env = CatanEnv(CatanEnvConfig(expansion_curriculum=True))
        env.game = game
        curriculum_mask = env.action_masks()
        self.assertFalse(curriculum_mask[action_to_id(BuyDevelopmentAction())])
        self.assertFalse(any(curriculum_mask[item.id] for item in ACTION_CATALOG
                             if isinstance(item.action, BuildRoadAction)))
        self.assertTrue(curriculum_mask.any())
        env.close()

    def test_expansion_penalty_keeps_actions_legal_but_marks_bad_spending(self):
        game = create_game(20260924)
        game.phase = "action"
        game.current_player_id = 1
        game.roads[30] = 1
        player = player_for(game, 1)
        player.resources.update(wood=4, brick=4, sheep=2, wheat=2, ore=2)
        normal_mask = get_action_mask(game, 1)
        development = BuyDevelopmentAction()
        road = next(item.action for item in ACTION_CATALOG
                    if normal_mask[item.id] and isinstance(item.action, BuildRoadAction))

        self.assertTrue(normal_mask[action_to_id(development)])
        self.assertTrue(normal_mask[action_to_id(road)])
        self.assertEqual(expansion_strategy_penalty(
            game, 1, normal_mask, development,
            early_development=0.03, unproductive_road=0.02), -0.03)
        self.assertEqual(expansion_strategy_penalty(
            game, 1, normal_mask, road,
            early_development=0.03, unproductive_road=0.02), -0.02)

    def test_expansion_curriculum_waits_for_connected_site_without_settlement_resources(self):
        game = create_game(20260925)
        game.phase = "action"
        game.current_player_id = 1
        game.roads[30] = 1
        player = player_for(game, 1)
        player.resources.update(wood=1, brick=1)
        self.assertTrue(legal_settlement_ids(game, 1, initial=False))
        normal_mask = get_action_mask(game, 1)
        self.assertFalse(any(normal_mask[item.id] for item in ACTION_CATALOG
                             if isinstance(item.action, BuildSettlementAction)))
        self.assertTrue(any(normal_mask[item.id] for item in ACTION_CATALOG
                            if isinstance(item.action, BuildRoadAction)))

        env = CatanEnv(CatanEnvConfig(expansion_curriculum=True))
        env.game = game
        curriculum_mask = env.action_masks()
        self.assertFalse(any(curriculum_mask[item.id] for item in ACTION_CATALOG
                             if isinstance(item.action, BuildRoadAction)))
        target_weights = expansion_target_weights(game, 1, normal_mask)
        self.assertTrue(all(target_weights[item.id] == 0.25 for item in ACTION_CATALOG
                            if normal_mask[item.id]
                            and isinstance(item.action, BuildRoadAction)))
        env.close()

    def test_expansion_curriculum_allows_development_during_estimated_stall(self):
        game = create_game(20260926)
        game.phase = "action"
        game.current_player_id = 1
        game.roads[30] = 1
        player = player_for(game, 1)
        player.resources.update(sheep=1, wheat=1, ore=1)
        normal_mask = get_action_mask(game, 1)
        development_id = action_to_id(BuyDevelopmentAction())
        self.assertTrue(normal_mask[development_id])

        env = CatanEnv(CatanEnvConfig(expansion_curriculum=True))
        env.game = game
        self.assertTrue(env.action_masks()[development_id])
        self.assertEqual(expansion_target_weights(
            game, 1, normal_mask)[development_id], 1.0)
        env.close()

    def test_masked_policy_rollout_reaches_game_over(self):
        """P1の合法手選択と固定Heuristic3人で、UIなしの1局が完走する。"""
        env = CatanEnv()
        _, info = env.reset(seed=101)
        while True:
            choices = np.flatnonzero(info["action_mask"])
            action_id = int(choices[(101 + env.policy_steps) % len(choices)])
            _, _, terminated, truncated, info = env.step(action_id)
            if terminated or truncated:
                break

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIn(env.game.winner_id, range(1, 5))

    def test_trade_auxiliary_and_trade_capable_opponents_finish_game(self):
        """PPO外の交渉を補助層へ委譲しても、Action Mask境界を保ち完走する。"""
        env = CatanEnv(
            CatanEnvConfig(allow_player_trades=True),
            learner_auxiliary_agent=HeuristicAgent(),
            opponent_agents={2: HeuristicAgent()},
        )
        _, info = env.reset(seed=20260926)
        while True:
            choices = np.flatnonzero(info["action_mask"])
            self.assertGreater(len(choices), 0)
            action_id = int(choices[(20260926 + env.policy_steps) % len(choices)])
            _, _, terminated, truncated, info = env.step(action_id)
            if terminated or truncated:
                break

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["player_trades_enabled"])
        self.assertGreater(info["learner_auxiliary_actions"], 0)
        self.assertTrue(any(entry["kind"].startswith("trade")
                            for entry in env.game.activity_log))
        env.close()


if __name__ == "__main__":
    unittest.main()
