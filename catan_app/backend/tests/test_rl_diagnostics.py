from copy import deepcopy
import importlib.util
import unittest

import numpy as np

from app.agents.random_agent import RandomAgent
from app.domain.actions import (BankTradeAction, BuildCityAction, BuildRoadAction,
                                BuildSettlementAction, BuyDevelopmentAction, EndTurnAction)
from app.domain.game import apply_action, create_game, player_for
from app.rl.action_space import action_to_id, get_action_mask, id_to_action
from app.rl.compare import PolicyCompatibleHeuristicAgent
from app.rl.diagnostics_metrics import EvaluationTelemetry, endgame_rank
from app.rl.env import CatanEnv, CatanEnvConfig
from app.rl.evaluate import EvaluationConfig, episode_config, evaluate_agent
from app.rl.observation import get_observation


class DiagnosticTests(unittest.TestCase):
    def test_city_upgrade_keeps_site_count_and_records_milestone_once(self):
        game = create_game(42)
        game.phase = "action"
        game.current_player_id = 1
        player = player_for(game, 1)
        game.settlements[0] = 1
        player.settlements = 4
        player.resources.update(wheat=2, ore=3)
        telemetry = EvaluationTelemetry(1)
        action = BuildCityAction(0)
        telemetry.before_action(game, action, get_action_mask(game, 1))
        apply_action(game, 1, action, game.revision, "city-milestone")
        telemetry.after_action(game, 1, action)
        self.assertEqual(player.settlements + player.cities, 4)
        self.assertEqual(telemetry.strategy["first_city_own_turn"], 1)
        self.assertEqual(telemetry.strategy["fourth_site_reached"], 1)
        telemetry._own_turns_seen.add(game.turn_number + 4)
        telemetry.after_action(game, 1, action)
        self.assertEqual(telemetry.strategy["first_city_own_turn"], 1)

    @unittest.skipUnless(importlib.util.find_spec("sb3_contrib"), "PPO runtime required")
    def test_policy_trace_groups_action_families(self):
        from app.rl.inspect_policy import action_family

        self.assertEqual(action_family(BuildRoadAction(0)), "road")
        self.assertEqual(action_family(BuildSettlementAction(0)), "settlement")
        self.assertEqual(action_family(BuildCityAction(0)), "city")
        self.assertEqual(action_family(BuyDevelopmentAction()), "development_purchase")
        self.assertEqual(action_family(BankTradeAction("wood", "brick")), "bank_trade")
        self.assertEqual(action_family(EndTurnAction()), "end_turn")

    def test_terminal_observation_does_not_double_count_revealed_victory_points(self):
        game = create_game(42)
        player = player_for(game, 1)
        player.cities = 4
        player.development_cards = ["victory_point", "victory_point"]
        game.phase = "game_over"
        game.winner_id = 1
        observed = next(p for p in get_observation(game, 1).players if p.id == 1)
        self.assertEqual(observed.public_score, 10)
        self.assertEqual(observed.actual_score, 10)

    def test_rank_respects_winner_ties_and_truncation(self):
        scores = {1: 10, 2: 8, 3: 8, 4: 3}
        self.assertEqual(endgame_rank(scores, 1, 1), (1.0, 0.0))
        self.assertEqual(endgame_rank(scores, 2, 1), (2.5, 0.5))
        self.assertEqual(endgame_rank(scores, 4, 1), (4.0, 0.0))
        self.assertEqual(endgame_rank(scores, 2, None), (None, 0.0))

    def test_mix_and_identity_rotation_are_balanced_and_reproducible(self):
        config = EvaluationConfig(heuristic_opponents=1, rotate_player_ids=True)
        assignments = [episode_config(config, i) for i in range(12)]
        for pid in range(1, 5):
            matching = [c for c in assignments if c.learning_player_id == pid]
            self.assertEqual(len(matching), 3)
            self.assertEqual({other for c in matching for other, kind in c.opponent_by_player.items()
                              if kind == "heuristic"}, set(range(1, 5)) - {pid})
        self.assertEqual(assignments, [episode_config(config, i) for i in range(12)])

    def test_telemetry_keeps_title_ever_acquired_and_detects_skipped_city(self):
        game = create_game(42)
        game.phase = "action"
        game.current_player_id = 1
        game.settlements[0] = 1
        player = player_for(game, 1)
        player.settlements = 1
        player.resources.update(wheat=2, ore=3)
        telemetry = EvaluationTelemetry(1)
        mask = get_action_mask(game, 1)
        self.assertTrue(mask[action_to_id(BuildCityAction(0))])
        telemetry.before_action(game, EndTurnAction(), mask)
        self.assertEqual(telemetry.opportunities["end_turn_with_city_available"], 1)
        game.longest_road_holder_id = 1
        telemetry.after_action(game, 1, EndTurnAction())
        game.longest_road_holder_id = 2
        telemetry.after_action(game, 2, EndTurnAction())
        self.assertEqual(telemetry.titles["longest_road"], {1, 2})

    def test_telemetry_records_settlement_quality_and_road_purpose(self):
        game = create_game(43)
        game.phase = "action"
        game.current_player_id = 1
        player = player_for(game, 1)
        player.resources.update(wood=2, brick=2, sheep=2, wheat=2)
        # 盤面中央付近の1本を自分のネットワークにして、次の合法道路を選ぶ。
        game.roads[30] = 1
        mask = get_action_mask(game, 1)
        road_ids = [index for index in np.flatnonzero(mask)
                    if isinstance(id_to_action(int(index)), BuildRoadAction)]
        self.assertTrue(road_ids)
        road = id_to_action(int(road_ids[0]))
        telemetry = EvaluationTelemetry(1)
        telemetry.before_action(game, road, mask)
        apply_action(game, 1, road, game.revision, "diagnostic-road")
        telemetry.after_action(game, 1, road)
        self.assertEqual(telemetry.strategy["road_builds"], 1)
        self.assertGreaterEqual(telemetry.strategy["road_endpoint_pips_total"], 0)
        self.assertEqual(
            telemetry.strategy["road_opens_settlement_site"]
            + telemetry.strategy["road_without_immediate_settlement_site"], 1,
        )

        # 道路の先に開拓地を建てられる状態なら、候補品質も記録される。
        player.resources.update(wood=2, brick=2, sheep=2, wheat=2)
        mask = get_action_mask(game, 1)
        settlement_ids = [index for index in np.flatnonzero(mask)
                          if isinstance(id_to_action(int(index)), BuildSettlementAction)]
        if settlement_ids:
            settlement = id_to_action(int(settlement_ids[0]))
            telemetry.before_action(game, settlement, mask)
            self.assertEqual(telemetry.opportunities["settlement_selected_when_available"], 1)
            self.assertEqual(telemetry.strategy["settlement_builds"], 1)
            self.assertGreaterEqual(telemetry.strategy["settlement_pips_total"], 1)

    def test_mixed_evaluation_records_assignments_and_no_hidden_score_in_policy(self):
        config = EvaluationConfig(heuristic_opponents=2, rotate_player_ids=True,
                                  policy_compatible_opponents=True)
        report = evaluate_agent(RandomAgent(), (20001, 20002), config=config)
        summary = report.to_dict()["summary"]
        self.assertEqual(summary["illegal_action_count"], 0)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual([e.learner_id for e in report.episodes], [1, 2])
        for episode in report.episodes:
            self.assertEqual(list(episode.opponent_controllers.values()).count("heuristic"), 2)
            self.assertEqual(episode.observation_range, (0.0, 1.0))
            self.assertIn(episode.learner_rank, (1, 2, 2.5, 3, 3.5, 4))
            self.assertIsInstance(episode.strategy, dict)

    def test_sampled_mask_candidates_apply_to_existing_rules_during_game(self):
        env = CatanEnv(CatanEnvConfig(policy_compatible_opponents=True))
        _, info = env.reset(seed=20005)
        agent = PolicyCompatibleHeuristicAgent()
        checked = 0
        while True:
            game = env.game
            if env.policy_steps % 10 == 0:
                for action_id in np.flatnonzero(info["action_mask"])[::3]:
                    cloned = deepcopy(game)
                    apply_action(cloned, 1, id_to_action(int(action_id)), cloned.revision, "mask-rule-check")
                    checked += 1
            action = agent.select_action(game, 1)
            _, _, terminated, truncated, info = env.step(action_to_id(action))
            if terminated or truncated:
                break
        self.assertGreater(checked, 5)

    @unittest.skipUnless(importlib.util.find_spec("sb3_contrib"), "PPO runtime required")
    def test_rollout_buffer_and_inference_both_use_masks(self):
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback

        test = self
        class CheckMasks(BaseCallback):
            checked = False

            def _on_step(self):
                return True

            def _on_rollout_end(self):
                buffer = self.model.rollout_buffer
                masks = buffer.action_masks.astype(bool)
                selected = buffer.actions.astype(int)
                test.assertTrue(np.take_along_axis(masks, selected, axis=-1).all())
                test.assertTrue((masks.sum(axis=-1) < 377).all())
                self.checked = True

        env = CatanEnv()
        model = MaskablePPO("MlpPolicy", env, n_steps=16, batch_size=16, n_epochs=1,
                            policy_kwargs={"net_arch": [16]}, seed=11, device="cpu")
        callback = CheckMasks()
        model.learn(total_timesteps=16, callback=callback)
        self.assertTrue(callback.checked)
        observation, info = env.reset(seed=11)
        action, _ = model.predict(observation, action_masks=info["action_mask"], deterministic=True)
        self.assertTrue(info["action_mask"][int(action)])
        env.close()


if __name__ == "__main__":
    unittest.main()
