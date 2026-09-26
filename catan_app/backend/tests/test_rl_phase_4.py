import importlib.util
import tempfile
import unittest
from pathlib import Path

from app.domain.actions import RollOrderAction, RespondToTradeAction
from app.domain.game import create_game


HAS_PPO_RUNTIME = importlib.util.find_spec("sb3_contrib") is not None


@unittest.skipUnless(HAS_PPO_RUNTIME, "PPO smoke test requires backend/.venv-rl")
class RlPhaseFourTests(unittest.TestCase):
    def test_maskable_ppo_trains_saves_and_selects_a_legal_action(self):
        from app.agents.ppo import PPOAgent
        from app.rl.train import TrainConfig, train_maskable_ppo

        with tempfile.TemporaryDirectory() as directory:
            result = train_maskable_ppo(TrainConfig(
                model_id="ppo_smoke",
                total_timesteps=64,
                seed=73,
                n_steps=32,
                batch_size=32,
                n_epochs=1,
                policy_net_arch=(32, 32),
                checkpoint_interval_steps=32,
                device="cpu",
                evaluation_seeds=(701,),
                models_root=Path(directory),
            ))

            self.assertTrue((result.model_directory / "model.zip").is_file())
            self.assertTrue((result.model_directory / "checkpoint_64_steps.zip").is_file())
            self.assertEqual(result.evaluation.illegal_action_count, 0)
            agent = PPOAgent("ppo_smoke", models_root=Path(directory), device="cpu")
            action = agent.select_action(create_game(73), player_id=1)
            self.assertIsInstance(action, RollOrderAction)

            game = create_game(73)
            game.phase = "trade_response"
            game.current_player_id = 1
            game.pending_trade = {
                "proposer_id": 1, "recipient_id": 2,
                "offers": [{"give": {"wood": 1}, "want": {"brick": 1}}],
                "is_counter": False, "awaiting_counter": False,
            }
            revision = game.revision
            answer = agent.select_action(game, player_id=2)
            self.assertIsInstance(answer, RespondToTradeAction)
            self.assertEqual(game.revision, revision)

            resumed = train_maskable_ppo(TrainConfig(
                model_id="ppo_resumed",
                total_timesteps=128,
                seed=73,
                n_steps=32,
                batch_size=32,
                n_epochs=1,
                policy_net_arch=(32, 32),
                checkpoint_interval_steps=32,
                resume_checkpoint_steps=64,
                resume_from_model_id="ppo_smoke",
                device="cpu",
                evaluation_seeds=(702,),
                models_root=Path(directory),
            ))
            self.assertGreaterEqual(resumed.manifest.training_steps, 128)
            self.assertEqual(resumed.evaluation.illegal_action_count, 0)

            mixed = train_maskable_ppo(TrainConfig(
                model_id="ppo_mixed_smoke", total_timesteps=96, seed=73,
                heuristic_opponents=1, n_steps=32, batch_size=32, n_epochs=1,
                policy_net_arch=(32, 32), checkpoint_interval_steps=32,
                resume_checkpoint_steps=64, resume_from_model_id="ppo_smoke",
                device="cpu", evaluation_seeds=(703,), models_root=Path(directory),
            ))
            self.assertEqual(mixed.evaluation.illegal_action_count, 0)
            opponents = list(mixed.evaluation.episodes[0].opponent_controllers.values())
            self.assertEqual(opponents.count("heuristic"), 1)
            self.assertEqual(opponents.count("random"), 2)

    def test_resumed_seed_reproduces_policy_weights(self):
        from sb3_contrib import MaskablePPO
        import torch

        from app.rl.train import TrainConfig, train_maskable_ppo

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = dict(total_timesteps=32, n_steps=32, batch_size=32,
                          n_epochs=1, policy_net_arch=(32, 32),
                          checkpoint_interval_steps=32, device="cpu",
                          evaluation_seeds=(704,), models_root=root)
            train_maskable_ppo(TrainConfig(model_id="source", seed=71, **common))
            resumed = dict(common, total_timesteps=64, resume_checkpoint_steps=32,
                           resume_from_model_id="source")
            first = train_maskable_ppo(TrainConfig(model_id="repeat_a", seed=81, **resumed))
            second = train_maskable_ppo(TrainConfig(model_id="repeat_b", seed=81, **resumed))
            third = train_maskable_ppo(TrainConfig(model_id="different_seed", seed=82, **resumed))
            a = MaskablePPO.load(str(first.model_directory / "model.zip"), device="cpu")
            b = MaskablePPO.load(str(second.model_directory / "model.zip"), device="cpu")
            c = MaskablePPO.load(str(third.model_directory / "model.zip"), device="cpu")
            for name, tensor in a.policy.state_dict().items():
                self.assertTrue(torch.equal(tensor, b.policy.state_dict()[name]), name)
            self.assertTrue(any(not torch.equal(tensor, c.policy.state_dict()[name])
                                for name, tensor in a.policy.state_dict().items()))

    def test_setup_trained_model_delegates_setup_in_inference(self):
        from app.agents.heuristic import HeuristicAgent
        from app.agents.ppo import PPOAgent
        from app.rl.train import TrainConfig, train_maskable_ppo

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_maskable_ppo(TrainConfig(
                model_id="setup_smoke", total_timesteps=32, seed=91,
                heuristic_initial_placement=True, n_steps=32, batch_size=32,
                n_epochs=1, policy_net_arch=(32, 32), checkpoint_interval_steps=32,
                device="cpu", evaluation_seeds=(705,), models_root=root,
            ))
            agent = PPOAgent("setup_smoke", models_root=root, device="cpu")
            game = create_game(91)
            game.phase = "setup_ready"
            game.setup_order = [1, 2, 3, 4, 4, 3, 2, 1]
            setup_player = 1
            self.assertEqual(agent.select_action(game, setup_player),
                             HeuristicAgent().select_action(game, setup_player))


if __name__ == "__main__":
    unittest.main()
