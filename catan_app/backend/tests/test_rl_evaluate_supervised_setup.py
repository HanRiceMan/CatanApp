from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from app.domain.actions import PlaceInitialSettlementAction
from app.rl.action_space import action_to_id
from app.rl.analyze_supervised_setup import paired_effect
from app.rl.compare import PolicyCompatibleHeuristicAgent
from app.rl.env import CatanEnv, CatanEnvConfig
from app.rl.evaluate_supervised_setup import SupervisedSetupAgent
from app.rl.setup_supervised import make_model


class SupervisedSetupEvaluationTests(unittest.TestCase):
    def test_supervised_setup_agent_chooses_legal_action_without_mutating_game(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_model.pt"
            torch.save(make_model("shared").state_dict(), path)
            agent = SupervisedSetupAgent(path, "shared")
            env = CatanEnv(CatanEnvConfig(learning_player_id=1, observation_version="v2"))
            _, info = env.reset(seed=36001)
            game = env.game
            assert game is not None
            for _ in range(12):
                if game.phase == "setup_ready" and game.pending_settlement_id is None:
                    break
                advance = PolicyCompatibleHeuristicAgent().select_action(game, 1)
                _, _, _, _, info = env.step(action_to_id(advance))
            else:
                raise AssertionError("初期開拓地選択に到達しませんでした。")
            before = (game.revision, dict(game.settlements), dict(game.roads))
            action = agent.select_action(game, 1)
            assert isinstance(action, PlaceInitialSettlementAction)
            assert info["action_mask"][action_to_id(action)]
            assert (game.revision, game.settlements, game.roads) == before
            env.close()


    def test_paired_effect_checks_seed_alignment(self):
        baseline = {"episodes": [{"seed": 1, "learner_id": 1, "learner_seat": 2,
                                   "learner_won": False, "learner_score": 5}]}
        candidate = {"episodes": [{"seed": 1, "learner_id": 1, "learner_seat": 2,
                                    "learner_won": True, "learner_score": 7}]}
        assert paired_effect(candidate, baseline, "learner_won")["candidate_minus_heuristic"] == 1
        candidate["episodes"][0]["seed"] = 2
        with self.assertRaises(ValueError):
            paired_effect(candidate, baseline, "learner_won")
