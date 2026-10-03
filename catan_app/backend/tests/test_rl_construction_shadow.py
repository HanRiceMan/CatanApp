"""Step 3A-7: state-based建設Surface、hard exception、保存再読込。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.domain.actions import BuildCityAction, BuildSettlementAction, EndTurnAction
from app.domain.game import create_game, player_for
from app.rl.action_space import ACTION_SPACE_SIZE, action_to_id, get_action_mask
from app.rl.collect_construction_shadow import verify_artifact_reload
from app.rl.construction_shadow import _shadow_proposal, detect_construction_surface
from app.rl.context_aware_policy import (
    ContextAwareGraphMaskablePolicy, ContextAwarePPOAgent,
    load_context_aware_artifact, save_context_aware_artifact,
)
from app.rl.env import CatanEnv, CatanEnvConfig
from app.rl.observation import encode_observation, get_observation


class ConstructionSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.game = create_game(20261101)
        self.game.phase = "action"
        self.game.current_player_id = 1
        self.mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        self.mask[action_to_id(BuildSettlementAction(0))] = True
        self.mask[action_to_id(EndTurnAction())] = True

    def test_state_predicate_is_independent_of_planner_action(self):
        settlement = detect_construction_surface(self.game, 1, self.mask)
        self.assertTrue(settlement.eligible)
        self.assertEqual(settlement.subtype, "SETTLEMENT_ONLY")
        self.mask[action_to_id(BuildCityAction(1))] = True
        both = detect_construction_surface(self.game, 1, self.mask)
        self.assertTrue(both.eligible)
        self.assertEqual(both.subtype, "SETTLEMENT_AND_CITY")
        self.mask[action_to_id(BuildSettlementAction(0))] = False
        self.assertEqual(detect_construction_surface(self.game, 1, self.mask).subtype,
                         "CITY_ONLY")

    def test_hard_exceptions_are_excluded_before_shadow(self):
        self.game.phase = "robber_move"
        self.assertEqual(detect_construction_surface(self.game, 1, self.mask).exclusion,
                         "OUTSIDE_ACTION_PHASE")
        self.game.phase = "action"
        self.game.free_road_remaining = 1
        self.assertEqual(detect_construction_surface(self.game, 1, self.mask).exclusion,
                         "FREE_ROAD_SEQUENCE")
        self.game.free_road_remaining = 0
        player_for(self.game, 1).development_cards.extend(["victory_point"] * 9)
        self.mask[action_to_id(BuildCityAction(1))] = True
        self.assertEqual(detect_construction_surface(self.game, 1, self.mask).exclusion,
                         "IMMEDIATE_BUILD_WIN")

    def test_immediate_title_win_and_no_build_are_not_delegated(self):
        player_for(self.game, 1).development_cards.extend(["victory_point"] * 8)
        title_context = SimpleNamespace(sections={
            "road_title": {"immediate_win": SimpleNamespace(value=True)},
            "knight_title": {"immediate_win": SimpleNamespace(value=False)},
        })
        with patch("app.rl.construction_shadow.build_structured_context_report",
                   return_value=title_context):
            self.assertEqual(detect_construction_surface(self.game, 1, self.mask).exclusion,
                             "IMMEDIATE_TITLE_WIN")
        self.mask[action_to_id(BuildSettlementAction(0))] = False
        self.assertEqual(detect_construction_surface(self.game, 1, self.mask).exclusion,
                         "NO_IMMEDIATE_CONSTRUCTION")

    def test_context_artifact_save_reload_does_not_overwrite(self):
        from sb3_contrib import MaskablePPO

        model = MaskablePPO(
            ContextAwareGraphMaskablePolicy,
            CatanEnv(CatanEnvConfig(observation_version="v7")),
            n_steps=8, batch_size=8,
            policy_kwargs={"net_arch": [], "ortho_init": False},
            device="cpu", seed=123,
        )
        parent = SimpleNamespace(
            manifest=SimpleNamespace(model_id="unit_test_parent"),
            initial_setup_policy=None,
            settlement_planning_config=None,
            deterministic=True,
        )
        original = ContextAwarePPOAgent(parent, model)
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "fresh_artifact"
            save_context_aware_artifact(original, path)
            self.assertTrue((path / "model.zip").exists())
            with self.assertRaises(FileExistsError):
                save_context_aware_artifact(original, path)
            loaded = load_context_aware_artifact(path, parent)
            self.assertTrue(all(verify_artifact_reload(original, loaded).values()))
            game = create_game(20261102)
            game.phase = "action"
            game.current_player_id = 1
            observation = encode_observation(get_observation(game, 1, version="v7"))
            mask = get_action_mask(game, 1)
            proposed = _shadow_proposal(loaded.model, observation, mask)
            predicted, _ = loaded.model.predict(
                observation, deterministic=True, action_masks=mask,
            )
            self.assertEqual(proposed["selected_action"]["catalog_id"],
                             int(np.asarray(predicted).item()))
            wrong_parent = SimpleNamespace(
                manifest=SimpleNamespace(model_id="different_parent"),
                initial_setup_policy=None, settlement_planning_config=None,
                deterministic=True,
            )
            with self.assertRaises(ValueError):
                load_context_aware_artifact(path, wrong_parent)


if __name__ == "__main__":
    unittest.main()
