"""Step 3A-6: v7決定的Contextと初期zero-impact Policy。"""

import json
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from app.domain.game import create_game, player_for
from app.rl.action_space import ACTION_SPACE_SIZE, get_action_mask
from app.rl.candidate_policy import GraphHierarchicalCandidateMaskablePolicy
from app.rl.context_aware_policy import (
    ContextAwareGraphMaskablePolicy, initialize_context_policy_from_champion,
)
from app.rl.env import CatanEnv, CatanEnvConfig
from app.rl.observation import (OBSERVATION_VECTOR_SIZES, encode_observation,
                                get_observation)
from app.rl.runtime_context import (CONTEXT_FIELDS, RUNTIME_CONTEXT_SIZE,
                                    encode_runtime_context,
                                    runtime_context_definition)


class RuntimeContextTests(unittest.TestCase):
    def test_definition_dimension_order_and_no_teacher_fields(self):
        definition = runtime_context_definition()
        snapshot = Path(__file__).resolve().parents[1] / "docs" / "rl_runtime_context_v7_definition.json"
        self.assertEqual(json.loads(snapshot.read_text(encoding="utf-8")), definition)
        self.assertEqual(definition["dimension"], RUNTIME_CONTEXT_SIZE)
        self.assertEqual(OBSERVATION_VECTOR_SIZES["v7"], 1667 + RUNTIME_CONTEXT_SIZE)
        self.assertLess(len(CONTEXT_FIELDS), RUNTIME_CONTEXT_SIZE)
        self.assertEqual(definition["field_order"][0]["vector_index"], 0)
        self.assertEqual(definition["field_order"][-1]["vector_index"],
                         RUNTIME_CONTEXT_SIZE - 1)
        self.assertEqual({field["name"] for field in definition["field_order"]},
                         {field.name for field in CONTEXT_FIELDS})
        self.assertEqual(
            next(field for field in definition["field_order"]
                 if field["name"] == "risk.port_advantage.wood")["range"], "2..4",
        )
        self.assertNotIn("plan_value", definition["field_order"])

    def test_context_is_side_effect_free_and_v7_has_bitwise_v2_prefix(self):
        game = create_game(20261031)
        game.phase = "action"
        game.current_player_id = 1
        before = deepcopy(game)
        old = encode_observation(get_observation(game, 1, version="v2"))
        new = encode_observation(get_observation(game, 1, version="v7"))
        self.assertEqual(game, before)
        self.assertTrue(np.array_equal(old, new[:len(old)]))
        self.assertTrue(np.array_equal(new[len(old):], encode_runtime_context(game, 1)))
        self.assertTrue(np.isfinite(new).all())
        self.assertTrue(((new[len(old):] >= 0) & (new[len(old):] <= 1)).all())
        self.assertEqual(game.random_source.counter, before.random_source.counter)

    def test_invalid_candidate_uses_value_zero_and_mask_zero(self):
        game = create_game(20261032)
        game.phase = "action"
        game.current_player_id = 1
        vector = encode_runtime_context(game, 1)
        fields = {field["name"]: field for field in runtime_context_definition()["field_order"]}
        for name in ("settlement.min_additional_roads", "settlement.eta",
                     "settlement.shortage.wood", "settlement.shortage.brick"):
            index = fields[name]["vector_index"]
            self.assertEqual(vector[index], 0)
            self.assertEqual(vector[index + 1], 0)
        self.assertEqual(vector[fields["risk.hand_size"]["vector_index"]], 0)

    def test_new_development_cards_are_own_known_state(self):
        game = create_game(20261033)
        game.phase = "turn_pre_roll"
        game.current_player_id = 1
        own = player_for(game, 1)
        own.development_cards.append("knight")
        own.new_development_cards.append("knight")
        other = player_for(game, 2)
        other.development_cards.append("monopoly")
        other.new_development_cards.append("monopoly")
        vector = encode_runtime_context(game, 1)
        fields = {field["name"]: field["vector_index"]
                  for field in runtime_context_definition()["field_order"]}
        self.assertEqual(vector[fields["new_development.knight"]], 1 / 14)
        self.assertEqual(vector[fields["new_development.monopoly"]], 0)
        self.assertEqual(vector[fields["knight.usable"]], 0)

    def test_zero_initialized_residual_preserves_actor_critic_and_mask(self):
        from sb3_contrib import MaskablePPO

        source = MaskablePPO(
            GraphHierarchicalCandidateMaskablePolicy,
            CatanEnv(CatanEnvConfig(observation_version="v2")),
            n_steps=8, batch_size=8, policy_kwargs={"net_arch": [], "ortho_init": False},
            device="cpu", seed=123,
        )
        target = MaskablePPO(
            ContextAwareGraphMaskablePolicy,
            CatanEnv(CatanEnvConfig(observation_version="v7")),
            n_steps=8, batch_size=8, policy_kwargs={"net_arch": [], "ortho_init": False},
            device="cpu", seed=456,
        )
        self.assertTrue(initialize_context_policy_from_champion(
            target.policy, source.policy,
        ))
        game = create_game(20261034)
        game.phase = "action"
        game.current_player_id = 1
        old = encode_observation(get_observation(game, 1, version="v2"))
        new = encode_observation(get_observation(game, 1, version="v7"))
        old_tensor, _ = source.policy.obs_to_tensor(old)
        new_tensor, _ = target.policy.obs_to_tensor(new)
        with torch.no_grad():
            old_logits = source.policy.mlp_extractor.forward_actor(old_tensor)
            new_logits = target.policy.mlp_extractor.forward_actor(new_tensor)
            old_value = source.policy.value_net(
                source.policy.mlp_extractor.forward_critic(old_tensor))
            new_value = target.policy.value_net(
                target.policy.mlp_extractor.forward_critic(new_tensor))
        self.assertTrue(torch.equal(old_logits, new_logits))
        self.assertTrue(torch.equal(old_value, new_value))
        mask = get_action_mask(game, 1)
        self.assertEqual(mask.shape, (ACTION_SPACE_SIZE,))
        if mask.any():
            with torch.no_grad():
                old_probs = source.policy.get_distribution(
                    old_tensor, action_masks=mask).distribution.probs
                new_probs = target.policy.get_distribution(
                    new_tensor, action_masks=mask).distribution.probs
            self.assertTrue(torch.equal(old_probs, new_probs))


if __name__ == "__main__":
    unittest.main()
