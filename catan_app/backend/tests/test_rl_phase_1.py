import unittest

import numpy as np

from app.domain.actions import BuildRoadAction, DiscardHalfAction, RollOrderAction
from app.domain.game import apply_action, create_game, player_for
from app.rl.action_space import (ACTION_CATALOG, ACTION_SPACE_SIZE, action_to_id,
                                 get_action_mask, id_to_action, legal_policy_actions)
from app.rl.observation import (OBSERVATION_VECTOR_SIZE, encode_observation,
                                get_observation)


class RlPhaseOneTests(unittest.TestCase):
    def test_observation_keeps_opponent_private_cards_private(self):
        game = create_game(42)
        self_player = player_for(game, 1)
        opponent = player_for(game, 2)
        self_player.resources["wood"] = 2
        self_player.development_cards.append("victory_point")
        self_player.settlements = 1
        opponent.resources["brick"] = 3
        opponent.development_cards.extend(("knight", "victory_point"))
        opponent.used_development_cards.append("road_building")

        observation = get_observation(game, 1)
        observed_self = next(player for player in observation.players if player.id == 1)
        observed_opponent = next(player for player in observation.players if player.id == 2)

        self.assertEqual(observed_self.resources, (2, 0, 0, 0, 0))
        self.assertEqual(observed_self.actual_score, 2)
        self.assertEqual(observed_opponent.resource_count, 3)
        self.assertEqual(observed_opponent.development_count, 2)
        self.assertIsNone(observed_opponent.resources)
        self.assertIsNone(observed_opponent.development_cards)
        self.assertIsNone(observed_opponent.actual_score)
        self.assertEqual(observed_opponent.public_score, 0)
        self.assertEqual(observed_opponent.used_development_cards, ("road_building",))

    def test_observation_encoder_is_fixed_size_and_normalized(self):
        first = encode_observation(get_observation(create_game(42), 1))
        second = encode_observation(get_observation(create_game(99), 4))

        self.assertEqual(first.shape, (OBSERVATION_VECTOR_SIZE,))
        self.assertEqual(second.shape, (OBSERVATION_VECTOR_SIZE,))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.all((0.0 <= first) & (first <= 1.0)))
        self.assertTrue(np.all((0.0 <= second) & (second <= 1.0)))

    def test_fixed_action_catalog_is_reversible(self):
        self.assertEqual(ACTION_SPACE_SIZE, 377)
        self.assertEqual(len(ACTION_CATALOG), ACTION_SPACE_SIZE)
        for definition in ACTION_CATALOG:
            with self.subTest(action_id=definition.id):
                self.assertEqual(action_to_id(id_to_action(definition.id)), definition.id)
                self.assertEqual(id_to_action(definition.id), definition.action)

    def test_action_mask_matches_policy_legal_actions_and_applies(self):
        game = create_game(42)
        mask = get_action_mask(game, 1)
        legal_actions = legal_policy_actions(game, 1)
        legal_ids = {action_to_id(action) for action in legal_actions}

        self.assertEqual(set(np.flatnonzero(mask)), legal_ids)
        self.assertEqual(legal_actions, [RollOrderAction()])
        action = id_to_action(next(iter(legal_ids)))
        apply_action(game, 1, action, game.revision, "rl-action-mask-roll")
        self.assertEqual(game.revision, 1)

    def test_fixed_discard_action_uses_existing_discard_rule(self):
        game = create_game(42)
        player = player_for(game, 1)
        player.resources["wood"] = 8
        game.bank["wood"] -= 8
        game.phase = "discarding"
        game.current_player_id = 1
        game.pending_discards = [1]

        action = legal_policy_actions(game, 1)[0]
        self.assertIsInstance(action, DiscardHalfAction)
        apply_action(game, 1, action, game.revision, "rl-discard-half")
        self.assertEqual(player.resources["wood"], 4)
        self.assertEqual(game.phase, "robber_move")

    def test_free_road_phase_only_masks_road_and_can_apply_it(self):
        game = create_game(42)
        player = player_for(game, 1)
        game.phase = "action"
        game.current_player_id = 1
        game.settlements[0] = 1
        player.settlements = 1
        player.resources.update(dict.fromkeys(player.resources, 5))
        game.free_road_remaining = 1

        options = legal_policy_actions(game, 1)
        self.assertTrue(options)
        self.assertTrue(all(isinstance(option, BuildRoadAction) for option in options))
        self.assertEqual(int(get_action_mask(game, 1).sum()), len(options))
        apply_action(game, 1, options[0], game.revision, "rl-free-road-mask")
        self.assertEqual(game.free_road_remaining, 0)


if __name__ == "__main__":
    unittest.main()
