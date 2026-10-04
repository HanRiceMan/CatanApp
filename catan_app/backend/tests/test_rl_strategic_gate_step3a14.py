"""Step 3A-14 sampling, safety attribution, and semi-MDP boundaries."""

from math import log
from types import SimpleNamespace
import random
import unittest

import numpy as np

from app.domain.actions import (BankTradeAction, BuildCityAction, BuildRoadAction,
                                EndTurnAction)
from app.domain.game import create_game, player_for
from app.rl.action_space import action_to_id, get_action_mask
from app.rl.run_strategic_gate_step3a14 import _safety_difference
from app.rl.strategic_gate_rollout_step3a14 import (
    _application_probe, _close_transition, _sample_family,
)


class StrategicIntegrationTests(unittest.TestCase):
    def test_family_sampling_uses_only_external_rng(self):
        distribution = np.array([0, 0.1, 0.7, 0, 0.2, 0], dtype=float)
        self.assertEqual(_sample_family(distribution, random.Random(3), False), 2)
        first = [_sample_family(distribution, random.Random(seed), True)
                 for seed in range(20)]
        second = [_sample_family(distribution, random.Random(seed), True)
                  for seed in range(20)]
        self.assertEqual(first, second)
        self.assertTrue(all(distribution[index] > 0 for index in first))

    def test_macro_reward_and_bootstrap_at_actual_next_gate(self):
        mask = (True, False, False, False, False, True)
        pending = {"decision_index": 0, "family_index": 0,
                   "candidate_mask": mask, "gate_log_prob": log(.7),
                   "value": .25, "start_policy_step": 4,
                   "rewards": [.1, .2, .3],
                   "vp_rewards": [.05, 0, 0],
                   "terminal_rewards": [0, 0, .3],
                   "protected_trades_crossed": 1,
                   "nondelegated_surfaces_crossed": 2}
        item = _close_transition(pending, .99, next_mask=mask,
                                 terminal=False, truncated=False,
                                 close_policy_step=7)
        self.assertEqual(item["macro_duration"], 3)
        self.assertAlmostEqual(item["reward_accumulated"], .1 + .99*.2 + .99**2*.3)
        self.assertAlmostEqual(item["discount"], .99**3)
        self.assertTrue(item["next_actual_delegated"])
        self.assertEqual(item["protected_trades_crossed"], 1)
        terminal = _close_transition(pending, .99, next_mask=None,
                                     terminal=True, truncated=False,
                                     close_policy_step=7)
        self.assertEqual(terminal["discount"], 0)
        with self.assertRaises(AssertionError):
            _close_transition(pending, .99, next_mask=mask,
                              terminal=False, truncated=False,
                              close_policy_step=6)

    def test_safety_difference_requires_resolver_at_first_difference(self):
        action = EndTurnAction()
        win = BuildCityAction(0)
        legacy = SimpleNamespace(action_trace=((1, action), (1, action)),
                                 winner=2, final_vp=9, final_rank=2,
                                 final_state="legacy", rng_counter=1,
                                 policy_steps=2, terminal=True, truncated=False)
        safety = SimpleNamespace(action_trace=((1, action), (1, win)),
                                 winner=1, learner_id=1, final_vp=10, final_rank=1,
                                 final_state="safety", rng_counter=1,
                                 policy_steps=2, terminal=True, truncated=False,
                                 resolver_events=({"changed_action": True,
                                                   "action_trace_index": 1,
                                                   "immediate_terminal": True,
                                                   "champion_counterfactual_action": repr(action),
                                                   "resolver_action": repr(win)},))
        self.assertTrue(_safety_difference(legacy, safety)["changed"])
        safety.resolver_events = ()
        with self.assertRaises(AssertionError):
            _safety_difference(legacy, safety)

    def test_bank_probe_respects_port_ratio_and_empty_bank_stock(self):
        game = create_game(20261401)
        game.phase = "action"
        game.current_player_id = 1
        port = next(port for port in game.board.ports
                    if port.ratio == 2 and port.resource is not None)
        vertex = game.board.edges[port.edge_id].vertex_ids[0]
        game.settlements[vertex] = 1
        player_for(game, 1).settlements = 1
        player_for(game, 1).resources[port.resource] = 4
        receive = next(resource for resource in ("wood", "brick", "sheep", "wheat", "ore")
                       if resource != port.resource)
        unavailable = next(resource for resource in ("wood", "brick", "sheep", "wheat", "ore")
                           if resource not in {port.resource, receive})
        game.bank[unavailable] = 0
        mask = get_action_mask(game, 1)
        self.assertFalse(mask[action_to_id(BankTradeAction(port.resource,
                                                           unavailable))])
        action = BankTradeAction(port.resource, receive)
        self.assertTrue(mask[action_to_id(action)])
        probe = _application_probe(game, 1, action)
        self.assertEqual(probe["bank_ratio"], 2)
        self.assertEqual(probe["resource_before"][port.resource]
                         - probe["resource_after"][port.resource], 2)
        self.assertEqual(probe["resource_after"][receive]
                         - probe["resource_before"][receive], 1)

    def test_road_probe_records_opponent_endpoint_blockage(self):
        game = create_game(20261402)
        game.phase = "action"
        game.current_player_id = 1
        edge = game.board.edges[0]
        near, far = edge.vertex_ids
        adjacent = next(edge_id for edge_id in game.board.vertices[near].edge_ids
                        if edge_id != edge.id)
        game.roads[adjacent] = 1
        player_for(game, 1).roads = 1
        game.settlements[far] = 2
        player_for(game, 2).settlements = 1
        player_for(game, 1).resources.update({"wood": 1, "brick": 1})
        action = BuildRoadAction(edge.id)
        self.assertTrue(get_action_mask(game, 1)[action_to_id(action)])
        probe = _application_probe(game, 1, action)
        self.assertTrue(probe["road_endpoint_opponent_building"])
        self.assertEqual(probe["road_owner_after"], 1)
        self.assertEqual(probe["resource_before"]["wood"]
                         - probe["resource_after"]["wood"], 1)


if __name__ == "__main__":
    unittest.main()
