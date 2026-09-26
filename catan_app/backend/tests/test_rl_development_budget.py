from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from app.domain.actions import BuyDevelopmentAction, BuildSettlementAction, EndTurnAction
from app.domain.game import create_game, player_for, apply_action, legal_settlement_ids
from app.rl.action_space import action_to_id, get_action_mask
from app.rl.candidate_policy import ExpansionGraphHierarchicalCandidateMaskablePolicy
from app.rl.development_budget import DevelopmentBudgetFeatures, budget_wait, FEATURE_SIZE
from app.rl.development_policy import (DevelopmentBudgetMaskablePolicy, DEVELOPMENT_LOGIT,
                                      initialize_development_policy, configure_development_only)
from app.rl.diagnostics_metrics import EvaluationTelemetry
from app.rl.env import CatanEnv, CatanEnvConfig
from app.rl.observation import get_observation, encode_observation


def scenario():
    game = create_game(241001)
    game.phase = "action"
    game.current_player_id = 1
    game.turn_number = 9
    game.roads[30] = 1
    player = player_for(game, 1)
    player.settlements = 2
    player.resources.update(wood=1, brick=1, sheep=1, wheat=1, ore=1)
    return game


def encoded(game, pid=1):
    return torch.from_numpy(encode_observation(get_observation(game, pid, version="v2")))[None]


class DevelopmentBudgetTests(unittest.TestCase):
    def test_trade_stock_is_reserved_and_cannot_pay_two_deficits_twice(self):
        stock = torch.tensor([[4., 0, 0, 0, 0], [8., 0, 0, 0, 0]])
        zero = torch.zeros_like(stock)
        ratios = torch.full_like(stock, 4)
        costs = torch.tensor([[0., 0, 1, 1, 0]]).expand_as(stock)
        bank = torch.full_like(stock, 19)
        waits = budget_wait(stock, zero, ratios, costs, bank)
        self.assertEqual(waits.tolist(), [12, 0])
        # woodを4枚建設に残すなら、8枚でも2種類の不足資源は交換できない。
        costs = costs.clone()
        costs[:, 0] = 4
        self.assertTrue((budget_wait(stock, zero, ratios, costs, bank) == 12).all())

    def test_connected_affordable_settlement_reserve_and_knight_exceptions(self):
        game = scenario()
        features = DevelopmentBudgetFeatures()
        assessment = features(encoded(game))
        self.assertEqual(assessment["features"].shape, (1, FEATURE_SIZE))
        self.assertTrue(assessment["connected"].item())
        self.assertTrue(assessment["reserve"].item())
        self.assertEqual(assessment["settlement_before"].item(), 0)
        self.assertGreater(assessment["settlement_after"].item(), 0)
        player_for(game, 1).played_knights = 2
        self.assertFalse(features(encoded(game))["reserve"].item())
        # 手札に騎士があるなら、自動的な追加購入優遇は解除。
        player_for(game, 1).development_cards = ["knight"]
        self.assertTrue(features(encoded(game))["reserve"].item())

    def test_city_production_doubles_and_robber_production_is_removed(self):
        game = scenario()
        vertex = next(v for v in game.board.vertices
                      if any(game.board.tiles[t].terrain != "desert" for t in v.hex_ids))
        game.settlements[vertex.id] = 1
        features = DevelopmentBudgetFeatures()
        before = features(encoded(game))["features"][:, 5:10]
        del game.settlements[vertex.id]
        game.cities[vertex.id] = 1
        after = features(encoded(game))["features"][:, 5:10]
        torch.testing.assert_close(after, before * 2)
        game.robber_tile_id = next(t for t in vertex.hex_ids if game.board.tiles[t].terrain != "desert")
        blocked = features(encoded(game))["features"][:, 5:10]
        self.assertLess(blocked.sum(), after.sum())

    def test_hidden_opponent_resources_and_deck_order_do_not_enter_assessment(self):
        game = scenario()
        other = deepcopy(game)
        player_for(game, 2).resources.update(wood=4)
        player_for(other, 2).resources.update(ore=4)
        other.development_deck.reverse()
        a, b = DevelopmentBudgetFeatures()(encoded(game)), DevelopmentBudgetFeatures()(encoded(other))
        for key in a:
            torch.testing.assert_close(a[key], b[key])

    def test_development_head_only_and_save_reload(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        kwargs = dict(n_steps=8, batch_size=8, n_epochs=1, device="cpu",
                      policy_kwargs={"net_arch": [], "ortho_init": False})
        source = MaskablePPO(ExpansionGraphHierarchicalCandidateMaskablePolicy, env, **kwargs)
        candidate = MaskablePPO(DevelopmentBudgetMaskablePolicy, env, **kwargs)
        initialize_development_policy(candidate.policy, source.policy)
        configure_development_only(candidate.policy)
        self.assertTrue(all("development_head" in name for name, parameter
                            in candidate.policy.named_parameters() if parameter.requires_grad))
        game = scenario()
        observation = encoded(game)
        with torch.no_grad():
            a = source.policy.mlp_extractor.forward_actor(observation)
            b = candidate.policy.mlp_extractor.forward_actor(observation)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            candidate.policy.mlp_extractor.development_head[-1].bias.fill_(-2)
            b = candidate.policy.mlp_extractor.forward_actor(observation)
        keep = torch.arange(a.shape[1]) != DEVELOPMENT_LOGIT
        torch.testing.assert_close(a[:, keep], b[:, keep], rtol=0, atol=0)
        self.assertLess(b[0, DEVELOPMENT_LOGIT], a[0, DEVELOPMENT_LOGIT])
        masks = get_action_mask(game, 1)[None]
        with torch.no_grad():
            pa = source.policy.get_distribution(observation, action_masks=masks).distribution.probs.clone()
            pb = candidate.policy.get_distribution(observation, action_masks=masks).distribution.probs.clone()
        dev = action_to_id(BuyDevelopmentAction())
        nondev = torch.arange(pa.shape[1]) != dev
        torch.testing.assert_close(pa[:, nondev] / (1-pa[:, dev:dev+1]),
                                   pb[:, nondev] / (1-pb[:, dev:dev+1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "development.zip"
            candidate.save(path)
            reloaded = MaskablePPO.load(path, device="cpu")
            torch.testing.assert_close(reloaded.policy.mlp_extractor.forward_actor(observation), b)
        env.close()

    def test_zero_initialized_head_actually_learns_and_preserves_source(self):
        from app.rl.train_development_budget import train, verify_frozen
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        kwargs = dict(n_steps=8, batch_size=8, n_epochs=1, device="cpu",
                      policy_kwargs={"net_arch": [], "ortho_init": False})
        source = MaskablePPO(ExpansionGraphHierarchicalCandidateMaskablePolicy, env, **kwargs)
        candidate = MaskablePPO(DevelopmentBudgetMaskablePolicy, env, **kwargs)
        initialize_development_policy(candidate.policy, source.policy)
        observations = encoded(scenario()).repeat(8, 1)
        data = {"observations": observations, **DevelopmentBudgetFeatures()(observations)}
        report = train(candidate, [data, data, data], epochs=40, seed=410)
        self.assertGreater(report["best_epoch"], 0)
        self.assertLess(report["best_validation_mse"], report["initial_validation_mse"] * 0.8)
        self.assertLess(candidate.policy.mlp_extractor.development_correction(observations).max(), 0)
        self.assertTrue(verify_frozen(source, candidate, data)["all_source_tensors_identical"])
        env.close()

    def test_reserve_spending_and_third_site_censoring_metrics(self):
        game = scenario()
        telemetry = EvaluationTelemetry(1)
        development = BuyDevelopmentAction()
        telemetry.before_action(game, development, get_action_mask(game, 1))
        self.assertEqual(telemetry.strategy["development_settlement_deficit_increase"], 2)
        self.assertEqual(telemetry.strategy["own_turns_observed"], 1)
        self.assertFalse(telemetry.strategy.get("third_site_reached", 0))
        # 一手番中に判断回数が増えても手番数を重複計上しない。
        telemetry.before_action(game, EndTurnAction(), get_action_mask(game, 1))
        self.assertEqual(telemetry.strategy["own_turns_observed"], 1)
        game.turn_number += 4
        settlement = BuildSettlementAction(legal_settlement_ids(game, 1, initial=False)[0])
        telemetry.before_action(game, settlement, get_action_mask(game, 1))
        apply_action(game, 1, settlement, game.revision, "third-site")
        telemetry.after_action(game, 1, settlement)
        self.assertEqual(telemetry.strategy["third_site_reached"], 1)
        self.assertEqual(telemetry.strategy["third_site_own_turn"], 2)


if __name__ == "__main__":
    unittest.main()
