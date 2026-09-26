import unittest

from app.domain.game import create_game, player_for
from app.rl.action_space import ACTION_CATALOG, get_action_mask
from app.rl.expansion_planner import player_production, vertex_production
from app.rl.setup_planner import expansion_coverage_mask
from app.domain.actions import PlaceInitialSettlementAction


class SetupPlannerTests(unittest.TestCase):
    def test_second_settlement_keeps_only_wood_brick_coverage_when_available(self):
        game = create_game(20260924)
        game.phase = "setup_ready"
        game.setup_order = [1]
        game.setup_index = 0
        game.settlements[0] = 1
        player_for(game, 1).settlements = 1
        mask = get_action_mask(game, 1)
        filtered = expansion_coverage_mask(game, 1, mask)
        current = player_production(game, 1)
        kept = [item.action.vertex_id for item in ACTION_CATALOG
                if filtered[item.id] and isinstance(item.action, PlaceInitialSettlementAction)]

        self.assertTrue(kept)
        for vertex_id in kept:
            candidate = vertex_production(game, vertex_id)
            self.assertGreater(current["wood"] + candidate["wood"], 0)
            self.assertGreater(current["brick"] + candidate["brick"], 0)

    def test_first_settlement_mask_is_unchanged(self):
        game = create_game(20260925)
        game.phase = "setup_ready"
        game.setup_order = [1]
        mask = get_action_mask(game, 1)
        self.assertTrue((expansion_coverage_mask(game, 1, mask) == mask).all())

    def test_infeasible_threshold_can_fallback_to_best_coverage(self):
        game = create_game(20260924)
        game.phase = "setup_ready"
        game.setup_order = [1]
        game.setup_index = 0
        game.settlements[0] = 1
        player_for(game, 1).settlements = 1
        mask = get_action_mask(game, 1)
        current = player_production(game, 1)
        filtered = expansion_coverage_mask(
            game, 1, mask, min_pips=5, fallback_to_best_coverage=True,
        )
        kept = [item.action.vertex_id for item in ACTION_CATALOG
                if filtered[item.id] and isinstance(item.action, PlaceInitialSettlementAction)]
        legal = [item.action.vertex_id for item in ACTION_CATALOG
                 if mask[item.id] and isinstance(item.action, PlaceInitialSettlementAction)]
        coverage = lambda vertex_id: min(
            current["wood"] + vertex_production(game, vertex_id)["wood"],
            current["brick"] + vertex_production(game, vertex_id)["brick"],
        )

        self.assertTrue(kept)
        self.assertEqual({coverage(vertex_id) for vertex_id in kept},
                         {max(map(coverage, legal))})

    def test_wheat_requirement_only_filters_expansion_candidates(self):
        selected = None
        for first_vertex_id in range(54):
            game = create_game(20260924)
            game.phase = "setup_ready"
            game.setup_order = [1]
            game.setup_index = 0
            game.settlements[first_vertex_id] = 1
            player_for(game, 1).settlements = 1
            mask = get_action_mask(game, 1)
            current = player_production(game, 1)
            has_candidate = any(
                mask[item.id]
                and isinstance(item.action, PlaceInitialSettlementAction)
                and current["wood"] + vertex_production(game, item.action.vertex_id)["wood"] >= 1
                and current["brick"] + vertex_production(game, item.action.vertex_id)["brick"] >= 1
                and current["wheat"] + vertex_production(game, item.action.vertex_id)["wheat"] >= 1
                for item in ACTION_CATALOG
            )
            if has_candidate:
                selected = game, mask, current
                break
        self.assertIsNotNone(selected)
        game, mask, current = selected
        filtered = expansion_coverage_mask(
            game, 1, mask, min_pips=1, min_wheat_pips=1,
        )
        kept = [item.action.vertex_id for item in ACTION_CATALOG
                if filtered[item.id] and isinstance(item.action, PlaceInitialSettlementAction)]

        self.assertTrue(kept)
        for vertex_id in kept:
            candidate = vertex_production(game, vertex_id)
            self.assertGreaterEqual(current["wood"] + candidate["wood"], 1)
            self.assertGreaterEqual(current["brick"] + candidate["brick"], 1)
            self.assertGreaterEqual(current["wheat"] + candidate["wheat"], 1)

    def test_ore_requirement_refines_wheat_candidates_when_available(self):
        selected = None
        for first_vertex_id in range(54):
            game = create_game(20260924)
            game.phase = "setup_ready"
            game.setup_order = [1]
            game.setup_index = 0
            game.settlements[first_vertex_id] = 1
            player_for(game, 1).settlements = 1
            mask = get_action_mask(game, 1)
            current = player_production(game, 1)
            has_candidate = any(
                mask[item.id]
                and isinstance(item.action, PlaceInitialSettlementAction)
                and current["wood"] + vertex_production(game, item.action.vertex_id)["wood"] >= 1
                and current["brick"] + vertex_production(game, item.action.vertex_id)["brick"] >= 1
                and current["wheat"] + vertex_production(game, item.action.vertex_id)["wheat"] >= 1
                and current["ore"] + vertex_production(game, item.action.vertex_id)["ore"] >= 1
                for item in ACTION_CATALOG
            )
            if has_candidate:
                selected = game, mask, current
                break
        self.assertIsNotNone(selected)
        game, mask, current = selected
        filtered = expansion_coverage_mask(
            game, 1, mask, min_pips=1, min_wheat_pips=1, min_ore_pips=1,
        )
        kept = [item.action.vertex_id for item in ACTION_CATALOG
                if filtered[item.id] and isinstance(item.action, PlaceInitialSettlementAction)]

        self.assertTrue(kept)
        for vertex_id in kept:
            candidate = vertex_production(game, vertex_id)
            self.assertGreaterEqual(current["wood"] + candidate["wood"], 1)
            self.assertGreaterEqual(current["brick"] + candidate["brick"], 1)
            self.assertGreaterEqual(current["wheat"] + candidate["wheat"], 1)
            self.assertGreaterEqual(current["ore"] + candidate["ore"], 1)


if __name__ == "__main__":
    unittest.main()
