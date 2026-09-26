import unittest

from app.domain.actions import PlaceInitialRoadAction
from app.domain.game import create_game, legal_road_ids, player_for
from app.rl.initial_road_planner import assess_initial_road, choose_initial_road


def setup_road_game(seed: int = 1, source_vertex_id: int = 0):
    game = create_game(seed)
    game.phase = "setup_ready"
    game.setup_order = [1]
    game.setup_index = 0
    game.pending_settlement_id = source_vertex_id
    game.settlements[source_vertex_id] = 1
    player_for(game, 1).settlements = 1
    return game


class InitialRoadPlannerTests(unittest.TestCase):
    def test_avoids_third_road_that_closes_the_shared_junction(self):
        game = setup_road_game()
        contested_edge_id = 1
        far_vertex_id = 4
        game.roads[2] = 2
        game.roads[7] = 3

        assessment = assess_initial_road(game, 1, contested_edge_id)
        selected_edge_id, _ = choose_initial_road(game, 1, contested_edge_id)

        self.assertEqual(assessment["far_vertex_id"], far_vertex_id)
        self.assertEqual(assessment["opponent_road_count"], 2)
        self.assertEqual(assessment["open_exit_count"], 0)
        self.assertEqual(assessment["target_count"], 0)
        self.assertNotEqual(selected_edge_id, contested_edge_id)

    def test_opponent_settlement_redirects_road_to_open_side(self):
        game = setup_road_game()
        blocked_edge_id = 0
        blocked_target_vertex_id = 7
        before = assess_initial_road(game, 1, blocked_edge_id)

        # 画像の右下の黄色と同様に、道路の先の有望地点を赤が先に確保した状態。
        game.settlements[blocked_target_vertex_id] = 2
        after = assess_initial_road(game, 1, blocked_edge_id)
        selected_edge_id, _ = choose_initial_road(game, 1, blocked_edge_id)

        self.assertGreater(before["best_target_value"], after["best_target_value"])
        self.assertNotIn(
            blocked_target_vertex_id,
            {target["vertex_id"] for target in after["best_targets"]},
        )
        self.assertEqual(set(legal_road_ids(game, 1)), {0, 1})
        self.assertEqual(selected_edge_id, 1)

    def test_planning_agent_override_can_be_enabled(self):
        from app.rl.settlement_planning_agent import SettlementPlanningAgent

        class FixedRoadPolicy:
            initial_setup_policy = None

            @staticmethod
            def select_action(game, player_id):
                return PlaceInitialRoadAction(1)

        game = setup_road_game()
        game.roads[2] = 2
        game.roads[7] = 3
        agent = SettlementPlanningAgent(
            FixedRoadPolicy(), collision_aware_initial_roads=True,
        )

        action = agent.select_action(game, 1)

        self.assertIsInstance(action, PlaceInitialRoadAction)
        self.assertEqual(action.edge_id, 0)


if __name__ == "__main__":
    unittest.main()
