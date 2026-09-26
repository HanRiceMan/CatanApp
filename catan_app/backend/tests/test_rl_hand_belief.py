import json
import unittest

from app.domain.game import create_game, game_view
from app.rl.hand_belief import BayesianHandEstimator


class BayesianHandEstimatorTests(unittest.TestCase):
    def test_known_public_events_reconstruct_hand(self):
        game = create_game(11)
        game.resource_events = [
            {"deltas": {"2": {"wood": 2, "brick": 1}}},
            {"deltas": {"2": {"wood": -1, "brick": -1}}},
            {"deltas": {"2": {"ore": 1}}},
        ]

        belief = BayesianHandEstimator().estimate(game, 1)[2]

        self.assertEqual(belief.expected("wood"), 1)
        self.assertEqual(belief.expected("brick"), 0)
        self.assertEqual(belief.expected("ore"), 1)
        self.assertEqual(len(belief.distribution), 1)

    def test_hidden_steal_does_not_leak_private_resource_to_third_party(self):
        def belief(private_resource):
            game = create_game(12)
            game.resource_events = [
                {"deltas": {"2": {"wood": 1, "ore": 1}}},
                {"hidden_transfer": {
                    "from_player_id": 2, "to_player_id": 1, "count": 1,
                    "private_resource": private_resource, "known_to": [1, 2],
                }},
            ]
            return BayesianHandEstimator().estimate(game, 3)

        wood = belief("wood")
        ore = belief("ore")
        self.assertEqual(wood[1].distribution, ore[1].distribution)
        self.assertEqual(wood[2].distribution, ore[2].distribution)
        self.assertAlmostEqual(wood[1].probability_has("wood"), 0.5)
        self.assertAlmostEqual(wood[1].probability_has("ore"), 0.5)

    def test_public_spending_conditions_uncertain_hand(self):
        game = create_game(13)
        game.resource_events = [
            {"deltas": {"1": {"wood": 1, "brick": 1}, "2": {"wood": 1}}},
            {"hidden_transfer": {
                "from_player_id": 1, "to_player_id": 2, "count": 1,
                "private_resource": "wood", "known_to": [1, 2],
            }},
            {"deltas": {"2": {"wood": -1, "brick": -1}}},
        ]

        belief = BayesianHandEstimator().estimate(game, 3)[2]

        self.assertEqual(len(belief.distribution), 1)
        self.assertEqual(belief.expected_total, 0)

    def test_joint_distribution_propagates_evidence_between_players(self):
        game = create_game(16)
        game.resource_events = [
            {"deltas": {"1": {"wood": 1, "ore": 1}}},
            {"hidden_transfer": {
                "from_player_id": 1, "to_player_id": 2, "count": 1,
                "private_resource": "wood", "known_to": [1, 2],
            }},
            # 第三者には盗まれた種類が不明だが、その後P1が木を使えたことで
            # 盗まれたのは鉱石、P2が受け取ったのも鉱石だと確定する。
            {"deltas": {"1": {"wood": -1}}},
        ]

        belief = BayesianHandEstimator().estimate(game, 3)

        self.assertEqual(belief[1].expected_total, 0)
        self.assertEqual(belief[2].expected("ore"), 1)
        self.assertEqual(belief[2].probability_has("wood"), 0)

    def test_multiple_hidden_steals_accumulate_and_later_evidence_resolves_all(self):
        game = create_game(18)
        game.resource_events = [
            {"deltas": {"1": {"wood": 2, "ore": 2}}},
            {"hidden_transfer": {
                "from_player_id": 1, "to_player_id": 2, "count": 1,
                "private_resource": "wood", "known_to": [1, 2],
            }},
            {"hidden_transfer": {
                "from_player_id": 1, "to_player_id": 3, "count": 1,
                "private_resource": "ore", "known_to": [1, 3],
            }},
            # P1が後から木2枚を公開支出できたため、第三者視点でも過去2回で
            # 盗まれたのは両方とも鉱石だったと確定する。
            {"deltas": {"1": {"wood": -2}}},
        ]

        belief = BayesianHandEstimator().estimate(game, 4)

        self.assertEqual(belief[2].expected("ore"), 1)
        self.assertEqual(belief[3].expected("ore"), 1)
        self.assertEqual(belief[2].probability_has("wood"), 0)
        self.assertEqual(belief[3].probability_has("wood"), 0)

    def test_public_bank_and_hand_totals_condition_legacy_unknown_loss(self):
        game = create_game(17)
        game.resource_events = [
            {"deltas": {"1": {"wood": 1, "ore": 1}}},
            {"unknown_loss": {"player_id": 1, "count": 1}},
        ]
        game.bank["ore"] = 18
        game.players[0].resources["ore"] = 1

        belief = BayesianHandEstimator().estimate(game, 2)[1]

        self.assertEqual(belief.expected("ore"), 1)
        self.assertEqual(belief.probability_has("wood"), 0)

    def test_private_resource_events_are_not_exposed_by_game_view(self):
        game = create_game(14)
        game.resource_events = [{
            "kind": "robber_steal",
            "hidden_transfer": {
                "from_player_id": 2,
                "to_player_id": 1,
                "count": 1,
                "private_resource": "ore",
                "known_to": [1, 2],
            },
        }]

        view = game_view(game, 3)
        serialized = json.dumps(view)

        self.assertNotIn("resource_events", view)
        self.assertNotIn("private_resource", serialized)

    def test_public_diagnostic_view_does_not_gain_party_only_knowledge(self):
        def public_belief(private_resource):
            game = create_game(15)
            game.resource_events = [
                {"deltas": {"2": {"wood": 1, "ore": 1}}},
                {"hidden_transfer": {
                    "from_player_id": 2,
                    "to_player_id": 1,
                    "count": 1,
                    "private_resource": private_resource,
                    "known_to": [1, 2],
                }},
            ]
            return BayesianHandEstimator().estimate(game, None)

        wood = public_belief("wood")
        ore = public_belief("ore")
        self.assertEqual(wood[1].distribution, ore[1].distribution)
        self.assertEqual(wood[2].distribution, ore[2].distribution)


if __name__ == "__main__":
    unittest.main()
