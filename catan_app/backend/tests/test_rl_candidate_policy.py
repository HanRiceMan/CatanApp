import numpy as np
import torch
import unittest

from sb3_contrib import MaskablePPO

from app.domain.actions import BankTradeAction, BuildRoadAction, EndTurnAction
from app.rl.action_space import ACTION_SPACE_SIZE, action_to_id
from app.rl.candidate_policy import (CandidateActorCritic, CandidateMaskablePolicy,
                                     ExpansionGraphHierarchicalCandidateMaskablePolicy,
                                     GraphFamilyHierarchicalCandidateMaskablePolicy,
                                     GraphHierarchicalCandidateMaskablePolicy,
                                     HeteroGraphHierarchicalCandidateMaskablePolicy,
                                     HierarchicalCandidateMaskablePolicy,
                                     SPATIAL_BLOCKS, VERTEX_START,
                                     configure_graph_candidate_pretrain,
                                     configure_expansion_graph_finetune,
                                     configure_hetero_graph_finetune,
                                     configure_graph_residual_finetune,
                                     initialize_graph_policy_from_hierarchical,
                                     initialize_expansion_policy_from_graph,
                                     initialize_hetero_policy_from_graph)
from app.rl.compare import PolicyCompatibleHeuristicAgent
from app.rl.env import CatanEnv, CatanEnvConfig
from app.rl.hierarchical_distribution import (ACTION_FAMILY_IDS, FAMILY_COUNT,
                                              HierarchicalMaskableCategoricalDistribution)
from app.rl.gnn_setup_policy import train_graph_initial_setup_policy
from app.rl.initial_setup_policy import (train_initial_setup_policy,
                                         train_initial_setup_policy_dagger,
                                         train_ranked_initial_setup_policy)
from app.rl.observation import OBSERVATION_VECTOR_SIZES
from app.rl.pretrain_candidate_setup import (INITIAL_START, pretrain_initial_settlement,
                                             pretrain_initial_setup_heads)
from app.rl.policy_selection import choose_action_id


class CandidatePolicyTests(unittest.TestCase):
    def test_hierarchical_distribution_normalizes_legal_families_and_candidates(self):
        distribution = HierarchicalMaskableCategoricalDistribution()
        logits = torch.zeros((1, ACTION_SPACE_SIZE + FAMILY_COUNT))
        mask = np.zeros((1, ACTION_SPACE_SIZE), dtype=np.bool_)
        roads = [action_to_id(BuildRoadAction(edge)) for edge in range(3)]
        end_turn = action_to_id(EndTurnAction())
        mask[0, roads + [end_turn]] = True
        road_family = ACTION_FAMILY_IDS[roads[0]]
        end_family = ACTION_FAMILY_IDS[end_turn]
        logits[0, ACTION_SPACE_SIZE + road_family] = 0.0
        logits[0, ACTION_SPACE_SIZE + end_family] = -0.2
        distribution.proba_distribution(logits)
        distribution.apply_masking(mask)
        probabilities = distribution.distribution.probs[0]
        expected_road_mass = torch.softmax(torch.tensor([0.0, -0.2]), dim=0)[0]
        self.assertTrue(torch.allclose(probabilities[roads].sum(), expected_road_mass))
        self.assertTrue(torch.allclose(probabilities[roads],
                                       torch.full((3,), expected_road_mass / 3)))
        self.assertEqual(int(distribution.mode().item()), roads[0])
        self.assertEqual(float(probabilities[~torch.from_numpy(mask[0])].sum()), 0.0)
        self.assertTrue(torch.allclose(distribution.entropy(),
                                       -(probabilities[probabilities > 0]
                                         * probabilities[probabilities > 0].log()).sum().reshape(1)))

    def test_family_selection_compares_mass_and_candidate_count_without_illegal_actions(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        probabilities = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
        roads = [action_to_id(BuildRoadAction(edge)) for edge in range(3)]
        end_turn = action_to_id(EndTurnAction())
        trade = action_to_id(BankTradeAction("wood", "brick"))
        for action_id in roads:
            mask[action_id] = True
            probabilities[action_id] = .2
        mask[end_turn] = mask[trade] = True
        probabilities[end_turn] = .35
        probabilities[trade] = .05
        probabilities[action_to_id(BuildRoadAction(3))] = 1.0  # Mask外は無視する。

        self.assertEqual(choose_action_id(probabilities, mask, "global_argmax"), end_turn)
        self.assertEqual(choose_action_id(probabilities, mask, "family_mass"), roads[0])
        self.assertEqual(choose_action_id(probabilities, mask, "family_l2"), end_turn)
        with self.assertRaises(ValueError):
            choose_action_id(probabilities, mask, "unknown")

    def test_candidate_actor_outputs_action_ids_and_shared_vertex_scores(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        observation, _ = env.reset(seed=37001)
        actor = CandidateActorCritic(OBSERVATION_VECTOR_SIZES["v2"])
        actor.eval()
        original = torch.from_numpy(observation[None, :].copy())
        swapped = original.clone()
        first = swapped[:, VERTEX_START:VERTEX_START + 14].clone()
        swapped[:, VERTEX_START:VERTEX_START + 14] = swapped[:, VERTEX_START + 14:VERTEX_START + 28]
        swapped[:, VERTEX_START + 14:VERTEX_START + 28] = first
        with torch.no_grad():
            logits = actor.forward_actor(original)
            swapped_logits = actor.forward_actor(swapped)
        initial_start, initial_count = SPATIAL_BLOCKS[0]
        assert logits.shape == (1, ACTION_SPACE_SIZE)
        assert initial_count == 54
        assert torch.allclose(logits[0, initial_start], swapped_logits[0, initial_start + 1])
        assert torch.allclose(logits[0, initial_start + 1], swapped_logits[0, initial_start])
        actor.train()
        actor.forward_actor(original)[:, initial_start:initial_start + initial_count].sum().backward()
        assert actor.vertex_encoder[0].weight.grad is not None
        assert actor.vertex_encoder[0].weight.grad.abs().sum() > 0
        env.close()


    def test_candidate_maskable_policy_predicts_legal_action_and_learns_one_rollout(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        model = MaskablePPO(CandidateMaskablePolicy, env, n_steps=32, batch_size=16,
                            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
                            seed=11, device="cpu", verbose=0)
        observation, info = env.reset(seed=37002)
        action, _ = model.predict(observation, deterministic=True, action_masks=info["action_mask"])
        assert info["action_mask"][int(np.asarray(action).item())]
        model.learn(total_timesteps=32)
        assert model.num_timesteps >= 32
        env.close()

    def test_hierarchical_candidate_policy_predicts_legal_action_and_learns_one_rollout(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        model = MaskablePPO(HierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
                            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
                            seed=13, device="cpu", verbose=0)
        observation, info = env.reset(seed=37003)
        action, _ = model.predict(observation, deterministic=True, action_masks=info["action_mask"])
        assert info["action_mask"][int(np.asarray(action).item())]
        model.learn(total_timesteps=32)
        assert model.num_timesteps >= 32
        env.close()

    def test_graph_policy_starts_from_identical_logits_and_learns_one_rollout(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        source = MaskablePPO(
            HierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
            seed=19, device="cpu", verbose=0,
        )
        graph = MaskablePPO(
            GraphHierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
            seed=20, device="cpu", verbose=0,
        )
        missing = initialize_graph_policy_from_hierarchical(graph.policy, source.policy)
        self.assertTrue(any("graph_layers" in key for key in missing))
        observation, info = env.reset(seed=37004)
        tensor = torch.from_numpy(observation[None, :])
        source.policy.eval()
        graph.policy.eval()
        with torch.no_grad():
            source_logits = source.policy.mlp_extractor.forward_actor(tensor)
            graph_logits = graph.policy.mlp_extractor.forward_actor(tensor)
        self.assertTrue(torch.equal(source_logits, graph_logits))
        action, _ = graph.predict(observation, deterministic=True,
                                  action_masks=info["action_mask"])
        self.assertTrue(info["action_mask"][int(np.asarray(action).item())])
        graph.learn(total_timesteps=32)
        self.assertGreater(
            sum(parameter.detach().abs().sum().item()
                for name, parameter in graph.policy.named_parameters()
                if "graph_" in name and "_head" in name),
            0,
        )
        env.close()

    def test_graph_residual_finetune_freezes_existing_actor(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        model = MaskablePPO(
            GraphHierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
            seed=21, device="cpu", verbose=0,
        )
        report = configure_graph_residual_finetune(model.policy)
        self.assertGreater(report["trainable_parameters"], 0)
        self.assertGreater(report["frozen_parameters"], 0)
        parameters = dict(model.policy.named_parameters())
        self.assertFalse(parameters["mlp_extractor.family_head.0.weight"].requires_grad)
        self.assertFalse(parameters["mlp_extractor.vertex_encoder.0.weight"].requires_grad)
        self.assertTrue(parameters["mlp_extractor.graph_layers.0.update.0.weight"].requires_grad)
        self.assertTrue(parameters["mlp_extractor.graph_road_head.2.weight"].requires_grad)
        self.assertTrue(parameters["mlp_extractor.critic.0.weight"].requires_grad)
        self.assertTrue(parameters["value_net.weight"].requires_grad)
        env.close()

    def test_graph_candidate_pretrain_keeps_family_and_base_candidate_heads_frozen(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        model = MaskablePPO(
            GraphHierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
            seed=24, device="cpu", verbose=0,
        )
        report = configure_graph_candidate_pretrain(model.policy)
        self.assertGreater(report["trainable_parameters"], 0)
        self.assertGreater(report["frozen_parameters"], 0)
        parameters = dict(model.policy.named_parameters())
        self.assertFalse(parameters["mlp_extractor.family_head.0.weight"].requires_grad)
        self.assertFalse(parameters["mlp_extractor.road_head.weight"].requires_grad)
        self.assertFalse(parameters["mlp_extractor.graph_initial_road_head.0.weight"].requires_grad)
        self.assertTrue(parameters["mlp_extractor.graph_layers.0.update.0.weight"].requires_grad)
        self.assertTrue(parameters["mlp_extractor.graph_road_head.0.weight"].requires_grad)
        self.assertTrue(parameters["mlp_extractor.graph_city_head.weight"].requires_grad)
        self.assertFalse(parameters["mlp_extractor.critic.0.weight"].requires_grad)
        env.close()

    def test_hetero_graph_v2_starts_identically_and_only_trains_new_residual(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        source = MaskablePPO(
            GraphHierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
            seed=25, device="cpu", verbose=0,
        )
        hetero = MaskablePPO(
            HeteroGraphHierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
            seed=26, device="cpu", verbose=0,
        )
        missing = initialize_hetero_policy_from_graph(hetero.policy, source.policy)
        self.assertTrue(any("hetero_layers" in key for key in missing))
        observation, info = env.reset(seed=37006)
        tensor = torch.from_numpy(observation[None, :])
        source.policy.eval()
        hetero.policy.eval()
        with torch.no_grad():
            source_logits = source.policy.mlp_extractor.forward_actor(tensor)
            hetero_logits = hetero.policy.mlp_extractor.forward_actor(tensor)
        self.assertTrue(torch.equal(source_logits, hetero_logits))
        action, _ = hetero.predict(observation, deterministic=True,
                                   action_masks=info["action_mask"])
        self.assertTrue(info["action_mask"][int(np.asarray(action).item())])
        report = configure_hetero_graph_finetune(hetero.policy)
        self.assertGreater(report["trainable_parameters"], 0)
        parameters = dict(hetero.policy.named_parameters())
        self.assertFalse(parameters["mlp_extractor.graph_road_head.0.weight"].requires_grad)
        self.assertFalse(parameters["mlp_extractor.family_head.0.weight"].requires_grad)
        self.assertTrue(parameters["mlp_extractor.hetero_layers.0.vertex_update.0.weight"].requires_grad)
        self.assertTrue(parameters["mlp_extractor.hetero_road_head.0.weight"].requires_grad)
        self.assertTrue(parameters["value_net.weight"].requires_grad)
        env.close()

    def test_expansion_graph_starts_identically_and_only_trains_strategy_residual(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        source = MaskablePPO(
            GraphHierarchicalCandidateMaskablePolicy, env, n_steps=32,
            batch_size=16, n_epochs=1,
            policy_kwargs={"net_arch": [], "ortho_init": False},
            seed=27, device="cpu", verbose=0,
        )
        expansion = MaskablePPO(
            ExpansionGraphHierarchicalCandidateMaskablePolicy, env, n_steps=32,
            batch_size=16, n_epochs=1,
            policy_kwargs={"net_arch": [], "ortho_init": False},
            seed=28, device="cpu", verbose=0,
        )
        missing = initialize_expansion_policy_from_graph(
            expansion.policy, source.policy
        )
        self.assertTrue(any("expansion_family_head" in key for key in missing))
        observation, info = env.reset(seed=37007)
        tensor = torch.from_numpy(observation[None, :])
        source.policy.eval()
        expansion.policy.eval()
        with torch.no_grad():
            source_logits = source.policy.mlp_extractor.forward_actor(tensor)
            expansion_logits = expansion.policy.mlp_extractor.forward_actor(tensor)
        self.assertTrue(torch.equal(source_logits, expansion_logits))
        action, _ = expansion.predict(
            observation, deterministic=True, action_masks=info["action_mask"]
        )
        self.assertTrue(info["action_mask"][int(np.asarray(action).item())])
        report = configure_expansion_graph_finetune(expansion.policy)
        self.assertGreater(report["trainable_parameters"], 0)
        parameters = dict(expansion.policy.named_parameters())
        self.assertFalse(parameters["mlp_extractor.graph_road_head.0.weight"].requires_grad)
        self.assertTrue(parameters["mlp_extractor.expansion_road_head.weight"].requires_grad)
        self.assertTrue(parameters["mlp_extractor.expansion_family_head.0.weight"].requires_grad)
        self.assertTrue(parameters["value_net.weight"].requires_grad)
        env.close()

    def test_graph_family_policy_starts_identically_and_trains_family_residual(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        source = MaskablePPO(
            HierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
            seed=22, device="cpu", verbose=0,
        )
        graph = MaskablePPO(
            GraphFamilyHierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
            seed=23, device="cpu", verbose=0,
        )
        missing = initialize_graph_policy_from_hierarchical(graph.policy, source.policy)
        self.assertTrue(any("graph_family_head" in key for key in missing))
        observation, _ = env.reset(seed=37005)
        tensor = torch.from_numpy(observation[None, :])
        source.policy.eval()
        graph.policy.eval()
        with torch.no_grad():
            source_logits = source.policy.mlp_extractor.forward_actor(tensor)
            graph_logits = graph.policy.mlp_extractor.forward_actor(tensor)
        self.assertTrue(torch.equal(source_logits, graph_logits))
        configure_graph_residual_finetune(graph.policy)
        parameters = dict(graph.policy.named_parameters())
        self.assertFalse(parameters["mlp_extractor.family_head.0.weight"].requires_grad)
        self.assertTrue(parameters["mlp_extractor.graph_family_head.2.weight"].requires_grad)
        env.close()


    def test_candidate_policy_setup_pretraining_runs_without_changing_action_space(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        model = MaskablePPO(CandidateMaskablePolicy, env, n_steps=32, batch_size=16,
                            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
                            seed=12, device="cpu", verbose=0)
        report = pretrain_initial_settlement(model.policy, train_boards=8,
                                             validation_boards=4, test_boards=4,
                                             seed_start=37011, max_epochs=2)
        assert report["after_test"]["examples"] == 8
        assert 0 <= report["after_test"]["teacher_agreement"] <= 1
        assert report["initial_test"]["mean_pips_regret"] >= 0
        assert report["after_test"]["mean_pips_regret"] >= 0
        observation, info = env.reset(seed=37020)
        action, _ = model.predict(observation, deterministic=True, action_masks=info["action_mask"])
        assert info["action_mask"][int(np.asarray(action).item())]
        env.close()

    def test_hierarchical_setup_pretraining_only_changes_two_setup_heads(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        model = MaskablePPO(HierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
                            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
                            seed=14, device="cpu", verbose=0)
        before = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
        report = pretrain_initial_setup_heads(model.policy, train_boards=8,
                                              validation_boards=4, test_boards=4,
                                              seed_start=37031, max_epochs=2)
        changed = []
        for name, value in model.policy.state_dict().items():
            if not torch.equal(before[name], value):
                changed.append(name)
                self.assertTrue(name.startswith(("mlp_extractor.initial_settlement_head",
                                                 "mlp_extractor.initial_road_head")), name)
        self.assertTrue(changed)
        self.assertEqual(report["after_test"]["settlement"]["examples"], 8)
        self.assertEqual(report["after_test"]["road"]["examples"], 8)
        self.assertTrue(all(parameter.requires_grad for parameter in model.policy.parameters()))
        observation, info = env.reset(seed=37040)
        action, _ = model.predict(observation, deterministic=True, action_masks=info["action_mask"])
        self.assertTrue(info["action_mask"][int(np.asarray(action).item())])
        env.close()

    def test_independent_setup_policy_does_not_change_ppo_and_selects_legal_action(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        model = MaskablePPO(HierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
                            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
                            seed=15, device="cpu", verbose=0)
        before = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
        setup_policy, report = train_initial_setup_policy(
            model.policy, train_boards=8, validation_boards=4, test_boards=4,
            seed_start=37051, max_epochs=2)
        for name, value in model.policy.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertEqual(report["after_test"]["settlement"]["examples"], 8)
        self.assertEqual(report["after_test"]["road"]["examples"], 8)
        observation, info = env.reset(seed=37070)
        teacher = PolicyCompatibleHeuristicAgent()
        for _ in range(8):
            if info["action_mask"][INITIAL_START:INITIAL_START + 54].any():
                break
            action = teacher.select_action(env.game, env.config.learning_player_id)
            observation, _, _, _, info = env.step(action_to_id(action))
        action_id = setup_policy.select_action_id(observation, info["action_mask"])
        self.assertTrue(info["action_mask"][action_id])
        env.close()

    def test_dagger_setup_policy_collects_its_own_states_without_changing_ppo(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        model = MaskablePPO(HierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
                            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
                            seed=16, device="cpu", verbose=0)
        before = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
        _, report = train_initial_setup_policy_dagger(
            model.policy, train_boards=8, validation_boards=4, test_boards=4,
            seed_start=37101, training_seed=16, dagger_rounds=1, dagger_boards=4,
            dagger_validation_boards=2, dagger_seed_start=37121, dagger_test_boards=2)
        for name, value in model.policy.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertEqual(len(report["rounds"]), 1)
        self.assertEqual(report["rounds"][0]["training_boards_total"], 12)
        self.assertEqual(report["on_policy_test"]["settlement"]["examples"], 4)
        self.assertEqual(report["on_policy_test"]["road"]["examples"], 4)
        env.close()

    def test_ranked_setup_policy_learns_all_candidate_values_without_changing_ppo(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        model = MaskablePPO(HierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
                            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
                            seed=17, device="cpu", verbose=0)
        before = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
        setup_policy, report = train_ranked_initial_setup_policy(
            model.policy, train_boards=8, validation_boards=4, test_boards=4,
            seed_start=37151, training_seed=17, max_epochs=2)
        for name, value in model.policy.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        for kind in ("settlement", "road"):
            self.assertEqual(report["after_test"][kind]["examples"], 8)
            self.assertGreaterEqual(report["after_test"][kind]["optimal_value_rate"], 0)
            self.assertGreaterEqual(report["after_test"][kind]["mean_value_regret"], 0)
        observation, info = env.reset(seed=37170)
        teacher = PolicyCompatibleHeuristicAgent()
        for _ in range(8):
            if info["action_mask"][INITIAL_START:INITIAL_START + 54].any():
                break
            action = teacher.select_action(env.game, env.config.learning_player_id)
            observation, _, _, _, info = env.step(action_to_id(action))
        action_id = setup_policy.select_action_id(observation, info["action_mask"])
        self.assertTrue(info["action_mask"][action_id])
        env.close()

    def test_graph_setup_policy_uses_legal_graph_candidates_without_changing_ppo(self):
        env = CatanEnv(CatanEnvConfig(observation_version="v2"))
        model = MaskablePPO(HierarchicalCandidateMaskablePolicy, env, n_steps=32, batch_size=16,
                            n_epochs=1, policy_kwargs={"net_arch": [], "ortho_init": False},
                            seed=18, device="cpu", verbose=0)
        before = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
        graph, report = train_graph_initial_setup_policy(
            model.policy, train_boards=8, validation_boards=4, test_boards=4,
            train_seed_start=37201, validation_seed_start=37211,
            test_seed_start=37221, training_seed=18, max_epochs=2)
        for name, value in model.policy.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertEqual(report["after_test"]["settlement"]["examples"], 8)
        observation, info = env.reset(seed=37231)
        teacher = PolicyCompatibleHeuristicAgent()
        for _ in range(8):
            if info["action_mask"][INITIAL_START:INITIAL_START + 54].any():
                break
            action = teacher.select_action(env.game, env.config.learning_player_id)
            observation, _, _, _, info = env.step(action_to_id(action))
        action_id = graph.select_action_id(observation, info["action_mask"])
        self.assertTrue(info["action_mask"][action_id])
        env.close()
