"""Step 3A-14: no-update, real-action Strategic Gate integration episodes."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite, log
import random
from typing import Any

import numpy as np
import torch

from app.agents.heuristic import HeuristicAgent
from app.domain.actions import Action, BankTradeAction, BuildRoadAction, ProposeTradeAction
from app.domain.game import (
    GameState, RESOURCES, _longest_road_length, _port_ratio, apply_action,
    create_game, legal_road_ids, legal_settlement_ids, pending_ai_player_id,
)

from .action_space import action_to_id, get_action_mask
from .construction_timing_rollout import _HeuristicRollOrder, _TradeResponseOnly, _normalized_state
from .diagnostics_metrics import endgame_rank
from .env import CatanEnv, CatanEnvConfig
from .macro_teacher_dataset import PROFILE_ROLES, _champion_id_for_seat
from .reward import actual_score
from .settlement_planning_agent import SettlementPlanningAgent
from .strategic_action_surface import FAMILIES, detect_strategic_action_surface, family_of
from .strategic_gate_step3a13 import (
    FrozenStrategicExecutors, StrategicActionGate, StrategicSurfaceCritic,
    StrategicTransition, family_priors, frozen_snapshot, resolve_immediate_win,
)
from .strategic_prior_calibration_step3a15 import masked_temperature_probabilities
from .trade_strategy import choose_proactive_trade_with_diagnostic


def _protected_trade_available(champion: SettlementPlanningAgent,
                               game: GameState, player_id: int) -> bool:
    if not champion.proactive_player_trade:
        return False
    return choose_proactive_trade_with_diagnostic(
        deepcopy(game), player_id,
        target_sites=champion.target_sites,
        max_plan_roads=champion.max_plan_roads,
        max_wait_rounds=champion.max_wait_rounds,
        max_city_wait_rounds=champion.max_city_wait_rounds,
        max_scarcity_deficit=champion.proactive_trade_max_scarcity_deficit,
        strategic_surplus_trade=champion.strategic_surplus_trade,
        opponent_score_limit=champion.proactive_trade_opponent_score_limit,
    ) is not None


def _sample_family(probabilities: np.ndarray, rng: random.Random,
                   stochastic: bool) -> int:
    if not stochastic:
        return int(np.argmax(probabilities))
    point = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += float(probability)
        if point < cumulative:
            return index
    return int(np.flatnonzero(probabilities > 0)[-1])


def _application_probe(game: GameState, player_id: int, action: Action) -> dict[str, Any]:
    """Audit the concrete Action's immediate effects before env auto-progression."""
    clone = deepcopy(game)
    player = game.players[player_id - 1]
    resources_before = {resource: player.resources[resource] for resource in RESOURCES}
    bank_before = dict(game.bank)
    road_before = _longest_road_length(game, player_id)
    settlement_before = len(legal_settlement_ids(game, player_id, initial=False))
    legal_road_before = len(legal_road_ids(game, player_id, initial=False))
    ratio = (_port_ratio(game, player_id, action.give_resource)
             if isinstance(action, BankTradeAction) else None)
    endpoint_opponent_building = (
        any(game.settlements.get(vertex) not in {None, player_id}
            or game.cities.get(vertex) not in {None, player_id}
            for vertex in game.board.edges[action.edge_id].vertex_ids)
        if isinstance(action, BuildRoadAction) else None)
    apply_action(clone, player_id, action, clone.revision, "strategic-apply-probe")
    after_player = clone.players[player_id - 1]
    resources_after = {resource: after_player.resources[resource] for resource in RESOURCES}
    bank_after = dict(clone.bank)
    if isinstance(action, BankTradeAction):
        for resource in RESOURCES:
            before = bank_before[resource] + sum(p.resources[resource] for p in game.players)
            after = bank_after[resource] + sum(p.resources[resource] for p in clone.players)
            if before != after or bank_after[resource] < 0 or any(
                    p.resources[resource] < 0 for p in clone.players):
                raise AssertionError("BankTrade resource conservation failed")
        if (resources_before[action.give_resource] - resources_after[action.give_resource]
                != ratio or resources_after[action.receive_resource]
                - resources_before[action.receive_resource] != 1):
            raise AssertionError("BankTrade ratio or inventory mismatch")
    if isinstance(action, BuildRoadAction) and clone.roads.get(action.edge_id) != player_id:
        raise AssertionError("Road ownership did not update")
    if isinstance(action, BuildRoadAction) and any(
            resources_before[resource] - resources_after[resource] != 1
            for resource in ("wood", "brick")):
        raise AssertionError("Paid road did not consume one wood and one brick")
    next_mask = get_action_mask(clone, player_id) if clone.phase == "action" else None
    return {
        "resource_before": resources_before, "resource_after": resources_after,
        "bank_before": bank_before if isinstance(action, BankTradeAction) else None,
        "bank_after": bank_after if isinstance(action, BankTradeAction) else None,
        "bank_ratio": ratio,
        "road_owner_after": (clone.roads.get(action.edge_id)
                             if isinstance(action, BuildRoadAction) else None),
        "road_length_before": road_before,
        "road_length_after": _longest_road_length(clone, player_id),
        "legal_road_count_before": legal_road_before,
        "legal_road_count_after": len(legal_road_ids(clone, player_id, initial=False)),
        "legal_settlement_count_before": settlement_before,
        "legal_settlement_count_after": len(legal_settlement_ids(
            clone, player_id, initial=False)),
        "road_endpoint_opponent_building": endpoint_opponent_building,
        "game_rng_before": game.random_source.counter,
        "probe_rng_after": clone.random_source.counter,
        "next_legal_action_count": (int(next_mask.sum())
                                    if next_mask is not None else None),
        "next_legal_bank_trade_count": (int(sum(
            next_mask[action_to_id(BankTradeAction(give, receive))]
            for give in RESOURCES for receive in RESOURCES if give != receive))
            if next_mask is not None else None),
    }


@dataclass(frozen=True)
class StrategicEpisode:
    seed: int
    seat: int
    profile: str
    mode: str
    delegation_probability: float
    delegation_seed: int
    gate_sampling_seed: int
    learner_id: int
    winner: int | None
    final_vp: int
    final_rank: int
    policy_steps: int
    terminal: bool
    truncated: bool
    action_trace: tuple[tuple[int, Action], ...]
    final_state: GameState
    rng_counter: int
    gate_surface_count: int
    delegation_lottery_count: int
    protected_available_count: int
    protected_preemption_count: int
    decisions: tuple[dict[str, Any], ...]
    resolver_events: tuple[dict[str, Any], ...]
    transitions: tuple[dict[str, Any], ...]
    learner_action_families: dict[str, int]
    forced_applied: bool = False
    prior_temperature: float = 1.0


def _seat_probe(seed: int) -> GameState:
    game = create_game(seed, ai_player_ids=(1, 2, 3, 4))
    heuristic = HeuristicAgent()
    for step in range(16):
        if game.phase != "rolling_order":
            break
        actor = pending_ai_player_id(game)
        action = heuristic.select_action(game, actor)
        apply_action(game, actor, action, game.revision, f"strategic-seat-probe-{step}")
    if game.phase == "rolling_order":
        raise RuntimeError("Seat probe did not finish")
    return game


def _close_transition(pending: dict[str, Any], gamma: float, *,
                      next_mask: tuple[bool, ...] | None,
                      terminal: bool, truncated: bool,
                      close_policy_step: int) -> dict[str, Any]:
    rewards = pending["rewards"]
    if not rewards or len(rewards) != close_policy_step - pending["start_policy_step"]:
        raise AssertionError("Macro duration must equal learner policy-step delta")
    item = StrategicTransition(
        family_index=pending["family_index"],
        candidate_mask=pending["candidate_mask"],
        gate_log_prob=pending["gate_log_prob"],
        value=pending["value"],
        reward_accumulated=sum(gamma ** i * reward for i, reward in enumerate(rewards)),
        macro_duration=len(rewards), done=terminal,
        next_actual_delegated=next_mask is not None,
        next_candidate_mask=next_mask, truncated=truncated,
    )
    if not isfinite(item.reward_accumulated):
        raise AssertionError("Nonfinite macro reward")
    return {
        "decision_index": pending["decision_index"],
        "family": FAMILIES[item.family_index],
        "candidate_mask": list(item.candidate_mask),
        "gate_log_prob": item.gate_log_prob,
        "value": item.value,
        "reward_accumulated": item.reward_accumulated,
        "macro_duration": item.macro_duration,
        "discount": item.bootstrap_discount(gamma),
        "done": item.done, "truncated": item.truncated,
        "next_actual_delegated": item.next_actual_delegated,
        "next_candidate_mask": (list(item.next_candidate_mask)
                                if item.next_candidate_mask is not None else None),
        "start_policy_step": pending["start_policy_step"],
        "close_policy_step": close_policy_step,
        "reward_steps": rewards.copy(),
        "vp_reward_accumulated": sum(gamma ** i * reward for i, reward in
                                     enumerate(pending["vp_rewards"])),
        "terminal_reward_accumulated": sum(gamma ** i * reward for i, reward in
                                           enumerate(pending["terminal_rewards"])),
        "protected_trades_crossed": pending["protected_trades_crossed"],
        "nondelegated_surfaces_crossed": pending["nondelegated_surfaces_crossed"],
    }


def run_strategic_episode(
    champion_policy, robber_policy, *, seed: int, seat: int, profile: str,
    mode: str, delegation_probability: float = 0.0,
    delegation_seed: int = 0, gate_sampling_seed: int = 0,
    stochastic_gate: bool = False, max_policy_steps: int = 5000,
    force_family: str | None = None, resolver_enabled: bool | None = None,
    prior_temperature: float = 1.0,
) -> StrategicEpisode:
    """Run one episode; only actual Gate decisions create semi-MDP samples."""
    if profile not in PROFILE_ROLES or mode not in {"legacy", "safety", "strategic"}:
        raise ValueError("Unknown profile or mode")
    if not 0 <= delegation_probability <= 1:
        raise ValueError("Delegation probability must be in [0, 1]")
    if not isfinite(prior_temperature) or prior_temperature <= 0:
        raise ValueError("Prior temperature must be finite and positive")
    if force_family is not None and force_family not in FAMILIES:
        raise ValueError("Unknown forced family")
    if force_family is not None and mode != "strategic":
        raise ValueError("Forced integration requires strategic mode")
    if resolver_enabled is None:
        resolver_enabled = mode != "legacy"
    learner_id = _champion_id_for_seat(seed, seat)
    planning = champion_policy.settlement_planning_config
    robber_planning = robber_policy.settlement_planning_config
    if planning is None or robber_planning is None:
        raise ValueError("Frozen PPO agents need their existing Planner config")
    champion = SettlementPlanningAgent(champion_policy, **planning)
    executors = FrozenStrategicExecutors(champion)
    latent_dim = champion_policy.model.policy.mlp_extractor.latent_dim_pi
    gate = StrategicActionGate(latent_dim).eval()
    critic = StrategicSurfaceCritic(latent_dim).eval()
    probe = _seat_probe(seed)
    opponents = {}
    ids = [pid for pid in probe.seat_order if pid != learner_id]
    for player_id, role in zip(ids, PROFILE_ROLES[profile], strict=True):
        if role == "rule":
            delegate = HeuristicAgent()
        elif role == "champion":
            delegate = SettlementPlanningAgent(champion_policy, **planning)
        else:
            delegate = SettlementPlanningAgent(robber_policy, **robber_planning)
        opponents[player_id] = _HeuristicRollOrder(delegate)
    trace: list[tuple[int, Action]] = []
    env = CatanEnv(
        CatanEnvConfig(learning_player_id=learner_id, allow_player_trades=True,
                       observation_version="v2", max_policy_steps=max_policy_steps),
        initial_placement_agent=champion, opponent_agents=opponents,
        learner_auxiliary_agent=_TradeResponseOnly(champion),
        action_observer=lambda _game, player_id, action: trace.append((player_id, action)),
    )
    env.reset(seed=seed)
    delegation_rng = random.Random(delegation_seed)
    gate_rng = random.Random(gate_sampling_seed)
    gamma = float(champion_policy.model.gamma)
    pending: dict[str, Any] | None = None
    decisions: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    resolver_events: list[dict[str, Any]] = []
    action_families: Counter[str] = Counter()
    surface_count = lottery_count = protected_available = protected_selected = 0
    forced_applied = terminal = truncated = False
    heuristic = HeuristicAgent()
    try:
        while not (terminal or truncated):
            game = env.game
            if game.phase != "rolling_order" and game.seat_order.index(learner_id) + 1 != seat:
                raise AssertionError("Learner seat drifted")
            if env.policy_steps >= max_policy_steps:
                raise AssertionError("Unsignalled policy-step truncation")
            pre_phase = game.phase
            pre_revision = game.revision
            pre_game_rng_counter = game.random_source.counter
            delegated = False
            selected_mask: tuple[bool, ...] | None = None
            selected_family_index: int | None = None
            selected_log_prob = selected_value = 0.0
            if pre_phase == "rolling_order":
                action = heuristic.select_action(game, learner_id)
            else:
                win = (resolve_immediate_win(game, learner_id)
                       if resolver_enabled and pre_phase in {"action", "turn_pre_roll"}
                       else None)
                if win is not None and win.action is not None:
                    counterfactual = champion.select_action(deepcopy(game), learner_id)
                    action = win.action
                    resolver_events.append({
                        "action_trace_index": len(trace),
                        "pre_policy_step": env.policy_steps, "turn": game.turn_number,
                        "revision": pre_revision, "phase": pre_phase,
                        "winning_ids": list(win.winning_action_ids),
                        "resolver_action": repr(action),
                        "champion_counterfactual_action": repr(counterfactual),
                        "champion_counterfactual_family": family_of(counterfactual),
                        "changed_action": action != counterfactual,
                    })
                elif pre_phase != "action" or mode == "legacy":
                    action = champion.select_action(game, learner_id)
                else:
                    mask = get_action_mask(game, learner_id)
                    surface = detect_strategic_action_surface(game, learner_id, mask)
                    champion_action = champion.select_action(game, learner_id)
                    action = champion_action
                    if surface.eligible:
                        surface_count += 1
                        available = _protected_trade_available(champion, game, learner_id)
                        selected = isinstance(champion_action, ProposeTradeAction)
                        protected_available += int(available)
                        protected_selected += int(selected)
                        if selected:
                            if pending is not None:
                                pending["protected_trades_crossed"] += 1
                        else:
                            lottery_count += 1
                            force = (force_family is not None and not forced_applied
                                     and surface.family_mask[FAMILIES.index(force_family)])
                            do_delegate = force or (
                                mode == "strategic"
                                and delegation_rng.random() < delegation_probability)
                            if do_delegate:
                                snapshot = frozen_snapshot(
                                    champion_policy, game, learner_id,
                                    max_plan_roads=champion.max_plan_roads)
                                prior = family_priors(snapshot, mask, surface.family_mask)
                                latent = snapshot.actor_latent
                                context = torch.as_tensor(
                                    snapshot.runtime_context, dtype=latent.dtype,
                                    device=latent.device).unsqueeze(0)
                                family_mask = torch.tensor([surface.family_mask],
                                                           dtype=torch.bool, device=latent.device)
                                scores = torch.tensor(
                                    [prior["raw_scores"]["native_family_logits"]],
                                    dtype=latent.dtype, device=latent.device)
                                with torch.no_grad():
                                    logits = gate(latent, context, family_mask,
                                                  scores, temperature=prior_temperature)
                                    probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                                    value = critic(snapshot.critic_value, latent,
                                                   context, family_mask)
                                expected = masked_temperature_probabilities(
                                    prior["raw_scores"]["native_family_logits"],
                                    surface.family_mask, prior_temperature)
                                if not np.allclose(probabilities, expected, rtol=0, atol=1e-6):
                                    raise AssertionError("Zero residual / calibrated prior mismatch")
                                family_index = (FAMILIES.index(force_family) if force
                                                else _sample_family(probabilities, gate_rng,
                                                                    stochastic_gate))
                                family = FAMILIES[family_index]
                                result = executors.execute(
                                    game, learner_id, family, method="B_FROZEN_PPO",
                                    mask=mask, snapshot=snapshot)
                                if not result.success:
                                    raise AssertionError(f"Executor failure: {family}: {result.failure}")
                                action = result.action
                                if family_of(action) != family:
                                    raise AssertionError("Executor returned family mismatch")
                                application_probe = _application_probe(
                                    game, learner_id, action)
                                selected_mask = surface.family_mask
                                selected_family_index = family_index
                                selected_log_prob = float(log(max(float(probabilities[family_index]), 1e-38)))
                                selected_value = float(value.item())
                                delegated = True
                                forced_applied |= force
                                decisions.append({
                                    "pre_policy_step": env.policy_steps,
                                    "turn": game.turn_number, "revision": pre_revision,
                                    "candidate_mask": list(selected_mask),
                                    "candidate_count": surface.candidate_count,
                                    "native_family_logits": scores[0].cpu().tolist(),
                                    "masked_probability": probabilities.tolist(),
                                    "selected_family": family,
                                    "selected_probability": float(probabilities[family_index]),
                                    "selected_is_native_top": family_index == int(
                                        np.argmax(prior["probabilities"]["native_family_logits"])),
                                    "prior_temperature": prior_temperature,
                                    "gate_log_prob": selected_log_prob,
                                    "concrete_action": repr(action),
                                    "concrete_action_family": family_of(action),
                                    "application_probe": application_probe,
                                    "champion_counterfactual_action": repr(champion_action),
                                    "champion_counterfactual_family": family_of(champion_action),
                                    "protected_player_trade_available": available,
                                    "protected_player_trade_selected": selected,
                                    "forced_integration": force,
                                    "entropy": float(-sum(p * log(p) for p in probabilities if p > 0)),
                                    "value": selected_value,
                                    "hand_size": sum(game.players[learner_id - 1].resources.values()),
                                })
                            elif pending is not None:
                                pending["nondelegated_surfaces_crossed"] += 1
                    # A protected trade is not an on-policy Strategic sample.
            if delegated:
                if pending is not None:
                    transitions.append(_close_transition(
                        pending, gamma, next_mask=selected_mask,
                        terminal=False, truncated=False,
                        close_policy_step=env.policy_steps))
                pending = {
                    "decision_index": len(decisions) - 1,
                    "family_index": selected_family_index,
                    "candidate_mask": selected_mask,
                    "gate_log_prob": selected_log_prob,
                    "value": selected_value,
                    "start_policy_step": env.policy_steps,
                    "rewards": [], "vp_rewards": [], "terminal_rewards": [],
                    "protected_trades_crossed": 0,
                    "nondelegated_surfaces_crossed": 0,
                }
            if pre_phase == "action":
                action_families[family_of(action)] += 1
            if game.random_source.counter != pre_game_rng_counter:
                raise AssertionError("Controller arbitration consumed GameState RNG")
            _, reward, terminal, truncated, info = env.step_action(action)
            if not isfinite(reward):
                raise AssertionError("Nonfinite environment reward")
            if resolver_events and resolver_events[-1]["revision"] == pre_revision:
                event = resolver_events[-1]
                event.update({"immediate_terminal": terminal,
                              "winner_after_step": env.game.winner_id,
                              "reward": reward})
                if not terminal or env.game.winner_id != learner_id:
                    raise AssertionError("ImmediateWinResolver did not win immediately")
            if delegated:
                decisions[-1]["immediate_reward"] = reward
                decisions[-1]["reward_components"] = info["reward_components"]
                decisions[-1]["post_policy_step"] = env.policy_steps
                decisions[-1]["post_phase"] = env.game.phase
            if pending is not None:
                pending["rewards"].append(float(reward))
                pending["vp_rewards"].append(float(info["reward_components"]["victory_point_reward"]))
                pending["terminal_rewards"].append(float(info["reward_components"]["terminal_reward"]))
            if terminal or truncated:
                if pending is not None:
                    transitions.append(_close_transition(
                        pending, gamma, next_mask=None, terminal=terminal,
                        truncated=truncated, close_policy_step=env.policy_steps))
                    pending = None
        if pending is not None:
            raise AssertionError("Episode left a Strategic transition open")
        scores = {player.id: actual_score(env.game, player.id) for player in env.game.players}
        rank, _ = endgame_rank(scores, learner_id, env.game.winner_id)
        return StrategicEpisode(
            seed, seat, profile, mode, delegation_probability,
            delegation_seed, gate_sampling_seed, learner_id,
            env.game.winner_id, scores[learner_id], rank, env.policy_steps,
            terminal, truncated, tuple(trace), _normalized_state(env.game),
            env.game.random_source.counter, surface_count, lottery_count,
            protected_available, protected_selected, tuple(decisions),
            tuple(resolver_events), tuple(transitions), dict(action_families),
            forced_applied, prior_temperature,
        )
    finally:
        env.close()
