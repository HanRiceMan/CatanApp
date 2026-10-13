"""Real-action, Investment-only hierarchical rollout with 3A safety layers."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from math import isfinite, log
import random

import numpy as np
import torch

from app.agents.heuristic import HeuristicAgent
from app.domain.actions import ProposeTradeAction
from app.domain.game import legal_actions

from .action_space import ACTION_CATALOG, get_action_mask
from .construction_timing_rollout import _HeuristicRollOrder, _TradeResponseOnly, _normalized_state
from .diagnostics_metrics import endgame_rank
from .env import CatanEnv, CatanEnvConfig
from .hierarchical_l3_shadow_step3b5 import encode_l3_decision_facts
from .hierarchical_ppo_step3b6 import (
    FAMILY_ORDER, GAMMA, HierarchicalInvestmentCritic,
    HierarchicalInvestmentPolicy, investment_training_mask,
)
from .hierarchical_shadow_controller_step3b4 import (
    HierarchicalRoute, classify_hierarchical_route,
)
from .macro_teacher_dataset import PROFILE_ROLES, _champion_id_for_seat
from .reward import actual_score
from .settlement_planning_agent import SettlementPlanningAgent
from .strategic_action_surface import family_of
from .strategic_gate_rollout_step3a14 import _application_probe, _seat_probe
from .strategic_gate_step3a13 import FrozenStrategicExecutors, frozen_snapshot, resolve_immediate_win
from .victory_plan_context import build_victory_plan_context
from .victory_plan_policy import encode_victory_plan_features


_EXECUTABLE = ("BUILD_ROAD", "BUY_DEVELOPMENT", "BANK_TRADE", "END_TURN")
_ACTION_IDS = {family: tuple(item.id for item in ACTION_CATALOG
                             if family_of(item.action) == family)
               for family in _EXECUTABLE}


def _sample(probs: np.ndarray, rng: random.Random) -> int:
    target = rng.random()
    total = 0.
    for i, p in enumerate(probs):
        total += float(p)
        if target < total:
            return i
    return int(np.flatnonzero(probs)[-1])


def _delegation_candidate(training_mask: tuple[bool, ...],
                          multi_choice_only: bool) -> bool:
    count = sum(training_mask)
    return count >= (2 if multi_choice_only else 1)


def _close(pending: dict, *, next_mask: tuple[bool, ...] | None,
           terminal: bool, truncated: bool, policy_step: int) -> dict:
    rewards = pending.pop("rewards")
    duration = policy_step - pending["start_policy_step"]
    if duration < 1 or len(rewards) != duration:
        raise AssertionError("Macro duration differs from learner policy steps")
    value = sum(GAMMA ** i * reward for i, reward in enumerate(rewards))
    if not isfinite(value):
        raise AssertionError("Nonfinite macro reward")
    return {**pending, "reward_steps": rewards, "reward_accumulated": value,
            "macro_duration": duration, "done": terminal, "truncated": truncated,
            "discount": 0. if terminal else GAMMA ** duration,
            "next_actual_delegated": next_mask is not None,
            "next_family_mask": list(next_mask) if next_mask is not None else None,
            "close_policy_step": policy_step}


def run_hierarchical_episode(
    champion_policy, robber_policy, policy: HierarchicalInvestmentPolicy,
    critic: HierarchicalInvestmentCritic, *, seed: int, seat: int,
    profile: str, delegation_probability: float,
    delegation_seed: int, family_seed: int,
    multi_choice_only: bool = False,
    max_policy_steps: int = 5000,
) -> dict:
    if profile not in PROFILE_ROLES or not 0 <= delegation_probability <= 1:
        raise ValueError("Invalid profile/delegation rate")
    learner_id = _champion_id_for_seat(seed, seat)
    planning = champion_policy.settlement_planning_config
    robber_planning = robber_policy.settlement_planning_config
    if planning is None or robber_planning is None:
        raise ValueError("Frozen agents require Planner config")
    champion = SettlementPlanningAgent(champion_policy, **planning)
    executors = FrozenStrategicExecutors(champion)
    probe = _seat_probe(seed)
    opponent_ids = [pid for pid in probe.seat_order if pid != learner_id]
    opponents = {}
    for player_id, role in zip(opponent_ids, PROFILE_ROLES[profile], strict=True):
        delegate = (HeuristicAgent() if role == "rule" else SettlementPlanningAgent(
            champion_policy if role == "champion" else robber_policy,
            **(planning if role == "champion" else robber_planning)))
        opponents[player_id] = _HeuristicRollOrder(delegate)
    trace = []
    env = CatanEnv(
        CatanEnvConfig(learning_player_id=learner_id, allow_player_trades=True,
                       observation_version="v2", max_policy_steps=max_policy_steps),
        initial_placement_agent=champion, opponent_agents=opponents,
        learner_auxiliary_agent=_TradeResponseOnly(champion),
        action_observer=lambda _game, player_id, action: trace.append((player_id, action)),
    )
    env.reset(seed=seed)
    delegation_rng, family_rng = random.Random(delegation_seed), random.Random(family_seed)
    heuristic = HeuristicAgent()
    decisions: list[dict] = []
    transitions: list[dict] = []
    pending = None
    action_families = Counter()
    routes = Counter()
    protected = resolver = eligible = lottery = single_choice_skipped = 0
    terminal = truncated = False
    try:
        while not (terminal or truncated):
            game = env.game
            pre_rng = game.random_source.counter
            pre_step = env.policy_steps
            delegated = False
            if game.phase == "rolling_order":
                action = heuristic.select_action(game, learner_id)
                routes["FORCED_PHASE"] += 1
            else:
                win = (resolve_immediate_win(game, learner_id)
                       if game.phase in {"action", "turn_pre_roll"} else None)
                if win is not None and win.action is not None:
                    action = win.action
                    resolver += 1
                    routes["IMMEDIATE_WIN"] += 1
                    if pending is not None:
                        pending["resolver_crossed"] += 1
                elif game.phase != "action":
                    action = champion.select_action(game, learner_id)
                    routes["FORCED_PHASE"] += 1
                else:
                    legal = legal_actions(game, learner_id)
                    route = classify_hierarchical_route(
                        game, learner_id, legal=legal, immediate_win_available=False)
                    routes[route.route.value] += 1
                    champion_action = champion.select_action(game, learner_id)
                    action = champion_action
                    if isinstance(champion_action, ProposeTradeAction):
                        protected += 1
                        if pending is not None:
                            pending["protected_crossed"] += 1
                    elif route.route is HierarchicalRoute.INVESTMENT:
                        catalog_mask = get_action_mask(game, learner_id)
                        executable = tuple(
                            any(catalog_mask[index] for index in _ACTION_IDS[family])
                            if family in _EXECUTABLE else False for family in FAMILY_ORDER)
                        training_mask = investment_training_mask(
                            route.route, route.comparison_family_mask, executable)
                        if multi_choice_only and sum(training_mask) == 1:
                            single_choice_skipped += 1
                        if _delegation_candidate(training_mask, multi_choice_only):
                            eligible += 1
                            lottery += 1
                            if delegation_rng.random() < delegation_probability:
                                snapshot = frozen_snapshot(
                                    champion_policy, game, learner_id,
                                    max_plan_roads=champion.max_plan_roads)
                                plan_context = build_victory_plan_context(game, learner_id)
                                encoded = encode_victory_plan_features(plan_context)
                                latent = snapshot.actor_latent
                                runtime = torch.as_tensor(snapshot.runtime_context.copy()).unsqueeze(0)
                                global_features = torch.as_tensor(encoded.global_features.copy()).unsqueeze(0)
                                candidates = torch.as_tensor(encoded.candidate_features.copy()).unsqueeze(0)
                                plan_mask = torch.as_tensor(encoded.candidate_mask.copy()).unsqueeze(0)
                                facts = encode_l3_decision_facts([route])
                                facts = replace(facts, comparison_family_mask=torch.tensor(
                                    [training_mask], dtype=torch.bool))
                                with torch.no_grad():
                                    plan, bridge, l3 = policy(
                                        latent, runtime, global_features, candidates,
                                        plan_mask, facts)
                                    value = critic(latent, runtime, global_features,
                                                   bridge, facts.comparison_family_mask)
                                if not bool(l3.available[0]):
                                    raise AssertionError("Eligible Investment lost its Actor mask")
                                probabilities = l3.probabilities[0].cpu().numpy()
                                family_index = _sample(probabilities, family_rng)
                                family = FAMILY_ORDER[family_index]
                                result = executors.execute(
                                    game, learner_id, family, method="B_FROZEN_PPO",
                                    mask=catalog_mask, snapshot=snapshot)
                                if not result.success or family_of(result.action) != family:
                                    raise AssertionError(f"Frozen Executor failure: {family}: {result.failure}")
                                action = result.action
                                application = _application_probe(game, learner_id, action)
                                if application["game_rng_before"] != pre_rng:
                                    raise AssertionError("Executor arbitration consumed GameState RNG")
                                # Applying BuyDevelopment to the audit clone may
                                # legitimately draw from that clone's deck RNG.
                                delegated = True
                                decision = {
                                    "start_policy_step": pre_step, "turn": game.turn_number,
                                    "revision": game.revision, "route": route.route.value,
                                    "phase": game.phase, "family_mask": list(training_mask),
                                    "legal_family_mask": list(route.legal_family_mask),
                                    "diagnostic_family_mask": list(route.comparison_family_mask),
                                    "selected_family": family, "selected_index": family_index,
                                    "old_log_prob": log(float(probabilities[family_index])),
                                    "old_probability": probabilities.tolist(),
                                    "l3_entropy": float(l3.entropy[0]),
                                    "plan_entropy": float(plan.entropy[0]),
                                    "expected_priorities": bridge.expected_priorities[0].tolist(),
                                    "soft_plan": bridge.weighted_features[0].tolist(),
                                    "plan_available": bool(bridge.available[0]),
                                    "old_value": float(value[0]),
                                    "concrete_action": repr(action),
                                    "champion_counterfactual_action": repr(champion_action),
                                    "champion_counterfactual_family": family_of(champion_action),
                                    "observation_v2": snapshot.observation_v2.tolist(),
                                    "actor_latent": latent[0].tolist(),
                                    "runtime_context": snapshot.runtime_context.tolist(),
                                    "global_plan_features": encoded.global_features.tolist(),
                                    "candidate_plan_features": encoded.candidate_features.tolist(),
                                    "candidate_plan_mask": encoded.candidate_mask.tolist(),
                                    "candidate_plan_ids": list(encoded.candidate_ids),
                                    "old_l3_logits": l3.raw_logits[0].tolist(),
                                    "game_rng_before": pre_rng,
                                }
                                decisions.append(decision)
                        if not delegated and pending is not None:
                            pending["nondelegated_investment_crossed"] += 1
            if delegated:
                if pending is not None:
                    transitions.append(_close(
                        pending, next_mask=tuple(decision["family_mask"]),
                        terminal=False, truncated=False, policy_step=pre_step))
                pending = {
                    "decision_index": len(decisions) - 1,
                    "selected_family": decision["selected_family"],
                    "selected_index": decision["selected_index"],
                    "family_mask": decision["family_mask"],
                    "old_log_prob": decision["old_log_prob"],
                    "value": decision["old_value"],
                    "start_policy_step": pre_step, "rewards": [],
                    "protected_crossed": 0, "resolver_crossed": 0,
                    "nondelegated_investment_crossed": 0,
                }
            if game.phase == "action":
                action_families[family_of(action)] += 1
            if game.random_source.counter != pre_rng:
                raise AssertionError("Controller arbitration consumed GameState RNG")
            _, reward, terminal, truncated, info = env.step_action(action)
            if not isfinite(reward):
                raise AssertionError("Nonfinite environment reward")
            if delegated:
                decisions[-1]["immediate_reward"] = float(reward)
                decisions[-1]["reward_components"] = info["reward_components"]
            if pending is not None:
                pending["rewards"].append(float(reward))
            if terminal or truncated:
                if pending is not None:
                    transitions.append(_close(pending, next_mask=None,
                                              terminal=terminal, truncated=truncated,
                                              policy_step=env.policy_steps))
                    pending = None
        scores = {player.id: actual_score(env.game, player.id) for player in env.game.players}
        rank, _ = endgame_rank(scores, learner_id, env.game.winner_id)
        return {
            "seed": seed, "seat": seat, "profile": profile,
            "delegation_probability": delegation_probability,
            "delegation_seed": delegation_seed, "family_seed": family_seed,
            "learner_id": learner_id, "winner": env.game.winner_id,
            "final_vp": scores[learner_id], "final_rank": rank,
            "policy_steps": env.policy_steps, "terminal": terminal,
            "truncated": truncated, "action_trace": tuple(trace),
            "final_state": _normalized_state(env.game),
            "rng_counter": env.game.random_source.counter,
            "eligible_investment": eligible, "delegation_lottery_count": lottery,
            "single_choice_skipped": single_choice_skipped,
            "actual_delegated": len(decisions), "protected_preemption": protected,
            "resolver_count": resolver, "route_counts": dict(routes),
            "learner_action_families": dict(action_families),
            "decisions": decisions, "transitions": transitions,
        }
    finally:
        env.close()
