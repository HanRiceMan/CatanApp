"""Step 3A-9: real, no-update Construction Timing episode collection."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
import random
from typing import Any

import numpy as np
import torch

from app.agents.heuristic import HeuristicAgent
from app.domain.actions import (Action, BuildCityAction, BuildRoadAction,
                                BuildSettlementAction, EndTurnAction,
                                ProposeTradeAction)
from app.domain.game import GameState

from .construction_timing import (
    BUILD_NOW, DEFER, ConstructionTimingDelegator, ConstructionTimingGate,
    DelegationChoice, SurfaceCritic, SurfaceRolloutBuffer,
    SurfaceTransitionBuilder, frozen_gate_input, timing_surface,
)
from .diagnostics_metrics import endgame_rank
from .env import CatanEnv, CatanEnvConfig
from .macro_teacher_dataset import PROFILE_ROLES, _champion_id_for_seat
from .observation import encode_observation, get_observation
from .reward import actual_score
from .settlement_planning_agent import SettlementPlanningAgent


class _TradeResponseOnly:
    """Preserve frozen response/counter but never pre-empt an action-phase Gate."""

    def __init__(self, champion: SettlementPlanningAgent):
        self.champion = champion

    def select_action(self, game: GameState, player_id: int) -> Action | None:
        if game.phase in {"trade_response", "trade_counter_offer"}:
            return self.champion.select_action(game, player_id)
        return None


class _HeuristicRollOrder:
    def __init__(self, delegate):
        self.delegate = delegate

    def select_action(self, game: GameState, player_id: int) -> Action:
        if game.phase == "rolling_order":
            return HeuristicAgent().select_action(game, player_id)
        return self.delegate.select_action(game, player_id)


@dataclass(frozen=True)
class TimingEpisode:
    seed: int
    seat: int
    profile: str
    delegation_probability: float
    delegation_rng_seed: int
    winner: int | None
    final_vp: int
    final_rank: int
    policy_steps: int
    terminal: bool
    truncated: bool
    action_trace: tuple[tuple[int, Action], ...]
    final_state: GameState
    transitions: tuple
    decisions: tuple[dict[str, Any], ...]
    rewards: tuple[float, ...]
    rng_counter: int
    integration_forced: bool = False
    learner_id: int = 0
    surface_reached: int = 0
    risk_actions: tuple[str, ...] = ()


def _normalized_state(game: GameState) -> GameState:
    state = deepcopy(game)
    state.id = ""
    state.player_tokens = {}
    return state


def run_timing_episode(
    champion_policy, robber_policy, gate: ConstructionTimingGate,
    critic: SurfaceCritic, *, seed: int, seat: int, profile: str,
    delegation_probability: float, delegation_rng_seed: int,
    stochastic_gate: bool = True, max_policy_steps: int = 5000,
    use_delegator: bool = True,
    integration_force: tuple[str, int, str] | None = None,
) -> TimingEpisode:
    """One complete episode; only actually delegated decisions enter Actor data."""
    if profile not in PROFILE_ROLES:
        raise ValueError(f"Unknown profile: {profile}")
    champion_id = _champion_id_for_seat(seed, seat)
    planning = champion_policy.settlement_planning_config
    robber_planning = robber_policy.settlement_planning_config
    if planning is None or robber_planning is None:
        raise ValueError("Frozen PPO agents require Planner configuration")
    champion = SettlementPlanningAgent(champion_policy, **planning)
    roles = PROFILE_ROLES[profile]
    # The order probe gives player IDs in actual seat order, independently of
    # the learner's later choice of construction timing.
    from app.domain.game import apply_action, create_game, pending_ai_player_id
    probe = create_game(seed, ai_player_ids=(1, 2, 3, 4))
    heuristic = HeuristicAgent()
    for step in range(16):
        if probe.phase != "rolling_order":
            break
        actor = pending_ai_player_id(probe)
        action = heuristic.select_action(probe, actor)
        apply_action(probe, actor, action, probe.revision, f"timing-seat-probe-{step}")
    if probe.phase == "rolling_order":
        raise RuntimeError("Seat probe did not finish")
    opponent_ids = [pid for pid in probe.seat_order if pid != champion_id]
    opponents = {}
    for player_id, role in zip(opponent_ids, roles, strict=True):
        if role == "rule":
            opponents[player_id] = _HeuristicRollOrder(HeuristicAgent())
        elif role == "champion":
            opponents[player_id] = _HeuristicRollOrder(
                SettlementPlanningAgent(champion_policy, **planning))
        else:
            opponents[player_id] = _HeuristicRollOrder(
                SettlementPlanningAgent(robber_policy, **robber_planning))

    trace: list[tuple[int, Action]] = []
    def observe(_game, player_id, action):
        trace.append((player_id, action))

    env = CatanEnv(
        CatanEnvConfig(learning_player_id=champion_id,
                       allow_player_trades=True, observation_version="v2",
                       max_policy_steps=max_policy_steps),
        initial_placement_agent=champion, opponent_agents=opponents,
        learner_auxiliary_agent=_TradeResponseOnly(champion),
        action_observer=observe,
    )
    delegator = ConstructionTimingDelegator(
        champion, gate, critic, delegation_probability=delegation_probability)
    delegation_rng = random.Random(delegation_rng_seed)
    buffer = SurfaceRolloutBuffer(champion_policy.model.gamma)
    builder = SurfaceTransitionBuilder(champion_policy.model.gamma)
    decisions: list[dict[str, Any]] = []
    rewards: list[float] = []
    env.reset(seed=seed)
    terminal = truncated = False
    forced_applied = False
    surface_reached = 0
    risk_actions: list[str] = []
    while not (terminal or truncated):
        game = env.game
        if game.phase != "rolling_order" and game.seat_order.index(champion_id) + 1 != seat:
            raise AssertionError("Actual learner seat differs from requested seat")
        if env.policy_steps >= max_policy_steps:
            raise AssertionError("Episode exceeded max_policy_steps without truncation")
        if game.phase == "rolling_order":
            action = heuristic.select_action(game, champion_id)
            choice = None
        elif not use_delegator and integration_force is None:
            surface_reached += int(timing_surface(game, champion_id).eligible)
            choice = None
            action = champion.select_action(game, champion_id)
        elif integration_force is not None:
            choice = None
            action = None
            target_subtype, target_gate_action, target_action_name = integration_force
            if not forced_applied:
                surface = timing_surface(game, champion_id)
                surface_reached += int(surface.eligible)
                if surface.eligible and surface.subtype == target_subtype:
                    candidates = delegator.executors.candidates(game, champion_id, surface)
                    selected = (candidates.build_action if target_gate_action == BUILD_NOW
                                else candidates.defer_action)
                    if (candidates.delegatable and selected is not None
                            and type(selected).__name__ == target_action_name):
                        latent, context, subtype, base_value = frozen_gate_input(
                            champion_policy, game, champion_id, surface.subtype,
                            max_plan_roads=champion.max_plan_roads)
                        with torch.no_grad():
                            distribution = torch.distributions.Categorical(
                                logits=gate(latent, context, subtype))
                            log_prob = float(distribution.log_prob(
                                torch.tensor([target_gate_action])).item())
                            value = float(critic(base_value, latent, context, subtype).item())
                        choice = DelegationChoice(
                            selected, True, target_gate_action, log_prob, value,
                            candidates, context[0].detach().cpu().numpy().copy())
                        action = selected
                        forced_applied = True
            if action is None:
                action = champion.select_action(game, champion_id)
        elif use_delegator:
            choice = delegator.select(
                game, champion_id, random_uniform=delegation_rng.random,
                stochastic=stochastic_gate)
            action = choice.action
            surface_reached += int(choice.candidates.surface.eligible)

        if choice is not None and choice.delegated:
            observation = encode_observation(get_observation(game, champion_id,
                                                              version="v2"))
            context = choice.runtime_context
            if context is None or not np.isfinite(context).all():
                raise AssertionError("Delegated Context is missing or nonfinite")
            if builder.pending is not None:
                buffer.append(builder.finish(
                    next_surface_or_terminal=choice.candidates.surface.subtype,
                    done=False, next_surface_observation=observation,
                    next_runtime_context=context,
                    next_surface_type=choice.candidates.surface.subtype))
            builder.start(
                surface_observation=observation, runtime_context=context,
                surface_type=choice.candidates.surface.subtype,
                gate_action=choice.gate_action,
                gate_log_prob=choice.gate_log_prob,
                value=choice.surface_value)
        elif (choice is not None and choice.candidates.surface.eligible
              and builder.pending is not None):
            builder.record_nondelegated_surface()

        hand_size = sum(game.players[champion_id - 1].resources.values())
        if hand_size > 7:
            risk_actions.append(type(action).__name__)
        was_trade = isinstance(action, ProposeTradeAction)
        _, reward, terminal, truncated, info = env.step_action(action)
        if not isfinite(reward):
            raise AssertionError("Nonfinite environment reward")
        rewards.append(reward)
        components = info["reward_components"]
        builder.add_step(reward, vp_delta=components["victory_point_reward"],
                         terminal=components["terminal_reward"],
                         player_trade=was_trade)
        if choice is not None and choice.delegated:
            decisions.append({
                "step": env.policy_steps,
                "subtype": choice.candidates.surface.subtype,
                "gate_action": choice.gate_action,
                "action_type": type(action).__name__,
                "hand_size": hand_size,
                "immediate_reward": reward,
                "vp_delta_reward": components["victory_point_reward"],
                "terminal_reward": components["terminal_reward"],
                "trade_proposed": was_trade,
                "trade_phase_after_step": env.game.phase,
            })
        if terminal or truncated:
            if builder.pending is not None:
                buffer.append(builder.finish(
                    next_surface_or_terminal=("TERMINAL" if terminal else "TRUNCATED"),
                    done=terminal, truncated=truncated))
    if builder.pending is not None:
        raise AssertionError("Terminal transition left open")
    scores = {player.id: actual_score(env.game, player.id)
              for player in env.game.players}
    rank, _ = endgame_rank(scores, champion_id, env.game.winner_id)
    result = TimingEpisode(
        seed, seat, profile, delegation_probability, delegation_rng_seed,
        env.game.winner_id, scores[champion_id], rank, env.policy_steps,
        terminal, truncated, tuple(trace), _normalized_state(env.game),
        tuple(buffer.transitions), tuple(decisions), tuple(rewards),
        env.game.random_source.counter, forced_applied, champion_id,
        surface_reached, tuple(risk_actions))
    env.close()
    return result


def summarize_episodes(episodes: list[TimingEpisode]) -> dict[str, Any]:
    transitions = [transition for episode in episodes
                   for transition in episode.transitions]
    decisions = [decision for episode in episodes for decision in episode.decisions]
    duration = np.array([item.macro_duration for item in transitions], dtype=float)
    reward = np.array([item.reward_accumulated for item in transitions], dtype=float)
    build = [row for row in decisions if row["gate_action"] == BUILD_NOW]
    defer = [row for row in decisions if row["gate_action"] != BUILD_NOW]
    def longest_defer_run(item: TimingEpisode) -> int:
        current = maximum = 0
        for row in item.decisions:
            current = current + 1 if row["gate_action"] == DEFER else 0
            maximum = max(maximum, current)
        return maximum
    def stats(values):
        return ({"count": len(values), "mean": float(np.mean(values)),
                 "median": float(np.median(values)),
                 "p95": float(np.percentile(values, 95)),
                 "max": float(np.max(values))} if len(values) else {"count": 0})
    return {
        "games": len(episodes), "transitions": len(transitions),
        "profile_games": dict(Counter(item.profile for item in episodes)),
        "seat_games": dict(Counter(item.seat for item in episodes)),
        "subtype": dict(Counter(item.surface_type for item in transitions)),
        "gate_action": {"BUILD_NOW": len(build), "DEFER": len(defer)},
        "macro_duration": stats(duration),
        "reward_accumulated": stats(reward),
        "immediate_reward_build": stats([item["immediate_reward"] for item in build]),
        "immediate_reward_defer": stats([item["immediate_reward"] for item in defer]),
        "vp_delta_build": stats([item["vp_delta_reward"] for item in build]),
        "vp_delta_defer": stats([item["vp_delta_reward"] for item in defer]),
        "terminal_contribution": sum(item["terminal_reward"] for item in decisions),
        "terminal_transitions": sum(item.done for item in transitions),
        "next_delegated_surface": sum(item.next_surface_type is not None
                                      for item in transitions),
        "nondelegated_surface_crossings": sum(
            item.nondelegated_surfaces_crossed for item in transitions),
        "truncated_transitions": sum(item.truncated for item in transitions),
        "player_trade_decisions": sum(item["trade_proposed"] for item in decisions),
        "player_trades_crossed": sum(item.player_trades_crossed for item in transitions),
        "vp_delta_contribution": sum(item.vp_delta_contribution for item in transitions),
        "macro_terminal_contribution": sum(item.terminal_contribution
                                           for item in transitions),
        "defer_immediate_same_surface": sum(
            item.gate_action != BUILD_NOW and item.macro_duration == 1
            and item.next_surface_type == item.surface_type for item in transitions),
        "defer_hand_over_7": sum(item["hand_size"] > 7 for item in defer),
        "all_terminal": all(item.terminal for item in episodes),
        "terminal_rate": (sum(item.terminal for item in episodes) / len(episodes)
                          if episodes else None),
        "max_consecutive_defer": max((longest_defer_run(item) for item in episodes),
                                     default=0),
        "average_max_consecutive_defer": (float(np.mean([
            longest_defer_run(item) for item in episodes])) if episodes else None),
        "win_rate": (sum(item.winner == item.learner_id for item in episodes)
                     / len(episodes) if episodes else None),
        "average_vp": float(np.mean([item.final_vp for item in episodes])) if episodes else None,
        "average_rank": float(np.mean([item.final_rank for item in episodes])) if episodes else None,
        "average_policy_steps": float(np.mean([item.policy_steps for item in episodes]))
        if episodes else None,
        "average_total_actions": float(np.mean([len(item.action_trace) for item in episodes]))
        if episodes else None,
        "surface_reached": sum(item.surface_reached for item in episodes),
        "construction_actions": dict(Counter(
            type(action).__name__ for item in episodes for player_id, action in item.action_trace
            if player_id == item.learner_id
            and isinstance(action, (BuildSettlementAction, BuildCityAction, BuildRoadAction)))),
        "defer_end_turn": sum(row["gate_action"] != BUILD_NOW
                              and row["action_type"] == EndTurnAction.__name__
                              for row in decisions),
        "risk_actions": dict(Counter(action for item in episodes
                                     for action in item.risk_actions)),
    }
