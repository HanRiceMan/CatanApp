"""Paired one-decision CITY_ONLY rollouts with the frozen Champion thereafter.

This is a diagnostic, not a teacher label or an RL update. Both branches start
from a deep-copied learner boundary including the GameRandomSource counter.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from statistics import mean, median
from typing import Any, Callable

import numpy as np

from app.agents.heuristic import HeuristicAgent
from app.domain.actions import Action, BuildCityAction, BuyDevelopmentAction
from app.domain.game import GameState, apply_action, create_game

from .construction_timing import FrozenConstructionExecutors, timing_surface
from .construction_timing_rollout import _HeuristicRollOrder, _TradeResponseOnly
from .diagnostics_metrics import endgame_rank
from .env import CatanEnv, CatanEnvConfig
from .macro_teacher_dataset import PROFILE_ROLES, _champion_id_for_seat
from .reward import actual_score
from .runtime_context import _source_values
from .settlement_planning_agent import SettlementPlanningAgent


GAMMA = 0.99
CONTEXT_NAMES = (
    "risk.hand_size", "risk.discard_exposure", "robber.blocked_pips",
    "city.best_production_gain", "city.eta", "settlement.candidate_exists",
    "settlement.eta", "settlement.shortage.wood",
    "settlement.shortage.brick", "settlement.shortage.sheep",
    "settlement.shortage.wheat", "road.acquisition_gap",
    "knight.plays_needed", "knight.development_affordable",
    "risk.bank_trade_options",
)


@dataclass(frozen=True)
class LearnerBoundary:
    game: GameState
    episode_seed: int
    policy_steps: int
    request_number: int
    learning_score: int
    learner_auxiliary_actions: int
    controllers: dict[int, str]


def capture_boundary(env: CatanEnv) -> LearnerBoundary:
    game = env._require_game()
    if game.phase == "game_over" or env.pending_external_player_id is not None:
        raise ValueError("A live internal learner boundary is required")
    if env._required_player_id(game) != env.learning_player_id:
        raise ValueError("Not the learner's decision boundary")
    if env.episode_seed is None or env._is_truncated:
        raise ValueError("Episode seed is missing or already truncated")
    return LearnerBoundary(deepcopy(game), env.episode_seed, env.policy_steps,
                           env._request_number, env._learning_score,
                           env._learner_auxiliary_actions, env.controllers)


def fork_boundary(source: CatanEnv, boundary: LearnerBoundary,
                  observer: Callable[[GameState, int, Action], None] | None = None,
                  *, continuation_seed: int | None = None) -> CatanEnv:
    """Copy all reward/time/RNG state; optional new future RNG is diagnostic only."""
    fork = CatanEnv(source.config,
                    initial_placement_agent=source._initial_placement_agent,
                    opponent_agents=source._opponent_agents,
                    learner_auxiliary_agent=source._learner_auxiliary_agent,
                    action_observer=observer)
    fork.game = deepcopy(boundary.game)
    if continuation_seed is not None:
        # Extra continuations keep the historical board/hand fixed, but draw a
        # new *future* random stream shared by BUILD and DEFER. Never use this
        # for the primary paired observation.
        fork.game.random_source.seed = continuation_seed
        fork.game.random_source.counter = 0
    fork.episode_seed = boundary.episode_seed
    fork.policy_steps = boundary.policy_steps
    fork._request_number = boundary.request_number
    fork._learning_score = boundary.learning_score
    fork._learner_auxiliary_actions = boundary.learner_auxiliary_actions
    fork._controllers = dict(boundary.controllers)
    fork._pending_external_player_id = None
    fork._is_truncated = False
    return fork


def create_profile_episode(champion_policy, robber_policy, *, seed: int,
                           seat: int, profile: str) -> tuple[CatanEnv, SettlementPlanningAgent, int]:
    if profile not in PROFILE_ROLES or seat not in range(1, 5):
        raise ValueError("Invalid profile or seat")
    learner_id = _champion_id_for_seat(seed, seat)
    planning = champion_policy.settlement_planning_config
    robber_planning = robber_policy.settlement_planning_config
    if planning is None or robber_planning is None:
        raise ValueError("Frozen PPO model lacks Planner configuration")
    champion = SettlementPlanningAgent(champion_policy, **planning)
    probe = create_game(seed, ai_player_ids=(1, 2, 3, 4))
    heuristic = HeuristicAgent()
    from app.domain.game import pending_ai_player_id
    for index in range(16):
        if probe.phase != "rolling_order":
            break
        pid = pending_ai_player_id(probe)
        apply_action(probe, pid, heuristic.select_action(probe, pid),
                     probe.revision, f"city-cf-seat-probe-{index}")
    if probe.phase == "rolling_order":
        raise RuntimeError("Seat probe did not finish")
    opponents = {}
    for pid, role in zip((pid for pid in probe.seat_order if pid != learner_id),
                         PROFILE_ROLES[profile], strict=True):
        agent = (HeuristicAgent() if role == "rule" else
                 SettlementPlanningAgent(champion_policy if role == "champion"
                                         else robber_policy,
                                         **(planning if role == "champion"
                                            else robber_planning)))
        opponents[pid] = _HeuristicRollOrder(agent)
    env = CatanEnv(
        CatanEnvConfig(learning_player_id=learner_id, allow_player_trades=True,
                       observation_version="v2", max_policy_steps=5000),
        initial_placement_agent=champion, opponent_agents=opponents,
        learner_auxiliary_agent=_TradeResponseOnly(champion))
    env.reset(seed=seed)
    if env.game.phase != "rolling_order" and env.game.seat_order.index(learner_id) + 1 != seat:
        raise AssertionError("Learner seat differs from requested seat")
    return env, champion, learner_id


def _outcome(fork: CatanEnv, learner_id: int, action_types: list[str],
             *, discounted_total: float, discounted_vp: float,
             discounted_terminal: float, duration: int,
             first_action: Action, truncated: bool) -> dict[str, Any]:
    game = fork._require_game()
    scores = {player.id: actual_score(game, player.id) for player in game.players}
    rank, _ = endgame_rank(scores, learner_id, game.winner_id)
    player = game.players[learner_id - 1]
    return {
        "first_action": type(first_action).__name__,
        "discounted_environment_return": discounted_total,
        "discounted_vp_component": discounted_vp,
        "discounted_terminal_component": discounted_terminal,
        "terminal_outcome": (1 if game.winner_id == learner_id else -1)
        if not truncated else None,
        "win": game.winner_id == learner_id if not truncated else None,
        "winner_id": game.winner_id,
        "final_vp": scores[learner_id], "final_rank": rank,
        "policy_steps_total": fork.policy_steps,
        "policy_steps_after_branch": duration,
        "city_count": player.cities, "settlement_count": player.settlements,
        "development_count": (len(player.development_cards)
                              + len(player.new_development_cards)
                              + len(player.used_development_cards)),
        "longest_road_title": game.longest_road_holder_id == learner_id,
        "largest_army_title": game.largest_army_holder_id == learner_id,
        "discard_actions_after_branch": sum(name in {"DiscardHalfAction",
                                                     "DiscardResourcesAction"}
                                             for name in action_types),
        "development_purchases_after_branch": action_types.count(
            BuyDevelopmentAction.__name__),
        "truncated": truncated,
        "final_rng_counter": game.random_source.counter,
    }


def continue_branch(source: CatanEnv, boundary: LearnerBoundary,
                    first_action: Action, champion: SettlementPlanningAgent,
                    *, gamma: float = GAMMA,
                    continuation_seed: int | None = None) -> dict[str, Any]:
    learner_id = source.learning_player_id
    action_types: list[str] = []
    def observe(_game, pid, action):
        if pid == learner_id:
            action_types.append(type(action).__name__)
    fork = fork_boundary(source, boundary, observe,
                         continuation_seed=continuation_seed)
    before_rng = fork.game.random_source.counter
    if continuation_seed is None and before_rng != boundary.game.random_source.counter:
        raise AssertionError("Fork RNG was not copied")
    total = vp = terminal_component = 0.0
    duration = 0
    done = truncated = False
    try:
        while not (done or truncated):
            action = first_action if duration == 0 else champion.select_action(
                fork.game, learner_id)
            _, reward, done, truncated, info = fork.step_action(action)
            components = info["reward_components"]
            if not all(isfinite(float(value)) for value in (reward,
                    components["victory_point_reward"], components["terminal_reward"])):
                raise AssertionError("Nonfinite counterfactual reward")
            if abs(reward - components["total"]) > 1e-7:
                raise AssertionError("Reward decomposition differs from Environment")
            if abs(reward - components["victory_point_reward"]
                   - components["terminal_reward"]) > 1e-7:
                raise AssertionError("Counterfactual unexpectedly enabled another reward")
            discount = gamma ** duration
            total += discount * reward
            vp += discount * components["victory_point_reward"]
            terminal_component += discount * components["terminal_reward"]
            duration += 1
        return _outcome(fork, learner_id, action_types,
                        discounted_total=total, discounted_vp=vp,
                        discounted_terminal=terminal_component,
                        duration=duration, first_action=first_action,
                        truncated=truncated)
    finally:
        fork.close()


def paired_city_rollout(source: CatanEnv, champion: SettlementPlanningAgent,
                        boundary: LearnerBoundary, *, profile: str, seat: int,
                        continuation_seed: int | None = None) -> dict[str, Any]:
    learner_id = source.learning_player_id
    if source.game != boundary.game:
        raise AssertionError("Source state changed since boundary capture")
    surface = timing_surface(source.game, learner_id)
    if not surface.eligible or surface.subtype != "CITY_ONLY":
        raise ValueError("Paired rollout requires CITY_ONLY")
    candidates = FrozenConstructionExecutors(champion).candidates(
        source.game, learner_id, surface)
    if not candidates.delegatable or not isinstance(candidates.build_action, BuildCityAction):
        raise ValueError("BUILD/DEFER candidate unavailable")
    before_rng = source.game.random_source.counter
    champion_action = champion.select_action(source.game, learner_id)
    if source.game != boundary.game or source.game.random_source.counter != before_rng:
        raise AssertionError("Candidate/Champion diagnostic mutated source GameState")
    values, valid = _source_values(source.game, learner_id,
                                   max_plan_roads=champion.max_plan_roads)
    context = {name: values[name] if valid[name] else None for name in CONTEXT_NAMES}
    context["score"] = actual_score(source.game, learner_id)
    player = source.game.players[learner_id - 1]
    context["city_count"] = player.cities
    context["settlement_count"] = player.settlements
    context["site_count"] = player.cities + player.settlements
    context["settlement_shortage_total"] = (sum(
        values[f"settlement.shortage.{resource}"]
        for resource in ("wood", "brick", "sheep", "wheat"))
        if all(valid[f"settlement.shortage.{resource}"]
               for resource in ("wood", "brick", "sheep", "wheat")) else None)
    build = continue_branch(source, boundary, candidates.build_action, champion,
                            continuation_seed=continuation_seed)
    defer = continue_branch(source, boundary, candidates.defer_action, champion,
                            continuation_seed=continuation_seed)
    if source.game != boundary.game:
        raise AssertionError("Counterfactual branch mutated source GameState")
    deltas = {
        "total": build["discounted_environment_return"] - defer["discounted_environment_return"],
        "vp": build["discounted_vp_component"] - defer["discounted_vp_component"],
        "terminal": build["discounted_terminal_component"] - defer["discounted_terminal_component"],
        "terminal_outcome": (build["terminal_outcome"] - defer["terminal_outcome"]
                             if not (build["truncated"] or defer["truncated"]) else None),
        "final_vp": build["final_vp"] - defer["final_vp"],
        "final_rank": build["final_rank"] - defer["final_rank"],
        "policy_steps_total": build["policy_steps_total"] - defer["policy_steps_total"],
    }
    return {
        "game_id": f"city-cf-{profile}-{boundary.episode_seed}-{learner_id}",
        "state_id": f"city-cf-{profile}-{boundary.episode_seed}-{learner_id}-{boundary.policy_steps}",
        "game_seed": boundary.episode_seed, "profile": profile, "seat": seat,
        "learner_id": learner_id, "pre_policy_steps": boundary.policy_steps,
        "pre_turn": boundary.game.turn_number,
        "pre_rng_seed": boundary.game.random_source.seed,
        "pre_rng_counter": boundary.game.random_source.counter,
        "continuation_seed": continuation_seed,
        "champion_action": type(champion_action).__name__,
        "champion_build": isinstance(champion_action, BuildCityAction),
        "defer_action_family": type(candidates.defer_action).__name__,
        "context": context, "build": build, "defer": defer, "delta": deltas,
    }


def difference_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)), "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)), "min": float(array.min()),
        "max": float(array.max()),
        "build_better": int((array > 1e-9).sum()),
        "defer_better": int((array < -1e-9).sum()),
        "tie": int((np.abs(array) <= 1e-9).sum()),
    }


def summarize_city_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"pairs": 0}
    if len({record["state_id"] for record in records}) != len(records):
        raise ValueError("Duplicate state_id in primary dataset")
    valid = [record for record in records
             if not (record["build"]["truncated"] or record["defer"]["truncated"])]
    paired_metrics = {key: difference_stats([record["delta"][key] for record in valid])
                      for key in ("total", "vp", "terminal", "terminal_outcome",
                                  "final_vp", "final_rank", "policy_steps_total")}
    branch_returns = {
        branch: {component: difference_stats([
            record[branch][component] for record in valid])
            for component in ("discounted_environment_return",
                              "discounted_vp_component",
                              "discounted_terminal_component")}
        for branch in ("build", "defer")}
    branch_outcomes = {
        branch: {
            "wins": sum(record[branch]["win"] is True for record in valid),
            **{field: difference_stats([record[branch][field]
                                       for record in valid])
               for field in ("final_vp", "final_rank", "policy_steps_total",
                             "city_count", "settlement_count", "development_count",
                             "discard_actions_after_branch")},
            "longest_road_title": sum(record[branch]["longest_road_title"]
                                      for record in valid),
            "largest_army_title": sum(record[branch]["largest_army_title"]
                                      for record in valid),
        } for branch in ("build", "defer")}
    alignment = Counter()
    outcome_alignment = Counter()
    for record in valid:
        total = record["delta"]["total"]
        terminal = record["delta"]["terminal"]
        if abs(total) <= 1e-9 or abs(terminal) <= 1e-9:
            alignment["TIE_OR_ZERO_TERMINAL"] += 1
        elif total > 0 and terminal > 0:
            alignment["A_BOTH_BUILD"] += 1
        elif total < 0 and terminal < 0:
            alignment["B_BOTH_DEFER"] += 1
        elif total > 0:
            alignment["C_REWARD_BUILD_TERMINAL_DEFER"] += 1
        else:
            alignment["D_REWARD_DEFER_TERMINAL_BUILD"] += 1
        outcome = record["delta"]["terminal_outcome"]
        if abs(total) <= 1e-9 or outcome == 0:
            outcome_alignment["TIE_OR_SAME_WINNER"] += 1
        elif total > 0 and outcome > 0:
            outcome_alignment["A_BOTH_BUILD"] += 1
        elif total < 0 and outcome < 0:
            outcome_alignment["B_BOTH_DEFER"] += 1
        elif total > 0:
            outcome_alignment["C_REWARD_BUILD_WIN_DEFER"] += 1
        else:
            outcome_alignment["D_REWARD_DEFER_WIN_BUILD"] += 1
    by_champion = {}
    for champion_build in (True, False):
        subset = [record for record in valid if record["champion_build"] == champion_build]
        align = [record for record in subset if abs(record["delta"]["total"]) > 1e-9]
        correct = sum((record["delta"]["total"] > 0) == champion_build
                      for record in align)
        by_champion["BUILD" if champion_build else "DEFER"] = {
            "count": len(subset), "higher_return_fraction_excluding_ties":
                correct / len(align) if align else None,
            "tie_count": len(subset) - len(align),
            "delta_total": difference_stats([r["delta"]["total"] for r in subset]),
            "delta_terminal": difference_stats([r["delta"]["terminal"] for r in subset]),
        }
    by_defer_family = {}
    for family in sorted({record["defer_action_family"] for record in valid}):
        subset = [record for record in valid if record["defer_action_family"] == family]
        by_defer_family[family] = {
            key: difference_stats([record["delta"][key] for record in subset])
            for key in ("total", "terminal", "final_vp")}
    by_profile = {}
    for profile in sorted({record["profile"] for record in valid}):
        subset = [record for record in valid if record["profile"] == profile]
        by_profile[profile] = {
            "pairs": len(subset),
            "champion_defer": sum(not record["champion_build"] for record in subset),
            "delta_total": difference_stats([record["delta"]["total"] for record in subset]),
            "delta_terminal": difference_stats([record["delta"]["terminal"] for record in subset]),
            "delta_final_vp": difference_stats([record["delta"]["final_vp"] for record in subset]),
        }
    by_seat = {
        str(seat): {"pairs": len(subset),
                    "champion_defer": sum(not record["champion_build"]
                                          for record in subset),
                    "delta_total": difference_stats(
                        [record["delta"]["total"] for record in subset])}
        for seat in sorted({record["seat"] for record in valid})
        for subset in [[record for record in valid if record["seat"] == seat]]}
    by_game: dict[str, list[float]] = {}
    for record in valid:
        by_game.setdefault(record.get("game_id", record["state_id"]), []).append(
            record["delta"]["total"])
    game_mean_deltas = [mean(deltas) for deltas in by_game.values()]
    context = {}
    for group_name, sign in (("BUILD_BETTER", 1), ("DEFER_BETTER", -1)):
        subset = [record for record in valid if record["delta"]["total"] * sign > 1e-9]
        context[group_name] = {name: {
            "count": len(vals), "mean": float(mean(vals)) if vals else None,
            "median": float(median(vals)) if vals else None,
        } for name in (*CONTEXT_NAMES, "score", "city_count", "settlement_count",
                       "site_count", "settlement_shortage_total")
            for vals in [[float(record["context"][name]) for record in subset
                          if record["context"].get(name) is not None]]}
    return {
        "pairs": len(records), "valid_pairs": len(valid),
        "truncated_pairs": len(records) - len(valid),
        "profile": dict(Counter(record["profile"] for record in records)),
        "seat": dict(Counter(record["seat"] for record in records)),
        "champion_decision": dict(Counter("BUILD" if r["champion_build"] else "DEFER"
                                          for r in valid)),
        "defer_family": dict(Counter(r["defer_action_family"] for r in valid)),
        "branch_returns": branch_returns, "branch_outcomes": branch_outcomes,
        "paired_delta": paired_metrics,
        "reward_alignment": dict(alignment),
        "reward_vs_undiscounted_win_alignment": dict(outcome_alignment),
        "near_ties_abs_delta_q_at_most_0_01": sum(
            abs(record["delta"]["total"]) <= 0.01 for record in valid),
        "game_cluster": {"games": len(by_game),
                         "states_per_game": difference_stats(
                             [len(deltas) for deltas in by_game.values()]),
                         "mean_delta_q_per_game": difference_stats(game_mean_deltas)},
        "champion_correspondence": by_champion,
        "by_defer_family": by_defer_family, "by_profile": by_profile,
        "by_seat": by_seat,
        "context_by_return_sign": context,
    }
