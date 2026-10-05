"""Read-only Step 3A-17 audit of the frozen T=2 Strategic prior.

The paired continuations intervene only on the first concrete action. Both
branches then use the same frozen Champion controller and future RNG seed.
Neither their returns nor Champion disagreements are training targets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from statistics import mean, median, stdev
from typing import Any

import numpy as np

from app.domain.actions import (BuildCityAction, BuildRoadAction,
                                BuildSettlementAction, BuyDevelopmentAction)
from app.domain.game import RESOURCES, apply_action

from .action_families import ACTION_FAMILIES
from .action_space import ACTION_SPACE_SIZE, get_action_mask
from .construction_city_counterfactual import (LearnerBoundary, capture_boundary,
                                                fork_boundary)
from .diagnostics_metrics import endgame_rank
from .hierarchical_distribution import ACTION_FAMILY_IDS
from .reward import actual_score
from .runtime_context import _source_values
from .strategic_action_surface import FAMILIES
from .strategic_gate_step3a13 import FrozenStrategicExecutors


DEV = FAMILIES.index("BUY_DEVELOPMENT")
_NATIVE_NAMES = ("settlement", "city", "road", "development_purchase",
                 "bank_trade", "end_turn")
_NATIVE_IDS = tuple(ACTION_FAMILIES.index(name) for name in _NATIVE_NAMES)
_COST = {"sheep": 1, "wheat": 1, "ore": 1}


def _softmax(values: np.ndarray, mask: np.ndarray, temperature: float) -> np.ndarray:
    if not mask.any():
        raise ValueError("Empty family mask")
    out = np.zeros(len(values), dtype=np.float64)
    scores = values[mask].astype(np.float64) / temperature
    scores -= scores.max()
    weights = np.exp(scores)
    out[mask] = weights / weights.sum()
    return out


def diagnostic_distribution(native15: np.ndarray, legal377: np.ndarray,
                            candidate6: tuple[bool, ...]) -> dict[str, Any]:
    """Decompose temperature and removed-family renormalization separately."""
    native15 = np.asarray(native15, dtype=np.float64)
    legal377 = np.asarray(legal377, dtype=bool)
    candidate6 = tuple(bool(value) for value in candidate6)
    if native15.shape != (15,) or legal377.shape != (ACTION_SPACE_SIZE,):
        raise ValueError("Wrong frozen-logit or legal-mask shape")
    family_ids = np.asarray(ACTION_FAMILY_IDS)
    full_mask = np.asarray([(legal377 & (family_ids == index)).any()
                            for index in range(15)], dtype=bool)
    strategic15_mask = np.zeros(15, dtype=bool)
    for local, native_id in enumerate(_NATIVE_IDS):
        strategic15_mask[native_id] = candidate6[local]
        if candidate6[local] != full_mask[native_id]:
            raise AssertionError("377 and Strategic masks disagree")
    full_t1 = _softmax(native15, full_mask, 1.0)
    full_t2 = _softmax(native15, full_mask, 2.0)
    six_t1 = _softmax(native15, strategic15_mask, 1.0)
    six_t2 = _softmax(native15, strategic15_mask, 2.0)
    dev_id = _NATIVE_IDS[DEV]
    removed = {name: float(full_t2[index]) for index, name in
               enumerate(ACTION_FAMILIES) if full_mask[index] and not strategic15_mask[index]}
    return {
        "native15_logits": native15.tolist(),
        "native15_legal_mask": full_mask.tolist(),
        "native15_t1_dev_probability": float(full_t1[dev_id]),
        "native15_t2_dev_probability": float(full_t2[dev_id]),
        "strategic6_t1_dev_probability": float(six_t1[dev_id]),
        "strategic6_t2_dev_probability": float(six_t2[dev_id]),
        "mask_inflation_same_t2": float(six_t2[dev_id] - full_t2[dev_id]),
        "temperature_shift_full15": float(full_t2[dev_id] - full_t1[dev_id]),
        "mask_inflation_ratio_same_t2": float(six_t2[dev_id] / full_t2[dev_id])
        if full_t2[dev_id] > 0 else None,
        "removed_mass_t2": float(sum(removed.values())),
        "removed_family_mass_t2": removed,
        "native15_top_legal_family": ACTION_FAMILIES[int(np.argmax(full_t1))],
    }


def _context(game, learner_id: int, max_plan_roads: int) -> dict[str, Any]:
    values, valid = _source_values(game, learner_id, max_plan_roads)
    names = ("settlement.candidate_count", "settlement.min_additional_roads",
             "settlement.eta", "city.eta", "city.best_production_gain",
             "road.acquisition_gap", "knight.plays_needed",
             "robber.blocked_pips", "risk.discard_exposure", "risk.hand_size")
    result = {name: values[name] if valid[name] else None for name in names}
    result["settlement_shortage"] = {
        key: values[f"settlement.shortage.{key}"]
        if valid[f"settlement.shortage.{key}"] else None
        for key in ("wood", "brick", "sheep", "wheat")}
    result["city_shortage"] = {key: values[f"city.shortage.{key}"]
                              if valid[f"city.shortage.{key}"] else None
                              for key in ("wheat", "ore")}
    result["score"] = actual_score(game, learner_id)
    result["new_development_cards"] = dict(Counter(
        game.players[learner_id - 1].new_development_cards))
    return result


def audit_dev_decision(env, champion, snapshot, legal377, record) -> dict[str, Any]:
    if record["selected_family"] != "BUY_DEVELOPMENT":
        raise ValueError("Only selected Development decisions belong here")
    game = env.game
    player_id = env.learning_player_id
    mask = tuple(record["candidate_mask"])
    distribution = diagnostic_distribution(snapshot.old_family_logits, legal377, mask)
    expected = distribution["strategic6_t2_dev_probability"]
    if abs(expected - record["selected_probability"]) > 1e-6:
        raise AssertionError("Replayed Gate probability differs from archived decision")
    after = deepcopy(game)
    apply_action(after, player_id, BuyDevelopmentAction(), after.revision,
                 "step3a17-read-only-cost-probe")
    after_legal = get_action_mask(after, player_id)
    from .strategic_action_surface import detect_strategic_action_surface
    after_surface = detect_strategic_action_surface(after, player_id, after_legal)
    pre_settlement, pre_city = mask[0], mask[1]
    post_settlement, post_city = after_surface.family_mask[0:2]
    candidate_names = [name for name, enabled in zip(FAMILIES, mask) if enabled]
    ranked = sorted((index for index, enabled in enumerate(mask) if enabled),
                    key=lambda index: record["masked_probability"][index], reverse=True)
    non_dev = max((index for index, enabled in enumerate(mask)
                   if enabled and index != DEV),
                  key=lambda index: record["native_family_logits"][index] / 2.0)
    row = {
        "profile": None, "seat": None, "seed": None,
        "pre_policy_step": record["pre_policy_step"],
        "turn": record["turn"], "revision": record["revision"],
        "candidate_mask": list(mask), "candidate_set": candidate_names,
        "native_strategic6_logits": record["native_family_logits"],
        "selected_dev_probability": record["selected_probability"],
        "runner_up_family": FAMILIES[ranked[1]] if len(ranked) > 1 else None,
        "runner_up_probability": (record["masked_probability"][ranked[1]]
                                  if len(ranked) > 1 else None),
        "strongest_non_dev_family": FAMILIES[non_dev],
        "strongest_non_dev_probability": record["masked_probability"][non_dev],
        "entropy": record["entropy"],
        "dev_native6_rank": 1 + sum(mask[index] and
            record["native_family_logits"][index] >
            record["native_family_logits"][DEV] for index in range(6)),
        "champion_counterfactual_family": record["champion_counterfactual_family"],
        "champion_counterfactual_action": record["champion_counterfactual_action"],
        "resource_before": record["application_probe"]["resource_before"],
        "resource_after": record["application_probe"]["resource_after"],
        "construction_before": {"settlement": pre_settlement, "city": pre_city,
                                "road": mask[2]},
        "construction_after": {"settlement": post_settlement,
                               "city": post_city,
                               "road": after_surface.family_mask[2]},
        "construction_opportunity_lost": (pre_settlement and not post_settlement)
                                         or (pre_city and not post_city),
        "context": _context(game, player_id, champion.max_plan_roads),
        **distribution,
    }
    return row


@dataclass(frozen=True)
class CapturedDevState:
    row: dict[str, Any]
    boundary: LearnerBoundary
    source_env: Any
    champion: Any
    dev_action: BuyDevelopmentAction
    alternative_action: Any


def capture_dev_state(env, champion, snapshot, legal377, record) -> CapturedDevState:
    row = audit_dev_decision(env, champion, snapshot, legal377, record)
    non_dev = FAMILIES.index(row["strongest_non_dev_family"])
    executor = FrozenStrategicExecutors(champion)
    alternative = executor.execute(env.game, env.learning_player_id,
                                   FAMILIES[non_dev], method="B_FROZEN_PPO",
                                   mask=legal377, snapshot=snapshot)
    if not alternative.success:
        raise AssertionError(f"Alternative executor failed: {alternative.failure}")
    row["alternative_action"] = repr(alternative.action)
    return CapturedDevState(row, capture_boundary(env), env, champion,
                            BuyDevelopmentAction(), alternative.action)


def _branch(state: CapturedDevState, first_action, future_seed: int,
            gamma: float = .99, horizon: int = 30) -> dict[str, Any]:
    learner_id = state.source_env.learning_player_id
    fork = fork_boundary(state.source_env, state.boundary,
                         continuation_seed=future_seed)
    source_resources = dict(state.boundary.game.players[learner_id - 1].resources)
    action_steps: dict[str, int | None] = {"settlement": None, "city": None,
                                           "next_development": None}
    road_count = 0
    resource_recovery = {name: None for name in _COST}
    total = 0.0
    terminal = truncated = False
    duration = 0
    try:
        while not (terminal or truncated):
            action = first_action if duration == 0 else state.champion.select_action(
                fork.game, learner_id)
            if isinstance(action, BuildSettlementAction) and action_steps["settlement"] is None:
                action_steps["settlement"] = duration
            if isinstance(action, BuildCityAction) and action_steps["city"] is None:
                action_steps["city"] = duration
            if isinstance(action, BuyDevelopmentAction) and duration > 0 and action_steps["next_development"] is None:
                action_steps["next_development"] = duration
            if isinstance(action, BuildRoadAction) and duration <= horizon:
                road_count += 1
            _, reward, terminal, truncated, _ = fork.step_action(action)
            if not isfinite(float(reward)):
                raise AssertionError("Nonfinite paired return")
            total += gamma ** duration * float(reward)
            duration += 1
            if duration <= horizon:
                resources = fork.game.players[learner_id - 1].resources
                for name in _COST:
                    if resource_recovery[name] is None and resources[name] >= source_resources[name]:
                        resource_recovery[name] = duration
        scores = {player.id: actual_score(fork.game, player.id)
                  for player in fork.game.players}
        rank, _ = endgame_rank(scores, learner_id, fork.game.winner_id)
        return {"discounted_return": total, "winner": fork.game.winner_id,
                "win": fork.game.winner_id == learner_id,
                "final_vp": scores[learner_id], "final_rank": rank,
                "policy_steps_after_branch": duration,
                "policy_steps_total": fork.policy_steps,
                "terminal": terminal, "truncated": truncated,
                "first_settlement_step": action_steps["settlement"],
                "first_city_step": action_steps["city"],
                "next_development_step": action_steps["next_development"],
                "road_builds_within_horizon": road_count,
                "resource_recovery_steps": resource_recovery}
    finally:
        fork.close()


def paired_dev_continuation(state: CapturedDevState, future_seed: int) -> dict[str, Any]:
    a = _branch(state, state.dev_action, future_seed)
    b = _branch(state, state.alternative_action, future_seed)
    if a["truncated"] or b["truncated"] or not (a["terminal"] and b["terminal"]):
        raise AssertionError("Paired continuation did not terminate")
    return {"future_seed": future_seed, "dev": a, "non_dev": b,
            "delta_return": a["discounted_return"] - b["discounted_return"],
            "delta_vp": a["final_vp"] - b["final_vp"],
            "delta_rank": a["final_rank"] - b["final_rank"],
            "winner_flip": a["winner"] != b["winner"]}


def select_representative_states(states: list[CapturedDevState], count: int = 24
                                 ) -> list[CapturedDevState]:
    """Deterministic profile-balanced round-robin over audit strata."""
    def labels(row):
        probability = row["selected_dev_probability"]
        result = [">.9" if probability > .9 else ".7-.9" if probability > .7
                  else ".5-.7" if probability > .5 else "<=.5"]
        if row["construction_before"]["settlement"] or row["construction_before"]["city"]:
            result.append("immediate_construction")
        if row["champion_counterfactual_family"] == "BUILD_ROAD":
            result.append("champion_road")
        if row["champion_counterfactual_family"] == "END_TURN":
            result.append("champion_end")
        return result
    profiles = sorted({state.row["profile"] for state in states})
    if not profiles:
        return []
    priority = ("immediate_construction", "<=.5", ".5-.7", ".7-.9",
                ">.9", "champion_road", "champion_end")
    per_profile: dict[str, list[CapturedDevState]] = defaultdict(list)
    for state in states:
        per_profile[state.row["profile"]].append(state)
    chosen: list[CapturedDevState] = []
    for profile_index, profile in enumerate(profiles):
        quota = count // len(profiles) + int(profile_index < count % len(profiles))
        available = sorted(per_profile[profile], key=lambda state: (
            state.row["seed"], state.row["pre_policy_step"]))
        pools = {label: [state for state in available if label in labels(state.row)]
                 for label in priority}
        selected_keys = set()
        index = 0
        while len(selected_keys) < min(quota, len(available)):
            for label in priority:
                pool = pools[label]
                if index >= len(pool):
                    continue
                state = pool[index]
                key = (state.row["seed"], state.row["pre_policy_step"])
                if key not in selected_keys:
                    chosen.append(state)
                    selected_keys.add(key)
                    if len(selected_keys) >= quota:
                        break
            index += 1
            if index > len(available):
                break
    return chosen


def _stats(numbers) -> dict[str, Any]:
    values = [float(value) for value in numbers if value is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": mean(values), "std": stdev(values)
            if len(values) > 1 else 0.0, "median": median(values),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "min": min(values), "max": max(values)}


def summarize_dev_audit(rows: list[dict[str, Any]],
                        paired: list[dict[str, Any]]) -> dict[str, Any]:
    profile = Counter(row["profile"] for row in rows)
    champion = Counter(row["champion_counterfactual_family"] for row in rows)
    rank = Counter(str(row["dev_native6_rank"]) for row in rows)
    bins = Counter(">.9" if row["selected_dev_probability"] > .9 else
                   ".7-.9" if row["selected_dev_probability"] > .7 else
                   ".5-.7" if row["selected_dev_probability"] > .5 else "<=.5"
                   for row in rows)
    by_set = defaultdict(list)
    logit = defaultdict(list)
    removed = Counter()
    for row in rows:
        by_set["+".join(row["candidate_set"])].append(row["selected_dev_probability"])
        for name, probability in row["removed_family_mass_t2"].items():
            removed[name] += probability
        for local, name in enumerate(FAMILIES):
            if row["candidate_mask"][local]:
                logit[name].append(row["native_strategic6_logits"][local])
    states = defaultdict(list)
    for pair in paired:
        states[pair["state_id"]].append(pair)
    state_summaries = []
    row_by_id = {f"{row['profile']}-{row['seed']}-{row['pre_policy_step']}": row
                 for row in rows}
    for state_id, items in states.items():
        deltas = [item["delta_return"] for item in items]
        positive = sum(value > 1e-9 for value in deltas)
        negative = sum(value < -1e-9 for value in deltas)
        local_std = stdev(deltas) if len(deltas) > 1 else 0.0
        ci_radius = 2.365 * local_std / len(deltas) ** .5
        state_summaries.append({"state_id": state_id, "n": len(items),
                                "mean_delta": mean(deltas),
                                "std_delta": local_std,
                                "approx_ci95": [mean(deltas) - ci_radius,
                                                mean(deltas) + ci_radius],
                                "dev_better_seeds": positive,
                                "non_dev_better_seeds": negative,
                                "tie_seeds": sum(abs(value) <= 1e-9 for value in deltas),
                                "probability": row_by_id[state_id]["selected_dev_probability"],
                                "alternative_family": row_by_id[state_id].get(
                                    "strongest_non_dev_family",
                                    row_by_id[state_id].get("second_family")),
                                "context": row_by_id[state_id]["context"]})
    correlations = {}
    if len(state_summaries) >= 3:
        for context_name in ("probability", "score", "risk.hand_size",
                             "risk.discard_exposure", "settlement.eta", "city.eta",
                             "robber.blocked_pips", "knight.plays_needed",
                             "road.acquisition_gap", "settlement.candidate_count"):
            points = []
            for item in state_summaries:
                x = (item["probability"] if context_name == "probability" else
                     item["context"].get(context_name))
                if x is not None:
                    points.append((float(x), item["mean_delta"]))
            if len(points) >= 3 and np.std([p[0] for p in points]) > 1e-12:
                correlations[context_name] = {"n": len(points),
                    "pearson": float(np.corrcoef(np.asarray(points).T)[0, 1])}
    within_variance = mean(item["std_delta"] ** 2 for item in state_summaries) if state_summaries else 0.0
    between_means_variance = (float(np.var([item["mean_delta"] for item in state_summaries], ddof=1))
                              if len(state_summaries) > 1 else 0.0)
    mean_sample_count = mean(item["n"] for item in state_summaries) if state_summaries else 1.0
    corrected_between = max(0.0, between_means_variance - within_variance / mean_sample_count)
    return {
        "dev_decisions": len(rows), "profile": dict(profile),
        "states_with_new_development_cards": sum(
            bool(row["context"]["new_development_cards"]) for row in rows),
        "dev_context": {name: _stats(row["context"].get(name) for row in rows)
                        for name in ("score", "risk.hand_size",
                                     "risk.discard_exposure", "settlement.eta",
                                     "city.eta", "settlement.candidate_count",
                                     "settlement.min_additional_roads",
                                     "city.best_production_gain", "road.acquisition_gap",
                                     "knight.plays_needed", "robber.blocked_pips")},
        "champion_counterfactual": dict(champion), "native6_rank": dict(rank),
        "probability_bins": dict(bins),
        "probabilities": {name: _stats(row[name] for row in rows) for name in
            ("selected_dev_probability", "native15_t1_dev_probability",
             "native15_t2_dev_probability", "strategic6_t1_dev_probability",
             "mask_inflation_same_t2", "temperature_shift_full15",
             "mask_inflation_ratio_same_t2", "removed_mass_t2")},
        "candidate_sets": {name: {"count": len(values), "dev_probability": _stats(values)}
                           for name, values in sorted(by_set.items(),
                                                     key=lambda item: -len(item[1]))},
        "removed_family_expected_mass_sum": dict(removed),
        "native_logit_by_available_family": {name: _stats(values)
                                              for name, values in logit.items()},
        "immediate_settlement_competition": sum(row["construction_before"]["settlement"] for row in rows),
        "immediate_city_competition": sum(row["construction_before"]["city"] for row in rows),
        "construction_opportunity_lost": sum(row["construction_opportunity_lost"] for row in rows),
        "paired_states": len(states), "paired_futures": len(paired),
        "paired_delta": _stats(item["delta_return"] for item in paired),
        "state_mean_delta": _stats(item["mean_delta"] for item in state_summaries),
        "state_summaries": state_summaries,
        "paired_variance_decomposition": {
            "mean_within_state_sample_variance": within_variance,
            "between_state_means_variance": between_means_variance,
            "within_divided_by_futures": within_variance / mean_sample_count,
            "corrected_between_state_variance": corrected_between,
            "icc_like": (corrected_between / (corrected_between + within_variance)
                         if corrected_between + within_variance > 0 else None)},
        "dev_advantage_states": sum(item["mean_delta"] > .02 for item in state_summaries),
        "non_dev_advantage_states": sum(item["mean_delta"] < -.02 for item in state_summaries),
        "near_tie_states": sum(abs(item["mean_delta"]) <= .02 for item in state_summaries),
        "sign_stable_7_of_8": sum(max(item["dev_better_seeds"], item["non_dev_better_seeds"]) >= 7
                                  for item in state_summaries),
        "approx_ci95_excludes_zero": sum(
            item["approx_ci95"][0] > 0 or item["approx_ci95"][1] < 0
            for item in state_summaries),
        "unstable_mixed_sign_2_each": sum(item["dev_better_seeds"] >= 2
                                          and item["non_dev_better_seeds"] >= 2
                                          for item in state_summaries),
        "context_pearson_exploratory": correlations,
        "winner_flips": sum(item["winner_flip"] for item in paired),
        "learner_win_flips": sum(item["dev"]["win"] != item["non_dev"]["win"]
                                 for item in paired),
        "dev_wins": sum(item["dev"]["win"] for item in paired),
        "non_dev_wins": sum(item["non_dev"]["win"] for item in paired),
        "own_win_flip_return_direction_mismatch": sum(
            (item["dev"]["win"] != item["non_dev"]["win"])
            and ((item["delta_return"] > 0) != item["dev"]["win"])
            for item in paired),
        "delta_vp": _stats(item["delta_vp"] for item in paired),
        "delta_rank": _stats(item["delta_rank"] for item in paired),
        "delta_policy_steps": _stats(item["dev"]["policy_steps_after_branch"]
                                     - item["non_dev"]["policy_steps_after_branch"]
                                     for item in paired),
        "construction_within_30": {
            branch: {name: sum(item[branch][f"first_{name}_step"] is not None
                               and item[branch][f"first_{name}_step"] <= 30
                               for item in paired)
                     for name in ("settlement", "city")}
            for branch in ("dev", "non_dev")},
        "construction_timing_delta_both_observed": {
            name: _stats(item["dev"][f"first_{name}_step"]
                         - item["non_dev"][f"first_{name}_step"]
                         for item in paired
                         if item["dev"][f"first_{name}_step"] is not None
                         and item["non_dev"][f"first_{name}_step"] is not None)
            for name in ("settlement", "city")},
        "road_build_count_delta_within_30": _stats(
            item["dev"]["road_builds_within_horizon"]
            - item["non_dev"]["road_builds_within_horizon"] for item in paired),
        "next_dev_within_30": {branch: sum(
            item[branch]["next_development_step"] is not None
            and item[branch]["next_development_step"] <= 30 for item in paired)
            for branch in ("dev", "non_dev")},
        "resource_recovery": {branch: {resource: _stats(
            item[branch]["resource_recovery_steps"][resource] for item in paired)
            for resource in _COST} for branch in ("dev", "non_dev")},
    }
