"""Step 3A-18 read-only trajectory, paired-policy, and prior screens.

No optimizer, reward, mask, executor, or Champion weights are modified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .construction_city_counterfactual import capture_boundary
from .observation import encode_observation, get_observation
from .run_strategic_gate_step3a14 import _episode_record, _variant_stats, PROFILES
from .run_strategic_gate_step3a16 import (
    EVAL_SEED_START, _agents, _latent_dim,
)
from .runtime_context import _source_values, encode_runtime_context
from .strategic_action_surface import FAMILIES, family_of
from .strategic_gate_rollout_step3a14 import run_strategic_episode
from .strategic_gate_ppo_step3a16 import load_strategic_artifact
from .strategic_prior_step3a18 import offline_prior_audit, prior_scores


DEFAULT_SOURCE = Path("experiments/strategic_gate_step3a16_20261005")
DEFAULT_SHADOW = Path("experiments/strategic_gate_shadow_step3a13_20261004/decisions.jsonl")
DEFAULT_DEV = Path("experiments/strategic_dev_audit_step3a17_20261005_balanced/dev_decisions.json")
DEFAULT_OUTPUT = Path("experiments/strategic_prior_step3a18_20261006")
SHORTLIST = ("native_t2", "rank_b1.0", "rank_b1.5", "gap_c2.0")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    data = np.asarray(values, dtype=float)
    return {"n": len(data), "mean": float(data.mean()), "median": float(np.median(data)),
            "std": float(data.std()), "p10": float(np.percentile(data, 10)),
            "p90": float(np.percentile(data, 90)), "min": float(data.min()),
            "max": float(data.max())}


def _readiness(game, learner_id: int, max_plan_roads: int) -> dict[str, Any]:
    values, valid = _source_values(game, learner_id, max_plan_roads)
    names = (
        "settlement.immediate_build", "city.immediate_build", "settlement.eta",
        "city.eta", "settlement.candidate_count", "city.upgradeable_count",
        "risk.hand_size", "risk.discard_exposure",
        "settlement.shortage.wood", "settlement.shortage.brick",
        "settlement.shortage.sheep", "settlement.shortage.wheat",
        "city.shortage.wheat", "city.shortage.ore",
    )
    return {name: values[name] if valid[name] else None for name in names}


def _future_seed(seed: int, index: int) -> int:
    return 318000000 + seed * 17 + index


def offline(output: Path) -> dict[str, Any]:
    shadow = [json.loads(line) for line in DEFAULT_SHADOW.read_text(encoding="utf-8").splitlines()]
    dev = _read(DEFAULT_DEV)["rows"]
    result = offline_prior_audit(shadow, dev)
    result["source"] = {"shadow": str(DEFAULT_SHADOW), "dev": str(DEFAULT_DEV)}
    _write(output / "offline_prior.json", result)
    return result


def _trace_window(rows: list[dict], first_step: int) -> dict[str, Any]:
    selected = [row for row in rows if first_step <= row["policy_step"] < first_step + 30]
    counts = Counter(row["family"] for row in selected if row["phase"] == "action")
    development = [row["policy_step"] for row in selected
                   if row["family"] == "BUY_DEVELOPMENT"]
    return {"action_family_counts": dict(counts), "readiness": selected,
            "additional_development_count": len(development),
            "next_development_step": (development[0] - first_step if development else None)}


def _outcome(episode) -> dict[str, Any]:
    return {"winner": episode.winner, "won": episode.winner == episode.learner_id,
            "vp": episode.final_vp, "rank": episode.final_rank,
            "policy_steps": episode.policy_steps, "terminal": episode.terminal,
            "truncated": episode.truncated, "return": episode.discounted_return_from_start}


def summarize_trajectory_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive cascade/readiness statistics, not causal state labels."""
    changed = [row for row in records if row["first_divergence"]]
    result: dict[str, Any] = {"games": len(records), "first_divergence": len(changed),
                              "first_family_pairs": dict(Counter(
                                  f"{row['first']['safety_family']} -> {row['first']['full_family']}"
                                  for row in changed)),
                              "first_dev_games": sum(row["first"]["full_family"] == "BUY_DEVELOPMENT"
                                                     for row in changed),
                              "outcome_difference": {key: _stats([row["outcome_difference"][key]
                                                                   for row in records])
                                                     for key in ("learner_win_delta", "vp", "rank", "policy_steps")},
                              "winner_changed": sum(row["outcome_difference"]["winner_changed"]
                                                    for row in records)}
    for branch in ("safety", "full_t2"):
        family = Counter()
        valid_values: dict[str, list[float]] = defaultdict(list)
        readiness_by_step: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        additional_dev = []
        next_dev = []
        for row in changed:
            window = row["window30"][branch]
            family.update(window["action_family_counts"])
            dev_count = window["additional_development_count"]
            if branch == "full_t2" and row["first"]["full_family"] == "BUY_DEVELOPMENT":
                dev_count -= 1
            additional_dev.append(dev_count)
            later_dev = [decision["policy_step"] - row["first"]["policy_step"]
                         for decision in window["readiness"]
                         if decision["family"] == "BUY_DEVELOPMENT" and
                         decision["policy_step"] > row["first"]["policy_step"]]
            if later_dev:
                next_dev.append(min(later_dev))
            for decision in window["readiness"]:
                relative = decision["policy_step"] - row["first"]["policy_step"]
                for key, value in decision["context"].items():
                    if value is not None:
                        valid_values[key].append(float(value))
                        readiness_by_step[relative][key].append(float(value))
        result[branch] = {
            "window30_action_families": dict(family),
            "additional_dev_after_first": _stats(additional_dev),
            "next_dev_step": _stats(next_dev),
            "readiness_all_steps": {key: _stats(values) for key, values in valid_values.items()},
            "readiness_by_relative_step": {
                str(step): {key: _stats(values) for key, values in mapping.items()}
                for step, mapping in sorted(readiness_by_step.items())},
        }
    return result


def summarize_policy_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_state[pair["state_id"]].append(pair)
    state_mean = {key: mean([pair["delta_return"] for pair in group])
                  for key, group in by_state.items()}
    within_variances = [float(np.var([pair["delta_return"] for pair in group], ddof=1))
                        for group in by_state.values() if len(group) > 1]
    observed_between = (float(np.var(list(state_mean.values()), ddof=1))
                        if len(state_mean) > 1 else 0.0)
    mean_within = mean(within_variances) if within_variances else 0.0
    future_per_state = min(map(len, by_state.values())) if by_state else 0
    return {"states": len(by_state), "paired_futures": len(pairs),
            "profile_states": dict(Counter(key.split("-")[0] for key in by_state)),
            "first_pair_states": dict(Counter(group[0]["first_pair"]
                                      for group in by_state.values())),
            "delta_return_all": _stats([pair["delta_return"] for pair in pairs]),
            "delta_return_state_mean": _stats(list(state_mean.values())),
            "mean_within_state_sample_variance": mean_within,
            "observed_between_state_sample_variance": observed_between,
            "within_variance_over_futures": (mean_within / future_per_state
                                             if future_per_state else None),
            "corrected_between_state_variance": max(
                0.0, observed_between - mean_within / future_per_state)
                if future_per_state else None,
            "full_mean_better": sum(value > .02 for value in state_mean.values()),
            "safety_mean_better": sum(value < -.02 for value in state_mean.values()),
            "mean_near_tie": sum(abs(value) <= .02 for value in state_mean.values()),
            "stable_7_of_8": sum(max(sum(pair["delta_return"] > .02 for pair in group),
                                     sum(pair["delta_return"] < -.02 for pair in group)) >= 7
                                 for group in by_state.values()),
            "mixed_sign_at_least_2_each": sum(
                sum(pair["delta_return"] > .02 for pair in group) >= 2 and
                sum(pair["delta_return"] < -.02 for pair in group) >= 2
                for group in by_state.values()),
            "winner_changed": sum(pair["safety"]["winner"] != pair["full"]["winner"]
                                  for pair in pairs),
            "safety_learner_wins": sum(pair["safety"]["won"] for pair in pairs),
            "full_learner_wins": sum(pair["full"]["won"] for pair in pairs),
            "delta_vp": _stats([pair["delta_vp"] for pair in pairs]),
            "delta_rank": _stats([pair["delta_rank"] for pair in pairs]),
            "delta_steps": _stats([pair["full"]["policy_steps"] - pair["safety"]["policy_steps"]
                                   for pair in pairs]),
            "full_better_by_first_pair": {
                family_pair: _stats([pair["delta_return"] for pair in pairs
                                     if pair["first_pair"] == family_pair])
                for family_pair in sorted({pair["first_pair"] for pair in pairs})},
        }


def trajectories(output: Path, *, pair_states: int = 24, future_seeds: int = 8) -> dict[str, Any]:
    champion, robber = _agents()
    gate, critic, _ = load_strategic_artifact(
        DEFAULT_SOURCE / "Strategic-T2-preupdate", _latent_dim(champion))
    captured: list[dict[str, Any]] = []
    comparisons = []
    for pindex, profile in enumerate(PROFILES):
        for index in range(12):
            seed = EVAL_SEED_START + pindex * 1000 + index
            seat = index % 4 + 1
            baseline = _read(DEFAULT_SOURCE / f"eval-safety-{profile}-{seed}.json")
            pre = _read(DEFAULT_SOURCE / f"eval-pre100-{profile}-{seed}.json")
            safety_rows: list[dict[str, Any]] = []
            full_rows: list[dict[str, Any]] = []
            divergence: dict[str, Any] = {}

            def observe_safety(env, action, _decision, trace_index):
                safety_rows.append({"trace_index": trace_index, "policy_step": env.policy_steps,
                                    "turn": env.game.turn_number, "revision": env.game.revision,
                                    "phase": env.game.phase, "action": repr(action),
                                    "family": family_of(action),
                                    "context": _readiness(env.game, env.learning_player_id,
                                                          champion.settlement_planning_config["max_plan_roads"])})

            safety = run_strategic_episode(
                champion, robber, seed=seed, seat=seat, profile=profile,
                mode="safety", pre_action_audit_observer=observe_safety)
            if (_outcome(safety)["vp"] != baseline["final_vp"] or
                    safety.winner != baseline["winner"] or
                    safety.policy_steps != baseline["policy_steps"] or
                    safety.rng_counter != baseline["rng_counter"] or
                    safety.learner_action_families != baseline["learner_action_families"]):
                raise AssertionError(f"Safety source parity failed: {profile}/{seed}")
            safe_trace = safety.action_trace
            safe_by_trace = {row["trace_index"]: row for row in safety_rows}

            def observe_full(env, action, decision, trace_index):
                full_rows.append({"trace_index": trace_index, "policy_step": env.policy_steps,
                                  "turn": env.game.turn_number, "revision": env.game.revision,
                                  "phase": env.game.phase, "action": repr(action),
                                  "family": family_of(action),
                                  "context": _readiness(env.game, env.learning_player_id,
                                                        champion.settlement_planning_config["max_plan_roads"])})
                if divergence or trace_index >= len(safe_trace):
                    return
                if safe_trace[trace_index] == (env.learning_player_id, action):
                    return
                if decision is None:
                    raise AssertionError("First trajectory divergence was not a delegated Gate action")
                safe_action = safe_trace[trace_index][1]
                if safe_trace[trace_index][0] != env.learning_player_id:
                    raise AssertionError("First divergence occurs at a different player")
                divergence.update({
                    "trace_index": trace_index, "policy_step": env.policy_steps,
                    "turn": env.game.turn_number, "revision": env.game.revision,
                    "phase": env.game.phase,
                    "safety_action": repr(safe_action), "safety_family": family_of(safe_action),
                    "full_action": repr(action), "full_family": family_of(action),
                    "candidate_mask": decision["candidate_mask"],
                    "native_logits": decision["native_family_logits"],
                    "t2_probabilities": decision["masked_probability"],
                    "selected_probability": decision["selected_probability"],
                    "entropy": decision["entropy"],
                    "observation_v2": encode_observation(get_observation(
                        env.game, env.learning_player_id, version="v2")).tolist(),
                    "runtime_context79": encode_runtime_context(
                        env.game, env.learning_player_id,
                        max_plan_roads=champion.settlement_planning_config["max_plan_roads"]).tolist(),
                })
                divergence["boundary"] = capture_boundary(env)
                divergence["safety_action_object"] = safe_action
                divergence["full_action_object"] = action

            full = run_strategic_episode(
                champion, robber, seed=seed, seat=seat, profile=profile,
                mode="strategic", delegation_probability=1.0,
                delegation_seed=seed + 12345, gate_sampling_seed=seed + 54321,
                stochastic_gate=True, prior_temperature=2.0,
                strategic_gate=gate, strategic_critic=critic,
                pre_action_audit_observer=observe_full)
            if (_outcome(full)["vp"] != pre["final_vp"] or
                    full.winner != pre["winner"] or
                    full.policy_steps != pre["policy_steps"] or
                    full.rng_counter != pre["rng_counter"] or
                    full.learner_action_families != pre["learner_action_families"] or
                    len(full.decisions) != pre["actual_delegated_count"]):
                raise AssertionError(f"Full source parity failed: {profile}/{seed}")
            actual_first = next((i for i, (left, right) in enumerate(zip(
                safe_trace, full.action_trace)) if left != right), None)
            if actual_first != divergence.get("trace_index"):
                raise AssertionError(f"First divergence detection failed: {profile}/{seed}")
            row = {"profile": profile, "seat": seat, "seed": seed,
                   "first_divergence": bool(divergence),
                   "safety": _outcome(safety), "full_t2": _outcome(full),
                   "outcome_difference": {"winner_changed": safety.winner != full.winner,
                                          "learner_win_delta": int(full.winner == full.learner_id) - int(safety.winner == safety.learner_id),
                                          "vp": full.final_vp - safety.final_vp,
                                          "rank": full.final_rank - safety.final_rank,
                                          "policy_steps": full.policy_steps - safety.policy_steps}}
            if divergence:
                row["first"] = {key: value for key, value in divergence.items()
                                if key not in {"boundary", "safety_action_object", "full_action_object"}}
                row["window30"] = {
                    "safety": _trace_window(safety_rows, divergence["policy_step"]),
                    "full_t2": _trace_window(full_rows, divergence["policy_step"]),
                }
                captured.append({"row": row, **{key: divergence[key] for key in
                                                  ("boundary", "safety_action_object", "full_action_object")}})
            comparisons.append(row)
            _write(output / "trajectory_games" / f"{profile}-{seed}.json", row)
            print(f"trajectory {profile} {seed} first={row['first_divergence']}", flush=True)
    _write(output / "trajectory_comparison.json", comparisons)
    if len(captured) < pair_states:
        raise AssertionError(f"Only {len(captured)} first-divergence states")
    # Eight per profile; prioritize Road/End/Bank -> Dev and non-Dev divergences.
    chosen = []
    targets = ("BUILD_ROAD", "END_TURN", "BANK_TRADE")
    for profile in PROFILES:
        pool = [item for item in captured if item["row"]["profile"] == profile]
        selected = []
        for target in targets:
            options = [item for item in pool if item not in selected and
                       item["row"]["first"]["safety_family"] == target and
                       item["row"]["first"]["full_family"] == "BUY_DEVELOPMENT"]
            selected.extend(options[:2])
        other = [item for item in pool if item not in selected and
                 item["row"]["first"]["full_family"] != "BUY_DEVELOPMENT"]
        selected.extend(other[:2])
        selected.extend(item for item in pool if item not in selected)
        chosen.extend(selected[:pair_states // len(PROFILES)])
    if len(chosen) != pair_states:
        raise AssertionError("Representative profile balance could not be met")
    pairs = []
    for state_index, item in enumerate(chosen):
        row = item["row"]
        state_id = f"{row['profile']}-{row['seed']}"
        for future_index in range(future_seeds):
            path = output / "policy_pairs" / f"{state_id}-{future_index}.json"
            if path.exists():
                pair = _read(path)
            else:
                future_seed = _future_seed(row["seed"], future_index)
                shared = dict(champion_policy=champion, robber_policy=robber,
                              seed=row["seed"], seat=row["seat"], profile=row["profile"],
                              initial_boundary=item["boundary"], continuation_seed=future_seed,
                              delegation_seed=future_seed + 12345,
                              gate_sampling_seed=future_seed + 54321)
                safety = run_strategic_episode(
                    **shared, mode="safety", first_action_override=item["safety_action_object"])
                full = run_strategic_episode(
                    **shared, mode="strategic", delegation_probability=1.0,
                    stochastic_gate=True, prior_temperature=2.0,
                    strategic_gate=gate, strategic_critic=critic,
                    first_action_override=item["full_action_object"])
                if not (safety.terminal and full.terminal) or safety.truncated or full.truncated:
                    raise AssertionError(f"Policy continuation failed: {state_id}/{future_index}")
                pair = {"state_id": state_id, "future_seed": future_seed,
                        "first_pair": f"{row['first']['safety_family']} -> {row['first']['full_family']}",
                        "safety": _outcome(safety), "full": _outcome(full),
                        "delta_return": full.discounted_return_from_start - safety.discounted_return_from_start,
                        "delta_vp": full.final_vp - safety.final_vp,
                        "delta_rank": full.final_rank - safety.final_rank}
                _write(path, pair)
            pairs.append(pair)
            print(f"pair {state_index+1}/{len(chosen)} future={future_index+1}/{future_seeds}", flush=True)
    _write(output / "policy_pairs.json", pairs)
    state_summaries = []
    for state_id in dict.fromkeys(pair["state_id"] for pair in pairs):
        group = [pair for pair in pairs if pair["state_id"] == state_id]
        differences = [pair["delta_return"] for pair in group]
        state_summaries.append({"state_id": state_id, "first_pair": group[0]["first_pair"],
                                "delta_return": _stats(differences),
                                "positive": sum(x > .02 for x in differences),
                                "negative": sum(x < -.02 for x in differences),
                                "near_tie": sum(abs(x) <= .02 for x in differences),
                                "win_difference": sum(int(pair["full"]["won"]) - int(pair["safety"]["won"])
                                                      for pair in group),
                                "delta_vp": _stats([pair["delta_vp"] for pair in group]),
                                "delta_rank": _stats([pair["delta_rank"] for pair in group])})
    summary = {"comparison_games": len(comparisons),
               "first_divergence_games": len(captured),
               "first_family_pairs": dict(Counter(
                   f"{row['first']['safety_family']} -> {row['first']['full_family']}"
                   for row in comparisons if row["first_divergence"])),
               "first_dev_games": sum(row["first_divergence"] and
                                      row["first"]["full_family"] == "BUY_DEVELOPMENT"
                                      for row in comparisons),
               "paired_states": len(chosen), "future_seeds_per_state": future_seeds,
               "paired_rollouts": len(pairs),
               "delta_return_all": _stats([pair["delta_return"] for pair in pairs]),
               "delta_return_state_means": _stats([row["delta_return"]["mean"] for row in state_summaries]),
               "delta_vp": _stats([pair["delta_vp"] for pair in pairs]),
               "delta_rank": _stats([pair["delta_rank"] for pair in pairs]),
               "state_summaries": state_summaries}
    _write(output / "trajectory_summary.json", summary)
    return summary


def screens(output: Path) -> dict[str, Any]:
    champion, robber = _agents()
    gate, critic, _ = load_strategic_artifact(
        DEFAULT_SOURCE / "Strategic-T2-preupdate", _latent_dim(champion))
    result = {}
    for spec in SHORTLIST:
        records = []
        for pindex, profile in enumerate(PROFILES):
            for index in range(12):
                seed = EVAL_SEED_START + pindex * 1000 + index
                seat = index % 4 + 1
                checkpoint = output / "prior_games" / spec / f"{profile}-{seed}.json"
                if checkpoint.exists():
                    row = _read(checkpoint)
                elif spec == "native_t2":
                    row = _read(DEFAULT_SOURCE / f"eval-pre100-{profile}-{seed}.json")
                else:
                    episode = run_strategic_episode(
                        champion, robber, seed=seed, seat=seat, profile=profile,
                        mode="strategic", delegation_probability=1.0,
                        delegation_seed=seed + 12345, gate_sampling_seed=seed + 54321,
                        stochastic_gate=True, prior_temperature=2.0,
                        strategic_gate=gate, strategic_critic=critic,
                        prior_calibrator=lambda raw, mask, name=spec: prior_scores(raw, mask, name),
                        prior_name=spec)
                    if not episode.terminal or episode.truncated:
                        raise AssertionError(f"Prior screen failed: {spec}/{seed}")
                    row = _episode_record(episode)
                    _write(checkpoint, row)
                records.append(row)
                print(f"screen {spec} {profile} {seed} n={len(row['decisions'])}", flush=True)
        family = Counter(decision["selected_family"] for row in records
                         for decision in row["decisions"])
        candidate_sets: dict[str, Counter[str]] = defaultdict(Counter)
        immediate = Counter()
        for row in records:
            for decision in row["decisions"]:
                available = "+".join(FAMILIES[i] for i, flag in enumerate(decision["candidate_mask"]) if flag)
                candidate_sets[available][decision["selected_family"]] += 1
                if decision["candidate_mask"][0] or decision["candidate_mask"][1]:
                    immediate[decision["selected_family"]] += 1
        result[spec] = {"games": len(records), "profile": dict(Counter(row["profile"] for row in records)),
                        "seat": dict(Counter(str(row["seat"]) for row in records)),
                        "outcome": _variant_stats(records),
                        "selected_family": dict(family),
                        "action_family": dict(sum((Counter(row["learner_action_families"]) for row in records), Counter())),
                        "candidate_set_selection": {key: dict(value) for key, value in candidate_sets.items()},
                        "immediate_build_competition": dict(immediate),
                        "terminal": sum(row["terminal"] for row in records),
                        "truncated": sum(row["truncated"] for row in records)}
        _write(output / "screen_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("offline", "trajectory", "screens", "summarize", "all"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair-states", type=int, default=24)
    parser.add_argument("--future-seeds", type=int, default=8)
    args = parser.parse_args()
    if args.phase in {"offline", "all"}:
        offline(args.output)
    if args.phase in {"trajectory", "all"}:
        trajectories(args.output, pair_states=args.pair_states, future_seeds=args.future_seeds)
    if args.phase in {"screens", "all"}:
        screens(args.output)
    if args.phase in {"summarize", "all"}:
        _write(args.output / "trajectory_cascade_summary.json",
               summarize_trajectory_records(_read(args.output / "trajectory_comparison.json")))
        if (args.output / "policy_pairs.json").exists():
            _write(args.output / "policy_extra_summary.json",
                   summarize_policy_pairs(_read(args.output / "policy_pairs.json")))


if __name__ == "__main__":
    main()
