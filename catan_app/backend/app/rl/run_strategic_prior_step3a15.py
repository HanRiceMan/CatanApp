"""No-update Step 3A-15 prior calibration and exploratory delegation audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from math import ceil, log
from pathlib import Path
from statistics import mean, variance
from typing import Any

from app.agents.ppo import PPOAgent

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .run_strategic_gate_step3a14 import (
    CHAMPION_ID, PROFILES, ROBBER_ID, _episode_record, _gate_stats,
    _variant_stats,
)
from .strategic_gate_rollout_step3a14 import run_strategic_episode
from .strategic_prior_calibration_step3a15 import (
    audit_records, load_step13, load_step14,
)


SCHEMA = "strategic_prior_calibration_step3a15_v1"
SELECTED_FOR_SMOKE = (2.0, 3.0)


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _variance(values: list[float]) -> float | None:
    return variance(values) if len(values) >= 2 else None


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or _variance(left) == 0 or _variance(right) == 0:
        return None
    left_mean, right_mean = mean(left), mean(right)
    covariance = sum((a - left_mean) * (b - right_mean)
                     for a, b in zip(left, right, strict=True))
    denominator = (sum((a - left_mean) ** 2 for a in left)
                   * sum((b - right_mean) ** 2 for b in right)) ** .5
    return covariance / denominator


def _return_diagnostics(rows: list[dict]) -> dict[str, Any]:
    transitions = [item for game in rows for item in game["transitions"]]
    decisions = [item for game in rows for item in game["decisions"]]
    if len(transitions) != len(decisions):
        raise AssertionError("Gate Actor samples and macro transitions differ")
    all_rewards = [item["reward_accumulated"] for item in transitions]
    durations = [item["macro_duration"] for item in transitions]
    by_family: dict[str, list[float]] = defaultdict(list)
    for game in rows:
        for transition in game["transitions"]:
            family = game["decisions"][transition["decision_index"]]["selected_family"]
            by_family[family].append(transition["reward_accumulated"])
    return {
        "macro_reward_mean": mean(all_rewards) if all_rewards else None,
        "macro_reward_sample_variance": _variance(all_rewards),
        "duration_reward_pearson": _pearson(durations, all_rewards),
        "terminal": {
            "n": sum(item["done"] for item in transitions),
            "mean": mean([item["reward_accumulated"] for item in transitions
                          if item["done"]]) if any(item["done"] for item in transitions)
            else None,
            "variance": _variance([item["reward_accumulated"] for item in transitions
                                   if item["done"]]),
        },
        "nonterminal": {
            "n": sum(not item["done"] for item in transitions),
            "mean": mean([item["reward_accumulated"] for item in transitions
                          if not item["done"]]) if any(not item["done"]
                                                        for item in transitions) else None,
            "variance": _variance([item["reward_accumulated"] for item in transitions
                                   if not item["done"]]),
        },
        "by_family": {family: {"n": len(values),
                               "mean": mean(values),
                               "variance": _variance(values)}
                      for family, values in by_family.items()},
    }


def _exploration(rows: list[dict]) -> dict[str, Any]:
    decisions = [item for game in rows for item in game["decisions"]]
    selected = Counter(item["selected_family"] for item in decisions)
    def selected_native_top(item: dict) -> bool:
        if "selected_is_native_top" in item:
            return bool(item["selected_is_native_top"])
        # Step 3A-14's persisted T=1 records predate the explicit flag.
        return item["selected_probability"] == max(item["masked_probability"])
    non_top = Counter(item["selected_family"] for item in decisions
                      if not selected_native_top(item))
    return {"actual_delegated": len(decisions),
            "selected": dict(selected),
            "non_top_selected": sum(non_top.values()),
            "non_top_by_family": dict(non_top),
            "non_top_fraction": (sum(non_top.values()) / len(decisions)
                                 if decisions else None)}


def _validate_on_policy_records(rows: list[dict]) -> int:
    """The logged probability and transition must describe the sampled Gate."""
    checked = 0
    for game in rows:
        transitions = {item["decision_index"]: item for item in game["transitions"]}
        if len(transitions) != len(game["decisions"]):
            raise AssertionError("Missing or duplicate delegated transition")
        for index, decision in enumerate(game["decisions"]):
            probabilities = decision["masked_probability"]
            if abs(sum(probabilities) - 1) > 1e-5:
                raise AssertionError("Gate distribution is not normalized")
            if any(probability != 0 for probability, enabled in zip(
                    probabilities, decision["candidate_mask"], strict=True)
                   if not enabled):
                raise AssertionError("Unavailable family received probability")
            if abs(decision["gate_log_prob"] - log(decision["selected_probability"])) > 1e-6:
                raise AssertionError("Stored Gate log probability is not on-policy")
            if abs(transitions[index]["gate_log_prob"] - decision["gate_log_prob"]) > 1e-8:
                raise AssertionError("Transition log probability differs from decision")
            checked += 1
    return checked


def _sample_estimate(actual: int, games: int) -> dict[str, Any]:
    rate10 = actual / games
    return {str(int(probability * 100)): {
        "projected_transitions_per_game": rate10 * (probability / .10),
        "games_for_256": ceil(256 / (rate10 * probability / .10))
        if rate10 > 0 else None,
    } for probability in (.20, .25)}


def collect(root: Path, *, existing_step13: Path, existing_step14: Path,
            seed_start: int = 2970000) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    offline_file = root / "offline.json"
    if offline_file.exists():
        offline = _read(offline_file)
    else:
        offline = {
            "step13_shadow_states": audit_records(load_step13(existing_step13)),
            "step14_delegated_states": audit_records(load_step14(existing_step14)),
        }
        if offline["step13_shadow_states"]["T=2"]["top1_retention"] != 1.0 or (
            offline["step13_shadow_states"]["T=3"]["top1_retention"] != 1.0):
            raise AssertionError("Temperature must retain native family rank")
        if not (offline["step13_shadow_states"]["T=3"]["by_candidate_count"]["4+"][
                "alternative_mass"]["mean"] >
                offline["step13_shadow_states"]["T=2"]["by_candidate_count"]["4+"][
                    "alternative_mass"]["mean"] >
                offline["step13_shadow_states"]["T=1"]["by_candidate_count"]["4+"][
                    "alternative_mass"]["mean"]):
            raise AssertionError("Selected temperatures did not improve 4+ alternatives")
        _write(offline_file, offline)
    champion = PPOAgent(CHAMPION_ID, device="cpu")
    robber = PPOAgent(ROBBER_ID, device="cpu")
    before_digest = _weight_digest(champion)
    all_rows: dict[str, list[dict]] = {}
    for temperature in SELECTED_FOR_SMOKE:
        variant = f"T={temperature:g}"
        rows = []
        for profile_index, profile in enumerate(PROFILES):
            for index in range(12):
                seed = seed_start + profile_index * 1000 + index
                seat = index % 4 + 1
                checkpoint = root / f"{variant.replace('=', '-')}-{profile}-{seed}.json"
                if checkpoint.exists():
                    row = _read(checkpoint)
                else:
                    episode = run_strategic_episode(
                        champion, robber, seed=seed, seat=seat, profile=profile,
                        mode="strategic", delegation_probability=.10,
                        delegation_seed=seed + 12345,
                        gate_sampling_seed=seed + 54321,
                        stochastic_gate=True, prior_temperature=temperature)
                    if not episode.terminal or episode.truncated:
                        raise AssertionError(f"Calibrated smoke failed: {variant}/{seed}")
                    row = _episode_record(episode)
                    row["prior_temperature"] = temperature
                    if any(item["prior_temperature"] != temperature
                           for item in row["decisions"]):
                        raise AssertionError("Gate sampled the wrong temperature")
                    _write(checkpoint, row)
                rows.append(row)
                print(f"{variant} {profile} seed={seed} seat={seat} "
                      f"gate={row['actual_delegated_count']}", flush=True)
        all_rows[variant] = rows
    after_digest = _weight_digest(champion)
    if before_digest != after_digest:
        raise AssertionError("Frozen Champion artifact changed")
    old_games = _read(existing_step14)["stochastic"]
    old_rows = [game["strategic"] for game in old_games]
    all_rows["T=1"] = old_rows
    gamma = float(champion.model.gamma)
    result = {
        "schema_version": SCHEMA,
        "temperature_variants_smoked": list(SELECTED_FOR_SMOKE),
        "seed_start": seed_start,
        "games_per_profile": 12,
        "champion_model_id": CHAMPION_ID,
        "robber_model_id": ROBBER_ID,
        "champion_weight_digest_before": before_digest,
        "champion_weight_digest_after": after_digest,
        "gamma": gamma,
        "variants": {},
    }
    for variant, rows in all_rows.items():
        validated = _validate_on_policy_records(rows)
        gate = _gate_stats(rows, gamma)
        if validated != gate["actual_delegated"]:
            raise AssertionError("Delegation / validation count mismatch")
        outcome = _variant_stats(rows)
        if outcome["terminal"] != len(rows) or outcome["truncated"]:
            raise AssertionError("Incomplete calibrated episode")
        result["variants"][variant] = {
            "outcome": outcome,
            "gate": gate,
            "exploration": _exploration(rows),
            "returns": _return_diagnostics(rows),
            "profile_games": dict(Counter(row["profile"] for row in rows)),
            "seat_games": dict(Counter(str(row["seat"]) for row in rows)),
            "sample_estimate": _sample_estimate(gate["actual_delegated"], len(rows)),
            "failure_counts": {"illegal": 0, "executor": gate["executor_failure"],
                               "family_outside": gate["family_outside"],
                               "deadlock": 0, "truncation": outcome["truncated"],
                               "nonfinite_reward": 0},
        }
    _write(root / "summary.json", result)
    _write(root / "games.json", {variant: rows for variant, rows in all_rows.items()
                                  if variant != "T=1"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="strategic_prior_step3a15_20261005")
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("Invalid experiment ID")
    backend = Path(__file__).resolve().parents[2]
    root = backend / "experiments" / args.experiment_id
    if (root / "summary.json").exists():
        parser.error("Experiment already complete; use a fresh ID")
    summary = collect(
        root,
        existing_step13=backend / "experiments" /
        "strategic_gate_shadow_step3a13_20261004" / "decisions.jsonl",
        existing_step14=backend / "experiments" /
        "strategic_gate_step3a14_20261004" / "games.json",
    )
    print(json.dumps({variant: {"games": values["outcome"]["games"],
                                "actual_delegated": values["gate"]["actual_delegated"],
                                "non_top": values["exploration"]["non_top_selected"]}
                      for variant, values in summary["variants"].items()},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
