"""Step 3A-22: frozen Pre10 operating-envelope and seed-variance audit.

This module never creates an optimizer. A/B are read from prior experiments;
C/D are paired Safety/Pre10 episodes with the same pre-update T=2 artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .run_strategic_curriculum_step3a20 import (
    _module_digest, paired_rows, stratified_paired_bootstrap, validate_episode,
)
from .run_strategic_curriculum_step3a21 import (
    PARENT, TEMPERATURE, _road_observer,
)
from .run_strategic_gate_step3a14 import _episode_record, _gate_stats, _variant_stats
from .run_strategic_gate_step3a16 import (
    _agents, _gzip_read, _gzip_write, _latent_dim, _read, _write,
)
from .run_strategic_road_dose_step3a19 import PROFILES
from .strategic_gate_ppo_step3a16 import load_strategic_artifact
from .strategic_gate_rollout_step3a14 import run_strategic_episode


OUTPUT = Path("experiments/strategic_stage0_envelope_step3a22_20261011")
SET_START = {"A": 32000000, "B": 32100000,
             "C": 32300000, "D": 32400000}
EXISTING_A = Path("experiments/strategic_curriculum_step3a20_20261010")
EXISTING_B = Path("experiments/strategic_curriculum_step3a21_20261011")
POST1 = EXISTING_B / "Strategic-Curriculum-Stage0-Post1"
RATE = .10
BOOTSTRAP_SEED = 32390022


def seed_rows(set_id: str):
    for profile_index, profile in enumerate(PROFILES):
        for index in range(24):
            yield profile, SET_START[set_id] + profile_index * 1000 + index, index % 4 + 1


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def existing_row(set_id: str, variant: str, profile: str, seed: int) -> dict[str, Any]:
    if set_id == "A":
        return _read(EXISTING_A / "games" / ("0" if variant == "safety" else "0.1") /
                     f"{profile}-{seed}.json")
    if set_id == "B":
        return _gzip_read(EXISTING_B / "evaluation_games" / variant /
                          f"{profile}-{seed}.json.gz")
    raise ValueError("Only A/B have existing rows")


def load_rows(root: Path, set_id: str, variant: str) -> list[dict[str, Any]]:
    rate = 0.0 if variant == "safety" else RATE
    rows = []
    for profile, seed, seat in seed_rows(set_id):
        row = (existing_row(set_id, variant, profile, seed) if set_id in "AB" else
               _gzip_read(root / "games" / set_id / variant / f"{profile}-{seed}.json.gz"))
        validate_episode(row, rate)
        if (row["profile"], row["seed"], row["seat"]) != (profile, seed, seat):
            raise AssertionError("Set/profile/seed/seat mismatch")
        rows.append(row)
    if len(rows) != 72:
        raise AssertionError("Each seed set must have 72 episodes")
    return rows


def collect(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    champion, robber = _agents()
    gate, critic, manifest = load_strategic_artifact(PARENT, _latent_dim(champion))
    if manifest["temperature"] != TEMPERATURE:
        raise AssertionError("The frozen parent is not native T=2")
    before = {"champion": _weight_digest(champion), "gate": _module_digest(gate),
              "critic": _module_digest(critic), "parent_sha256": _sha(PARENT / "weights.pt")}
    for set_id in ("C", "D"):
        for profile, seed, seat in seed_rows(set_id):
            for variant in ("safety", "pre10"):
                rate = 0.0 if variant == "safety" else RATE
                path = root / "games" / set_id / variant / f"{profile}-{seed}.json.gz"
                if path.exists():
                    row = _gzip_read(path)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    road: list[dict[str, Any]] = []
                    episode = run_strategic_episode(
                        champion, robber, seed=seed, seat=seat, profile=profile,
                        mode="safety" if variant == "safety" else "strategic",
                        delegation_probability=rate,
                        delegation_seed=seed + 12345,
                        gate_sampling_seed=seed + 54321,
                        stochastic_gate=variant == "pre10",
                        prior_temperature=TEMPERATURE,
                        strategic_gate=gate if variant == "pre10" else None,
                        strategic_critic=critic if variant == "pre10" else None,
                        decision_audit_observer=_road_observer(road) if rate else None)
                    row = {**_episode_record(episode), "road_diagnostics": road,
                           "envelope_set": set_id, "envelope_variant": variant}
                    validate_episode(row, rate)
                    _gzip_write(path, row)
                validate_episode(row, rate)
                if (row["profile"], row["seed"], row["seat"]) != (profile, seed, seat):
                    raise AssertionError("Checkpoint identity mismatch")
                print(f"envelope {set_id} {profile} {seed} seat={seat} {variant} "
                      f"delegated={row['actual_delegated_count']}", flush=True)
    after = {"champion": _weight_digest(champion), "gate": _module_digest(gate),
             "critic": _module_digest(critic), "parent_sha256": _sha(PARENT / "weights.pt")}
    if before != after:
        raise AssertionError("Frozen Champion/Gate/Critic/parent weights changed")
    for set_id in "CD":
        for variant in ("safety", "pre10"):
            load_rows(root, set_id, variant)
    result = {"new_games_per_variant": 144, "new_pairs": 144,
              "sets": {set_id: {"seed_start": SET_START[set_id], "pairs": 72}
                       for set_id in "CD"},
              "weights_before": before, "weights_after": after,
              "optimizer_created": False}
    _write(root / "collection.json", result)
    return result


FIELDS = ("vp", "rank", "policy_steps", "construction", "development")


def set_profile_bootstrap(rows: list[dict], *, seed: int,
                          resamples: int = 5000) -> dict[str, dict[str, float]]:
    """Paired bootstrap with each seed-set/profile cell fixed at 24 games."""
    groups = {(set_id, profile): [row for row in rows
                                  if row["set_id"] == set_id and row["profile"] == profile]
              for set_id in "ABCD" for profile in PROFILES}
    if any(len(group) != 24 for group in groups.values()):
        raise AssertionError("Pooled bootstrap requires 12 set/profile strata of 24")
    rng = np.random.default_rng(seed)
    arrays = {field: np.stack([np.asarray([row[f"delta_{field}"] for row in group], dtype=float)
                               for group in groups.values()]) for field in FIELDS}
    indices = rng.integers(0, 24, (resamples, 12, 24))
    result = {}
    for field, values in arrays.items():
        sampled = np.take_along_axis(np.broadcast_to(values, (resamples, 12, 24)), indices, axis=2)
        means = sampled.mean(axis=(1, 2))
        result[field] = {"mean": float(values.mean()),
                         "ci95_low": float(np.quantile(means, .025)),
                         "ci95_high": float(np.quantile(means, .975))}
    return result


def _variant_summary(rows: list[dict]) -> dict[str, Any]:
    actions = Counter()
    for row in rows:
        actions.update(row["learner_action_families"])
    road = [item for row in rows for item in row["road_diagnostics"]]
    return {"outcome": _variant_stats(rows), "actions": dict(actions),
            "construction": actions["BUILD_SETTLEMENT"] + actions["BUILD_CITY"],
            "gate": _gate_stats(rows, .99),
            "hard_safety": {"terminal": sum(row["terminal"] for row in rows),
                            "truncation": sum(row["truncated"] for row in rows),
                            "illegal": 0, "executor_failure": 0, "family_mismatch": 0,
                            "deadlock": 0, "nonfinite_reward": 0,
                            "game_rng_arbitration_consumption": 0},
            "road_b": {"delegated": len(road),
                       "neither": sum(item["structural_class"] == "neither" for item in road),
                       "same_edge": sum(item["same_edge"] is True for item in road),
                       "different_edge": sum(item["same_edge"] is False for item in road),
                       "structural_class": dict(Counter(item["structural_class"] for item in road))}}


def _first_divergence(row: dict, safety: dict) -> dict[str, Any] | None:
    for decision in row["decisions"]:
        if decision["concrete_action"] != decision["champion_counterfactual_action"]:
            return {"policy_step": decision["pre_policy_step"],
                    "champion_family": decision["champion_counterfactual_family"],
                    "gate_family": decision["selected_family"],
                    "champion_action": decision["champion_counterfactual_action"],
                    "gate_action": decision["concrete_action"],
                    "pair": f"{decision['champion_counterfactual_family']} -> "
                            f"{decision['selected_family']}" + (
                                " edge difference" if decision["selected_family"] ==
                                decision["champion_counterfactual_family"] == "BUILD_ROAD" else "")}
    comparable = ("winner", "won", "final_vp", "final_rank", "policy_steps", "rng_counter")
    if any(row[key] != safety[key] for key in comparable) or row["learner_action_families"] != safety["learner_action_families"]:
        raise AssertionError("No concrete delegated divergence but episode outcome/state differs")
    return None


def _mean_metrics(rows: list[dict]) -> dict[str, Any]:
    return {"n": len(rows), **{field: float(np.mean([row[f"delta_{field}"] for row in rows]))
                              if rows else None for field in FIELDS}}


def _correlation(rows: list[dict], exposure: str) -> dict[str, Any]:
    x = np.asarray([row[exposure] for row in rows], dtype=float)
    y = np.asarray([row["delta_vp"] for row in rows], dtype=float)
    corr = (float(np.corrcoef(x, y)[0, 1])
            if x.std() > 1e-12 and y.std() > 1e-12 else None)
    return {"n": len(rows), "pearson_with_delta_vp": corr,
            "exposure_mean": float(x.mean()), "delta_vp_mean": float(y.mean())}


def analyze(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    champion, _ = _agents()
    gate, critic, manifest = load_strategic_artifact(PARENT, _latent_dim(champion))
    before = {"champion": _weight_digest(champion), "gate": _module_digest(gate),
              "critic": _module_digest(critic), "parent_sha256": _sha(PARENT / "weights.pt")}
    if manifest["temperature"] != TEMPERATURE:
        raise AssertionError("Unexpected Policy parent")
    collected = _read(root / "collection.json")
    if before != collected["weights_before"] or before != collected["weights_after"]:
        raise AssertionError("Frozen weights changed before/after new sets")
    post1_expected = _read(EXISTING_B / "update_result.json")["artifact_weight_sha256"]
    post1_actual = _sha(POST1 / "weights.pt")
    if post1_actual != post1_expected:
        raise AssertionError("Step 3A-21 Post1 artifact changed")
    by_set, pooled, details = {}, [], []
    for set_id in "ABCD":
        safety = load_rows(root, set_id, "safety")
        pre10 = load_rows(root, set_id, "pre10")
        differences = paired_rows(pre10, safety)
        bmap = {(row["profile"], row["seed"]): row for row in safety}
        divergence = Counter()
        for row, diff in zip(pre10, differences):
            first = _first_divergence(row, bmap[(row["profile"], row["seed"])])
            diff["set_id"] = set_id
            diff["actual_delegated_count"] = row["actual_delegated_count"]
            diff["first_divergence"] = first
            diff["diverged"] = first is not None
            exposure = Counter(item["selected_family"] for item in row["decisions"])
            diff["family_exposure"] = {family: exposure[family] for family in
                                       ("BUY_DEVELOPMENT", "BUILD_ROAD", "BANK_TRADE", "END_TURN")}
            for family, count in diff["family_exposure"].items():
                diff[f"exposure_{family}"] = count
            divergence[first["pair"] if first else "NO_DIVERGENCE"] += 1
        paired = stratified_paired_bootstrap(differences, resamples=5000,
                                              seed=BOOTSTRAP_SEED + ord(set_id))
        by_set[set_id] = {"seed_start": SET_START[set_id],
                          "safety": _variant_summary(safety),
                          "pre10": _variant_summary(pre10),
                          "paired_bootstrap": paired,
                          "first_divergence_pairs": dict(divergence),
                          "paired_means": _mean_metrics(differences)}
        pooled.extend(differences)
        details.extend(differences)
    if len(pooled) != 288:
        raise AssertionError("Exactly 288 independent paired games required")
    bootstrap = set_profile_bootstrap(pooled, seed=BOOTSTRAP_SEED)
    bootstrap_sensitivity = {
        str(seed): set_profile_bootstrap(pooled, seed=seed, resamples=20000)["vp"]
        for seed in (BOOTSTRAP_SEED + 1, BOOTSTRAP_SEED + 2)}
    set_means = {field: np.asarray([by_set[set_id]["paired_means"][field]
                                    for set_id in "ABCD"], dtype=float) for field in FIELDS}
    heterogeneity = {field: {"set_means": dict(zip("ABCD", values.tolist())),
                             "between_set_sd": float(values.std(ddof=1)),
                             "between_set_variance": float(values.var(ddof=1)),
                             "min": float(values.min()), "max": float(values.max()),
                             "range": float(values.max() - values.min()),
                             "within_set_se": {set_id: float(np.std(
                                 [row[f"delta_{field}"] for row in pooled if row["set_id"] == set_id],
                                 ddof=1) / np.sqrt(72)) for set_id in "ABCD"}}
                     for field, values in set_means.items()}
    subgroups = {
        "profile": {profile: _mean_metrics([row for row in pooled if row["profile"] == profile])
                    for profile in PROFILES},
        "seat": {str(seat): _mean_metrics([row for row in pooled if row["seat"] == seat])
                 for seat in range(1, 5)},
        "profile_seat": {f"{profile}/{seat}": _mean_metrics([
            row for row in pooled if row["profile"] == profile and row["seat"] == seat])
                         for profile in PROFILES for seat in range(1, 5)}}
    diverged = {label: _mean_metrics([row for row in pooled if row["diverged"] == status])
                for label, status in (("no_divergence", False), ("diverged", True))}
    first_pairs = Counter(row["first_divergence"]["pair"] if row["first_divergence"]
                          else "NO_DIVERGENCE" for row in pooled)
    first_pair_outcomes = {pair: _mean_metrics([row for row in pooled if (
        row["first_divergence"]["pair"] if row["first_divergence"] else "NO_DIVERGENCE") == pair])
                           for pair, count in first_pairs.items() if count >= 5}
    def bucket(n: int) -> str:
        return "0" if n == 0 else "1-2" if n <= 2 else "3-4" if n <= 4 else "5+"
    count_buckets = {label: _mean_metrics([row for row in pooled
                                           if bucket(row["actual_delegated_count"]) == label])
                     for label in ("0", "1-2", "3-4", "5+")}
    exposures = {family: _correlation(pooled, f"exposure_{family}") for family in
                 ("BUY_DEVELOPMENT", "BUILD_ROAD", "BANK_TRADE", "END_TURN")}
    exposure_groups = {family: {"0": _mean_metrics([row for row in pooled if
                      row[f"exposure_{family}"] == 0]),
                      "1+": _mean_metrics([row for row in pooled if
                      row[f"exposure_{family}"] >= 1])} for family in exposures}
    safety_construction = sum(by_set[s]["safety"]["construction"] for s in "ABCD")
    pre_construction = sum(by_set[s]["pre10"]["construction"] for s in "ABCD")
    safety_steps = np.mean([row["policy_steps"] for s in "ABCD" for row in load_rows(root, s, "safety")])
    pre_steps = np.mean([row["policy_steps"] for s in "ABCD" for row in load_rows(root, s, "pre10")])
    construction_ratio = pre_construction / safety_construction
    step_ratio = pre_steps / safety_steps
    point = bool(bootstrap["vp"]["mean"] > -.30 and construction_ratio >= .90 and step_ratio <= 1.15)
    ci = bool(bootstrap["vp"]["ci95_low"] > -.30 and
          bootstrap["construction"]["ci95_low"] > -.10 * safety_construction / 288 and
          bootstrap["policy_steps"]["ci95_high"] < .15 * safety_steps)
    hard = all(by_set[s][v]["hard_safety"]["terminal"] == 72 and all(
        count == 0 for key, count in by_set[s][v]["hard_safety"].items()
        if key != "terminal") for s in "ABCD" for v in ("safety", "pre10"))
    status = ("HARD_SAFETY_FAILED" if not hard else "REQUALIFIED" if point and ci else
              "BORDERLINE" if point else "FAILED")
    result = {"schema": "strategic_stage0_envelope_step3a22_v1", "sets": by_set,
              "pooled_pairs": 288, "bootstrap_resamples": 5000,
              "pooled_bootstrap": bootstrap, "heterogeneity": heterogeneity,
              "pooled_vp_bootstrap_sensitivity_20000": bootstrap_sensitivity,
              "subgroups": subgroups, "divergence": diverged,
              "first_divergence_pairs": dict(first_pairs),
              "first_pair_outcomes_n_ge_5": first_pair_outcomes,
              "delegation_count_buckets": count_buckets,
              "delegation_count_vs_delta_vp": _correlation(pooled, "actual_delegated_count"),
              "family_exposure_vs_delta_vp": exposures,
              "family_exposure_groups": exposure_groups,
              "pooled_road_b": {"delegated": sum(by_set[s]["pre10"]["road_b"]["delegated"] for s in "ABCD"),
                                "neither": sum(by_set[s]["pre10"]["road_b"]["neither"] for s in "ABCD"),
                                "same_edge": sum(by_set[s]["pre10"]["road_b"]["same_edge"] for s in "ABCD"),
                                "different_edge": sum(by_set[s]["pre10"]["road_b"]["different_edge"] for s in "ABCD")},
              "stage0_guardrail": {"mean_delta_vp": bootstrap["vp"]["mean"],
                                   "construction_ratio": construction_ratio,
                                   "policy_steps_ratio": float(step_ratio),
                                   "point_pass": point, "ci_excludes_failure": ci},
              "hard_safety_pass": hard, "qualification_status": status,
              "weights_before": before,
              "post1_artifact_sha256": post1_actual,
              "weights_after": {"champion": _weight_digest(champion),
                                "gate": _module_digest(gate), "critic": _module_digest(critic),
                                "parent_sha256": _sha(PARENT / "weights.pt")},
              "optimizer_created": False}
    if result["weights_before"] != result["weights_after"]:
        raise AssertionError("Analysis changed weights")
    _write(root / "analysis.json", result)
    _write(root / "paired_games.json", details)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("collect", "analyze"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "collect":
        collect(args.output)
    else:
        analyze(args.output)


if __name__ == "__main__":
    main()
