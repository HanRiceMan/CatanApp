"""PAUSED Step 3A-23: final, pre-registered 10% Stage 0 qualification.

Collection was interrupted by the Step 3B-0 redesign; do not treat the
partial E/F checkpoints as a final qualification. No optimizer is instantiated.
A-D are reused; only E/F would be collected if this study were resumed.
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
from .run_strategic_curriculum_step3a21 import PARENT, TEMPERATURE, _road_observer
from .run_strategic_gate_step3a14 import _episode_record
from .run_strategic_gate_step3a16 import (
    _agents, _gzip_read, _gzip_write, _latent_dim, _read, _write,
)
from .run_strategic_road_dose_step3a19 import PROFILES
from .run_strategic_stage0_envelope_step3a22 import (
    BOOTSTRAP_SEED, EXISTING_B, FIELDS, OUTPUT as PRIOR_OUTPUT,
    POST1, SET_START as PRIOR_SETS, _correlation, _first_divergence,
    _mean_metrics, _sha, _variant_summary, load_rows as load_prior_rows,
)
from .strategic_gate_ppo_step3a16 import load_strategic_artifact
from .strategic_gate_rollout_step3a14 import run_strategic_episode


OUTPUT = Path("experiments/strategic_stage0_final_step3a23_20261011")
SET_START = {**PRIOR_SETS, "E": 32500000, "F": 32600000}
NEW_SETS = "EF"
ALL_SETS = "ABCDEF"
RATE = .10
BOOT_SEED = BOOTSTRAP_SEED + 1000


def seed_rows(set_id: str):
    for profile_index, profile in enumerate(PROFILES):
        for index in range(24):
            yield profile, SET_START[set_id] + profile_index * 1000 + index, index % 4 + 1


def _digests(champion, gate, critic) -> dict[str, str]:
    return {"champion": _weight_digest(champion), "gate": _module_digest(gate),
            "critic": _module_digest(critic), "parent_sha256": _sha(PARENT / "weights.pt"),
            "post1_sha256": _sha(POST1 / "weights.pt")}


def _load_new(root: Path, set_id: str, variant: str) -> list[dict[str, Any]]:
    rate = 0.0 if variant == "safety" else RATE
    rows = []
    for profile, seed, seat in seed_rows(set_id):
        row = _gzip_read(root / "games" / set_id / variant / f"{profile}-{seed}.json.gz")
        validate_episode(row, rate)
        if (row["profile"], row["seed"], row["seat"]) != (profile, seed, seat):
            raise AssertionError("Set/profile/seed/seat mismatch")
        rows.append(row)
    if len(rows) != 72:
        raise AssertionError("Expected 72 episodes per set and variant")
    return rows


def load_rows(root: Path, set_id: str, variant: str) -> list[dict[str, Any]]:
    return (load_prior_rows(PRIOR_OUTPUT, set_id, variant) if set_id in PRIOR_SETS else
            _load_new(root, set_id, variant))


def collect(root: Path, *, set_ids: str = NEW_SETS, finalize: bool = True) -> dict[str, Any] | None:
    root.mkdir(parents=True, exist_ok=True)
    champion, robber = _agents()
    gate, critic, manifest = load_strategic_artifact(PARENT, _latent_dim(champion))
    if manifest["temperature"] != TEMPERATURE:
        raise AssertionError("Parent temperature differs from pre-registered T=2")
    before = _digests(champion, gate, critic)
    expected_post = _read(EXISTING_B / "update_result.json")["artifact_weight_sha256"]
    if before["post1_sha256"] != expected_post:
        raise AssertionError("Post1 artifact changed")
    if any(set_id not in NEW_SETS for set_id in set_ids):
        raise ValueError("Only the pre-registered E/F sets may be collected")
    for set_id in set_ids:
        for profile, seed, seat in seed_rows(set_id):
            for variant in ("safety", "pre10"):
                rate = 0.0 if variant == "safety" else RATE
                path = root / "games" / set_id / variant / f"{profile}-{seed}.json.gz"
                if path.exists():
                    row = _gzip_read(path)
                else:
                    road: list[dict[str, Any]] = []
                    episode = run_strategic_episode(
                        champion, robber, seed=seed, seat=seat, profile=profile,
                        mode="safety" if variant == "safety" else "strategic",
                        delegation_probability=rate,
                        delegation_seed=seed + 12345,
                        gate_sampling_seed=seed + 54321,
                        stochastic_gate=variant == "pre10", prior_temperature=TEMPERATURE,
                        strategic_gate=gate if variant == "pre10" else None,
                        strategic_critic=critic if variant == "pre10" else None,
                        decision_audit_observer=_road_observer(road) if rate else None)
                    row = {**_episode_record(episode), "road_diagnostics": road,
                           "envelope_set": set_id, "envelope_variant": variant}
                    validate_episode(row, rate)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _gzip_write(path, row)
                validate_episode(row, rate)
                if (row["profile"], row["seed"], row["seat"]) != (profile, seed, seat):
                    raise AssertionError("Checkpoint identity mismatch")
                print(f"final {set_id} {profile} {seed} seat={seat} {variant} "
                      f"delegated={row['actual_delegated_count']}", flush=True)
    after = _digests(champion, gate, critic)
    if before != after:
        raise AssertionError("Frozen weights changed")
    if not finalize:
        return None
    for set_id in NEW_SETS:
        for variant in ("safety", "pre10"):
            _load_new(root, set_id, variant)
    result = {"schema": "strategic_stage0_final_step3a23_v1",
              "pre_registered_seed_ranges": {s: SET_START[s] for s in NEW_SETS},
              "new_pairs": 144, "weights_before": before, "weights_after": after,
              "optimizer_created": False}
    _write(root / "collection.json", result)
    return result


def stratified_bootstrap(rows: list[dict], *, seed: int, resamples: int = 5000,
                         set_ids: str = ALL_SETS) -> dict[str, dict[str, float]]:
    """Two-sided paired bootstrap, retaining each set x profile cell of 24."""
    groups = [[row for row in rows if row["set_id"] == sid and row["profile"] == profile]
              for sid in set_ids for profile in PROFILES]
    if any(len(group) != 24 for group in groups):
        raise AssertionError(f"Expected {len(set_ids) * 3} set/profile strata of 24")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 24, (resamples, len(groups), 24))
    result = {}
    for field in FIELDS:
        values = np.asarray([[row[f"delta_{field}"] for row in group] for group in groups], dtype=float)
        sampled = np.take_along_axis(np.broadcast_to(values, indices.shape), indices, axis=2)
        means = sampled.mean(axis=(1, 2))
        result[field] = {"mean": float(values.mean()),
                         "ci95_low": float(np.quantile(means, .025)),
                         "ci95_high": float(np.quantile(means, .975))}
    return result


def _subgroup(rows: list[dict], label: str) -> dict[str, Any]:
    result = _mean_metrics(rows)
    if rows:
        values = np.asarray([row["delta_vp"] for row in rows], dtype=float)
        rng = np.random.default_rng(BOOT_SEED + sum(map(ord, label)))
        draws = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
        result["vp_ci95"] = {"mean": float(values.mean()),
                            "ci95_low": float(np.quantile(draws, .025)),
                            "ci95_high": float(np.quantile(draws, .975))}
    return result


def analyze(root: Path) -> dict[str, Any]:
    champion, _ = _agents()
    gate, critic, manifest = load_strategic_artifact(PARENT, _latent_dim(champion))
    if manifest["temperature"] != TEMPERATURE:
        raise AssertionError("Unexpected Policy parent")
    before = _digests(champion, gate, critic)
    collected = _read(root / "collection.json")
    if before != collected["weights_before"] or before != collected["weights_after"]:
        raise AssertionError("Frozen digest mismatch")
    by_set, pooled = {}, []
    all_safety, all_pre = [], []
    for set_id in ALL_SETS:
        safety, pre = load_rows(root, set_id, "safety"), load_rows(root, set_id, "pre10")
        all_safety.extend(safety)
        all_pre.extend(pre)
        diffs = paired_rows(pre, safety)
        divergence = Counter()
        for row, base, diff in zip(pre, safety, diffs):
            first = _first_divergence(row, base)
            diff.update(set_id=set_id, actual_delegated_count=row["actual_delegated_count"],
                        first_divergence=first, diverged=first is not None)
            exposure = Counter(d["selected_family"] for d in row["decisions"])
            for family in ("BUY_DEVELOPMENT", "BUILD_ROAD", "BANK_TRADE", "END_TURN"):
                diff[f"exposure_{family}"] = exposure[family]
            divergence[first["pair"] if first else "NO_DIVERGENCE"] += 1
        by_set[set_id] = {"seed_start": SET_START[set_id],
                          "safety": _variant_summary(safety), "pre10": _variant_summary(pre),
                          "paired_bootstrap": stratified_paired_bootstrap(
                              diffs, resamples=5000, seed=BOOT_SEED + ord(set_id)),
                          "paired_means": _mean_metrics(diffs),
                          "first_divergence_pairs": dict(divergence)}
        pooled.extend(diffs)
    if len(pooled) != 432:
        raise AssertionError("Final qualification requires exactly 432 paired games")
    bootstrap = stratified_bootstrap(pooled, seed=BOOT_SEED)
    vp_sensitivity = {str(seed): stratified_bootstrap(
        pooled, seed=seed, resamples=20000)["vp"] for seed in (BOOT_SEED + 1, BOOT_SEED + 2)}
    heterogeneity = {}
    for field in ("vp", "construction", "development"):
        values = np.asarray([by_set[s]["paired_means"][field] for s in ALL_SETS])
        heterogeneity[field] = {"set_means": dict(zip(ALL_SETS, values.tolist())),
                                "between_set_sd": float(values.std(ddof=1)),
                                "between_set_variance": float(values.var(ddof=1)),
                                "min": float(values.min()), "max": float(values.max()),
                                "range": float(values.max() - values.min())}
    loo = {s: stratified_bootstrap(pooled, seed=BOOT_SEED + 100 + ord(s),
                                    set_ids=ALL_SETS.replace(s, ""))
           for s in ALL_SETS}
    subgroups = {
        "profile": {p: _subgroup([r for r in pooled if r["profile"] == p], p)
                    for p in PROFILES},
        "seat": {str(seat): _mean_metrics([r for r in pooled if r["seat"] == seat])
                 for seat in range(1, 5)},
        "profile_seat": {f"{p}/{seat}": _mean_metrics([
            r for r in pooled if r["profile"] == p and r["seat"] == seat])
            for p in PROFILES for seat in range(1, 5)}}
    pairs = Counter(r["first_divergence"]["pair"] if r["first_divergence"] else
                    "NO_DIVERGENCE" for r in pooled)
    pair_outcomes = {pair: _mean_metrics([r for r in pooled if
                     (r["first_divergence"]["pair"] if r["first_divergence"] else
                      "NO_DIVERGENCE") == pair]) for pair, n in pairs.items() if n >= 5}
    def bucket(n):
        return "0" if n == 0 else "1-2" if n <= 2 else "3-4" if n <= 4 else "5+"
    buckets = {label: _mean_metrics([r for r in pooled if
               bucket(r["actual_delegated_count"]) == label])
               for label in ("0", "1-2", "3-4", "5+")}
    exposures = {f: _correlation(pooled, f"exposure_{f}") for f in
                 ("BUY_DEVELOPMENT", "BUILD_ROAD", "BANK_TRADE", "END_TURN")}
    base_construction = sum(s["learner_action_families"].get("BUILD_SETTLEMENT", 0) +
                            s["learner_action_families"].get("BUILD_CITY", 0) for s in all_safety)
    pre_construction = sum(s["learner_action_families"].get("BUILD_SETTLEMENT", 0) +
                           s["learner_action_families"].get("BUILD_CITY", 0) for s in all_pre)
    base_steps = float(np.mean([s["policy_steps"] for s in all_safety]))
    pre_steps = float(np.mean([s["policy_steps"] for s in all_pre]))
    construction_ratio = pre_construction / base_construction
    steps_ratio = pre_steps / base_steps
    point = (bootstrap["vp"]["mean"] > -.30 and construction_ratio >= .90 and
             steps_ratio <= 1.15)
    vp_ci_stable_pass = all(value["ci95_low"] > -.30 for value in
                            [bootstrap["vp"], *vp_sensitivity.values()])
    ci = (vp_ci_stable_pass and
          bootstrap["construction"]["ci95_low"] > -.10 * base_construction / 432 and
          bootstrap["policy_steps"]["ci95_high"] < .15 * base_steps)
    hard = all(by_set[s][v]["hard_safety"]["terminal"] == 72 and all(
        n == 0 for key, n in by_set[s][v]["hard_safety"].items() if key != "terminal")
        for s in ALL_SETS for v in ("safety", "pre10"))
    status = ("HARD_SAFETY_FAILED" if not hard else
              "FINAL_REQUALIFIED_10_PERCENT" if point and ci else
              "FINAL_BORDERLINE_10_PERCENT" if point else "FINAL_FAILED_10_PERCENT")
    road = {key: sum(by_set[s]["pre10"]["road_b"][key] for s in ALL_SETS)
            for key in ("delegated", "neither", "same_edge", "different_edge")}
    road["structural_class"] = dict(sum((Counter(by_set[s]["pre10"]["road_b"]["structural_class"])
                                          for s in ALL_SETS), Counter()))
    result = {"schema": "strategic_stage0_final_step3a23_v1",
              "pre_registered_new_seed_ranges": {s: SET_START[s] for s in NEW_SETS},
              "sets": by_set, "pooled_pairs": 432, "bootstrap_resamples": 5000,
              "pooled_bootstrap": bootstrap,
              "pooled_vp_bootstrap_sensitivity_20000": vp_sensitivity,
              "heterogeneity": heterogeneity, "leave_one_set_out": loo,
              "subgroups": subgroups, "first_divergence_pairs": dict(pairs),
              "first_pair_outcomes_n_ge_5": pair_outcomes,
              "divergence": {"no_divergence": _mean_metrics([r for r in pooled if not r["diverged"]]),
                             "diverged": _mean_metrics([r for r in pooled if r["diverged"]])},
              "delegation_count_buckets": buckets,
              "delegation_count_vs_delta_vp": _correlation(pooled, "actual_delegated_count"),
              "family_exposure_vs_delta_vp": exposures, "pooled_road_b": road,
              "stage0_guardrail": {"mean_delta_vp": bootstrap["vp"]["mean"],
                                   "construction_ratio": construction_ratio,
                                   "policy_steps_ratio": steps_ratio,
                                   "construction_ci_lower_ratio": 1 + bootstrap["construction"]["ci95_low"] * 432 / base_construction,
                                   "steps_ci_upper_ratio": 1 + bootstrap["policy_steps"]["ci95_high"] / base_steps,
                                   "point_pass": bool(point), "vp_ci_stable_pass": bool(vp_ci_stable_pass),
                                   "ci_pass": bool(ci)},
              "hard_safety_pass": hard, "qualification_status": status,
              "weights_before": before, "weights_after": _digests(champion, gate, critic),
              "optimizer_created": False}
    if result["weights_before"] != result["weights_after"]:
        raise AssertionError("Analysis changed frozen weights")
    _write(root / "analysis.json", result)
    _write(root / "paired_games.json", pooled)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("collect", "analyze"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sets", choices=("E", "F", "EF", ""), default="EF",
                        help="Collection partition; empty finalizes previously collected E/F")
    args = parser.parse_args()
    collect(args.output, set_ids=args.sets, finalize=args.sets in ("EF", "")) if args.phase == "collect" else analyze(args.output)


if __name__ == "__main__":
    main()
