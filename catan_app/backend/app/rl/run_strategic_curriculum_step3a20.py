"""Step 3A-20 qualification only: frozen T=2 Strategic Gate, no optimizer.

Games are checkpointed individually. Safety, 10%, and 20% use the same game
seed and independent, identically seeded external arbitration RNG streams.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import ceil, isfinite
from pathlib import Path
from typing import Any

import numpy as np

from .run_strategic_gate_step3a14 import _episode_record, _gate_stats
from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .run_strategic_road_dose_step3a19 import (
    PROFILES, SOURCE, _edge, _models, edge_diagnostics,
)
from .run_strategic_prior_step3a18 import _read, _stats, _write
from .strategic_action_surface import FAMILIES
from .strategic_gate_rollout_step3a14 import run_strategic_episode


OUTPUT = Path("experiments/strategic_curriculum_step3a20_20261010")
SEED_START = 32000000
EVALUATION_SEED_START = 32100000  # Reserved for held-out Stage 0 evaluation.
RATES = (0.0, 0.1, 0.2)
GAMMA = 0.99


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    delegation_probability: float
    prior: str
    parent_artifact: str
    training_transition_target: list[int]
    qualification_seed_range: list[int]
    evaluation_seed_range: list[int]
    promotion_status: str
    promotion_reason: str


def curriculum_manifest() -> dict[str, Any]:
    last = SEED_START + (len(PROFILES) - 1) * 1000 + 23
    return {
        "schema_version": "strategic_curriculum_v1",
        "decision": "Stage 0 restarts from pre-update T=2, not the prior 20% post-update artifact",
        "qualification_profile_order": list(PROFILES),
        "qualification_games_per_profile": 24,
        "seat_assignment": "seat = local_index % 4 + 1 (six games per seat and profile)",
        "delegation_rng": "game_seed + 12345; external to GameState",
        "family_rng": "game_seed + 54321; external to GameState",
        "stage0": asdict(CurriculumStage(
            "strategic_curriculum_stage0", 0.10, "native_t2",
            "Strategic-T2-preupdate", [256, 320], [SEED_START, last],
            [EVALUATION_SEED_START, EVALUATION_SEED_START + 2023],
            "QUALIFICATION_PENDING", "No training or automatic promotion in Step 3A-20")),
        "stage1_candidate": asdict(CurriculumStage(
            "strategic_curriculum_stage1", 0.20, "native_t2",
            "stage0_trained_if_approved", [256, 320], [], [], "LOCKED",
            "Requires Stage 0 trained 10% held-out safety/entropy/family checks and 20% probe")),
        "promotion_criteria": {
            "stage0_qualification": ["terminal=100%", "illegal/executor/family/deadlock/truncation/nonfinite/RNG_arbitration=0",
                "mean_delta_vp > -0.30", "construction_ratio >= 0.90", "policy_steps_ratio <= 1.15",
                "report paired 95% CI; wide CI means inconclusive, not proven safety"],
            "stage0_trained": ["10% held-out hard safety", "no family/entropy collapse",
                "no major loss versus pre-update Stage 0", "20% probe without construction collapse"],
            "full_surface_required": False,
            "automatic_promotion": False,
        },
        "frozen_components": ["Champion", "Strategic Gate and Critic", "Reward", "candidate mask",
                              "79-d Runtime Context", "Frozen B Executors", "ImmediateWinResolver",
                              "protected PlayerTrade"],
    }


def seed_rows():
    for profile_index, profile in enumerate(PROFILES):
        for index in range(24):
            yield profile, SEED_START + profile_index * 1000 + index, index % 4 + 1


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _module_digest(module) -> str:
    digest = sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def collect(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifest = curriculum_manifest()
    parent = SOURCE / "Strategic-T2-preupdate"
    manifest["parent_weight_sha256_before"] = _hash_file(parent / "weights.pt")
    _write(output / "curriculum_manifest.json", manifest)
    champion, robber, gate, critic = _models()
    champion_digest_before = _weight_digest(champion)
    gate_digest_before = _module_digest(gate)
    critic_digest_before = _module_digest(critic)
    records: dict[str, list[dict]] = {f"{rate:g}": [] for rate in RATES}
    for profile, seed, seat in seed_rows():
        for rate in RATES:
            label = f"{rate:g}"
            path = output / "games" / label / f"{profile}-{seed}.json"
            if path.exists():
                row = _read(path)
            else:
                roads: list[dict[str, Any]] = []

                def observe(env, _champion, snapshot, _mask, record):
                    if record["selected_family"] != "BUILD_ROAD":
                        return
                    player_id = env.learning_player_id
                    chosen = _edge(record["concrete_action"])
                    b = edge_diagnostics(env.game, player_id, chosen,
                                         snapshot.candidate_logits)
                    if not b["legal"]:
                        raise AssertionError("Delegated Road B is illegal")
                    same_edge = None
                    champion_edge = None
                    if record["champion_counterfactual_family"] == "BUILD_ROAD":
                        champion_edge = _edge(record["champion_counterfactual_action"])
                        same_edge = chosen == champion_edge
                    roads.append({"pre_policy_step": record["pre_policy_step"],
                                  "b_edge": chosen, "structural_class": b["structural_class"],
                                  "b_legal": b["legal"], "b_logit_rank": b["candidate_logit_rank"],
                                  "road_application_probe": record["application_probe"],
                                  "champion_road_edge": champion_edge,
                                  "same_edge": same_edge})

                shared = dict(champion_policy=champion, robber_policy=robber,
                              seed=seed, seat=seat, profile=profile)
                if rate == 0.0:
                    episode = run_strategic_episode(**shared, mode="safety")
                else:
                    episode = run_strategic_episode(
                        **shared, mode="strategic", delegation_probability=rate,
                        delegation_seed=seed + 12345, gate_sampling_seed=seed + 54321,
                        stochastic_gate=True, prior_temperature=2.0,
                        strategic_gate=gate, strategic_critic=critic,
                        decision_audit_observer=observe)
                row = {**_episode_record(episode), "road_diagnostics": roads,
                       "qualification_schema": "strategic_curriculum_v1"}
                validate_episode(row, rate)
                _write(path, row)
            validate_episode(row, rate)
            records[label].append(row)
            print(f"qualification {label} {profile} {seed} seat={seat} "
                  f"delegated={row['actual_delegated_count']}", flush=True)
    manifest["parent_weight_sha256_after"] = _hash_file(parent / "weights.pt")
    if manifest["parent_weight_sha256_before"] != manifest["parent_weight_sha256_after"]:
        raise AssertionError("Pre-update Strategic artifact changed")
    manifest["in_memory_weight_digest_before"] = {
        "champion": champion_digest_before, "gate": gate_digest_before,
        "critic": critic_digest_before}
    manifest["in_memory_weight_digest_after"] = {
        "champion": _weight_digest(champion), "gate": _module_digest(gate),
        "critic": _module_digest(critic)}
    if manifest["in_memory_weight_digest_before"] != manifest["in_memory_weight_digest_after"]:
        raise AssertionError("Frozen Champion/Gate/Critic weights changed")
    _write(output / "curriculum_manifest.json", manifest)
    summary = summarize(records)
    summary["parent_weight_sha256"] = manifest["parent_weight_sha256_after"]
    manifest["stage0_qualification"] = summary["stage0_qualification"]
    manifest["stage0"]["promotion_status"] = "NOT_PROMOTED"
    manifest["stage0"]["promotion_reason"] = "Qualified for first Stage 0 update; no training or promotion in Step 3A-20"
    _write(output / "curriculum_manifest.json", manifest)
    _write(output / "qualification_summary.json", summary)
    return summary


def validate_episode(row: dict, rate: float) -> None:
    if row["delegation_probability"] != rate:
        raise AssertionError("Qualification checkpoint uses a different delegation rate")
    if rate == 0 and row["mode"] != "safety":
        raise AssertionError("0% qualification must use Safety Champion")
    if rate > 0 and row["mode"] != "strategic":
        raise AssertionError("Delegated qualification must use Strategic controller")
    if rate > 0 and (row["delegation_seed"] != row["seed"] + 12345 or
                     row["gate_sampling_seed"] != row["seed"] + 54321):
        raise AssertionError("Qualification external RNG seed mismatch")
    if not row["terminal"] or row["truncated"]:
        raise AssertionError("Qualification episode failed terminal/truncation invariant")
    if not all(isfinite(float(row[key])) for key in ("final_vp", "final_rank", "policy_steps")):
        raise AssertionError("Nonfinite outcome")
    if row["actual_delegated_count"] != len(row["decisions"]) or len(row["decisions"]) != len(row["transitions"]):
        raise AssertionError("Delegation/transition count mismatch")
    if rate == 0 and row["decisions"]:
        raise AssertionError("Safety mode contains Strategic Actor samples")
    if len(row["road_diagnostics"]) != sum(
            item["selected_family"] == "BUILD_ROAD" for item in row["decisions"]):
        raise AssertionError("Not every delegated Road has a structural diagnostic")
    for item in row["decisions"]:
        if item["forced_integration"] or item["concrete_action_family"] != item["selected_family"]:
            raise AssertionError("Off-policy/forced or family-mismatched sample")
        if item["protected_player_trade_selected"]:
            raise AssertionError("Protected trade entered Actor sample")
        probe = item["application_probe"]
        if probe["game_rng_before"] != probe["probe_rng_after"]:
            raise AssertionError("Concrete Action application probe failed")
        if item["selected_family"] == "BUILD_ROAD" and (
                probe["road_owner_after"] != row["learner_id"] or
                any(probe["resource_before"][resource] -
                    probe["resource_after"][resource] != 1
                    for resource in ("wood", "brick"))):
            raise AssertionError("Road B ownership or resource invariant failed")
    for item in row["road_diagnostics"]:
        if not item["b_legal"]:
            raise AssertionError("Road B diagnostic found illegal action")


def stratified_paired_bootstrap(paired: list[dict], *, resamples: int = 5000,
                                seed: int = 32002010) -> dict[str, dict[str, float]]:
    if resamples < 2000:
        raise ValueError("At least 2000 paired bootstrap resamples required")
    groups = {profile: [row for row in paired if row["profile"] == profile]
              for profile in PROFILES}
    if not all(groups.values()):
        raise ValueError("Every profile must be represented")
    rng = np.random.default_rng(seed)
    fields = ("vp", "rank", "policy_steps", "construction", "development")
    samples = {field: np.empty(resamples, dtype=np.float64) for field in fields}
    for index in range(resamples):
        selected = [row for group in groups.values()
                    for row in (group[j] for j in rng.integers(0, len(group), len(group)))]
        for field in fields:
            samples[field][index] = np.mean([row[f"delta_{field}"] for row in selected])
    return {field: {"mean": float(np.mean([row[f"delta_{field}"] for row in paired])),
                    "ci95_low": float(np.quantile(samples[field], .025)),
                    "ci95_high": float(np.quantile(samples[field], .975))}
            for field in fields}


def _construction(row: dict) -> int:
    actions = row["learner_action_families"]
    return actions.get("BUILD_SETTLEMENT", 0) + actions.get("BUILD_CITY", 0)


def paired_rows(rows: list[dict], safety: list[dict]) -> list[dict]:
    base = {(row["profile"], row["seed"]): row for row in safety}
    result = []
    for row in rows:
        b = base[(row["profile"], row["seed"])]
        if row["seat"] != b["seat"]:
            raise AssertionError("Paired seat mismatch")
        result.append({"profile": row["profile"], "seed": row["seed"], "seat": row["seat"],
                       "delta_vp": row["final_vp"] - b["final_vp"],
                       "delta_rank": row["final_rank"] - b["final_rank"],
                       "delta_policy_steps": row["policy_steps"] - b["policy_steps"],
                       "delta_construction": _construction(row) - _construction(b),
                       "delta_development": row["learner_action_families"].get("BUY_DEVELOPMENT", 0)
                                            - b["learner_action_families"].get("BUY_DEVELOPMENT", 0)})
    return result


def summarize(records: dict[str, list[dict]]) -> dict[str, Any]:
    safety = records["0"]
    if len(safety) != 72 or any(len(records[f"{rate:g}"]) != 72 for rate in RATES):
        raise AssertionError("Qualification requires exactly 72 games per rate")
    result: dict[str, Any] = {"seed_start": SEED_START,
                              "seed_end": SEED_START + 2023,
                              "bootstrap_resamples": 5000, "rates": {}}
    for rate in RATES:
        label = f"{rate:g}"
        rows = records[label]
        for row in rows:
            validate_episode(row, rate)
        gates = _gate_stats(rows, GAMMA)
        actions = Counter()
        for row in rows:
            actions.update(row["learner_action_families"])
        roads = [item for row in rows for item in row["road_diagnostics"]]
        delegates = sum(row["actual_delegated_count"] for row in rows)
        entry: dict[str, Any] = {
            "games": len(rows), "profile": dict(Counter(row["profile"] for row in rows)),
            "profile_seat": {f"{p}/{seat}": sum(row["profile"] == p and row["seat"] == seat
                                               for row in rows)
                             for p in PROFILES for seat in range(1, 5)},
            "terminal": sum(row["terminal"] for row in rows),
            "truncated": sum(row["truncated"] for row in rows),
            "wins": sum(row["won"] for row in rows),
            "final_vp": _stats([row["final_vp"] for row in rows]),
            "rank": _stats([row["final_rank"] for row in rows]),
            "policy_steps": _stats([row["policy_steps"] for row in rows]),
            "actions": dict(actions), "construction": actions["BUILD_SETTLEMENT"] + actions["BUILD_CITY"],
            "actual_delegated": delegates, "delegated_per_game": delegates / len(rows),
            "selected_per_game": {f: gates["family_selected"].get(f, 0) / len(rows)
                                  for f in FAMILIES},
            "gate": gates,
            "road_b": {"delegated": len(roads),
                       "structural_class": dict(Counter(item["structural_class"] for item in roads)),
                       "neither": sum(item["structural_class"] == "neither" for item in roads),
                       "neither_rate": (sum(item["structural_class"] == "neither" for item in roads) / len(roads)
                                        if roads else None),
                       "both_road": sum(item["same_edge"] is not None for item in roads),
                       "same_edge": sum(item["same_edge"] is True for item in roads),
                       "different_edge": sum(item["same_edge"] is False for item in roads),
                       "application_probe_failures": 0},
            "hard_safety": {"illegal": 0, "executor_failure": 0, "family_mismatch": 0,
                            "deadlock": 0, "truncation": 0, "nonfinite_reward": 0,
                            "game_rng_arbitration_consumption": 0},
        }
        if rate > 0:
            paired = paired_rows(rows, safety)
            entry["paired_bootstrap"] = stratified_paired_bootstrap(paired)
            entry["paired_game_differences"] = paired
            if rate == .1:
                base_actions = Counter()
                for row in safety:
                    base_actions.update(row["learner_action_families"])
                base_construction = base_actions["BUILD_SETTLEMENT"] + base_actions["BUILD_CITY"]
                vp_diff = entry["paired_bootstrap"]["vp"]
                construction_ratio = entry["construction"] / base_construction
                steps_ratio = entry["policy_steps"]["mean"] / _stats([
                    row["policy_steps"] for row in safety])["mean"]
                entry["stage0_guardrail"] = {
                    "point_pass": vp_diff["mean"] > -.30 and construction_ratio >= .90 and steps_ratio <= 1.15,
                    "mean_delta_vp": vp_diff, "construction_ratio": construction_ratio,
                    "policy_steps_ratio": steps_ratio,
                    "vp_ci_excludes_guardrail_failure": vp_diff["ci95_low"] > -.30,
                    "construction_ci_excludes_guardrail_failure": (
                        entry["paired_bootstrap"]["construction"]["ci95_low"] > -.10 * base_construction / 72),
                    "policy_steps_ci_excludes_guardrail_failure": (
                        entry["paired_bootstrap"]["policy_steps"]["ci95_high"] < .15 *
                        _stats([row["policy_steps"] for row in safety])["mean"]),
                }
                per_game = entry["delegated_per_game"]
                entry["stage0_training_estimate"] = {
                    "games_for_256": ceil(256 / per_game) if per_game else None,
                    "games_for_320": ceil(320 / per_game) if per_game else None,
                    "balanced_games_for_256": 12 * ceil(256 / per_game / 12) if per_game else None,
                    "balanced_games_for_320": 12 * ceil(320 / per_game / 12) if per_game else None,
                    "family_samples_at_256": {f: 256 * gates["family_selected"].get(f, 0) / delegates
                                              if delegates else 0 for f in FAMILIES},
                    "family_samples_at_320": {f: 320 * gates["family_selected"].get(f, 0) / delegates
                                              if delegates else 0 for f in FAMILIES},
                }
        result["rates"][label] = entry
    result["road_b_provisionally_acceptable"] = all(
        result["rates"][label]["road_b"]["application_probe_failures"] == 0 and
        result["rates"][label]["hard_safety"]["illegal"] == 0 and
        result["rates"][label]["hard_safety"]["executor_failure"] == 0
        for label in ("0.1", "0.2"))
    guardrail = result["rates"]["0.1"]["stage0_guardrail"]
    family_projection = result["rates"]["0.1"]["stage0_training_estimate"]["family_samples_at_256"]
    hard_safety = all(result["rates"][f"{rate:g}"]["terminal"] == 72 and
                      not result["rates"][f"{rate:g}"]["truncated"] and
                      all(value == 0 for value in result["rates"][f"{rate:g}"]["hard_safety"].values())
                      for rate in RATES)
    interval_pass = all(guardrail[field] for field in (
        "vp_ci_excludes_guardrail_failure", "construction_ci_excludes_guardrail_failure",
        "policy_steps_ci_excludes_guardrail_failure"))
    qualified = (hard_safety and guardrail["point_pass"] and interval_pass and
                 result["road_b_provisionally_acceptable"] and
                 min(family_projection.values()) >= 5)
    result["stage0_qualification"] = {
        "status": "QUALIFIED_FOR_STAGE0_TRAINING" if qualified else "INCONCLUSIVE_OR_FAILED",
        "hard_safety": hard_safety, "point_guardrails_pass": guardrail["point_pass"],
        "ci_excludes_guardrail_failure": interval_pass,
        "road_b_provisionally_acceptable": result["road_b_provisionally_acceptable"],
        "minimum_predicted_family_samples_at_256": min(family_projection.values()),
        "no_training_or_promotion_performed": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("collect", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "collect":
        collect(args.output)
    else:
        records = {f"{rate:g}": [_read(args.output / "games" / f"{rate:g}" / f"{p}-{s}.json")
                                    for p, s, _ in seed_rows()] for rate in RATES}
        _write(args.output / "qualification_summary.json", summarize(records))


if __name__ == "__main__":
    main()
