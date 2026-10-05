"""Staged Step 3A-16: pre-eval, on-policy collection, one update, post-eval."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from app.agents.ppo import PPOAgent

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .run_strategic_gate_step3a14 import (
    CHAMPION_ID, PROFILES, ROBBER_ID, _episode_record, _gate_stats, _variant_stats,
)
from .strategic_action_surface import FAMILIES
from .runtime_context import RUNTIME_CONTEXT_VERSION
from .strategic_gate_ppo_step3a16 import (
    TEMPERATURE, load_strategic_artifact, one_strategic_ppo_update,
    preflight_strategic_update,
    rollout_diagnostics, save_strategic_artifact,
)
from .strategic_gate_rollout_step3a14 import run_strategic_episode
from .strategic_gate_step3a13 import StrategicActionGate, StrategicSurfaceCritic


SCHEMA = "strategic_gate_ppo_step3a16_v1"
TRAIN_SEED_START = 3160000
EVAL_SEED_START = 3260000
MINIBATCH_SEED = 3169001
PRE_INIT_SEED = 3169000


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _gzip_write(path: Path, payload: Any) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False)


def _gzip_read(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _manifest(champion_digest: str, *, training: bool) -> dict[str, Any]:
    base = {
        "schema": SCHEMA,
        "name": "Strategic-T2-postupdate" if training else "Strategic-T2-preupdate",
        "champion_parent_model_id": CHAMPION_ID,
        "champion_weight_digest": champion_digest,
        "gate_schema": "strategic_gate_step3a13_v1",
        "temperature": TEMPERATURE,
        "observation_version": "v2_prefix_plus_runtime_context_v7",
        "runtime_context_version": RUNTIME_CONTEXT_VERSION,
        "runtime_context_dimension": 79,
        "strategic_family_order": list(FAMILIES),
        "executor_version": "step3a14_B_FROZEN_PPO",
        "immediate_win_resolver_version": "step3a13_single_action_v1",
        "protected_player_trade": "step3a14_planner_selected_trade_preempts_gate",
        "reward": {"win": 1, "loss": -1, "victory_point_delta": .05,
                   "gamma": .99},
        "evaluation_seed_start": EVAL_SEED_START,
    }
    if training:
        base.update({"preupdate_parent_artifact": "Strategic-T2-preupdate",
                     "training_seed_start": TRAIN_SEED_START,
                     "ppo_minibatch_seed": MINIBATCH_SEED,
                     "delegation_probability": .20,
                     "ppo": {"actor_lr": 1e-4, "critic_lr": 3e-4,
                             "batch_size": 64, "epochs": 2, "clip_range": .1,
                             "entropy_coef": .005, "gamma": .99,
                             "gae_lambda": .95}})
    else:
        base["initialization_seed"] = PRE_INIT_SEED
    return base


def _agents():
    return PPOAgent(CHAMPION_ID, device="cpu"), PPOAgent(ROBBER_ID, device="cpu")


def _latent_dim(champion) -> int:
    return champion.model.policy.mlp_extractor.latent_dim_pi


def prepare_preupdate(root: Path, champion) -> None:
    path = root / "Strategic-T2-preupdate"
    if path.exists():
        load_strategic_artifact(path, _latent_dim(champion))
        return
    torch.manual_seed(PRE_INIT_SEED)
    gate = StrategicActionGate(_latent_dim(champion))
    critic = StrategicSurfaceCritic(_latent_dim(champion))
    save_strategic_artifact(path, gate, critic,
                            _manifest(_weight_digest(champion), training=False))
    loaded_gate, loaded_critic, _ = load_strategic_artifact(path, _latent_dim(champion))
    for source, reloaded in ((gate, loaded_gate), (critic, loaded_critic)):
        if any(not torch.equal(value, reloaded.state_dict()[key])
               for key, value in source.state_dict().items()):
            raise AssertionError("Strategic pre-update artifact failed roundtrip")


def evaluate(root: Path, champion, robber, variant: str) -> dict[str, Any]:
    if variant not in {"safety", "pre20", "pre100", "post20", "post100"}:
        raise ValueError("Unknown evaluation variant")
    file = root / f"eval_{variant}.json"
    if file.exists():
        return _read(file)
    if variant == "safety":
        gate = critic = None
    else:
        artifact = ("Strategic-T2-preupdate" if variant.startswith("pre")
                    else "Strategic-T2-postupdate")
        gate, critic, _ = load_strategic_artifact(root / artifact,
                                                 _latent_dim(champion))
    delegation = (0.0 if variant == "safety" else
                  .20 if variant.endswith("20") else 1.0)
    rows: list[dict] = []
    for profile_index, profile in enumerate(PROFILES):
        for index in range(12):
            seed = EVAL_SEED_START + profile_index * 1000 + index
            seat = index % 4 + 1
            checkpoint = root / f"eval-{variant}-{profile}-{seed}.json"
            if checkpoint.exists():
                row = _read(checkpoint)
            else:
                episode = run_strategic_episode(
                    champion, robber, seed=seed, seat=seat, profile=profile,
                    mode="safety" if variant == "safety" else "strategic",
                    delegation_probability=delegation,
                    delegation_seed=seed + 12345,
                    gate_sampling_seed=seed + 54321,
                    stochastic_gate=variant != "safety",
                    prior_temperature=TEMPERATURE,
                    strategic_gate=gate, strategic_critic=critic)
                if not episode.terminal or episode.truncated:
                    raise AssertionError(f"Evaluation failed to finish: {variant}/{seed}")
                row = _episode_record(episode)
                _write(checkpoint, row)
            rows.append(row)
            print(f"eval {variant} {profile} seed={seed} seat={seat} "
                  f"gate={row['actual_delegated_count']}", flush=True)
    result = {"variant": variant, "seed_start": EVAL_SEED_START,
              "delegation_probability": delegation,
              "profile_games": dict(Counter(row["profile"] for row in rows)),
              "seat_games": dict(Counter(str(row["seat"]) for row in rows)),
              "outcome": _variant_stats(rows),
              "gate": _gate_stats(rows, float(champion.model.gamma))}
    _write(file, result)
    return result


def collect_rollout(root: Path, champion, robber) -> dict[str, Any]:
    file = root / "rollout.json.gz"
    gate, critic, _ = load_strategic_artifact(
        root / "Strategic-T2-preupdate", _latent_dim(champion))
    episodes: list[dict] = _gzip_read(file) if file.exists() else []
    chosen = Counter(item["selected_family"] for episode in episodes
                     for item in episode["decisions"])
    non_top = sum(not item["selected_is_native_top"] for episode in episodes
                  for item in episode["decisions"])
    existing_count = sum(len(item["decisions"]) for item in episodes)
    existing_three_plus = sum(item["candidate_count"] >= 3
                              for episode in episodes for item in episode["decisions"])
    if (len(episodes) >= 72 and existing_count >= 256 and
            all(chosen[family] >= 5 for family in FAMILIES) and
            non_top >= 30 and existing_three_plus >= 30):
        return {"episodes": episodes, "diagnostics": rollout_diagnostics(episodes)}
    index = len(episodes)
    while True:
        profile_index = index % len(PROFILES)
        local_index = index // len(PROFILES)
        profile = PROFILES[profile_index]
        seed = TRAIN_SEED_START + profile_index * 1000 + local_index
        seat = local_index % 4 + 1
        checkpoint = root / f"train-{profile}-{seed}.json.gz"
        if checkpoint.exists():
            with gzip.open(checkpoint, "rt", encoding="utf-8") as stream:
                row = json.load(stream)
        else:
            episode = run_strategic_episode(
                champion, robber, seed=seed, seat=seat, profile=profile,
                mode="strategic", delegation_probability=.20,
                delegation_seed=seed + 12345,
                gate_sampling_seed=seed + 54321,
                stochastic_gate=True, prior_temperature=TEMPERATURE,
                strategic_gate=gate, strategic_critic=critic,
                record_training_inputs=True)
            if not episode.terminal or episode.truncated:
                raise AssertionError(f"Training episode failed: {profile}/{seed}")
            row = _episode_record(episode)
            _gzip_write(checkpoint, row)
        episodes.append(row)
        chosen.update(item["selected_family"] for item in row["decisions"])
        non_top += sum(not item["selected_is_native_top"] for item in row["decisions"])
        count = sum(len(item["decisions"]) for item in episodes)
        print(f"train {profile} seed={seed} seat={seat} gate={len(row['decisions'])} "
              f"total={count}", flush=True)
        coverage = (all(chosen[family] >= 5 for family in FAMILIES)
                    and non_top >= 30 and sum(item["candidate_count"] >= 3
                                              for episode in episodes
                                              for item in episode["decisions"]) >= 30)
        if (len(episodes) >= 72 and count >= 256 and coverage) or count >= 384:
            break
        index += 1
    diagnostics = rollout_diagnostics(episodes)
    diagnostics["coverage_gate"] = {"each_family_at_least_5": all(
        chosen[family] >= 5 for family in FAMILIES),
        "non_top_at_least_30": non_top >= 30,
        "candidate_3plus_at_least_30": sum(
            item["candidate_count"] >= 3 for episode in episodes
            for item in episode["decisions"]) >= 30}
    _gzip_write(file, episodes)
    _write(root / "rollout_diagnostics.json", diagnostics)
    return {"episodes": episodes, "diagnostics": diagnostics}


def update_once(root: Path, champion) -> dict[str, Any]:
    file = root / "update_result.json"
    attempt = root / "update_attempt_started.json"
    if file.exists() or attempt.exists():
        raise ValueError("Step 3A-16 update already attempted; never run twice")
    if any(not (root / f"eval_{variant}.json").exists()
           for variant in ("safety", "pre20", "pre100")):
        raise ValueError("All same-seed pre-update baselines must finish first")
    preflight = _read(root / "preupdate_gate.json")
    if preflight["frozen_champion_digest"] != _weight_digest(champion):
        raise ValueError("Champion changed after the read-only preflight")
    episodes = _gzip_read(root / "rollout.json.gz")
    diagnostics = rollout_diagnostics(episodes)
    if diagnostics["transitions"] < 256:
        raise ValueError("Need at least 256 actual delegated transitions")
    gate, critic, _ = load_strategic_artifact(
        root / "Strategic-T2-preupdate", _latent_dim(champion))
    _write(attempt, {"schema": SCHEMA, "minibatch_seed": MINIBATCH_SEED,
                     "transitions": diagnostics["transitions"],
                     "champion_digest": _weight_digest(champion)})
    result = one_strategic_ppo_update(
        champion, gate, critic, episodes, training_seed=MINIBATCH_SEED)
    payload = {"accepted": result.accepted, "reason": result.reason,
               "metrics": result.metrics}
    if result.accepted:
        manifest = _manifest(_weight_digest(champion), training=True)
        manifest["training_games"] = len(episodes)
        manifest["training_transitions"] = diagnostics["transitions"]
        manifest["training_seed_ranges_by_profile"] = {
            profile: [min(item["seed"] for item in episodes if item["profile"] == profile),
                      max(item["seed"] for item in episodes if item["profile"] == profile)]
            for profile in PROFILES}
        manifest["external_rng_seed_rule"] = {
            "delegation_seed": "game_seed + 12345",
            "family_sampling_seed": "game_seed + 54321"}
        save_strategic_artifact(root / "Strategic-T2-postupdate", gate, critic,
                                manifest)
        reloaded_gate, reloaded_critic, _ = load_strategic_artifact(
            root / "Strategic-T2-postupdate", _latent_dim(champion))
        for source, reloaded in ((gate, reloaded_gate), (critic, reloaded_critic)):
            if any(not torch.equal(value, reloaded.state_dict()[key])
                   for key, value in source.state_dict().items()):
                raise AssertionError("Post-update artifact failed roundtrip")
    _write(file, payload)
    return payload


def preflight(root: Path, champion) -> dict[str, Any]:
    file = root / "preupdate_gate.json"
    if file.exists():
        return _read(file)
    episodes = _gzip_read(root / "rollout.json.gz")
    gate, critic, _ = load_strategic_artifact(
        root / "Strategic-T2-preupdate", _latent_dim(champion))
    report = preflight_strategic_update(champion, gate, critic, episodes)
    _write(file, report)
    return report


def compare_evaluations(root: Path) -> dict[str, Any]:
    variants = ("safety", "pre20", "post20", "pre100", "post100")
    rows = {variant: [
        _read(root / f"eval-{variant}-{profile}-{EVAL_SEED_START + profile_index * 1000 + index}.json")
        for profile_index, profile in enumerate(PROFILES) for index in range(12)]
        for variant in variants}
    summaries = {variant: _read(root / f"eval_{variant}.json")
                 for variant in variants}
    comparison: dict[str, Any] = {"evaluation_seed_start": EVAL_SEED_START,
                                  "variants": {}}
    for variant in variants:
        games = rows[variant]
        decisions = [item for game in games for item in game["decisions"]]
        comparison["variants"][variant] = {
            "outcome": summaries[variant]["outcome"],
            "gate": summaries[variant]["gate"],
            "profile": {profile: {
                "games": len(subset), "wins": sum(item["won"] for item in subset),
                "mean_vp": mean(item["final_vp"] for item in subset),
                "mean_rank": mean(item["final_rank"] for item in subset),
                "action_counts": dict(sum((Counter(item["learner_action_families"])
                                           for item in subset), Counter())),
            } for profile in PROFILES
                if (subset := [item for item in games if item["profile"] == profile])},
            "non_top_selected": sum(not item["selected_is_native_top"]
                                    for item in decisions),
            "champion_counterfactual_family": dict(Counter(
                item["champion_counterfactual_family"] for item in decisions)),
            "gate_development_vs_champion": dict(Counter(
                item["champion_counterfactual_family"] for item in decisions
                if item["selected_family"] == "BUY_DEVELOPMENT")),
        }
    for left, right in (("pre20", "post20"), ("pre100", "post100")):
        paired = list(zip(rows[left], rows[right], strict=True))
        same_decision = [[(item["selected_family"], item["concrete_action"])
                          for item in a["decisions"]] ==
                         [(item["selected_family"], item["concrete_action"])
                          for item in b["decisions"]]
                         for a, b in paired]
        same_outcome = [all(a[key] == b[key] for key in (
            "winner", "final_vp", "final_rank", "policy_steps",
            "learner_action_families")) for a, b in paired]
        comparison[f"{left}_versus_{right}"] = {
            "same_delegated_decision_sequence_games": sum(bool(v) for v in same_decision),
            "same_outcome_and_action_family_counts_games": sum(same_outcome),
            "changed_seeds": [a["seed"] for (a, _), same in zip(
                paired, same_outcome, strict=True) if not same],
        }
    _write(root / "evaluation_comparison.json", comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("pre", "rollout", "preflight", "update", "post", "compare"))
    parser.add_argument("--experiment-id", default="strategic_gate_step3a16_20261005")
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("Invalid experiment ID")
    root = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    root.mkdir(parents=True, exist_ok=True)
    champion, robber = _agents()
    digest = _weight_digest(champion)
    if args.stage == "pre":
        prepare_preupdate(root, champion)
        for variant in ("safety", "pre20", "pre100"):
            evaluate(root, champion, robber, variant)
    elif args.stage == "rollout":
        result = collect_rollout(root, champion, robber)
        print(json.dumps(result["diagnostics"], ensure_ascii=False), flush=True)
    elif args.stage == "preflight":
        print(json.dumps(preflight(root, champion), ensure_ascii=False), flush=True)
    elif args.stage == "update":
        print(json.dumps(update_once(root, champion), ensure_ascii=False), flush=True)
    elif args.stage == "post":
        if not _read(root / "update_result.json")["accepted"]:
            raise ValueError("Rejected update has no post-update policy to evaluate")
        for variant in ("post20", "post100"):
            evaluate(root, champion, robber, variant)
    else:
        result = compare_evaluations(root)
        print(json.dumps({"pre20_post20": result["pre20_versus_post20"],
                          "pre100_post100": result["pre100_versus_post100"]},
                         ensure_ascii=False), flush=True)
    if _weight_digest(champion) != digest:
        raise AssertionError("Champion changed during Step 3A-16")


if __name__ == "__main__":
    main()
