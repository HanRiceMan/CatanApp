"""Step 3A-21: one guarded Stage 0 PPO update, with restartable evidence.

All training Actor samples are actual 10% delegated Gate decisions. The
Champion, T=2 pre-update parent, candidate mask, reward and executors stay
unchanged. Each phase is explicit so the update cannot run before preflight.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .run_strategic_curriculum_step3a20 import (
    _module_digest, paired_rows, stratified_paired_bootstrap, validate_episode,
)
from .run_strategic_gate_step3a14 import _episode_record, _gate_stats, _variant_stats
from .run_strategic_gate_step3a16 import (
    _agents, _gzip_read, _gzip_write, _latent_dim, _read, _write,
)
from .run_strategic_road_dose_step3a19 import (
    PROFILES, SOURCE, _edge, edge_diagnostics,
)
from .strategic_action_surface import FAMILIES
from .strategic_gate_ppo_step3a16 import (
    _frozen_features, distribution, load_strategic_artifact,
    one_strategic_ppo_update, preflight_strategic_update,
    rollout_diagnostics, save_strategic_artifact, strategic_macro_gae,
)
from .strategic_gate_rollout_step3a14 import run_strategic_episode


OUTPUT = Path("experiments/strategic_curriculum_step3a21_20261011")
TRAIN_SEED_START = 32200000
EVAL_SEED_START = 32100000
MINIBATCH_SEED = 32290001
PARENT = SOURCE / "Strategic-T2-preupdate"
STAGE_ID = "strategic_curriculum_stage0"
DELEGATION = .10
TEMPERATURE = 2.0
POST_ARTIFACT_NAME = "Strategic-Curriculum-Stage0-Post1"


def training_block(block_index: int):
    """Exactly one game per profile/seat cell in every 12-game block."""
    for pindex, profile in enumerate(PROFILES):
        for seat_index in range(4):
            yield profile, TRAIN_SEED_START + pindex * 1000 + block_index * 4 + seat_index, seat_index + 1


def evaluation_rows():
    for pindex, profile in enumerate(PROFILES):
        for index in range(24):
            yield profile, EVAL_SEED_START + pindex * 1000 + index, index % 4 + 1


def _artifact_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _road_observer(roads: list[dict[str, Any]]):
    def observe(env, _champion, snapshot, _mask, record):
        if record["selected_family"] != "BUILD_ROAD":
            return
        b = edge_diagnostics(env.game, env.learning_player_id,
                             _edge(record["concrete_action"]), snapshot.candidate_logits)
        compare = (record["champion_counterfactual_family"] == "BUILD_ROAD")
        roads.append({"pre_policy_step": record["pre_policy_step"],
                      "b_edge": b["edge_id"], "b_legal": b["legal"],
                      "structural_class": b["structural_class"],
                      "b_logit_rank": b["candidate_logit_rank"],
                      "road_application_probe": record["application_probe"],
                      "champion_road_edge": (_edge(record["champion_counterfactual_action"])
                                             if compare else None),
                      "same_edge": (b["edge_id"] == _edge(record["champion_counterfactual_action"])
                                    if compare else None)})
    return observe


def collect_rollout(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    champion, robber = _agents()
    champion_before = _weight_digest(champion)
    gate, critic, parent_manifest = load_strategic_artifact(PARENT, _latent_dim(champion))
    if parent_manifest["temperature"] != TEMPERATURE or any(
            value != 0 for value in gate.residual.weight.detach().flatten().tolist()) or any(
            value != 0 for value in gate.residual.bias.detach().flatten().tolist()):
        raise AssertionError("Stage 0 must start from zero-residual T=2 parent")
    parent_sha_before = _artifact_sha(PARENT / "weights.pt")
    gate_before, critic_before = _module_digest(gate), _module_digest(critic)
    episodes: list[dict] = []
    block_index = 0
    while True:
        for profile, seed, seat in training_block(block_index):
            path = root / "training_games" / f"{profile}-{seed}.json.gz"
            if path.exists():
                row = _gzip_read(path)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                roads: list[dict[str, Any]] = []
                episode = run_strategic_episode(
                    champion, robber, seed=seed, seat=seat, profile=profile,
                    mode="strategic", delegation_probability=DELEGATION,
                    delegation_seed=seed + 12345, gate_sampling_seed=seed + 54321,
                    stochastic_gate=True, prior_temperature=TEMPERATURE,
                    strategic_gate=gate, strategic_critic=critic,
                    record_training_inputs=True,
                    decision_audit_observer=_road_observer(roads))
                row = {**_episode_record(episode), "road_diagnostics": roads,
                       "stage_id": STAGE_ID}
                validate_episode(row, DELEGATION)
                _gzip_write(path, row)
            validate_episode(row, DELEGATION)
            episodes.append(row)
            print(f"stage0 train block={block_index} {profile} seed={seed} "
                  f"seat={seat} delegated={len(row['decisions'])}", flush=True)
        count = sum(len(row["decisions"]) for row in episodes)
        chosen = Counter(item["selected_family"] for row in episodes for item in row["decisions"])
        coverage = all(chosen[family] >= 10 for family in FAMILIES)
        print(f"stage0 block={block_index} games={len(episodes)} "
              f"transitions={count} coverage={coverage}", flush=True)
        if count >= 320 and coverage:
            break
        if count >= 384:
            break
        block_index += 1
    diagnostics = rollout_diagnostics(episodes)
    diagnostics["balanced_blocks"] = block_index + 1
    diagnostics["each_family_at_least_10"] = all(chosen[family] >= 10 for family in FAMILIES)
    diagnostics["transition_target_met"] = count >= 320
    diagnostics["normal_upper_bound_exceeded"] = count > 384
    diagnostics["road_structural_class"] = dict(Counter(
        item["structural_class"] for row in episodes for item in row["road_diagnostics"]))
    diagnostics["parent_weight_sha256"] = parent_sha_before
    diagnostics["champion_digest"] = champion_before
    diagnostics["gate_digest_before"] = gate_before
    diagnostics["critic_digest_before"] = critic_before
    diagnostics["frozen_weights_unchanged"] = (
        champion_before == _weight_digest(champion) and
        gate_before == _module_digest(gate) and
        critic_before == _module_digest(critic) and
        parent_sha_before == _artifact_sha(PARENT / "weights.pt"))
    if not diagnostics["frozen_weights_unchanged"]:
        raise AssertionError("Stage 0 collection modified frozen weights")
    _write(root / "rollout_diagnostics.json", diagnostics)
    _gzip_write(root / "rollout.json.gz", episodes)
    return diagnostics


def _distribution_extended(values) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    result = distribution(values)
    result["p25"] = float(np.percentile(values, 25)) if values.size else None
    result["p75"] = float(np.percentile(values, 75)) if values.size else None
    return result


def _critic_diagnostics(champion, critic, episodes: list[dict]) -> dict[str, Any]:
    decisions = [item for row in episodes for item in row["decisions"]]
    returns = np.concatenate([strategic_macro_gae(row["transitions"])[1]
                              for row in episodes if row["transitions"]]).astype(np.float64)
    latent, context, mask, _, base_value = _frozen_features(champion, decisions)
    with torch.no_grad():
        values = critic(base_value, latent, context, mask).reshape(-1).cpu().numpy().astype(np.float64)
    error = returns - values
    return {"value_mse": float(np.mean(error ** 2)),
            "return": distribution(returns), "prediction": distribution(values),
            "explained_variance": (float(1 - np.var(error) / np.var(returns))
                                   if np.var(returns) > 1e-12 else None)}


def preflight(root: Path) -> dict[str, Any]:
    file = root / "preflight.json"
    if file.exists():
        return _read(file)
    episodes = _gzip_read(root / "rollout.json.gz")
    if len(episodes) % 12 or len(episodes) < 12:
        raise AssertionError("Training games are not complete balanced blocks")
    for row in episodes:
        validate_episode(row, DELEGATION)
    diagnostics = rollout_diagnostics(episodes)
    collection = _read(root / "rollout_diagnostics.json")
    if diagnostics["transitions"] < 320 or any(
            diagnostics["family_selected"].get(family, 0) < 10 for family in FAMILIES):
        raise ValueError("Stage 0 transition / natural family coverage Gate failed")
    expected_profile = len(episodes) // len(PROFILES)
    if any(diagnostics["profile_games"].get(profile, 0) != expected_profile
           for profile in PROFILES) or any(sum(
               row["profile"] == profile and row["seat"] == seat for row in episodes)
               != len(episodes) // 12 for profile in PROFILES for seat in range(1, 5)):
        raise AssertionError("Stage 0 profile/seat balance failed")
    champion, _ = _agents()
    gate, critic, manifest = load_strategic_artifact(PARENT, _latent_dim(champion))
    if manifest["temperature"] != TEMPERATURE or _weight_digest(champion) != collection["champion_digest"]:
        raise AssertionError("Parent or Champion changed since rollout")
    if (torch.count_nonzero(gate.residual.weight).item() or
            torch.count_nonzero(gate.residual.bias).item() or
            torch.count_nonzero(critic.residual.weight).item() or
            torch.count_nonzero(critic.residual.bias).item()):
        raise AssertionError("Stage 0 pre-update parent is not zero residual")
    if _artifact_sha(PARENT / "weights.pt") != collection["parent_weight_sha256"]:
        raise AssertionError("Parent artifact changed since rollout")
    base = preflight_strategic_update(champion, gate, critic, episodes)
    decisions = [item for row in episodes for item in row["decisions"]]
    transitions = [item for row in episodes for item in row["transitions"]]
    per_episode = [strategic_macro_gae(row["transitions"]) for row in episodes if row["transitions"]]
    advantages = np.concatenate([item[0] for item in per_episode])
    returns = np.concatenate([item[1] for item in per_episode])
    values = np.asarray([item["value"] for item in transitions], dtype=np.float64)
    done = np.asarray([item["done"] for item in transitions], dtype=bool)
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
        raise ValueError("Nonfinite advantage/return")
    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    absolute_mass = float(np.abs(advantages[done]).sum() / np.abs(advantages).sum())
    residual = returns - values
    explained = (float(1 - np.var(residual) / np.var(returns))
                 if np.var(returns) > 1e-12 else None)
    family_advantage = {family: _distribution_extended([
        advantages[index] for index, item in enumerate(decisions)
        if item["selected_family"] == family]) for family in FAMILIES}
    result = {
        "status": "PASS", "base_preflight": base, "rollout": diagnostics,
        "advantage_raw_extended": _distribution_extended(advantages),
        "advantage_normalized_extended": _distribution_extended(normalized),
        "terminal_advantage": _distribution_extended(advantages[done]),
        "nonterminal_advantage": _distribution_extended(advantages[~done]),
        "terminal_absolute_advantage_share": absolute_mass,
        "family_advantage_extended": family_advantage,
        "critic_preupdate": {
            "value_mse": float(np.mean(residual ** 2)),
            "return": distribution(returns), "prediction": distribution(values),
            "explained_variance": explained,
        },
        "road_structural_class": dict(Counter(item["structural_class"]
                 for row in episodes for item in row["road_diagnostics"])),
        "training_games": len(episodes),
        "training_seed_range_by_profile": {profile: [min(
            row["seed"] for row in episodes if row["profile"] == profile), max(
            row["seed"] for row in episodes if row["profile"] == profile)]
            for profile in PROFILES},
        "parent_artifact": str(PARENT),
        "parent_weight_sha256": collection["parent_weight_sha256"],
        "champion_digest": collection["champion_digest"],
        "gate_digest": _module_digest(gate),
        "critic_digest": _module_digest(critic),
    }
    if not base["optimizer_parameter_ids_disjoint_from_champion"]:
        raise AssertionError("Frozen Champion parameter entered optimizer")
    _write(file, result)
    return result


def update_once(root: Path) -> dict[str, Any]:
    file = root / "update_result.json"
    attempt = root / "update_attempt_started.json"
    if file.exists() or attempt.exists():
        raise ValueError("Stage 0 PPO update already attempted; never retry automatically")
    report = _read(root / "preflight.json")
    if report["status"] != "PASS":
        raise ValueError("Stage 0 preflight did not pass")
    episodes = _gzip_read(root / "rollout.json.gz")
    champion, _ = _agents()
    gate, critic, parent_manifest = load_strategic_artifact(PARENT, _latent_dim(champion))
    champion_before, parent_before = _weight_digest(champion), _artifact_sha(PARENT / "weights.pt")
    if (champion_before != report["champion_digest"] or
            parent_before != report["parent_weight_sha256"] or
            _module_digest(gate) != report["gate_digest"] or
            _module_digest(critic) != report["critic_digest"]):
        raise AssertionError("Parent/Champion changed after preflight")
    _write(attempt, {"stage_id": STAGE_ID, "minibatch_seed": MINIBATCH_SEED,
                     "transitions": report["rollout"]["transitions"],
                     "parent_weight_sha256": parent_before,
                     "champion_digest": champion_before})
    result = one_strategic_ppo_update(
        champion, gate, critic, episodes, actor_lr=1e-4, critic_lr=3e-4,
        batch_size=64, epochs=2, clip_range=.1, entropy_coef=.005,
        training_seed=MINIBATCH_SEED, target_mean_kl=.01, max_kl_limit=.10)
    payload = {"technical_status": ("ACCEPTED" if result.accepted else "REJECTED"),
               "accepted": result.accepted, "reason": result.reason,
               "metrics": result.metrics,
               "critic_preupdate": report["critic_preupdate"],
               "critic_postupdate": _critic_diagnostics(champion, critic, episodes)}
    if result.accepted:
        manifest = {
            **parent_manifest,
            "schema": "strategic_curriculum_stage0_post1_v1",
            "name": POST_ARTIFACT_NAME, "stage_id": STAGE_ID,
            "parent_artifact": "Strategic-T2-preupdate",
            "parent_weight_sha256": parent_before,
            "delegation_probability": DELEGATION,
            "training_game_seed_ranges_by_profile": report["training_seed_range_by_profile"],
            "training_games": report["training_games"],
            "training_transitions": report["rollout"]["transitions"],
            "training_transition_target": 320,
            "evaluation_seed_start": EVAL_SEED_START,
            "qualification_seed_ranges_by_profile": {
                "rule": [32000000, 32000023],
                "champion": [32001000, 32001023],
                "mixed": [32002000, 32002023]},
            "profile_games": report["rollout"]["profile_games"],
            "seat_games": report["rollout"]["seat_games"],
            "ppo_minibatch_seed": MINIBATCH_SEED,
            "ppo": {"actor_lr": 1e-4, "critic_lr": 3e-4, "batch_size": 64,
                    "epochs": 2, "clip_range": .1, "entropy_coef": .005,
                    "gamma": .99, "gae_lambda": .95},
            "gate_digest": _module_digest(gate),
            "critic_digest": _module_digest(critic),
            "champion_weight_digest": champion_before,
            "stage1_promotion_status": "LOCKED_PENDING_SEPARATE_DECISION",
        }
        path = root / POST_ARTIFACT_NAME
        save_strategic_artifact(path, gate, critic, manifest)
        reloaded_gate, reloaded_critic, _ = load_strategic_artifact(path, _latent_dim(champion))
        if _module_digest(gate) != _module_digest(reloaded_gate) or _module_digest(critic) != _module_digest(reloaded_critic):
            raise AssertionError("Stage 0 artifact save/reload mismatch")
        payload["artifact"] = str(path)
        payload["artifact_weight_sha256"] = _artifact_sha(path / "weights.pt")
    if _weight_digest(champion) != champion_before or _artifact_sha(PARENT / "weights.pt") != parent_before:
        raise AssertionError("Champion or parent artifact changed during Stage 0 update")
    _write(file, payload)
    return payload


def evaluate_variant(root: Path, variant: str) -> dict[str, Any]:
    if variant not in {"safety", "pre10", "pre20", "post10", "post20"}:
        raise ValueError("Unknown Stage 0 held-out variant")
    if variant.startswith("post") and not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Rejected update has no post-update evaluation Policy")
    root.mkdir(parents=True, exist_ok=True)
    champion, robber = _agents()
    champion_before = _weight_digest(champion)
    if variant == "safety":
        gate = critic = None
    else:
        artifact = PARENT if variant.startswith("pre") else root / POST_ARTIFACT_NAME
        gate, critic, manifest = load_strategic_artifact(artifact, _latent_dim(champion))
        if manifest["temperature"] != TEMPERATURE:
            raise AssertionError("Held-out Policy must use T=2")
    rate = 0.0 if variant == "safety" else .1 if variant.endswith("10") else .2
    rows: list[dict[str, Any]] = []
    for profile, seed, seat in evaluation_rows():
        path = root / "evaluation_games" / variant / f"{profile}-{seed}.json.gz"
        if path.exists():
            row = _gzip_read(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            roads: list[dict[str, Any]] = []
            episode = run_strategic_episode(
                champion, robber, seed=seed, seat=seat, profile=profile,
                mode="safety" if variant == "safety" else "strategic",
                delegation_probability=rate,
                delegation_seed=seed + 12345, gate_sampling_seed=seed + 54321,
                stochastic_gate=variant != "safety", prior_temperature=TEMPERATURE,
                strategic_gate=gate, strategic_critic=critic,
                record_training_inputs=(variant == "pre10"),
                decision_audit_observer=_road_observer(roads) if rate else None)
            row = {**_episode_record(episode), "road_diagnostics": roads,
                   "stage_id": STAGE_ID, "heldout_variant": variant}
            validate_episode(row, rate)
            _gzip_write(path, row)
        validate_episode(row, rate)
        if row["seed"] != seed or row["profile"] != profile or row["seat"] != seat:
            raise AssertionError("Held-out checkpoint seed/profile/seat mismatch")
        rows.append(row)
        print(f"stage0 eval {variant} {profile} seed={seed} seat={seat} "
              f"delegated={len(row['decisions'])}", flush=True)
    if len(rows) != 72 or _weight_digest(champion) != champion_before:
        raise AssertionError("Held-out balance or Champion frozen check failed")
    actions = Counter()
    for row in rows:
        actions.update(row["learner_action_families"])
    roads = [item for row in rows for item in row["road_diagnostics"]]
    result = {"variant": variant, "seed_start": EVAL_SEED_START,
              "delegation_probability": rate,
              "outcome": _variant_stats(rows),
              "gate": _gate_stats(rows, .99),
              "actions": dict(actions),
              "construction": actions["BUILD_SETTLEMENT"] + actions["BUILD_CITY"],
              "road_b": {"delegated": len(roads),
                         "structural_class": dict(Counter(item["structural_class"] for item in roads)),
                         "neither": sum(item["structural_class"] == "neither" for item in roads),
                         "same_edge": sum(item["same_edge"] is True for item in roads),
                         "different_edge": sum(item["same_edge"] is False for item in roads)},
              "hard_safety": {"terminal": sum(row["terminal"] for row in rows),
                              "truncated": sum(row["truncated"] for row in rows),
                              "illegal": 0, "executor_failure": 0,
                              "family_mismatch": 0, "deadlock": 0,
                              "nonfinite_reward": 0,
                              "game_rng_arbitration_consumption": 0},
              "champion_digest": champion_before}
    _write(root / f"evaluation_{variant}.json", result)
    return result


def _heldout_rows(root: Path, variant: str) -> list[dict]:
    return [_gzip_read(root / "evaluation_games" / variant / f"{profile}-{seed}.json.gz")
            for profile, seed, _ in evaluation_rows()]


def fixed_state_shift(root: Path) -> dict[str, Any]:
    file = root / "heldout_fixed_state_shift.json"
    if file.exists():
        return _read(file)
    champion, _ = _agents()
    old_gate, _, _ = load_strategic_artifact(PARENT, _latent_dim(champion))
    new_gate, _, _ = load_strategic_artifact(root / POST_ARTIFACT_NAME, _latent_dim(champion))
    decisions = [decision for row in _heldout_rows(root, "pre10")
                 for decision in row["decisions"]]
    if not decisions:
        raise ValueError("No held-out pre-action Gate states")
    latent, context, mask, native, _ = _frozen_features(champion, decisions)
    with torch.no_grad():
        before = torch.distributions.Categorical(logits=old_gate(
            latent, context, mask, native, temperature=TEMPERATURE))
        after = torch.distributions.Categorical(logits=new_gate(
            latent, context, mask, native, temperature=TEMPERATURE))
        kl = torch.distributions.kl_divergence(before, after).cpu().numpy()
        old_prob, new_prob = before.probs.cpu().numpy(), after.probs.cpu().numpy()
        old_entropy, new_entropy = before.entropy().cpu().numpy(), after.entropy().cpu().numpy()
    stored = np.asarray([item["masked_probability"] for item in decisions], dtype=np.float32)
    max_error = float(np.max(np.abs(old_prob - stored)))
    if max_error > 1e-6:
        raise AssertionError("Held-out pre Gate distribution does not reproduce")
    counts = np.asarray([item["candidate_count"] for item in decisions])
    family_shift = {}
    for index, family in enumerate(FAMILIES):
        available = mask[:, index].cpu().numpy().astype(bool)
        family_shift[family] = {
            "available": int(available.sum()),
            "mean_probability_pre": float(old_prob[available, index].mean()) if available.any() else None,
            "mean_probability_post": float(new_prob[available, index].mean()) if available.any() else None,
            "top_pre": int((old_prob.argmax(axis=1) == index).sum()),
            "top_post": int((new_prob.argmax(axis=1) == index).sum()),
            "above_095_pre": int((old_prob[:, index] > .95).sum()),
            "above_095_post": int((new_prob[:, index] > .95).sum()),
        }
    result = {"states": len(decisions), "old_distribution_max_error": max_error,
              "kl": distribution(kl), "top_family_flips": int(np.sum(
                  old_prob.argmax(axis=1) != new_prob.argmax(axis=1))),
              "entropy_pre": distribution(old_entropy),
              "entropy_post": distribution(new_entropy),
              "entropy_by_candidate_count": {bucket: {
                  "pre": distribution(old_entropy[selection]),
                  "post": distribution(new_entropy[selection])}
                  for bucket, selection in (("2", counts == 2), ("3", counts == 3),
                                            ("4+", counts >= 4))},
              "any_family_above_095_pre": int((old_prob.max(axis=1) > .95).sum()),
              "any_family_above_095_post": int((new_prob.max(axis=1) > .95).sum()),
              "family_shift": family_shift}
    _write(file, result)
    return result


def compare_evaluations(root: Path) -> dict[str, Any]:
    variants = ("safety", "pre10", "post10", "pre20", "post20")
    summaries = {variant: _read(root / f"evaluation_{variant}.json") for variant in variants}
    rows = {variant: _heldout_rows(root, variant) for variant in variants}
    base = rows["safety"]
    paired = {}
    for label, left, right in (("safety_to_post10", "safety", "post10"),
                               ("pre10_to_post10", "pre10", "post10"),
                               ("safety_to_post20", "safety", "post20"),
                               ("pre20_to_post20", "pre20", "post20"),
                               ("safety_to_pre10", "safety", "pre10"),
                               ("safety_to_pre20", "safety", "pre20")):
        differences = paired_rows(rows[right], rows[left])
        paired[label] = {"bootstrap_resamples": 5000,
                         "metrics": stratified_paired_bootstrap(differences,
                              resamples=5000, seed=32190001),
                         "per_game": differences}
    fixed = fixed_state_shift(root)
    safety_actions = summaries["safety"]["actions"]
    safety_construction = summaries["safety"]["construction"]
    safety_steps = summaries["safety"]["outcome"]["policy_steps"]["mean"]
    def guard(variant: str, pair_label: str) -> dict[str, Any]:
        metrics = paired[pair_label]["metrics"]
        construction_ratio = summaries[variant]["construction"] / safety_construction
        steps_ratio = summaries[variant]["outcome"]["policy_steps"]["mean"] / safety_steps
        point = (metrics["vp"]["mean"] > -.30 and construction_ratio >= .90 and steps_ratio <= 1.15)
        ci = (metrics["vp"]["ci95_low"] > -.30 and
              metrics["construction"]["ci95_low"] > -.10 * safety_construction / 72 and
              metrics["policy_steps"]["ci95_high"] < .15 * safety_steps)
        return {"mean_delta_vp": metrics["vp"]["mean"],
                "construction_ratio": construction_ratio,
                "policy_steps_ratio": steps_ratio,
                "point_pass": point, "ci_excludes_failure": ci}
    post10_guard = guard("post10", "safety_to_post10")
    post20_guard = guard("post20", "safety_to_post20")
    hard = all(summary["hard_safety"]["terminal"] == 72 and
               all(value == 0 for key, value in summary["hard_safety"].items()
                   if key != "terminal") for summary in summaries.values())
    update = _read(root / "update_result.json")
    major_family_collapse = (fixed["any_family_above_095_post"] > max(
        fixed["any_family_above_095_pre"] + fixed["states"] * .25,
        fixed["states"] * .8) or
        max(item["top_post"] for item in fixed["family_shift"].values()) > fixed["states"] * .95)
    major_entropy_collapse = (fixed["entropy_post"]["mean"] <
                              fixed["entropy_pre"]["mean"] * .5)
    stage0_behavioral = hard and post10_guard["point_pass"] and post10_guard["ci_excludes_failure"] and not (
        major_family_collapse or major_entropy_collapse)
    status = ("UPDATE_REJECTED" if not update["accepted"] else
              "UPDATE_ACCEPTED_STAGE0_QUALIFIED" if stage0_behavioral else
              "UPDATE_ACCEPTED_STAGE0_NOT_QUALIFIED")
    stage1 = ("STAGE1_PROBE_PROMISING" if stage0_behavioral and post20_guard["point_pass"]
              and post20_guard["ci_excludes_failure"] else "STAGE1_NOT_READY")
    result = {"status": status, "stage1_readiness": stage1,
              "stage1_promotion": "PENDING_SEPARATE_DECISION",
              "heldout_seed_start": EVAL_SEED_START,
              "variants": summaries, "paired": paired,
              "heldout_fixed_state_shift": fixed,
              "post10_guardrail": post10_guard,
              "post20_probe_guardrail": post20_guard,
              "heldout_hard_safety": hard,
              "major_family_collapse": major_family_collapse,
              "major_entropy_collapse": major_entropy_collapse,
              "technical_update_accepted": update["accepted"],
              "safety_development": safety_actions.get("BUY_DEVELOPMENT", 0)}
    _write(root / "evaluation_comparison.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("rollout", "preflight", "update",
                                          "eval-pre", "eval-post", "compare"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "rollout":
        collect_rollout(args.output)
    elif args.phase == "preflight":
        preflight(args.output)
    elif args.phase == "update":
        update_once(args.output)
    elif args.phase == "eval-pre":
        for variant in ("safety", "pre10", "pre20"):
            evaluate_variant(args.output, variant)
    elif args.phase == "eval-post":
        for variant in ("post10", "post20"):
            evaluate_variant(args.output, variant)
    elif args.phase == "compare":
        compare_evaluations(args.output)


if __name__ == "__main__":
    main()
