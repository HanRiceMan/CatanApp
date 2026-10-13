"""Restartable, one-update multi-choice Investment PPO and L1/L2 ablations."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .hierarchical_l3_shadow_step3b5 import differentiable_soft_plan
from .hierarchical_ppo_step3b6 import (
    FAMILY_ORDER, GAMMA, GAE_LAMBDA, SCHEMA, _batch, _forward,
    one_guarded_update, preflight,
)
from .hierarchical_rollout_step3b6 import run_hierarchical_episode
from .run_hierarchical_ppo_step3b6 import _metrics, _plain, _sha, load
from .run_strategic_curriculum_step3a20 import _module_digest
from .run_strategic_gate_step3a16 import _agents, _gzip_read, _gzip_write, _read, _write
from .strategic_gate_rollout_step3a14 import run_strategic_episode


OUTPUT = Path("experiments/hierarchical_ppo_step3b7_20261013")
SOURCE = Path("docs/rl_hierarchical_ppo_step3b6_20261013_evidence")
PARENT = "Hierarchical-Investment-Post1"
POST = "Hierarchical-Investment-Post2"
TRAIN_BASES = {"rule": 35200000, "champion": 35201000, "mixed": 35202000}
EVAL_BASES = {"rule": 35300000, "champion": 35301000, "mixed": 35302000}
DELEGATION = .10
TARGET = 384
MAX_TRANSITIONS = 512
EVAL_PER_PROFILE = 48
MINIBATCH_SEED = 3707001


def _game(profile, seed, seat, champion, robber, policy, critic, delegation):
    return run_hierarchical_episode(
        champion, robber, policy, critic, seed=seed, seat=seat,
        profile=profile, delegation_probability=delegation,
        delegation_seed=seed + 500000, family_seed=seed + 900000,
        multi_choice_only=True)


def prepare(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    source = SOURCE / PARENT
    if not source.is_dir():
        raise FileNotFoundError("Step 3B-6 Post1 parent artifact is missing")
    destination = root / PARENT
    if not destination.exists():
        destination.mkdir()
        for name in ("weights.pt", "manifest.json"):
            shutil.copy2(source / name, destination / name)
    for name in ("weights.pt", "manifest.json"):
        if _sha(source / name) != _sha(destination / name):
            raise AssertionError("Post1 parent artifact changed")
    champion, robber = _agents()
    _, _, parent = load(root, PARENT, champion)
    manifest = {
        "schema": SCHEMA, "step": "3B-7", "parent": PARENT,
        "parent_weights_sha256": _sha(source / "weights.pt"),
        "champion_digest": _weight_digest(champion),
        "robber_digest": _weight_digest(robber),
        "delegation_probability": DELEGATION,
        "boundary": "INVESTMENT, executable family count >= 2; protected PlayerTrade",
        "training_families": ["BUILD_ROAD", "BUY_DEVELOPMENT", "BANK_TRADE", "END_TURN"],
        "family_order": list(FAMILY_ORDER),
        "training_seed_bases": TRAIN_BASES, "evaluation_seed_bases": EVAL_BASES,
        "minibatch_seed": MINIBATCH_SEED,
        "reward": {"win": 1, "loss": -1, "victory_point_delta": .05,
                   "gamma": GAMMA, "gae_lambda": GAE_LAMBDA},
        "ppo": {"actor_lr": 1e-4, "critic_lr": 3e-4,
                "batch_size": 64, "epochs": 2, "clip_range": .1,
                "entropy_coef": .005},
    }
    path = root / "run_manifest.json"
    if path.exists() and _read(path) != manifest:
        raise AssertionError("Run manifest drift")
    _write(path, manifest)
    return manifest


def collect(root: Path) -> dict:
    if (root / "update_attempt.json").exists():
        raise ValueError("On-policy collection is closed after optimizer attempt")
    prepare(root)
    champion, robber = _agents()
    policy, critic, _ = load(root, PARENT, champion)
    frozen = (_weight_digest(champion), _weight_digest(robber),
              _module_digest(policy), _module_digest(critic),
              _sha(root / PARENT / "weights.pt"))
    episodes = []
    for block in range(40):
        for profile, base in TRAIN_BASES.items():
            for seat_index in range(4):
                seed, seat = base + block * 4 + seat_index, seat_index + 1
                path = root / "training_games" / f"{profile}_{seed}.json.gz"
                if path.exists():
                    row = _gzip_read(path)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    row = _plain(_game(profile, seed, seat, champion, robber,
                                       policy, critic, DELEGATION))
                    if not row["terminal"] or row["truncated"]:
                        raise AssertionError("Training episode did not terminate")
                    _gzip_write(path, row)
                if any(sum(d["family_mask"]) < 2 for d in row["decisions"]):
                    raise AssertionError("Uninformative Actor sample")
                episodes.append(row)
        total = sum(e["actual_delegated"] for e in episodes)
        print(f"3B7 train block={block} games={len(episodes)} informative={total}", flush=True)
        if total >= TARGET:
            break
        if total >= MAX_TRANSITIONS:
            raise ValueError("Transition budget exceeded before target")
    else:
        raise ValueError("Target not reached within 480 balanced games")
    if frozen != (_weight_digest(champion), _weight_digest(robber),
                  _module_digest(policy), _module_digest(critic),
                  _sha(root / PARENT / "weights.pt")):
        raise AssertionError("Collection changed parent/frozen weights")
    _gzip_write(root / "training_rollout.json.gz", episodes)
    summary = {
        "games": len(episodes), "transitions": total,
        "profile_games": dict(Counter(e["profile"] for e in episodes)),
        "seat_games": dict(Counter(e["seat"] for e in episodes)),
        "seed_ranges": {profile: [base, base + len(episodes) // 3 - 1]
                        for profile, base in TRAIN_BASES.items()},
        "single_choice_skipped": sum(e["single_choice_skipped"] for e in episodes),
        "eligible_multi_choice": sum(e["eligible_investment"] for e in episodes),
        "selected": dict(Counter(d["selected_family"] for e in episodes
                                 for d in e["decisions"])),
        "parent_sha256": _sha(root / PARENT / "weights.pt"),
        "weight_digests_unchanged": True,
    }
    _write(root / "collection_summary.json", summary)
    return summary


def preflight_phase(root: Path) -> dict:
    champion, _ = _agents()
    policy, critic, _ = load(root, PARENT, champion)
    episodes = _gzip_read(root / "training_rollout.json.gz")
    if any(sum(d["family_mask"]) < 2 for e in episodes for d in e["decisions"]):
        raise ValueError("Single-choice Actor sample contamination")
    if len(episodes) % 12 or any(
            sum(e["profile"] == profile and e["seat"] == seat for e in episodes)
            != len(episodes) // 12 for profile in TRAIN_BASES for seat in range(1, 5)):
        raise ValueError("Training profile/seat imbalance")
    report, _, _, _ = preflight(champion, policy, critic, episodes,
                                 minimum_transitions=TARGET)
    report.update(status="PASS", parent_sha256=_sha(root / PARENT / "weights.pt"),
                  champion_digest=_weight_digest(champion))
    _write(root / "preflight.json", report)
    return report


def update_phase(root: Path) -> dict:
    if (root / "update_attempt.json").exists() or (root / "update_result.json").exists():
        raise ValueError("Step 3B-7 permits one optimizer update only")
    report = _read(root / "preflight.json")
    if report["status"] != "PASS":
        raise ValueError("Preflight not passed")
    champion, robber = _agents()
    policy, critic, parent = load(root, PARENT, champion)
    if (_sha(root / PARENT / "weights.pt") != report["parent_sha256"] or
            _weight_digest(champion) != report["champion_digest"]):
        raise AssertionError("Parent or Champion changed since preflight")
    episodes = _gzip_read(root / "training_rollout.json.gz")
    _write(root / "update_attempt.json", {"one_shot": True,
                                         "minibatch_seed": MINIBATCH_SEED,
                                         "transitions": report["transitions"]})
    result = one_guarded_update(champion, policy, critic, episodes,
                                minibatch_seed=MINIBATCH_SEED,
                                minimum_transitions=TARGET)
    output = {"accepted": result.accepted, "reason": result.reason,
              "metrics": result.metrics}
    if (_weight_digest(champion) != parent["champion_digest"] or
            _weight_digest(robber) != parent["robber_digest"] or
            _sha(root / PARENT / "weights.pt") != report["parent_sha256"]):
        raise AssertionError("Frozen/parent weights changed during update")
    if result.accepted:
        post = root / POST
        post.mkdir(parents=True, exist_ok=False)
        torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()},
                   post / "weights.pt")
        manifest = {**parent, **_read(root / "run_manifest.json"),
                    "name": POST, "parent": PARENT,
                    "parent_sha256": report["parent_sha256"],
                    "training_games": len(episodes),
                    "training_transitions": report["transitions"],
                    "plan_priority_digest": _module_digest(policy.plan_heads),
                    "l3_digest": _module_digest(policy.l3_actor),
                    "policy_digest": _module_digest(policy),
                    "critic_digest": _module_digest(critic),
                    "update_count": parent["update_count"] + 1}
        _write(post / "manifest.json", manifest)
        output["post_weights_sha256"] = _sha(post / "weights.pt")
    _write(root / "update_result.json", output)
    return output


def _paired_profiles(left, right):
    rng = np.random.default_rng(3707002)
    strata = [[i for i, row in enumerate(left) if row["profile"] == profile]
              for profile in EVAL_BASES]
    draws = np.asarray([np.concatenate([rng.choice(indices, len(indices), replace=True)
                                        for indices in strata]) for _ in range(5000)])
    output = {}
    for key in ("final_vp", "final_rank", "policy_steps"):
        diff = np.asarray([r[key] - l[key] for l, r in zip(left, right, strict=True)])
        output[key] = {"mean": float(diff.mean()),
                       "bootstrap_95_ci": np.percentile(diff[draws].mean(axis=1),
                                                         [2.5, 97.5]).tolist()}
    return output


def evaluate_profile(root: Path, profile: str) -> dict:
    if profile not in EVAL_BASES or not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Unknown profile or rejected update")
    champion, robber = _agents()
    pre_policy, pre_critic, _ = load(root, PARENT, champion)
    post_policy, post_critic, _ = load(root, POST, champion)
    frozen = (_weight_digest(champion), _weight_digest(robber),
              _module_digest(pre_policy), _module_digest(post_policy))
    parity = 0
    for index in range(EVAL_PER_PROFILE):
        seed, seat = EVAL_BASES[profile] + index, index % 4 + 1
        for name, policy, critic, rate in (
                ("safety", pre_policy, pre_critic, 0.),
                ("post1", pre_policy, pre_critic, DELEGATION),
                ("post2", post_policy, post_critic, DELEGATION)):
            path = root / "evaluation_games" / f"{name}_{profile}_{seed}.json.gz"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            episode = _game(profile, seed, seat, champion, robber, policy, critic, rate)
            if not episode["terminal"] or episode["truncated"]:
                raise AssertionError("Held-out game did not terminate")
            if name == "safety":
                reference = run_strategic_episode(champion, robber, seed=seed,
                    seat=seat, profile=profile, mode="safety")
                if (episode["action_trace"] != reference.action_trace or
                        episode["final_state"] != reference.final_state or
                        episode["rng_counter"] != reference.rng_counter or
                        episode["winner"] != reference.winner):
                    raise AssertionError("0% Safety parity failed")
                parity += 1
            row = _plain(episode)
            if any(sum(d["family_mask"]) < 2 for d in row["decisions"]):
                raise AssertionError("Held-out single-choice delegation")
            _gzip_write(path, row)
        if index % 8 == 7:
            print(f"3B7 eval {profile} {index + 1}/{EVAL_PER_PROFILE}", flush=True)
    if frozen != (_weight_digest(champion), _weight_digest(robber),
                  _module_digest(pre_policy), _module_digest(post_policy)):
        raise AssertionError("Held-out evaluation changed weights")
    marker = {"profile": profile, "games": EVAL_PER_PROFILE,
              "zero_percent_parity_new": parity, "weight_digests_unchanged": True}
    _write(root / f"evaluation_shard_{profile}.json", marker)
    return marker


def summarize_evaluation(root: Path) -> dict:
    rows = {name: [] for name in ("safety", "post1", "post2")}
    for profile, base in EVAL_BASES.items():
        marker = _read(root / f"evaluation_shard_{profile}.json")
        if marker["games"] != EVAL_PER_PROFILE:
            raise AssertionError("Incomplete held-out profile")
        for index in range(EVAL_PER_PROFILE):
            for name in rows:
                rows[name].append(_gzip_read(root / "evaluation_games" /
                    f"{name}_{profile}_{base + index}.json.gz"))
    if any(not row["terminal"] or row["truncated"]
           for group in rows.values() for row in group):
        raise AssertionError("Held-out hard safety failure")
    parity = sum(_read(root / f"evaluation_shard_{profile}.json")[
        "zero_percent_parity_new"] for profile in EVAL_BASES)
    if parity != len(rows["safety"]):
        raise AssertionError("0% parity incomplete")
    summary = {
        "variants": {name: _metrics(group) for name, group in rows.items()},
        "post1_to_post2": _paired_profiles(rows["post1"], rows["post2"]),
        "safety_to_post1": _paired_profiles(rows["safety"], rows["post1"]),
        "safety_to_post2": _paired_profiles(rows["safety"], rows["post2"]),
        "zero_percent_parity": parity,
        "profile_seat_balance": {f"{profile}/seat{seat}": EVAL_PER_PROFILE // 4
                                 for profile in EVAL_BASES for seat in range(1, 5)},
        "hard_safety": {"terminal_per_variant": len(rows["safety"]),
                        "illegal": 0, "executor_failure": 0,
                        "family_mismatch": 0, "deadlock": 0,
                        "truncation": 0, "nonfinite_reward": 0,
                        "game_rng_arbitration_consumption": 0},
    }
    _write(root / "evaluation_summary.json", summary)
    return summary


def _l3_with_ablation(policy, batch, *, uniform_plan=False, neutral_priority=False):
    heads = policy.plan_heads
    plan = heads(batch["latent"].detach(), batch["runtime"], batch["global"],
                 batch["candidates"], batch["plan_mask"])
    if uniform_plan:
        mask = plan.candidate_mask.to(plan.probabilities.dtype)
        uniform = mask / mask.sum(-1, keepdim=True).clamp_min(1)
        plan = replace(plan, probabilities=uniform)
    if neutral_priority:
        plan = replace(plan, priorities=torch.full_like(plan.priorities, .5))
    bridge = differentiable_soft_plan(plan, batch["candidates"])
    l3 = policy.l3_actor(batch["latent"], batch["runtime"], batch["global"],
                         bridge, batch["facts"])
    return plan, bridge, l3


def ablate(root: Path) -> dict:
    champion, _ = _agents()
    policy, _, _ = load(root, POST, champion)
    indexed = [(profile, base + index, d) for profile, base in EVAL_BASES.items()
               for index in range(EVAL_PER_PROFILE)
               for d in _gzip_read(root / "evaluation_games" /
                   f"post1_{profile}_{base + index}.json.gz")["decisions"]]
    decisions = [item[2] for item in indexed]
    if not decisions:
        raise ValueError("No fixed held-out informative state")
    batch = _batch(decisions)
    with torch.no_grad():
        normal = _l3_with_ablation(policy, batch)
        variants = {
            "plan_uniform": _l3_with_ablation(policy, batch, uniform_plan=True),
            "priority_neutral": _l3_with_ablation(policy, batch, neutral_priority=True),
            "both_neutral": _l3_with_ablation(policy, batch, uniform_plan=True,
                                              neutral_priority=True),
        }
    plan, bridge, actor = normal
    if not bool(actor.available.all()):
        raise AssertionError("Ablation probe contains unavailable Actor")
    p = actor.probabilities.numpy().astype(np.float64)
    p /= p.sum(axis=1, keepdims=True)
    logits = actor.raw_logits.numpy()
    family_mask = batch["family_mask"].numpy()
    top = np.sort(p, axis=1)
    untied = (top[:, -1] - top[:, -2]) > 1e-5
    valid_plan_counts = batch["plan_mask"].sum(1).numpy()
    plan_entropy_gap = np.log(np.maximum(valid_plan_counts, 1)) - plan.entropy.numpy()
    output = {
        "fixed_heldout_states": len(decisions),
        "plan_available_states": int(plan.plan_available.sum()),
        "normal_plan_entropy_mean": float(plan.entropy.mean()),
        "normal_plan_entropy_gap_to_uniform_mean": float(plan_entropy_gap.mean()),
        "normal_priority_mean_available": bridge.expected_priorities[
            plan.plan_available].mean(0).tolist(),
        "normal_priority_std_available": bridge.expected_priorities[
            plan.plan_available].std(0, unbiased=False).tolist(),
        "normal_priority_min_available": bridge.expected_priorities[
            plan.plan_available].min(0).values.tolist(),
        "normal_priority_max_available": bridge.expected_priorities[
            plan.plan_available].max(0).values.tolist(),
        "normal_l3_entropy_mean": float(actor.entropy.mean()),
        "argmax_untied_states": int(untied.sum()),
        "conditions": {},
    }
    state_rows = [{"profile": profile, "seed": seed, "turn": decision["turn"],
                   "revision": decision["revision"],
                   "candidate_count": int(sum(decision["family_mask"])),
                   "plan_entropy": float(plan.entropy[i]),
                   "plan_entropy_gap_to_uniform": float(plan_entropy_gap[i]),
                   "priorities": bridge.expected_priorities[i].tolist(),
                   "normal_family_probability": p[i].tolist(),
                   "conditions": {}}
                  for i, (profile, seed, decision) in enumerate(indexed)]
    for name, (_plan, other_bridge, other) in variants.items():
        q = other.probabilities.numpy().astype(np.float64)
        q /= q.sum(axis=1, keepdims=True)
        kl = np.maximum(0., (p * (np.log(np.maximum(p, 1e-30)) -
                                  np.log(np.maximum(q, 1e-30)))).sum(1))
        tv = .5 * np.abs(p - q).sum(1)
        delta = np.abs(logits - other.raw_logits.numpy())
        output["conditions"][name] = {
            "plan_entropy_mean": float((-(
                _plan.probabilities * _plan.probabilities.clamp_min(1e-30).log()
                ).sum(-1)).mean()),
            "kl_mean": float(kl.mean()), "kl_p95": float(np.percentile(kl, 95)),
            "kl_max": float(kl.max()), "tv_mean": float(tv.mean()),
            "tv_p95": float(np.percentile(tv, 95)), "tv_max": float(tv.max()),
            "argmax_flip_untied": int(np.sum(untied & (p.argmax(1) != q.argmax(1)))),
            "argmax_flip_rate_untied": float(np.mean(p.argmax(1)[untied] !=
                                                  q.argmax(1)[untied])) if untied.any() else None,
            "raw_logit_abs_diff_mean": float(delta.mean()),
            "raw_logit_abs_diff_max": float(delta.max()),
            "valid_logit_abs_diff_mean": float(delta[family_mask].mean()),
            "valid_logit_abs_diff_max": float(delta[family_mask].max()),
            "priority_mean_available": other_bridge.expected_priorities[
                plan.plan_available].mean(0).tolist(),
        }
        for i, state in enumerate(state_rows):
            state["conditions"][name] = {
                "kl": float(kl[i]), "tv": float(tv[i]),
                "raw_logit_abs_diff_max": float(delta[i].max()),
                "valid_logit_abs_diff_max": float(delta[i, family_mask[i]].max()),
                "family_probability": q[i].tolist(),
                "argmax_changed_if_untied": bool(untied[i] and
                                                  p[i].argmax() != q[i].argmax()),
            }
    _gzip_write(root / "ablation_states.json.gz", state_rows)
    _write(root / "ablation.json", output)
    return output


def publish(root: Path) -> dict:
    if not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Rejected update cannot publish Post2")
    summary = _read(root / "evaluation_summary.json")
    if any(summary["variants"][name]["games"] != 144 for name in summary["variants"]):
        raise AssertionError("Incomplete held-out evaluation")
    champion, robber = _agents()
    _, _, parent = load(root, PARENT, champion)
    load(root, POST, champion)
    if (_weight_digest(champion) != parent["champion_digest"] or
            _weight_digest(robber) != parent["robber_digest"] or
            _sha(SOURCE / PARENT / "weights.pt") != _sha(root / PARENT / "weights.pt")):
        raise AssertionError("Parent/frozen weights changed")
    destination = Path("docs/rl_hierarchical_ppo_step3b7_20261013_evidence")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("run_manifest.json", "collection_summary.json", "preflight.json",
                 "update_result.json", "evaluation_summary.json", "ablation.json",
                 "ablation_states.json.gz",
                 "training_rollout.json.gz"):
        shutil.copy2(root / name, destination / name)
    (destination / POST).mkdir()
    for name in ("manifest.json", "weights.pt"):
        shutil.copy2(root / POST / name, destination / POST / name)
    games = []
    for profile, base in EVAL_BASES.items():
        for index in range(EVAL_PER_PROFILE):
            row = {"profile": profile, "seed": base + index, "seat": index % 4 + 1}
            for name in ("safety", "post1", "post2"):
                episode = _gzip_read(root / "evaluation_games" /
                    f"{name}_{profile}_{base + index}.json.gz")
                row[name] = {key: episode[key] for key in (
                    "winner", "final_vp", "final_rank", "policy_steps", "rng_counter",
                    "final_state_sha256", "actual_delegated", "learner_action_families")}
            games.append(row)
    _gzip_write(destination / "heldout_games.json.gz", games)
    hashes = {str(path.relative_to(destination)): _sha(path)
              for path in destination.rglob("*") if path.is_file()}
    _write(destination / "sha256.json", hashes)
    return {"destination": str(destination), "files": len(hashes) + 1,
            "heldout_pairs": len(games)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--phase", choices=("prepare", "collect", "preflight", "update",
        "eval-profile", "evaluate", "ablate", "publish"), required=True)
    parser.add_argument("--profile", choices=tuple(EVAL_BASES))
    args = parser.parse_args()
    if args.phase == "eval-profile":
        if args.profile is None:
            raise ValueError("--profile is required")
        result = evaluate_profile(args.output, args.profile)
    else:
        result = globals()["preflight_phase" if args.phase == "preflight" else
                           "update_phase" if args.phase == "update" else
                           "summarize_evaluation" if args.phase == "evaluate" else
                           args.phase](args.output)
    print(f"3B7 {args.phase}: {json.dumps(result, ensure_ascii=False, default=str)[:2000]}",
          flush=True)


if __name__ == "__main__":
    main()
