"""Restartable Step 3B-8: fixed-state probe, one on-policy update, paired eval."""

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
from .hierarchical_ppo_step3b6 import (
    GAMMA, GAE_LAMBDA, HierarchicalInvestmentCritic, _batch,
    one_guarded_update, preflight,
)
from .hierarchical_rollout_step3b6 import run_hierarchical_episode
from .plan_conditioned_policy_step3b8 import (
    SCHEMA, PlanConditionedInvestmentPolicy,
)
from .run_hierarchical_ppo_step3b6 import _metrics, _plain, _sha, load as load_old
from .run_hierarchical_ppo_step3b7 import _paired_profiles, _l3_with_ablation
from .run_strategic_curriculum_step3a20 import _module_digest
from .run_strategic_gate_step3a16 import _agents, _gzip_read, _gzip_write, _read, _write
from .strategic_gate_rollout_step3a14 import run_strategic_episode


OUTPUT = Path("experiments/plan_conditioned_step3b8_20261014")
PARENT_ROOT = Path("docs/rl_hierarchical_ppo_step3b7_20261013_evidence")
PARENT = "Hierarchical-Investment-Post2"
OLD_EVAL_ROOT = Path("experiments/hierarchical_ppo_step3b7_20261013/evaluation_games")
PRE = "Plan-Conditioned-Investment-Pre"
POST = "Plan-Conditioned-Investment-Post1"
TRAIN_BASES = {"rule": 35400000, "champion": 35401000, "mixed": 35402000}
EVAL_BASES = {"rule": 35500000, "champion": 35501000, "mixed": 35502000}
TARGET = 256
MAX_TRANSITIONS = 512
DELEGATION = .10
MINIBATCH_SEED = 3808001
INIT_SEED = 3808
EVAL_PER_PROFILE = 24


def _new_models(champion, *, parent=False):
    latent_dim = champion.model.policy.mlp_extractor.latent_dim_pi
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(INIT_SEED)
        policy = PlanConditionedInvestmentPolicy(latent_dim)
        critic = HierarchicalInvestmentCritic(latent_dim)
    if parent:
        old_policy, old_critic, _ = load_old(PARENT_ROOT, PARENT, champion)
        policy.plan_heads.load_state_dict(old_policy.plan_heads.state_dict())
        critic.load_state_dict(old_critic.state_dict())
    return policy.eval(), critic.eval()


def _load_new(root: Path, name: str, champion):
    manifest = _read(root / name / "manifest.json")
    if manifest["schema"] != SCHEMA or manifest["champion_digest"] != _weight_digest(champion):
        raise ValueError("New artifact schema or Champion mismatch")
    policy, critic = _new_models(champion)
    weights = torch.load(root / name / "weights.pt", map_location="cpu", weights_only=True)
    policy.load_state_dict(weights["policy"])
    critic.load_state_dict(weights["critic"])
    return policy.eval(), critic.eval(), manifest


def prepare(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    champion, robber = _agents()
    source = PARENT_ROOT / PARENT / "weights.pt"
    if not source.is_file():
        raise FileNotFoundError(source)
    policy, critic = _new_models(champion, parent=True)
    manifest = {
        "schema": SCHEMA, "step": "3B-8", "name": PRE, "parent": PARENT,
        "parent_sha256": _sha(source), "champion_digest": _weight_digest(champion),
        "robber_digest": _weight_digest(robber), "l3_initialization_seed": INIT_SEED,
        "actor_mixture": "sum_i P(plan_i|state)*P(family|state,plan_i)",
        "plan_sampling": False, "delegation_probability": DELEGATION,
        "boundary": "INVESTMENT executable multi-choice, protected PlayerTrade",
        "training_seed_bases": TRAIN_BASES, "evaluation_seed_bases": EVAL_BASES,
        "minibatch_seed": MINIBATCH_SEED,
        "reward": {"win": 1, "loss": -1, "victory_point_delta": .05,
                   "gamma": GAMMA, "gae_lambda": GAE_LAMBDA},
        "ppo": {"actor_lr": 1e-4, "critic_lr": 3e-4, "batch_size": 64,
                "epochs": 2, "clip_range": .1, "entropy_coef": .005},
        "policy_digest": _module_digest(policy), "critic_digest": _module_digest(critic),
    }
    folder = root / PRE
    if folder.exists():
        if _read(folder / "manifest.json") != manifest:
            raise AssertionError("Pre artifact manifest drift")
        _load_new(root, PRE, champion)
    else:
        folder.mkdir()
        torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()},
                   folder / "weights.pt")
        _write(folder / "manifest.json", manifest)
    return {"pre_artifact": str(folder), "pre_sha256": _sha(folder / "weights.pt"),
            "parent_sha256": manifest["parent_sha256"]}


def _probe_stats(reference: np.ndarray, alternative: np.ndarray,
                 mask: np.ndarray) -> dict:
    p = reference.astype(np.float64)
    q = alternative.astype(np.float64)
    kl = np.maximum(0., (p * (np.log(np.maximum(p, 1e-30)) -
                             np.log(np.maximum(q, 1e-30)))).sum(-1))
    tv = .5 * np.abs(p - q).sum(-1)
    active = mask.sum(-1) >= 2
    flips = (p.argmax(-1) != q.argmax(-1)) & active
    return {"n": len(p), "kl_mean": float(kl.mean()),
            "kl_p95": float(np.percentile(kl, 95)),
            "tv_mean": float(tv.mean()), "tv_p95": float(np.percentile(tv, 95)),
            "tv_max": float(tv.max()), "argmax_flips": int(flips.sum())}


def _probe(root: Path, model_name: str, output_name: str):
    prepare(root)
    champion, _ = _agents()
    old_policy, _, _ = load_old(PARENT_ROOT, PARENT, champion)
    new_policy, _, _ = _load_new(root, model_name, champion)
    decisions = []
    for profile, base in {"rule": 35300000, "champion": 35301000,
                          "mixed": 35302000}.items():
        for index in range(48):
            path = OLD_EVAL_ROOT / f"post1_{profile}_{base + index}.json.gz"
            if not path.exists():
                raise FileNotFoundError(f"Fixed held-out states missing: {path}")
            decisions.extend(_gzip_read(path)["decisions"])
    if not decisions:
        raise ValueError("No fixed held-out state")
    fixed_path = root / "fixed_heldout_states.json.gz"
    if fixed_path.exists():
        if _gzip_read(fixed_path) != decisions:
            raise AssertionError("Fixed held-out state input drift")
    else:
        _gzip_write(fixed_path, decisions)
    batch = _batch(decisions)
    with torch.no_grad():
        old = _l3_with_ablation(old_policy, batch)
        plan = new_policy.plan_heads(batch["latent"].detach(), batch["runtime"],
                                     batch["global"], batch["candidates"],
                                     batch["plan_mask"])
        actor = new_policy.l3_actor
        base_args = (batch["latent"], batch["runtime"], batch["global"],
                     batch["candidates"], plan, batch["facts"])
        normal = actor(*base_args)
        valid = plan.candidate_mask.to(plan.probabilities.dtype)
        uniform = valid / valid.sum(-1, keepdim=True).clamp_min(1)
        variants = {
            "plan_uniform": actor(*base_args, plan_probabilities=uniform),
            "priority_neutral": actor(*base_args,
                                      priorities=torch.full_like(plan.priorities, .5)),
            "both_neutral": actor(*base_args, plan_probabilities=uniform,
                                  priorities=torch.full_like(plan.priorities, .5)),
        }
        counts = plan.candidate_mask.sum(-1)
        double = counts >= 2
        first = torch.zeros_like(plan.probabilities)
        last = torch.zeros_like(plan.probabilities)
        first_index = plan.candidate_mask.to(torch.long).argmax(-1).unsqueeze(-1)
        last_index = (plan.candidate_mask.to(torch.long) *
                      torch.arange(plan.candidate_mask.shape[1])).max(-1).values.unsqueeze(-1)
        first.scatter_(1, first_index, double.to(first.dtype).unsqueeze(-1))
        last.scatter_(1, last_index, double.to(last.dtype).unsqueeze(-1))
        # A/B intervention is measured only where both distinct valid Plans exist.
        if double.any():
            indices = double.nonzero().squeeze(-1).tolist()
            def sliced(tensor): return tensor[indices]
            facts = batch["facts"]
            small_facts = type(facts)(*(sliced(getattr(facts, name)) for name in (
                "route_one_hot", "legal_family_mask", "comparison_family_mask",
                "construction_commit", "l3_route_available")))
            small_plan = replace(plan, raw_logits=sliced(plan.raw_logits),
                                 probabilities=sliced(plan.probabilities),
                                 candidate_mask=sliced(plan.candidate_mask),
                                 plan_available=sliced(plan.plan_available),
                                 entropy=sliced(plan.entropy), priorities=sliced(plan.priorities))
            small_args = (sliced(batch["latent"]), sliced(batch["runtime"]),
                          sliced(batch["global"]), sliced(batch["candidates"]),
                          small_plan, small_facts)
            a = actor(*small_args, plan_probabilities=sliced(first))
            b = actor(*small_args, plan_probabilities=sliced(last))
            intervention = _probe_stats(a.probabilities.numpy(), b.probabilities.numpy(),
                                         sliced(batch["family_mask"]).numpy())
            examples = [{"candidate_a": decisions[index]["candidate_plan_ids"][int(first[index].argmax())],
                         "candidate_b": decisions[index]["candidate_plan_ids"][int(last[index].argmax())],
                         "family_a": a.probabilities[j].tolist(),
                         "family_b": b.probabilities[j].tolist()}
                        for j, index in enumerate(indices[:5])]
        else:
            intervention = {"n": 0}
            examples = []
    p = normal.probabilities.numpy()
    mask = batch["family_mask"].numpy()
    result = {
        "fixed_heldout_states": len(decisions),
        "plan_available": int(plan.plan_available.sum()),
        "states_with_two_plans": int(double.sum()),
        "old_post2_vs_new_actor": _probe_stats(old[-1].probabilities.numpy(), p, mask),
        "new_actor_ablation": {name: _probe_stats(p, output.probabilities.numpy(), mask)
                             for name, output in variants.items()},
        "valid_plan_a_vs_b": intervention,
        "plan_intervention_examples": examples,
        "plan_entropy_mean": float(plan.entropy.mean()),
        "priority_mean": plan.priorities[plan.candidate_mask].mean(0).tolist(),
        "new_l3_entropy_mean": float(normal.entropy.mean()),
        "unavailable_fallback_states": int((~plan.plan_available).sum()),
        "model_name": model_name,
        "model_sha256": _sha(root / model_name / "weights.pt"),
    }
    _write(root / output_name, result)
    return result


def probe(root: Path):
    return _probe(root, PRE, "probe_pre.json")


def probe_post(root: Path):
    if not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Cannot probe rejected update")
    return _probe(root, POST, "probe_post.json")


def _game(profile, seed, seat, champion, robber, policy, critic, rate):
    return run_hierarchical_episode(
        champion, robber, policy, critic, seed=seed, seat=seat, profile=profile,
        delegation_probability=rate, delegation_seed=seed + 500000,
        family_seed=seed + 900000, multi_choice_only=True)


def collect(root: Path):
    if (root / "update_attempt.json").exists():
        raise ValueError("On-policy collection closed after optimizer attempt")
    if not (root / "probe_pre.json").exists():
        raise ValueError("Fixed-state probe required before collection")
    champion, robber = _agents()
    policy, critic, manifest = _load_new(root, PRE, champion)
    frozen = (_weight_digest(champion), _weight_digest(robber),
              _module_digest(policy), _module_digest(critic),
              _sha(root / PRE / "weights.pt"))
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
                    raise AssertionError("Single-choice sample contamination")
                episodes.append(row)
        total = sum(e["actual_delegated"] for e in episodes)
        print(f"3B8 train block={block} games={len(episodes)} transitions={total}", flush=True)
        if total >= TARGET:
            break
        if total >= MAX_TRANSITIONS:
            raise ValueError("Transition budget exceeded")
    else:
        raise ValueError("Transition target not reached")
    if frozen != (_weight_digest(champion), _weight_digest(robber),
                  _module_digest(policy), _module_digest(critic),
                  _sha(root / PRE / "weights.pt")):
        raise AssertionError("Collection mutated frozen weights")
    _gzip_write(root / "training_rollout.json.gz", episodes)
    result = {"games": len(episodes), "transitions": total,
              "profiles": dict(Counter(e["profile"] for e in episodes)),
              "seats": dict(Counter(e["seat"] for e in episodes)),
              "selected": dict(Counter(d["selected_family"] for e in episodes
                                       for d in e["decisions"])),
              "single_choice_skipped": sum(e["single_choice_skipped"] for e in episodes),
              "pre_sha256": frozen[-1], "frozen_unchanged": True}
    _write(root / "collection_summary.json", result)
    return result


def preflight_phase(root: Path):
    champion, _ = _agents()
    policy, critic, _ = _load_new(root, PRE, champion)
    episodes = _gzip_read(root / "training_rollout.json.gz")
    if len(episodes) % 12:
        raise ValueError("Unbalanced training block")
    result, _, _, _ = preflight(champion, policy, critic, episodes,
                                 minimum_transitions=TARGET)
    result.update(status="PASS", pre_sha256=_sha(root / PRE / "weights.pt"),
                  champion_digest=_weight_digest(champion))
    _write(root / "preflight.json", result)
    return result


def update(root: Path):
    if (root / "update_attempt.json").exists():
        raise ValueError("Only one PPO update is permitted")
    ready = _read(root / "preflight.json")
    if ready["status"] != "PASS":
        raise ValueError("Preflight did not pass")
    champion, robber = _agents()
    policy, critic, parent = _load_new(root, PRE, champion)
    if (_sha(root / PRE / "weights.pt") != ready["pre_sha256"] or
            _sha(PARENT_ROOT / PARENT / "weights.pt") != parent["parent_sha256"]):
        raise AssertionError("Frozen parent changed")
    episodes = _gzip_read(root / "training_rollout.json.gz")
    _write(root / "update_attempt.json", {"one_shot": True,
                                         "transitions": ready["transitions"],
                                         "minibatch_seed": MINIBATCH_SEED})
    result = one_guarded_update(champion, policy, critic, episodes,
                                minibatch_seed=MINIBATCH_SEED,
                                minimum_transitions=TARGET)
    report = {"accepted": result.accepted, "reason": result.reason,
              "metrics": result.metrics}
    if (_weight_digest(champion) != parent["champion_digest"] or
            _weight_digest(robber) != parent["robber_digest"] or
            _sha(root / PRE / "weights.pt") != ready["pre_sha256"]):
        raise AssertionError("Frozen weights changed")
    if result.accepted:
        folder = root / POST
        folder.mkdir(parents=True, exist_ok=False)
        torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()},
                   folder / "weights.pt")
        manifest = {**parent, "name": POST, "training_games": len(episodes),
                    "training_transitions": ready["transitions"],
                    "pre_sha256": ready["pre_sha256"], "update_count": 1,
                    "policy_digest": _module_digest(policy),
                    "critic_digest": _module_digest(critic)}
        _write(folder / "manifest.json", manifest)
        report["post_sha256"] = _sha(folder / "weights.pt")
    _write(root / "update_result.json", report)
    return report


def evaluate_profile(root: Path, profile: str):
    if profile not in EVAL_BASES or not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Unknown profile or rejected update")
    champion, robber = _agents()
    old_policy, old_critic, _ = load_old(PARENT_ROOT, PARENT, champion)
    pre_policy, pre_critic, _ = _load_new(root, PRE, champion)
    post_policy, post_critic, _ = _load_new(root, POST, champion)
    variants = {"safety": (old_policy, old_critic, 0.),
                "post2": (old_policy, old_critic, DELEGATION),
                "new_pre": (pre_policy, pre_critic, DELEGATION),
                "new_post": (post_policy, post_critic, DELEGATION)}
    frozen = (_weight_digest(champion), _weight_digest(robber),
              _module_digest(old_policy), _module_digest(pre_policy),
              _module_digest(post_policy))
    parity = 0
    for index in range(EVAL_PER_PROFILE):
        seed, seat = EVAL_BASES[profile] + index, index % 4 + 1
        for name, (policy, critic, rate) in variants.items():
            path = root / "evaluation_games" / f"{name}_{profile}_{seed}.json.gz"
            if path.exists():
                if name == "safety":
                    parity += 1  # verified when this immutable row was first written
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            episode = _game(profile, seed, seat, champion, robber, policy, critic, rate)
            if not episode["terminal"] or episode["truncated"]:
                raise AssertionError("Held-out safety failure")
            if name == "safety":
                reference = run_strategic_episode(champion, robber, seed=seed,
                    seat=seat, profile=profile, mode="safety")
                if (episode["action_trace"] != reference.action_trace or
                        episode["final_state"] != reference.final_state or
                        episode["rng_counter"] != reference.rng_counter or
                        episode["winner"] != reference.winner):
                    raise AssertionError("Zero-delegation parity failure")
                parity += 1
            _gzip_write(path, _plain(episode))
        if index % 8 == 7:
            print(f"3B8 eval {profile}: {index+1}/{EVAL_PER_PROFILE}", flush=True)
    if frozen != (_weight_digest(champion), _weight_digest(robber),
                  _module_digest(old_policy), _module_digest(pre_policy),
                  _module_digest(post_policy)):
        raise AssertionError("Evaluation changed frozen weights")
    marker = {"profile": profile, "games": EVAL_PER_PROFILE,
              "new_zero_percent_parity": parity, "weights_unchanged": True}
    _write(root / f"evaluation_shard_{profile}.json", marker)
    return marker


def summarize(root: Path):
    names = ("safety", "post2", "new_pre", "new_post")
    rows = {name: [] for name in names}
    for profile, base in EVAL_BASES.items():
        if _read(root / f"evaluation_shard_{profile}.json")["games"] != EVAL_PER_PROFILE:
            raise AssertionError("Held-out shard incomplete")
        for index in range(EVAL_PER_PROFILE):
            for name in names:
                rows[name].append(_gzip_read(root / "evaluation_games" /
                    f"{name}_{profile}_{base+index}.json.gz"))
    if any(not e["terminal"] or e["truncated"] for group in rows.values() for e in group):
        raise AssertionError("Held-out hard safety failure")
    summary = {"variants": {name: _metrics(group) for name, group in rows.items()},
               "post2_to_new_pre": _paired_profiles(rows["post2"], rows["new_pre"]),
               "post2_to_new_post": _paired_profiles(rows["post2"], rows["new_post"]),
               "new_pre_to_new_post": _paired_profiles(rows["new_pre"], rows["new_post"]),
               "safety_to_new_post": _paired_profiles(rows["safety"], rows["new_post"]),
               "identical_action_trace_games": {
                   "post2_vs_new_pre": sum(a["action_trace"] == b["action_trace"]
                                           for a, b in zip(rows["post2"], rows["new_pre"], strict=True)),
                   "new_pre_vs_new_post": sum(a["action_trace"] == b["action_trace"]
                                               for a, b in zip(rows["new_pre"], rows["new_post"], strict=True)),
               },
               "zero_percent_parity": sum(_read(root / f"evaluation_shard_{p}.json")[
                   "new_zero_percent_parity"] for p in EVAL_BASES),
               "hard_safety": "PASS", "profile_seat_balanced": True}
    _write(root / "evaluation_summary.json", summary)
    return summary


def publish(root: Path):
    """Keep compact reproducible evidence and both independent artifacts."""
    if not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Rejected update has no publishable Post artifact")
    summary = _read(root / "evaluation_summary.json")
    if any(item["games"] != 72 for item in summary["variants"].values()):
        raise AssertionError("Held-out evaluation incomplete")
    champion, robber = _agents()
    pre_policy, _, manifest = _load_new(root, PRE, champion)
    post_policy, _, _ = _load_new(root, POST, champion)
    if (_weight_digest(champion) != manifest["champion_digest"] or
            _weight_digest(robber) != manifest["robber_digest"] or
            _sha(PARENT_ROOT / PARENT / "weights.pt") != manifest["parent_sha256"]):
        raise AssertionError("Frozen parent digest changed")
    destination = Path("docs/rl_plan_conditioned_step3b8_20261014_evidence")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("probe_pre.json", "probe_post.json", "collection_summary.json",
                 "preflight.json", "update_attempt.json", "update_result.json",
                 "evaluation_summary.json", "training_rollout.json.gz",
                 "fixed_heldout_states.json.gz"):
        shutil.copy2(root / name, destination / name)
    for name in (PRE, POST):
        (destination / name).mkdir()
        for filename in ("weights.pt", "manifest.json"):
            shutil.copy2(root / name / filename, destination / name / filename)
    games = []
    for profile, base in EVAL_BASES.items():
        for index in range(EVAL_PER_PROFILE):
            row = {"profile": profile, "seed": base + index, "seat": index % 4 + 1}
            for name in ("safety", "post2", "new_pre", "new_post"):
                episode = _gzip_read(root / "evaluation_games" /
                    f"{name}_{profile}_{base+index}.json.gz")
                row[name] = {key: episode[key] for key in (
                    "winner", "learner_id", "final_vp", "final_rank", "policy_steps",
                    "rng_counter", "final_state_sha256", "actual_delegated",
                    "learner_action_families")}
            games.append(row)
    _gzip_write(destination / "heldout_games.json.gz", games)
    hashes = {str(path.relative_to(destination)): _sha(path)
              for path in destination.rglob("*") if path.is_file()}
    _write(destination / "sha256.json", hashes)
    return {"destination": str(destination), "heldout_pairs": len(games),
            "files": len(hashes) + 1,
            "pre_policy_digest": _module_digest(pre_policy),
            "post_policy_digest": _module_digest(post_policy),
            "champion_digest": _weight_digest(champion),
            "robber_digest": _weight_digest(robber)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--phase", choices=("prepare", "probe", "probe-post", "collect",
        "preflight", "update", "eval-profile", "summarize", "publish"), required=True)
    parser.add_argument("--profile", choices=tuple(EVAL_BASES))
    args = parser.parse_args()
    if args.phase == "eval-profile":
        result = evaluate_profile(args.output, args.profile)
    elif args.phase == "preflight":
        result = preflight_phase(args.output)
    elif args.phase == "probe-post":
        result = probe_post(args.output)
    else:
        result = globals()[args.phase](args.output)
    print(f"3B8 {args.phase}: {json.dumps(result, ensure_ascii=False, default=str)[:1800]}",
          flush=True)


if __name__ == "__main__":
    main()
