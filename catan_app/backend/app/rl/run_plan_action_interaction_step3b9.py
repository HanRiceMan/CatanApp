"""Restartable Step 3B-9: interaction probe, one on-policy PPO, paired eval."""

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
from .hierarchical_ppo_step3b6 import GAMMA, GAE_LAMBDA, HierarchicalInvestmentCritic, _batch, one_guarded_update, preflight
from .plan_action_interaction_step3b9 import SCHEMA, PlanActionInvestmentPolicy
from .run_hierarchical_ppo_step3b6 import _metrics, _plain, _sha
from .run_hierarchical_ppo_step3b7 import _paired_profiles
from .run_plan_conditioned_step3b8 import _game, _load_new as load_parent, _probe_stats
from .run_strategic_curriculum_step3a20 import _module_digest
from .run_strategic_gate_step3a16 import _agents, _gzip_read, _gzip_write, _read, _write
from .strategic_gate_rollout_step3a14 import run_strategic_episode


OUTPUT = Path("experiments/plan_action_interaction_step3b9_20261014")
PARENT_ROOT = Path("docs/rl_plan_conditioned_step3b8_20261014_evidence")
PARENT = "Plan-Conditioned-Investment-Post1"
PRE = "Plan-Action-Interaction-Pre"
POST = "Plan-Action-Interaction-Post1"
TRAIN_BASES = {"rule": 35600000, "champion": 35601000, "mixed": 35602000}
EVAL_BASES = {"rule": 35700000, "champion": 35701000, "mixed": 35702000}
INIT_SEED = 3909
MINIBATCH_SEED = 3909001
TARGET = 256
DELEGATION = .10
EVAL_PER_PROFILE = 24


def _models(champion, *, from_parent=False):
    latent_dim = champion.model.policy.mlp_extractor.latent_dim_pi
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(INIT_SEED)
        policy = PlanActionInvestmentPolicy(latent_dim)
        critic = HierarchicalInvestmentCritic(latent_dim)
    if from_parent:
        old_policy, old_critic, _ = load_parent(PARENT_ROOT, PARENT, champion)
        policy.plan_heads.load_state_dict(old_policy.plan_heads.state_dict())
        critic.load_state_dict(old_critic.state_dict())
    return policy.eval(), critic.eval()


def _load(root: Path, name: str, champion):
    manifest = _read(root / name / "manifest.json")
    if manifest["schema"] != SCHEMA or manifest["champion_digest"] != _weight_digest(champion):
        raise ValueError("Step 3B-9 artifact schema or Champion mismatch")
    policy, critic = _models(champion)
    weights = torch.load(root / name / "weights.pt", map_location="cpu", weights_only=True)
    policy.load_state_dict(weights["policy"])
    critic.load_state_dict(weights["critic"])
    return policy.eval(), critic.eval(), manifest


def prepare(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    champion, robber = _agents()
    parent_weights = PARENT_ROOT / PARENT / "weights.pt"
    if not parent_weights.is_file():
        raise FileNotFoundError(parent_weights)
    policy, critic = _models(champion, from_parent=True)
    manifest = {
        "schema": SCHEMA, "step": "3B-9", "name": PRE, "parent": PARENT,
        "parent_sha256": _sha(parent_weights), "champion_digest": _weight_digest(champion),
        "robber_digest": _weight_digest(robber), "l3_initialization_seed": INIT_SEED,
        "actor_mixture": "sum_i P(plan_i|state)*P(family|state,plan_i)",
        "interaction": "base_a(state)+alpha*dot(query_a(state),plan_embedding_i)/sqrt(dim)",
        "initial_scale": .1, "plan_sampling": False,
        "boundary": "INVESTMENT executable multi-choice, protected PlayerTrade",
        "delegation_probability": DELEGATION, "training_seed_bases": TRAIN_BASES,
        "evaluation_seed_bases": EVAL_BASES, "minibatch_seed": MINIBATCH_SEED,
        "executor": "step3a14_B_FROZEN_PPO", "resolver": "step3a13_single_action_v1",
        "protected_trade": "champion_selected_player_trade_preempts",
        "runtime_context": "runtime_context_v1_79",
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
        _load(root, PRE, champion)
    else:
        folder.mkdir()
        torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()},
                   folder / "weights.pt")
        _write(folder / "manifest.json", manifest)
    return {"pre_sha256": _sha(folder / "weights.pt"),
            "parent_sha256": manifest["parent_sha256"]}


def _conditioned(policy, batch, *, uniform=False, neutral=False):
    plan = policy.plan_heads(batch["latent"].detach(), batch["runtime"],
                             batch["global"], batch["candidates"], batch["plan_mask"])
    valid = plan.candidate_mask.to(plan.probabilities.dtype)
    weights = valid / valid.sum(-1, keepdim=True).clamp_min(1) if uniform else None
    priorities = torch.full_like(plan.priorities, .5) if neutral else None
    action = policy.l3_actor(batch["latent"], batch["runtime"], batch["global"],
                             batch["candidates"], plan, batch["facts"],
                             plan_probabilities=weights, priorities=priorities)
    return plan, action


def _per_plan_stats(output, plan, family_mask):
    distributions = output.per_plan_probabilities.detach().numpy()
    log_probabilities = np.log(np.maximum(distributions, 1e-30))
    raw_logits = (output.per_plan_logits.detach().numpy()
                  if hasattr(output, "per_plan_logits") else None)
    valid = plan.candidate_mask.detach().numpy()
    legal = family_mask.detach().numpy()
    pairwise_mean, pairwise_max, log_probability_std, raw_logit_std = [], [], [], []
    family_std = [[] for _ in range(distributions.shape[-1])]
    for row in range(len(valid)):
        indexes = np.flatnonzero(valid[row])
        if len(indexes) < 2:
            continue
        q = distributions[row, indexes]
        pair = .5 * np.abs(q[:, None] - q[None, :]).sum(-1)
        upper = pair[np.triu_indices(len(indexes), 1)]
        pairwise_mean.append(float(upper.mean()))
        pairwise_max.append(float(upper.max()))
        log_probability_std.append(float(log_probabilities[row, indexes][
            :, legal[row]].std(0).mean()))
        if raw_logits is not None:
            raw_logit_std.append(float(raw_logits[row, indexes][
                :, legal[row]].std(0).mean()))
        for family in range(len(family_std)):
            if legal[row, family]:
                family_std[family].append(float(q[:, family].std()))
    return {"states_with_two_plans": len(pairwise_mean),
            "pairwise_tv_mean": float(np.mean(pairwise_mean)),
            "pairwise_tv_p95": float(np.percentile(pairwise_mean, 95)),
            "pairwise_tv_max": float(np.max(pairwise_max)),
            "conditional_log_probability_std_mean": float(np.mean(log_probability_std)),
            "raw_plan_logit_std_mean": float(np.mean(raw_logit_std)) if raw_logit_std else None,
            "family_probability_std_by_plan": {str(i): float(np.mean(values)) if values else None
                                                for i, values in enumerate(family_std)}}


def diagnose(root: Path, *, model_name=PRE, filename="diagnosis_pre.json"):
    prepare(root)
    champion, _ = _agents()
    old, _, _ = load_parent(PARENT_ROOT, PARENT, champion)
    new, _, _ = _load(root, model_name, champion)
    fixed = _gzip_read(PARENT_ROOT / "fixed_heldout_states.json.gz")
    batch = _batch(fixed)
    mask = batch["family_mask"].numpy()
    with torch.no_grad():
        old_plan, old_action = _conditioned(old, batch)
        plan, action = _conditioned(new, batch)
        old_stats = _per_plan_stats(old_action, old_plan, batch["family_mask"])
        new_stats = _per_plan_stats(action, plan, batch["family_mask"])
        ablations = {name: _conditioned(new, batch, uniform=u, neutral=n)[1]
                     for name, u, n in (("plan_uniform", True, False),
                                        ("priority_neutral", False, True),
                                        ("both_neutral", True, True))}
        valid = plan.candidate_mask
        count = valid.sum(-1)
        twice = count >= 2
        first = torch.zeros_like(plan.probabilities)
        last = torch.zeros_like(plan.probabilities)
        first_idx = valid.long().argmax(-1).unsqueeze(-1)
        last_idx = (valid.long() * torch.arange(valid.shape[1])).max(-1).values.unsqueeze(-1)
        first.scatter_(1, first_idx, twice.to(first.dtype).unsqueeze(-1))
        last.scatter_(1, last_idx, twice.to(last.dtype).unsqueeze(-1))
        if twice.any():
            indices = twice.nonzero().squeeze(-1).tolist()
            def take(tensor): return tensor[indices]
            facts = batch["facts"]
            small_facts = type(facts)(*(take(getattr(facts, name)) for name in (
                "route_one_hot", "legal_family_mask", "comparison_family_mask",
                "construction_commit", "l3_route_available")))
            small_plan = replace(plan, **{name: take(getattr(plan, name)) for name in (
                "raw_logits", "probabilities", "candidate_mask", "plan_available",
                "entropy", "priorities")})
            small_args = (take(batch["latent"]), take(batch["runtime"]),
                          take(batch["global"]), take(batch["candidates"]),
                          small_plan, small_facts)
            actor = new.l3_actor
            a = actor(*small_args, plan_probabilities=take(first))
            b = actor(*small_args, plan_probabilities=take(last))
            intervention = _probe_stats(a.probabilities.numpy(), b.probabilities.numpy(),
                                         take(batch["family_mask"]).numpy())
        else:
            intervention = {"n": 0}
    gap_old = (np.log(np.maximum(old_plan.candidate_mask.sum(-1).numpy(), 1)) -
               old_plan.entropy.numpy())
    gap_new = (np.log(np.maximum(plan.candidate_mask.sum(-1).numpy(), 1)) -
               plan.entropy.numpy())
    previous = _read(PARENT_ROOT / "update_result.json")["metrics"]
    result = {
        "fixed_heldout_states": len(fixed), "plan_available": int(plan.plan_available.sum()),
        "old_post1_per_plan": old_stats, "new_per_plan": new_stats,
        "old_plan_entropy_gap_to_uniform_mean": float(gap_old.mean()),
        "new_plan_entropy_gap_to_uniform_mean": float(gap_new.mean()),
        "parent_gradient_norm": {key: previous["gradient_norm"][key]["mean"]
                                 for key in ("l3", "priority", "plan_scorer")},
        "new_ablation": {name: _probe_stats(action.probabilities.numpy(),
                                            alt.probabilities.numpy(), mask)
                         for name, alt in ablations.items()},
        "new_plan_a_vs_b": intervention,
        "old_vs_new": _probe_stats(old_action.probabilities.numpy(),
                                    action.probabilities.numpy(), mask),
        "new_entropy_mean": float(action.entropy.mean()),
        "interaction_scale": float(new.l3_actor.interaction_scale.detach()),
        "embedding_scale": {
            "state_rms_mean": float(action.state_embedding_rms.mean()),
            "plan_rms_mean_available": float(action.plan_embedding_rms[
                plan.plan_available].mean()),
            "query_rms_mean": float(action.action_query_rms.mean()),
            "base_logit_std_mean": float(action.base_logits.std(-1, unbiased=False).mean()),
            "interaction_logit_std_mean": float(action.interaction_logits[
                plan.candidate_mask].std(unbiased=False)),
        },
        "model_name": model_name, "model_sha256": _sha(root / model_name / "weights.pt"),
    }
    _write(root / filename, result)
    return result


def collect(root: Path):
    diagnosis = _read(root / "diagnosis_pre.json")
    if (diagnosis["new_plan_a_vs_b"]["tv_mean"] <= 1e-4 or
            diagnosis["new_per_plan"]["pairwise_tv_mean"] <=
            diagnosis["old_post1_per_plan"]["pairwise_tv_mean"]):
        raise ValueError("Plan–Action interaction sensitivity did not improve")
    if (root / "update_attempt.json").exists():
        raise ValueError("Collection closed after optimizer attempt")
    champion, robber = _agents()
    policy, critic, _ = _load(root, PRE, champion)
    frozen = (_weight_digest(champion), _weight_digest(robber),
              _module_digest(policy), _module_digest(critic), _sha(root / PRE / "weights.pt"))
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
                    raise AssertionError("Single-choice Actor sample")
                episodes.append(row)
        total = sum(e["actual_delegated"] for e in episodes)
        print(f"3B9 train block={block} games={len(episodes)} transitions={total}", flush=True)
        if total >= TARGET:
            break
        if total >= 512:
            raise ValueError("Transition budget exceeded")
    else:
        raise ValueError("Target not reached")
    if frozen != (_weight_digest(champion), _weight_digest(robber),
                  _module_digest(policy), _module_digest(critic), _sha(root / PRE / "weights.pt")):
        raise AssertionError("Collection changed frozen weights")
    _gzip_write(root / "training_rollout.json.gz", episodes)
    summary = {"games": len(episodes), "transitions": total,
               "profiles": dict(Counter(e["profile"] for e in episodes)),
               "seats": dict(Counter(e["seat"] for e in episodes)),
               "family_selected": dict(Counter(d["selected_family"] for e in episodes
                                              for d in e["decisions"])),
               "single_choice_skipped": sum(e["single_choice_skipped"] for e in episodes),
               "parent_unchanged": True, "pre_sha256": frozen[-1]}
    _write(root / "collection_summary.json", summary)
    return summary


def preflight_phase(root: Path):
    champion, _ = _agents()
    policy, critic, _ = _load(root, PRE, champion)
    episodes = _gzip_read(root / "training_rollout.json.gz")
    if len(episodes) % 12:
        raise ValueError("Unbalanced training block")
    report, _, _, _ = preflight(champion, policy, critic, episodes,
                                 minimum_transitions=TARGET)
    report.update(status="PASS", pre_sha256=_sha(root / PRE / "weights.pt"),
                  champion_digest=_weight_digest(champion))
    _write(root / "preflight.json", report)
    return report


def update(root: Path):
    if (root / "update_attempt.json").exists():
        raise ValueError("Step 3B-9 permits one optimizer update")
    ready = _read(root / "preflight.json")
    if ready["status"] != "PASS":
        raise ValueError("Preflight failed")
    champion, robber = _agents()
    policy, critic, parent = _load(root, PRE, champion)
    if (_sha(root / PRE / "weights.pt") != ready["pre_sha256"] or
            _sha(PARENT_ROOT / PARENT / "weights.pt") != parent["parent_sha256"]):
        raise AssertionError("Frozen parent changed")
    episodes = _gzip_read(root / "training_rollout.json.gz")
    _write(root / "update_attempt.json", {"one_shot": True,
                                         "minibatch_seed": MINIBATCH_SEED,
                                         "transitions": ready["transitions"]})
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
        _write(folder / "manifest.json", {**parent, "name": POST,
            "training_games": len(episodes), "training_transitions": ready["transitions"],
            "pre_sha256": ready["pre_sha256"], "update_count": 1,
            "policy_digest": _module_digest(policy), "critic_digest": _module_digest(critic)})
        report["post_sha256"] = _sha(folder / "weights.pt")
    _write(root / "update_result.json", report)
    return report


def diagnose_post(root: Path):
    if not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Rejected update has no Post diagnosis")
    return diagnose(root, model_name=POST, filename="diagnosis_post.json")


def evaluate_profile(root: Path, profile: str):
    if profile not in EVAL_BASES or not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Unknown profile or rejected update")
    champion, robber = _agents()
    old_policy, old_critic, _ = load_parent(PARENT_ROOT, PARENT, champion)
    pre_policy, pre_critic, _ = _load(root, PRE, champion)
    post_policy, post_critic, _ = _load(root, POST, champion)
    variants = {"safety": (old_policy, old_critic, 0.),
                "old_post1": (old_policy, old_critic, DELEGATION),
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
                    parity += 1
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            episode = _game(profile, seed, seat, champion, robber, policy, critic, rate)
            if not episode["terminal"] or episode["truncated"]:
                raise AssertionError("Held-out hard safety failure")
            if name == "safety":
                reference = run_strategic_episode(champion, robber, seed=seed,
                    seat=seat, profile=profile, mode="safety")
                if (episode["action_trace"] != reference.action_trace or
                        episode["final_state"] != reference.final_state or
                        episode["rng_counter"] != reference.rng_counter or
                        episode["winner"] != reference.winner):
                    raise AssertionError("0% Safety parity failure")
                parity += 1
            _gzip_write(path, _plain(episode))
        if index % 8 == 7:
            print(f"3B9 eval {profile} {index+1}/{EVAL_PER_PROFILE}", flush=True)
    if frozen != (_weight_digest(champion), _weight_digest(robber),
                  _module_digest(old_policy), _module_digest(pre_policy),
                  _module_digest(post_policy)):
        raise AssertionError("Evaluation changed frozen weights")
    marker = {"profile": profile, "games": EVAL_PER_PROFILE,
              "zero_percent_parity": parity, "weights_unchanged": True}
    _write(root / f"evaluation_shard_{profile}.json", marker)
    return marker


def summarize(root: Path):
    names = ("safety", "old_post1", "new_pre", "new_post")
    rows = {name: [] for name in names}
    for profile, base in EVAL_BASES.items():
        if _read(root / f"evaluation_shard_{profile}.json")["games"] != EVAL_PER_PROFILE:
            raise AssertionError("Held-out shard incomplete")
        for index in range(EVAL_PER_PROFILE):
            for name in names:
                rows[name].append(_gzip_read(root / "evaluation_games" /
                    f"{name}_{profile}_{base+index}.json.gz"))
    if any(not row["terminal"] or row["truncated"] for group in rows.values() for row in group):
        raise AssertionError("Held-out safety failed")
    report = {"variants": {name: _metrics(group) for name, group in rows.items()},
              "old_post1_to_new_pre": _paired_profiles(rows["old_post1"], rows["new_pre"]),
              "old_post1_to_new_post": _paired_profiles(rows["old_post1"], rows["new_post"]),
              "new_pre_to_new_post": _paired_profiles(rows["new_pre"], rows["new_post"]),
              "safety_to_new_post": _paired_profiles(rows["safety"], rows["new_post"]),
              "identical_action_trace_games": {
                  "old_vs_new_pre": sum(a["action_trace"] == b["action_trace"]
                                        for a, b in zip(rows["old_post1"], rows["new_pre"], strict=True)),
                  "new_pre_vs_post": sum(a["action_trace"] == b["action_trace"]
                                          for a, b in zip(rows["new_pre"], rows["new_post"], strict=True))},
              "zero_percent_parity": sum(_read(root / f"evaluation_shard_{p}.json")[
                  "zero_percent_parity"] for p in EVAL_BASES),
              "hard_safety": "PASS", "profile_seat_balanced": True}
    _write(root / "evaluation_summary.json", report)
    return report


def publish(root: Path):
    if not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Rejected update cannot publish")
    summary = _read(root / "evaluation_summary.json")
    if any(item["games"] != 72 for item in summary["variants"].values()):
        raise AssertionError("Held-out evaluation incomplete")
    champion, robber = _agents()
    pre_policy, _, manifest = _load(root, PRE, champion)
    post_policy, _, _ = _load(root, POST, champion)
    if (_weight_digest(champion) != manifest["champion_digest"] or
            _weight_digest(robber) != manifest["robber_digest"] or
            _sha(PARENT_ROOT / PARENT / "weights.pt") != manifest["parent_sha256"]):
        raise AssertionError("Frozen parent changed")
    destination = Path("docs/rl_plan_action_interaction_step3b9_20261014_evidence")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("diagnosis_pre.json", "diagnosis_post.json", "collection_summary.json",
                 "preflight.json", "update_attempt.json", "update_result.json",
                 "evaluation_summary.json", "training_rollout.json.gz"):
        shutil.copy2(root / name, destination / name)
    for name in (PRE, POST):
        (destination / name).mkdir()
        for filename in ("weights.pt", "manifest.json"):
            shutil.copy2(root / name / filename, destination / name / filename)
    games = []
    for profile, base in EVAL_BASES.items():
        for index in range(EVAL_PER_PROFILE):
            row = {"profile": profile, "seed": base + index, "seat": index % 4 + 1}
            for name in ("safety", "old_post1", "new_pre", "new_post"):
                episode = _gzip_read(root / "evaluation_games" /
                    f"{name}_{profile}_{base+index}.json.gz")
                row[name] = {key: episode[key] for key in (
                    "winner", "learner_id", "final_vp", "final_rank", "policy_steps",
                    "rng_counter", "final_state_sha256", "actual_delegated",
                    "learner_action_families")}
            games.append(row)
    _gzip_write(destination / "heldout_games.json.gz", games)
    _write(destination / "sha256.json", {str(path.relative_to(destination)): _sha(path)
                                      for path in destination.rglob("*") if path.is_file()})
    return {"destination": str(destination), "heldout_pairs": len(games),
            "pre_policy_digest": _module_digest(pre_policy),
            "post_policy_digest": _module_digest(post_policy),
            "champion_digest": _weight_digest(champion)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--phase", choices=("prepare", "diagnose", "collect", "preflight",
        "update", "diagnose-post", "eval-profile", "summarize", "publish"), required=True)
    parser.add_argument("--profile", choices=tuple(EVAL_BASES))
    args = parser.parse_args()
    if args.phase == "preflight":
        result = preflight_phase(args.output)
    elif args.phase == "diagnose-post":
        result = diagnose_post(args.output)
    elif args.phase == "eval-profile":
        result = evaluate_profile(args.output, args.profile)
    else:
        result = globals()[args.phase](args.output)
    print(f"3B9 {args.phase}: {json.dumps(result, ensure_ascii=False, default=str)[:2200]}",
          flush=True)


if __name__ == "__main__":
    main()
