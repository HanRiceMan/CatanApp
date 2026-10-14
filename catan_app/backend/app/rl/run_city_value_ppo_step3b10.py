"""Restartable City-only on-policy pilot; one guarded PPO update at most."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .city_value_ppo_step3b10 import CITY_FAMILIES, SCHEMA, CityValueCritic
from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .hierarchical_ppo_step3b6 import GAMMA, GAE_LAMBDA, _batch, one_guarded_update, preflight
from .hierarchical_rollout_step3b6 import run_hierarchical_episode
from .hierarchical_shadow_controller_step3b4 import HierarchicalRoute
from .plan_action_interaction_step3b9 import PlanActionInvestmentPolicy
from .run_hierarchical_ppo_step3b6 import _metrics, _plain, _sha
from .run_hierarchical_ppo_step3b7 import _paired_profiles
from .run_plan_action_interaction_step3b9 import (
    _conditioned, _load as load_parent, _per_plan_stats,
)
from .run_plan_conditioned_step3b8 import _probe_stats
from .run_strategic_curriculum_step3a20 import _module_digest
from .run_strategic_gate_step3a16 import _agents, _gzip_read, _gzip_write, _read, _write
from .strategic_gate_rollout_step3a14 import run_strategic_episode


ROOT = Path("experiments/city_value_ppo_step3b10_20261014")
PARENT_ROOT = Path("docs/rl_plan_action_interaction_step3b9_20261014_evidence")
PARENT = "Plan-Action-Interaction-Post1"
PRE = "City-Value-Pre"
POST = "City-Value-Post1"
ROUTE = HierarchicalRoute.CITY_VALUE_COMPARISON
TRAIN_BASES = {"rule": 35800000, "champion": 35801000, "mixed": 35802000}
EVAL_BASES = {"rule": 35900000, "champion": 35901000, "mixed": 35902000}
TRAIN_DELEGATION = .25
EVAL_DELEGATION = .10
TARGET = 128
MAX_BLOCKS = 41  # 492 games; stop only on a balanced 12-game block.
EVAL_PER_PROFILE = 48
INIT_SEED = 3910
MINIBATCH_SEED = 3910001


def _models(champion, *, parent=False):
    latent_dim = champion.model.policy.mlp_extractor.latent_dim_pi
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(INIT_SEED)
        policy = PlanActionInvestmentPolicy(latent_dim)
        critic = CityValueCritic(latent_dim)
    if parent:
        source, _, _ = load_parent(PARENT_ROOT, PARENT, champion)
        policy.load_state_dict(source.state_dict())
    return policy.eval(), critic.eval()


def _load(root: Path, name: str, champion):
    manifest = _read(root / name / "manifest.json")
    if manifest["schema"] != SCHEMA or manifest["champion_digest"] != _weight_digest(champion):
        raise ValueError("City artifact schema or frozen Champion mismatch")
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
    policy, critic = _models(champion, parent=True)
    manifest = {
        "schema": SCHEMA, "step": "3B-10", "name": PRE, "parent": PARENT,
        "parent_sha256": _sha(parent_weights), "champion_digest": _weight_digest(champion),
        "robber_digest": _weight_digest(robber), "city_critic_initialization_seed": INIT_SEED,
        "actor_parent": "3B-9 Post1 L1/L2/L3 exactly", "critic_parent": None,
        "train_route": ROUTE.value, "investment_delegation": 0.,
        "city_training_delegation": TRAIN_DELEGATION,
        "city_evaluation_delegation": EVAL_DELEGATION,
        "training_seed_bases": TRAIN_BASES, "evaluation_seed_bases": EVAL_BASES,
        "minibatch_seed": MINIBATCH_SEED, "target_transitions": TARGET,
        "max_games": 12 * MAX_BLOCKS, "evaluation_games_per_variant": 3 * EVAL_PER_PROFILE,
        "family_order": list(CITY_FAMILIES), "executor": "step3a14_B_FROZEN_PPO",
        "player_trade": "protected_champion_preemption",
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


def _city_game(profile, seed, seat, champion, robber, policy, critic, delegation):
    return run_hierarchical_episode(
        champion, robber, policy, critic, seed=seed, seat=seat, profile=profile,
        delegation_probability=delegation, delegation_seed=seed + 500000,
        family_seed=seed + 900000, multi_choice_only=True, delegated_route=ROUTE)


def collect(root: Path):
    if (root / "update_attempt.json").exists():
        raise ValueError("Collection closed after an optimizer attempt")
    champion, robber = _agents()
    policy, critic, _ = _load(root, PRE, champion)
    frozen = (_weight_digest(champion), _weight_digest(robber),
              _module_digest(policy), _module_digest(critic), _sha(root / PRE / "weights.pt"))
    episodes = []
    for block in range(MAX_BLOCKS):
        for profile, base in TRAIN_BASES.items():
            for seat_index in range(4):
                seed, seat = base + block * 4 + seat_index, seat_index + 1
                path = root / "training_games" / f"{profile}_{seed}.json.gz"
                if path.exists():
                    row = _gzip_read(path)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    row = _plain(_city_game(profile, seed, seat, champion, robber,
                                            policy, critic, TRAIN_DELEGATION))
                    if not row["terminal"] or row["truncated"]:
                        raise AssertionError("Training episode did not terminate")
                    _gzip_write(path, row)
                if any(d["route"] != ROUTE.value or sum(d["family_mask"]) < 2
                       for d in row["decisions"]):
                    raise AssertionError("Wrong route or uninformative City Actor sample")
                episodes.append(row)
        total = sum(e["actual_delegated"] for e in episodes)
        print(f"3B10 training block={block+1} games={len(episodes)} transitions={total}",
              flush=True)
        if total >= TARGET:
            break
    if frozen != (_weight_digest(champion), _weight_digest(robber),
                  _module_digest(policy), _module_digest(critic), _sha(root / PRE / "weights.pt")):
        raise AssertionError("Collection changed frozen weights")
    summary = {"games": len(episodes), "transitions": total,
               "profile_games": dict(Counter(e["profile"] for e in episodes)),
               "seat_games": dict(Counter(e["seat"] for e in episodes)),
               "family_selected": dict(Counter(d["selected_family"] for e in episodes
                                              for d in e["decisions"])),
               "city_opportunities": sum(len(e["city_opportunities"]) for e in episodes),
               "protected_preemption": sum(e["protected_preemption"] for e in episodes),
               "target_reached": total >= TARGET,
               "parent_unchanged": True}
    _write(root / "collection_summary.json", summary)
    if total >= TARGET:
        _gzip_write(root / "training_rollout.json.gz", episodes)
    return summary


def preflight_phase(root: Path):
    if not _read(root / "collection_summary.json")["target_reached"]:
        raise ValueError("Insufficient natural City transitions; PPO forbidden")
    champion, _ = _agents()
    policy, critic, _ = _load(root, PRE, champion)
    episodes = _gzip_read(root / "training_rollout.json.gz")
    report, _, _, _ = preflight(
        champion, policy, critic, episodes, minimum_transitions=TARGET,
        route=ROUTE, allowed_families=CITY_FAMILIES,
        delegation_probability=TRAIN_DELEGATION)
    report.update(status="PASS", pre_sha256=_sha(root / PRE / "weights.pt"),
                  champion_digest=_weight_digest(champion))
    _write(root / "preflight.json", report)
    return report


def update(root: Path):
    if (root / "update_attempt.json").exists():
        raise ValueError("Step 3B-10 permits only one optimizer attempt")
    ready = _read(root / "preflight.json")
    if ready["status"] != "PASS":
        raise ValueError("City preflight failed")
    champion, robber = _agents()
    policy, critic, parent = _load(root, PRE, champion)
    if (_sha(root / PRE / "weights.pt") != ready["pre_sha256"] or
            _sha(PARENT_ROOT / PARENT / "weights.pt") != parent["parent_sha256"]):
        raise AssertionError("Frozen parent changed")
    episodes = _gzip_read(root / "training_rollout.json.gz")
    _write(root / "update_attempt.json", {"one_shot": True,
                                         "minibatch_seed": MINIBATCH_SEED,
                                         "transitions": ready["transitions"]})
    result = one_guarded_update(
        champion, policy, critic, episodes, minibatch_seed=MINIBATCH_SEED,
        minimum_transitions=TARGET, route=ROUTE,
        allowed_families=CITY_FAMILIES,
        delegation_probability=TRAIN_DELEGATION)
    output = {"accepted": result.accepted, "reason": result.reason,
              "metrics": result.metrics}
    if (_weight_digest(champion) != parent["champion_digest"] or
            _weight_digest(robber) != parent["robber_digest"] or
            _sha(root / PRE / "weights.pt") != ready["pre_sha256"]):
        raise AssertionError("Frozen parent changed during update")
    if result.accepted:
        folder = root / POST
        folder.mkdir(parents=True, exist_ok=False)
        torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()},
                   folder / "weights.pt")
        _write(folder / "manifest.json", {**parent, "name": POST,
            "training_games": len(episodes), "training_transitions": ready["transitions"],
            "pre_sha256": ready["pre_sha256"], "update_count": 1,
            "policy_digest": _module_digest(policy), "critic_digest": _module_digest(critic)})
        output["post_sha256"] = _sha(folder / "weights.pt")
    _write(root / "update_result.json", output)
    return output


def evaluate_profile(root: Path, profile: str):
    if profile not in EVAL_BASES or not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Unknown profile or rejected City update")
    champion, robber = _agents()
    old_policy, old_critic, _ = load_parent(PARENT_ROOT, PARENT, champion)
    pre_policy, pre_critic, _ = _load(root, PRE, champion)
    post_policy, post_critic, _ = _load(root, POST, champion)
    frozen = (_weight_digest(champion), _weight_digest(robber),
              _module_digest(old_policy), _module_digest(pre_policy),
              _module_digest(post_policy))
    variants = {"safety": (pre_policy, pre_critic, 0., ROUTE),
                "investment_post1": (old_policy, old_critic, .10,
                                     HierarchicalRoute.INVESTMENT),
                "city_pre": (pre_policy, pre_critic, EVAL_DELEGATION, ROUTE),
                "city_post": (post_policy, post_critic, EVAL_DELEGATION, ROUTE)}
    parity = 0
    for index in range(EVAL_PER_PROFILE):
        seed, seat = EVAL_BASES[profile] + index, index % 4 + 1
        for name, (policy, critic, rate, route) in variants.items():
            path = root / "evaluation_games" / f"{name}_{profile}_{seed}.json.gz"
            if path.exists():
                if name == "safety":
                    parity += 1
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            episode = run_hierarchical_episode(
                champion, robber, policy, critic, seed=seed, seat=seat, profile=profile,
                delegation_probability=rate, delegation_seed=seed + 500000,
                family_seed=seed + 900000, multi_choice_only=True,
                delegated_route=route)
            if not episode["terminal"] or episode["truncated"]:
                raise AssertionError("Held-out City episode safety failure")
            if name == "safety":
                reference = run_strategic_episode(champion, robber, seed=seed,
                    seat=seat, profile=profile, mode="safety")
                if (episode["action_trace"] != reference.action_trace or
                        episode["final_state"] != reference.final_state or
                        episode["rng_counter"] != reference.rng_counter or
                        episode["winner"] != reference.winner):
                    raise AssertionError("0% Safety Action/State/RNG parity failure")
                parity += 1
            _gzip_write(path, _plain(episode))
        if index % 8 == 7:
            print(f"3B10 evaluation {profile} {index+1}/{EVAL_PER_PROFILE}", flush=True)
    if frozen != (_weight_digest(champion), _weight_digest(robber),
                  _module_digest(old_policy), _module_digest(pre_policy),
                  _module_digest(post_policy)):
        raise AssertionError("Evaluation changed frozen weights")
    marker = {"profile": profile, "games": EVAL_PER_PROFILE,
              "zero_percent_parity": parity, "weights_unchanged": True}
    _write(root / f"evaluation_shard_{profile}.json", marker)
    return marker


def _city_metrics(rows):
    opportunities = [item for row in rows for item in row["city_opportunities"]]
    selected = [item for item in opportunities if item["first_action_delegated"]]
    return {"opportunities": len(opportunities),
            "same_turn_city": sum(item["city_same_turn"] for item in opportunities),
            "same_turn_city_rate": float(np.mean([item["city_same_turn"]
                                                    for item in opportunities])),
            "immediate_city": sum(item["city_immediately"] for item in opportunities),
            "other_score_build_without_city": sum(
                "BUILD_SETTLEMENT" in item["action_families"] and
                not item["city_same_turn"] for item in opportunities),
            "deferred_without_city": sum(not item["city_same_turn"] for item in opportunities),
            "first_action_families": dict(Counter(item["first_action_family"]
                                                   for item in opportunities)),
            "delegated_opportunities": len(selected),
            "delegated_same_turn_city_rate": (float(np.mean([item["city_same_turn"]
                                                        for item in selected]))
                                              if selected else None),
            "delegated_first_action_families": dict(Counter(
                item["first_action_family"] for item in selected))}


def summarize(root: Path):
    names = ("safety", "investment_post1", "city_pre", "city_post")
    rows = {name: [] for name in names}
    for profile, base in EVAL_BASES.items():
        if _read(root / f"evaluation_shard_{profile}.json")["games"] != EVAL_PER_PROFILE:
            raise AssertionError("Incomplete held-out evaluation")
        for index in range(EVAL_PER_PROFILE):
            for name in names:
                rows[name].append(_gzip_read(root / "evaluation_games" /
                    f"{name}_{profile}_{base+index}.json.gz"))
    if any(not row["terminal"] or row["truncated"]
           for group in rows.values() for row in group):
        raise AssertionError("Held-out terminal/truncation failure")
    result = {"variants": {name: {**_metrics(group), **_city_metrics(group)}
                            for name, group in rows.items()},
              "city_pre_to_post": _paired_profiles(rows["city_pre"], rows["city_post"]),
              "safety_to_city_post": _paired_profiles(rows["safety"], rows["city_post"]),
              "investment_post1_to_city_post": _paired_profiles(
                  rows["investment_post1"], rows["city_post"]),
              "identical_city_pre_post_action_traces": sum(
                  a["action_trace"] == b["action_trace"] for a, b in zip(
                      rows["city_pre"], rows["city_post"], strict=True)),
              "zero_percent_parity": sum(_read(root / f"evaluation_shard_{profile}.json")[
                  "zero_percent_parity"] for profile in EVAL_BASES),
              "hard_safety": "PASS", "profile_seat_balanced": True}
    _write(root / "evaluation_summary.json", result)
    return result


def diagnose(root: Path):
    champion, _ = _agents()
    pre_policy, _, _ = _load(root, PRE, champion)
    post_policy, _, _ = _load(root, POST, champion)
    city_decisions = []
    for profile, base in EVAL_BASES.items():
        for index in range(EVAL_PER_PROFILE):
            episode = _gzip_read(root / "evaluation_games" /
                                 f"city_pre_{profile}_{base+index}.json.gz")
            city_decisions.extend(episode["decisions"])
    if not city_decisions:
        raise AssertionError("No held-out City state for fixed-state diagnosis")
    city_batch = _batch(city_decisions)
    investment_batch = _batch(_gzip_read(
        Path("docs/rl_plan_conditioned_step3b8_20261014_evidence") /
        "fixed_heldout_states.json.gz"))

    def probe(batch):
        mask = batch["family_mask"].numpy()
        with torch.no_grad():
            pre_plan, pre_action = _conditioned(pre_policy, batch)
            post_plan, post_action = _conditioned(post_policy, batch)
            uniform = _conditioned(post_policy, batch, uniform=True)[1]
            neutral = _conditioned(post_policy, batch, neutral=True)[1]
            both = _conditioned(post_policy, batch, uniform=True, neutral=True)[1]
            plan_stats = _per_plan_stats(post_action, post_plan, batch["family_mask"])
            valid = post_plan.candidate_mask
            twice = valid.sum(-1) >= 2
            if twice.any():
                indices = twice.nonzero().squeeze(-1).tolist()
                def take(tensor): return tensor[indices]
                facts = batch["facts"]
                small_facts = type(facts)(*(take(getattr(facts, key)) for key in (
                    "route_one_hot", "legal_family_mask", "comparison_family_mask",
                    "construction_commit", "l3_route_available")))
                small_plan = replace(post_plan, **{key: take(getattr(post_plan, key))
                    for key in ("raw_logits", "probabilities", "candidate_mask",
                                "plan_available", "entropy", "priorities")})
                first = torch.zeros_like(post_plan.probabilities)
                last = torch.zeros_like(post_plan.probabilities)
                first.scatter_(1, valid.long().argmax(-1).unsqueeze(-1),
                               twice.to(first.dtype).unsqueeze(-1))
                last.scatter_(1, (valid.long() * torch.arange(valid.shape[1])).max(-1)
                              .values.unsqueeze(-1), twice.to(last.dtype).unsqueeze(-1))
                args = (take(batch["latent"]), take(batch["runtime"]),
                        take(batch["global"]), take(batch["candidates"]),
                        small_plan, small_facts)
                a = post_policy.l3_actor(*args, plan_probabilities=take(first))
                b = post_policy.l3_actor(*args, plan_probabilities=take(last))
                intervention = _probe_stats(a.probabilities.numpy(), b.probabilities.numpy(),
                                             take(batch["family_mask"]).numpy())
            else:
                intervention = {"n": 0}
        priorities_pre = pre_plan.priorities.detach().numpy()
        priorities_post = post_plan.priorities.detach().numpy()
        weights_pre = pre_plan.probabilities.detach().numpy()
        weights_post = post_plan.probabilities.detach().numpy()
        expected_pre = (weights_pre[:, :, None] * priorities_pre).sum(1)
        expected_post = (weights_post[:, :, None] * priorities_post).sum(1)
        plan_available = post_plan.plan_available.detach().numpy().astype(bool)
        count = post_plan.candidate_mask.detach().sum(-1).numpy()
        entropy_gap = (np.log(count[plan_available]) -
                       post_plan.entropy.detach().numpy()[plan_available])
        return {"states": len(mask), "per_plan_post": plan_stats,
                "pre_to_post": _probe_stats(pre_action.probabilities.numpy(),
                                             post_action.probabilities.numpy(), mask),
                "plan_uniform": _probe_stats(post_action.probabilities.numpy(),
                                               uniform.probabilities.numpy(), mask),
                "priority_neutral": _probe_stats(post_action.probabilities.numpy(),
                                                   neutral.probabilities.numpy(), mask),
                "both_neutral": _probe_stats(post_action.probabilities.numpy(),
                                              both.probabilities.numpy(), mask),
                "plan_a_vs_b": intervention,
                "plan_available": int(plan_available.sum()),
                "plan_entropy_pre": float(pre_plan.entropy.mean()),
                "plan_entropy_post": float(post_plan.entropy.mean()),
                "plan_entropy_gap_to_uniform_post": float(entropy_gap.mean()),
                "priority_pre_mean": expected_pre[plan_available].mean(0).tolist(),
                "priority_post_mean": expected_post[plan_available].mean(0).tolist(),
                "priority_post_std": expected_post[plan_available].std(0).tolist()}

    result = {"city_fixed": probe(city_batch),
              "investment_fixed": probe(investment_batch),
              "investment_delegation_during_city_training": 0,
              "investment_delegation_during_city_evaluation": 0}
    _write(root / "fixed_state_diagnosis.json", result)
    return result


def publish(root: Path):
    if not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Rejected City artifact cannot publish")
    summary = _read(root / "evaluation_summary.json")
    if any(row["games"] != 144 for row in summary["variants"].values()):
        raise AssertionError("Evaluation incomplete")
    champion, robber = _agents()
    _, _, manifest = _load(root, PRE, champion)
    if (_weight_digest(champion) != manifest["champion_digest"] or
            _weight_digest(robber) != manifest["robber_digest"] or
            _sha(PARENT_ROOT / PARENT / "weights.pt") != manifest["parent_sha256"]):
        raise AssertionError("Frozen parent changed")
    destination = Path("docs/rl_city_value_ppo_step3b10_20261014_evidence")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("collection_summary.json", "preflight.json", "update_attempt.json",
                 "update_result.json", "evaluation_summary.json",
                 "fixed_state_diagnosis.json", "training_rollout.json.gz"):
        shutil.copy2(root / name, destination / name)
    for name in (PRE, POST):
        (destination / name).mkdir()
        for filename in ("manifest.json", "weights.pt"):
            shutil.copy2(root / name / filename, destination / name / filename)
    games = []
    for profile, base in EVAL_BASES.items():
        for index in range(EVAL_PER_PROFILE):
            row = {"profile": profile, "seed": base + index, "seat": index % 4 + 1}
            for name in ("safety", "investment_post1", "city_pre", "city_post"):
                episode = _gzip_read(root / "evaluation_games" /
                                     f"{name}_{profile}_{base+index}.json.gz")
                row[name] = {key: episode[key] for key in (
                    "winner", "learner_id", "final_vp", "final_rank", "policy_steps",
                    "rng_counter", "final_state_sha256", "actual_delegated",
                    "learner_action_families", "city_opportunities")}
            games.append(row)
    _gzip_write(destination / "heldout_games.json.gz", games)
    _write(destination / "sha256.json", {str(path.relative_to(destination)): _sha(path)
                                      for path in destination.rglob("*") if path.is_file()})
    return {"destination": str(destination), "heldout_games": len(games),
            "champion_digest": manifest["champion_digest"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--phase", choices=("prepare", "collect", "preflight", "update",
        "eval-profile", "summarize", "diagnose", "publish"), required=True)
    parser.add_argument("--profile", choices=tuple(EVAL_BASES))
    args = parser.parse_args()
    if args.phase == "preflight":
        result = preflight_phase(args.output)
    elif args.phase == "eval-profile":
        result = evaluate_profile(args.output, args.profile)
    else:
        result = globals()[args.phase](args.output)
    print(f"3B10 {args.phase}: {json.dumps(result, ensure_ascii=False, default=str)[:1800]}",
          flush=True)


if __name__ == "__main__":
    main()
