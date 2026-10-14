"""Restartable route-isolated City PPO: new rollout, one guarded update."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .city_value_ppo_step3b10 import CITY_FAMILIES, CityValueCritic
from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .hierarchical_ppo_step3b6 import GAMMA, GAE_LAMBDA, _batch, one_guarded_update, preflight
from .hierarchical_rollout_step3b6 import run_hierarchical_episode
from .hierarchical_shadow_controller_step3b4 import HierarchicalRoute
from .route_isolated_city_step3b11 import RouteIsolatedCityPolicy, SCHEMA
from .run_city_value_ppo_step3b10 import _city_metrics, _load as load_3b10
from .run_hierarchical_ppo_step3b6 import _metrics, _plain, _sha
from .run_hierarchical_ppo_step3b7 import _paired_profiles
from .run_plan_action_interaction_step3b9 import _load as load_parent, _per_plan_stats
from .run_plan_conditioned_step3b8 import _probe_stats
from .run_strategic_curriculum_step3a20 import _module_digest
from .run_strategic_gate_step3a16 import _agents, _gzip_read, _gzip_write, _read, _write
from .strategic_gate_rollout_step3a14 import run_strategic_episode


ROOT = Path("experiments/route_isolated_city_step3b11_20261014")
PARENT_ROOT = Path("docs/rl_plan_action_interaction_step3b9_20261014_evidence")
PARENT = "Plan-Action-Interaction-Post1"
OLD_CITY_ROOT = Path("docs/rl_city_value_ppo_step3b10_20261014_evidence")
OLD_CITY_POST = "City-Value-Post1"
PRE = "Route-Isolated-City-Pre"
POST = "Route-Isolated-City-Post1"
ROUTE = HierarchicalRoute.CITY_VALUE_COMPARISON
TRAIN_BASES = {"rule": 36000000, "champion": 36001000, "mixed": 36002000}
EVAL_BASES = {"rule": 36100000, "champion": 36101000, "mixed": 36102000}
TRAIN_DELEGATION = .25
EVAL_DELEGATION = .10
TARGET = 192
MAX_BLOCKS = 54  # 648 games; stop only on a balanced 12-game block.
EVAL_PER_PROFILE = 48
INIT_SEED = 3911
MINIBATCH_SEED = 3911001
INVESTMENT_FIXED = (Path("docs/rl_plan_conditioned_step3b8_20261014_evidence") /
                    "fixed_heldout_states.json.gz")


def _models(champion, *, parent=False):
    latent_dim = champion.model.policy.mlp_extractor.latent_dim_pi
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(INIT_SEED)
        policy = RouteIsolatedCityPolicy(latent_dim)
        critic = CityValueCritic(latent_dim)
    if parent:
        source, _, _ = load_parent(PARENT_ROOT, PARENT, champion)
        policy.load_base(source)
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


def _investment_parity(champion, policy):
    """Exact fixed-state parity with frozen 3B-9 Investment Actor."""
    parent, _, _ = load_parent(PARENT_ROOT, PARENT, champion)
    batch = _batch(_gzip_read(INVESTMENT_FIXED))
    args = (batch["latent"], batch["runtime"], batch["global"],
            batch["candidates"], batch["plan_mask"], batch["facts"])
    with torch.no_grad():
        actual = policy(*args)[-1].probabilities
        expected = parent(*args)[-1].probabilities
    if not torch.equal(actual, expected):
        raise AssertionError("City Adapter changed frozen Investment probabilities")
    return {"states": len(actual), "max_abs_probability_difference": 0.,
            "top_family_flips": 0, "exact": True}


def prepare(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    champion, robber = _agents()
    parent_weights = PARENT_ROOT / PARENT / "weights.pt"
    if not parent_weights.is_file():
        raise FileNotFoundError(parent_weights)
    policy, critic = _models(champion, parent=True)
    investment_parity = _investment_parity(champion, policy)
    manifest = {
        "schema": SCHEMA, "step": "3B-11", "name": PRE, "parent": PARENT,
        "parent_sha256": _sha(parent_weights), "champion_digest": _weight_digest(champion),
        "robber_digest": _weight_digest(robber), "city_critic_initialization_seed": INIT_SEED,
        "actor_parent": "3B-9 Post1 L1/L2/L3 frozen; City adapter zero-init",
        "critic_parent": None,
        "train_route": ROUTE.value, "investment_delegation": 0.,
        "city_training_delegation": TRAIN_DELEGATION,
        "city_evaluation_delegation": EVAL_DELEGATION,
        "training_seed_bases": TRAIN_BASES, "evaluation_seed_bases": EVAL_BASES,
        "minibatch_seed": MINIBATCH_SEED, "target_transitions": TARGET,
        "max_games": 12 * MAX_BLOCKS, "evaluation_games_per_variant": 3 * EVAL_PER_PROFILE,
        "family_order": list(CITY_FAMILIES), "executor": "step3a14_B_FROZEN_PPO",
        "player_trade": "protected_champion_preemption",
        "trainable_actor_parameters": sum(p.numel() for p in policy.parameters()
                                           if p.requires_grad),
        "reward": {"win": 1, "loss": -1, "victory_point_delta": .05,
                   "gamma": GAMMA, "gae_lambda": GAE_LAMBDA},
        "ppo": {"actor_lr": 1e-4, "critic_lr": 3e-4, "batch_size": 64,
                "epochs": 2, "clip_range": .1, "entropy_coef": .005},
        "policy_digest": _module_digest(policy), "critic_digest": _module_digest(critic),
        "investment_parity_pre": investment_parity,
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


def collect_profile(root: Path, profile: str, *, blocks: int = 32):
    """Disjoint fixed-seed shards; collect() later applies balanced-block stop."""
    if profile not in TRAIN_BASES or (root / "update_attempt.json").exists():
        raise ValueError("Unknown profile or collection already closed")
    champion, robber = _agents()
    policy, critic, _ = _load(root, PRE, champion)
    frozen = (_weight_digest(champion), _weight_digest(robber),
              _module_digest(policy), _module_digest(critic))
    total = 0
    for block in range(blocks):
        for seat_index in range(4):
            seed, seat = TRAIN_BASES[profile] + block * 4 + seat_index, seat_index + 1
            path = root / "training_games" / f"{profile}_{seed}.json.gz"
            if path.exists():
                row = _gzip_read(path)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                row = _plain(_city_game(profile, seed, seat, champion, robber,
                                        policy, critic, TRAIN_DELEGATION))
                if not row["terminal"] or row["truncated"]:
                    raise AssertionError("Shard episode did not terminate")
                _gzip_write(path, row)
            total += row["actual_delegated"]
        if block % 8 == 7:
            print(f"3B11 shard {profile} blocks={block+1} transitions={total}", flush=True)
    if frozen != (_weight_digest(champion), _weight_digest(robber),
                  _module_digest(policy), _module_digest(critic)):
        raise AssertionError("Shard collection changed frozen weights")
    return {"profile": profile, "games": 4 * blocks,
            "transitions": total, "weights_unchanged": True}


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
        print(f"3B11 training block={block+1} games={len(episodes)} transitions={total}",
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
        raise ValueError("Step 3B-11 permits only one optimizer attempt")
    ready = _read(root / "preflight.json")
    if ready["status"] != "PASS":
        raise ValueError("City preflight failed")
    champion, robber = _agents()
    policy, critic, parent = _load(root, PRE, champion)
    base_digests = {"plan_heads": _module_digest(policy.plan_heads),
                    "shared_actor": _module_digest(policy.l3_actor),
                    "investment_parity": _investment_parity(champion, policy)}
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
    if (result.metrics["l3_delta"] != 0 or result.metrics["priority_delta"] != 0 or
            result.metrics["plan_scorer_delta"] != 0 or
            result.metrics["city_adapter_delta"] <= 0 or
            _module_digest(policy.plan_heads) != base_digests["plan_heads"] or
            _module_digest(policy.l3_actor) != base_digests["shared_actor"]):
        raise AssertionError("City-only trainable boundary violated")
    output_parity = _investment_parity(champion, policy)
    output = {"accepted": result.accepted, "reason": result.reason,
              "metrics": result.metrics, "investment_parity": output_parity,
              "frozen_base_digests": base_digests}
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


def evaluate_profile(root: Path, profile: str, *, start: int = 0, stop: int = EVAL_PER_PROFILE):
    if profile not in EVAL_BASES or not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Unknown profile or rejected City update")
    if not 0 <= start < stop <= EVAL_PER_PROFILE:
        raise ValueError("Held-out shard must be within fixed profile seed range")
    champion, robber = _agents()
    old_policy, old_critic, _ = load_3b10(OLD_CITY_ROOT, OLD_CITY_POST, champion)
    pre_policy, pre_critic, _ = _load(root, PRE, champion)
    post_policy, post_critic, _ = _load(root, POST, champion)
    frozen = (_weight_digest(champion), _weight_digest(robber),
              _module_digest(old_policy), _module_digest(pre_policy),
              _module_digest(post_policy))
    variants = {"safety": (pre_policy, pre_critic, 0., ROUTE),
                "city_post_3b10": (old_policy, old_critic, EVAL_DELEGATION, ROUTE),
                "adapter_pre": (pre_policy, pre_critic, EVAL_DELEGATION, ROUTE),
                "adapter_post": (post_policy, post_critic, EVAL_DELEGATION, ROUTE)}
    parity = 0
    for index in range(start, stop):
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
            print(f"3B11 evaluation {profile} {index+1}/{EVAL_PER_PROFILE}", flush=True)
    if frozen != (_weight_digest(champion), _weight_digest(robber),
                  _module_digest(old_policy), _module_digest(pre_policy),
                  _module_digest(post_policy)):
        raise AssertionError("Evaluation changed frozen weights")
    if start == 0 and stop == EVAL_PER_PROFILE:
        return finalize_profile(root, profile)
    return {"profile": profile, "start": start, "stop": stop,
            "zero_percent_parity_checked": parity, "weights_unchanged": True}


def finalize_profile(root: Path, profile: str):
    if profile not in EVAL_BASES:
        raise ValueError("Unknown profile")
    variants = ("safety", "city_post_3b10", "adapter_pre", "adapter_post")
    for index in range(EVAL_PER_PROFILE):
        seed = EVAL_BASES[profile] + index
        for name in variants:
            if not (root / "evaluation_games" /
                    f"{name}_{profile}_{seed}.json.gz").is_file():
                raise ValueError("Incomplete held-out profile shard")
    marker = {"profile": profile, "games": EVAL_PER_PROFILE,
              "zero_percent_parity": EVAL_PER_PROFILE, "weights_unchanged": True}
    _write(root / f"evaluation_shard_{profile}.json", marker)
    return marker


def summarize(root: Path):
    names = ("safety", "city_post_3b10", "adapter_pre", "adapter_post")
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
              "adapter_pre_to_post": _paired_profiles(rows["adapter_pre"], rows["adapter_post"]),
              "safety_to_adapter_post": _paired_profiles(rows["safety"], rows["adapter_post"]),
              "city_post_3b10_to_adapter_post": _paired_profiles(
                  rows["city_post_3b10"], rows["adapter_post"]),
              "identical_adapter_pre_post_action_traces": sum(
                  a["action_trace"] == b["action_trace"] for a, b in zip(
                      rows["adapter_pre"], rows["adapter_post"], strict=True)),
              "zero_percent_parity": sum(_read(root / f"evaluation_shard_{profile}.json")[
                  "zero_percent_parity"] for profile in EVAL_BASES),
              "hard_safety": "PASS", "profile_seat_balanced": True}
    _write(root / "evaluation_summary.json", result)
    return result


def diagnose(root: Path):
    champion, _ = _agents()
    parent, _, _ = load_parent(PARENT_ROOT, PARENT, champion)
    pre_policy, _, _ = _load(root, PRE, champion)
    post_policy, _, _ = _load(root, POST, champion)
    city_decisions = []
    for profile, base in EVAL_BASES.items():
        for index in range(EVAL_PER_PROFILE):
            episode = _gzip_read(root / "evaluation_games" /
                                 f"adapter_pre_{profile}_{base+index}.json.gz")
            city_decisions.extend(episode["decisions"])
    if not city_decisions:
        raise AssertionError("No fixed held-out City decisions")
    city_batch = _batch(city_decisions)
    investment_batch = _batch(_gzip_read(INVESTMENT_FIXED))

    def probe(batch, *, city: bool):
        args = (batch["latent"], batch["runtime"], batch["global"],
                batch["candidates"], batch["plan_mask"], batch["facts"])
        mask = batch["family_mask"].numpy()
        with torch.no_grad():
            pre_plan, _, pre_action = pre_policy(*args)
            post_plan, _, post_action = post_policy(*args)
            if not city:
                reference = parent(*args)[-1]
                if (not torch.equal(pre_action.probabilities, reference.probabilities) or
                        not torch.equal(post_action.probabilities, reference.probabilities)):
                    raise AssertionError("Investment changed after City-only update")
            else:
                reference = parent(*args)[-1]
                if not torch.equal(pre_action.probabilities, reference.probabilities):
                    raise AssertionError("Zero City adapter does not match 3B-9 parent")
            valid = post_plan.candidate_mask.bool()
            uniform_weights = valid.to(torch.float32) / valid.sum(-1, keepdim=True).clamp_min(1)
            uniform = post_policy(*args, plan_probabilities=uniform_weights)[-1]
            neutral = post_policy(*args, priorities=torch.full_like(
                post_plan.priorities, .5))[-1]
            both = post_policy(*args, plan_probabilities=uniform_weights,
                               priorities=torch.full_like(post_plan.priorities, .5))[-1]
            twice = valid.sum(-1) >= 2
            first = post_plan.probabilities.clone()
            last = post_plan.probabilities.clone()
            if twice.any():
                first[twice] = 0
                last[twice] = 0
                first_index = valid.to(torch.long).argmax(-1)
                last_index = (valid.to(torch.long) * torch.arange(valid.shape[1])).max(-1).values
                rows = twice.nonzero().squeeze(-1)
                first[rows, first_index[rows]] = 1
                last[rows, last_index[rows]] = 1
            plan_a = post_policy(*args, plan_probabilities=first)[-1]
            plan_b = post_policy(*args, plan_probabilities=last)[-1]
            low_priority = post_policy(*args, priorities=torch.zeros_like(
                post_plan.priorities))[-1]
            high_priority = post_policy(*args, priorities=torch.ones_like(
                post_plan.priorities))[-1]
            plan_stats = _per_plan_stats(post_action, post_plan, batch["family_mask"])
        available = post_plan.plan_available.numpy().astype(bool)
        expected_priorities = (post_plan.probabilities.unsqueeze(-1) *
                               post_plan.priorities).sum(1).numpy()
        entropy_gap = (np.log(valid.sum(-1).numpy()[available]) -
                       post_plan.entropy.numpy()[available])
        return {
            "states": len(mask), "plan_available": int(available.sum()),
            "per_plan_post": plan_stats,
            "pre_to_post": _probe_stats(pre_action.probabilities.numpy(),
                                         post_action.probabilities.numpy(), mask),
            "pre_to_parent": _probe_stats(pre_action.probabilities.numpy(),
                                           reference.probabilities.numpy(), mask),
            "plan_uniform": _probe_stats(post_action.probabilities.numpy(),
                                           uniform.probabilities.numpy(), mask),
            "priority_neutral": _probe_stats(post_action.probabilities.numpy(),
                                               neutral.probabilities.numpy(), mask),
            "both_neutral": _probe_stats(post_action.probabilities.numpy(),
                                          both.probabilities.numpy(), mask),
            "plan_a_vs_b": _probe_stats(plan_a.probabilities.numpy()[twice.numpy()],
                                         plan_b.probabilities.numpy()[twice.numpy()],
                                         mask[twice.numpy()]) if twice.any() else {"n": 0},
            "priority_zero_vs_one": _probe_stats(
                low_priority.probabilities.numpy(), high_priority.probabilities.numpy(), mask),
            "plan_entropy_post": float(post_plan.entropy.mean()),
            "plan_entropy_gap_to_uniform_post": float(entropy_gap.mean()) if len(entropy_gap) else None,
            "priority_post_mean": expected_priorities[available].mean(0).tolist()
                if available.any() else None,
            "priority_post_std": expected_priorities[available].std(0).tolist()
                if available.any() else None,
        }

    result = {"city_fixed": probe(city_batch, city=True),
              "investment_fixed": probe(investment_batch, city=False),
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
    destination = Path("docs/rl_route_isolated_city_step3b11_20261014_evidence")
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
            for name in ("safety", "city_post_3b10", "adapter_pre", "adapter_post"):
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
    parser.add_argument("--phase", choices=("prepare", "collect", "collect-profile", "preflight", "update",
        "eval-profile", "finalize-profile", "summarize", "diagnose", "publish"), required=True)
    parser.add_argument("--profile", choices=tuple(EVAL_BASES))
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=EVAL_PER_PROFILE)
    args = parser.parse_args()
    if args.phase == "collect-profile":
        result = collect_profile(args.output, args.profile, blocks=args.blocks)
    elif args.phase == "preflight":
        result = preflight_phase(args.output)
    elif args.phase == "eval-profile":
        result = evaluate_profile(args.output, args.profile,
                                  start=args.start, stop=args.stop)
    elif args.phase == "finalize-profile":
        result = finalize_profile(args.output, args.profile)
    else:
        result = globals()[args.phase](args.output)
    print(f"3B11 {args.phase}: {json.dumps(result, ensure_ascii=False, default=str)[:1800]}",
          flush=True)


if __name__ == "__main__":
    main()
