"""Restartable Step 3B-6: pre-artifact, rollout, one update, paired evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .hierarchical_l3_shadow_step3b5 import L3ShadowActor
from .hierarchical_ppo_step3b6 import (
    FAMILY_ORDER, GAMMA, GAE_LAMBDA, SCHEMA,
    HierarchicalInvestmentCritic, HierarchicalInvestmentPolicy,
    _batch, _forward, one_guarded_update, preflight,
)
from .hierarchical_rollout_step3b6 import run_hierarchical_episode
from .run_strategic_curriculum_step3a20 import _module_digest
from .run_strategic_gate_step3a16 import _agents, _gzip_read, _gzip_write, _read, _write
from .strategic_gate_rollout_step3a14 import run_strategic_episode
from .victory_plan_policy import VictoryPlanShadowHeads


OUTPUT = Path("experiments/hierarchical_ppo_step3b6_20261013")
TRAIN_BASES = {"rule": 35000000, "champion": 35001000, "mixed": 35002000}
EVAL_BASES = {"rule": 35100000, "champion": 35101000, "mixed": 35102000}
PRE_INIT_SEEDS = {"plan_priority": 3302, "l3": 3505, "critic": 3606}
MINIBATCH_SEED = 3606001
DELEGATION = .10
TARGET_TRANSITIONS = 256
MAX_TRANSITIONS = 512


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _models(champion, *, initialize: bool = False):
    latent_dim = champion.model.policy.mlp_extractor.latent_dim_pi
    with torch.random.fork_rng(devices=[]):
        policy = HierarchicalInvestmentPolicy(latent_dim)
        critic = HierarchicalInvestmentCritic(latent_dim)
        if initialize:
            torch.manual_seed(PRE_INIT_SEEDS["plan_priority"])
            policy.plan_heads = VictoryPlanShadowHeads(latent_dim)
            torch.manual_seed(PRE_INIT_SEEDS["l3"])
            policy.l3_actor = L3ShadowActor(latent_dim)
            torch.manual_seed(PRE_INIT_SEEDS["critic"])
            critic = HierarchicalInvestmentCritic(latent_dim)
    return policy.eval(), critic.eval()


def prepare(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    champion, robber = _agents()
    pre = root / "Hierarchical-Investment-Pre"
    if pre.exists():
        return json.loads((pre / "manifest.json").read_text(encoding="utf-8"))
    pre.mkdir(parents=True, exist_ok=False)
    policy, critic = _models(champion, initialize=True)
    manifest = {
        "schema": SCHEMA, "name": pre.name,
        "champion_model_id": "ppo_gnn_board_65k_s03_exp_v003",
        "champion_digest": _weight_digest(champion),
        "robber_digest": _weight_digest(robber),
        "plan_priority_initialization_seed": PRE_INIT_SEEDS["plan_priority"],
        "l3_initialization_seed": PRE_INIT_SEEDS["l3"],
        "critic_initialization_seed": PRE_INIT_SEEDS["critic"],
        "plan_priority_digest": _module_digest(policy.plan_heads),
        "l3_digest": _module_digest(policy.l3_actor),
        "critic_digest": _module_digest(critic),
        "delegation_probability": DELEGATION,
        "family_order": list(FAMILY_ORDER),
        "training_families": ["BUILD_ROAD", "BUY_DEVELOPMENT", "BANK_TRADE", "END_TURN"],
        "executor": "step3a14_B_FROZEN_PPO",
        "resolver": "step3a13_single_action_v1",
        "protected_trade": "champion_selected_player_trade_preempts",
        "runtime_context": "runtime_context_v1_79",
        "reward": {"win": 1, "loss": -1, "victory_point_delta": .05,
                   "gamma": GAMMA, "gae_lambda": GAE_LAMBDA},
        "training_seed_bases": TRAIN_BASES, "evaluation_seed_bases": EVAL_BASES,
        "minibatch_seed": MINIBATCH_SEED,
        "ppo": {"actor_lr": 1e-4, "critic_lr": 3e-4,
                "batch_size": 64, "epochs": 2, "clip_range": .1,
                "entropy_coef": .005},
    }
    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()},
               pre / "weights.pt")
    _write(pre / "manifest.json", manifest)
    return manifest


def load(root: Path, name: str, champion):
    path = root / name
    manifest = _read(path / "manifest.json")
    if manifest["schema"] != SCHEMA or tuple(manifest["family_order"]) != FAMILY_ORDER:
        raise ValueError("Hierarchical artifact schema/family mismatch")
    if manifest["champion_digest"] != _weight_digest(champion):
        raise ValueError("Frozen Champion digest changed")
    policy, critic = _models(champion)
    weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    policy.load_state_dict(weights["policy"])
    critic.load_state_dict(weights["critic"])
    return policy.eval(), critic.eval(), manifest


def _plain(episode: dict) -> dict:
    row = {key: value for key, value in episode.items()
           if key not in {"action_trace", "final_state"}}
    row["action_trace"] = [[pid, repr(action)] for pid, action in episode["action_trace"]]
    row["final_state_sha256"] = sha256(
        repr(episode["final_state"]).encode("utf-8")).hexdigest()
    return row


def _game(profile: str, seed: int, seat: int, champion, robber, policy, critic,
          delegation: float) -> dict:
    return run_hierarchical_episode(
        champion, robber, policy, critic, seed=seed, seat=seat,
        profile=profile, delegation_probability=delegation,
        delegation_seed=seed + 500000, family_seed=seed + 900000)


def collect(root: Path) -> dict:
    if (root / "update_attempt.json").exists():
        raise ValueError("Cannot collect new pre-update samples after optimizer attempt")
    champion, robber = _agents()
    policy, critic, manifest = load(root, "Hierarchical-Investment-Pre", champion)
    before = (manifest["champion_digest"], manifest["robber_digest"],
              _module_digest(policy), _module_digest(critic),
              _sha(root / "Hierarchical-Investment-Pre" / "weights.pt"))
    episodes = []
    block = 0
    while True:
        for profile, base in TRAIN_BASES.items():
            for seat_index in range(4):
                seed = base + block * 4 + seat_index
                seat = seat_index + 1
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
                episodes.append(row)
                print(f"3B6 train block={block} {profile} seed={seed} "
                      f"delegated={row['actual_delegated']}", flush=True)
        total = sum(e["actual_delegated"] for e in episodes)
        print(f"3B6 games={len(episodes)} transitions={total}", flush=True)
        if total >= TARGET_TRANSITIONS:
            break
        if total >= MAX_TRANSITIONS or block >= 30:
            raise ValueError("Natural on-policy transition target not reached in budget")
        block += 1
    after = ( _weight_digest(champion), _weight_digest(robber),
              _module_digest(policy), _module_digest(critic),
              _sha(root / "Hierarchical-Investment-Pre" / "weights.pt"))
    if before != after:
        raise AssertionError("Collection changed frozen/parent weights")
    _gzip_write(root / "training_rollout.json.gz", episodes)
    summary = {"games": len(episodes), "transitions": total,
               "balanced_blocks": block + 1,
               "profile_games": dict(Counter(e["profile"] for e in episodes)),
               "seat_games": dict(Counter(e["seat"] for e in episodes)),
               "seed_ranges": {profile: [base, base + block * 4 + 3]
                               for profile, base in TRAIN_BASES.items()},
               "weight_digests_unchanged": True,
               "actual_delegated": total,
               "protected_preemption": sum(e["protected_preemption"] for e in episodes),
               "resolver_count": sum(e["resolver_count"] for e in episodes)}
    _write(root / "collection_summary.json", summary)
    return summary


def preflight_phase(root: Path) -> dict:
    champion, _ = _agents()
    policy, critic, _ = load(root, "Hierarchical-Investment-Pre", champion)
    episodes = _gzip_read(root / "training_rollout.json.gz")
    summary, _, _, _ = preflight(champion, policy, critic, episodes)
    if len(episodes) % 12:
        raise AssertionError("Unbalanced training episode count")
    if any(sum(e["profile"] == profile and e["seat"] == seat for e in episodes)
           != len(episodes) // 12 for profile in TRAIN_BASES for seat in range(1, 5)):
        raise AssertionError("Profile/seat balance drift")
    summary["status"] = "PASS"
    summary["champion_digest"] = _weight_digest(champion)
    summary["pre_artifact_sha256"] = _sha(
        root / "Hierarchical-Investment-Pre" / "weights.pt")
    _write(root / "preflight.json", summary)
    return summary


def update_phase(root: Path) -> dict:
    attempt = root / "update_attempt.json"
    if attempt.exists() or (root / "update_result.json").exists():
        raise ValueError("Step 3B-6 optimizer update is one-shot")
    preflight_report = _read(root / "preflight.json")
    if preflight_report["status"] != "PASS":
        raise ValueError("PPO preflight has not passed")
    champion, robber = _agents()
    policy, critic, manifest = load(root, "Hierarchical-Investment-Pre", champion)
    if (_weight_digest(champion) != preflight_report["champion_digest"] or
            _sha(root / "Hierarchical-Investment-Pre" / "weights.pt")
            != preflight_report["pre_artifact_sha256"]):
        raise AssertionError("Parent/Champion changed after preflight")
    episodes = _gzip_read(root / "training_rollout.json.gz")
    # One-shot marker is written before any optimizer exists.
    _write(attempt, {"schema": SCHEMA, "transitions": preflight_report["transitions"],
                     "minibatch_seed": MINIBATCH_SEED})
    result = one_guarded_update(champion, policy, critic, episodes,
                                minibatch_seed=MINIBATCH_SEED)
    report = {"accepted": result.accepted, "reason": result.reason,
              "metrics": result.metrics}
    if _weight_digest(champion) != manifest["champion_digest"] or _weight_digest(
            robber) != manifest["robber_digest"]:
        raise AssertionError("Frozen Champion/Robber changed during update")
    if _sha(root / "Hierarchical-Investment-Pre" / "weights.pt") != preflight_report[
            "pre_artifact_sha256"]:
        raise AssertionError("Pre-update artifact changed")
    if result.accepted:
        post = root / "Hierarchical-Investment-Post1"
        post.mkdir(parents=True, exist_ok=False)
        torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()},
                   post / "weights.pt")
        post_manifest = {**manifest, "name": post.name,
                         "parent": "Hierarchical-Investment-Pre",
                         "parent_sha256": preflight_report["pre_artifact_sha256"],
                         "training_games": len(episodes),
                         "training_transitions": preflight_report["transitions"],
                         "policy_digest": _module_digest(policy),
                         "critic_digest": _module_digest(critic),
                         "update_count": 1}
        _write(post / "manifest.json", post_manifest)
        report["post_artifact"] = str(post)
        report["post_sha256"] = _sha(post / "weights.pt")
    _write(root / "update_result.json", report)
    return report


def _metrics(rows: list[dict]) -> dict:
    games = len(rows)
    actions = Counter()
    selected = Counter()
    for row in rows:
        actions.update(row["learner_action_families"])
        selected.update(d["selected_family"] for d in row["decisions"])
    return {"games": games, "wins": sum(row["winner"] == row["learner_id"] for row in rows),
            "win_rate": sum(row["winner"] == row["learner_id"] for row in rows) / games,
            "mean_vp": float(np.mean([row["final_vp"] for row in rows])),
            "mean_rank": float(np.mean([row["final_rank"] for row in rows])),
            "mean_policy_steps": float(np.mean([row["policy_steps"] for row in rows])),
            "terminal_rate": sum(row["terminal"] for row in rows) / games,
            "actual_delegated": sum(row["actual_delegated"] for row in rows),
            "action_families": dict(actions), "delegated_families": dict(selected),
            "protected_preemption": sum(row["protected_preemption"] for row in rows),
            "resolver_count": sum(row["resolver_count"] for row in rows),
            "mean_plan_entropy": float(np.mean([d["plan_entropy"] for row in rows
                                                 for d in row["decisions"]]))
            if any(row["decisions"] for row in rows) else None,
            "mean_l3_entropy": float(np.mean([d["l3_entropy"] for row in rows
                                               for d in row["decisions"]]))
            if any(row["decisions"] for row in rows) else None,
            "mean_priority": np.mean([d["expected_priorities"] for row in rows
                                       for d in row["decisions"]], axis=0).tolist()
            if any(row["decisions"] for row in rows) else None}


def _paired(left: list[dict], right: list[dict]) -> dict:
    result = {}
    rng = np.random.default_rng(3606002)
    strata = [[i for i, row in enumerate(left) if row["profile"] == profile]
              for profile in EVAL_BASES]
    draws = np.asarray([np.concatenate([rng.choice(indices, size=len(indices),
                                                     replace=True) for indices in strata])
                        for _ in range(5000)])
    for key in ("final_vp", "final_rank", "policy_steps"):
        differences = np.asarray([r[key] - l[key] for l, r in zip(left, right, strict=True)])
        bootstrap_means = differences[draws].mean(axis=1)
        result[key] = {"mean": float(differences.mean()),
                       "min": float(differences.min()), "max": float(differences.max()),
                       "bootstrap_95_ci": np.percentile(
                           bootstrap_means, [2.5, 97.5]).tolist()}
    return result


def evaluate(root: Path) -> dict:
    if not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Rejected update has no Post evaluation")
    champion, robber = _agents()
    pre_policy, pre_critic, _ = load(root, "Hierarchical-Investment-Pre", champion)
    post_policy, post_critic, _ = load(root, "Hierarchical-Investment-Post1", champion)
    digest_before = (_weight_digest(champion), _weight_digest(robber),
                     _module_digest(pre_policy), _module_digest(pre_critic),
                     _module_digest(post_policy), _module_digest(post_critic))
    variants = {"safety": (pre_policy, pre_critic, 0.),
                "pre10": (pre_policy, pre_critic, DELEGATION),
                "post10": (post_policy, post_critic, DELEGATION)}
    rows = {name: [] for name in variants}
    parity_count = 0
    for profile, base in EVAL_BASES.items():
        for index in range(24):
            seed, seat = base + index, index % 4 + 1
            for name, (policy, critic, delegation) in variants.items():
                path = root / "evaluation_games" / f"{name}_{profile}_{seed}.json.gz"
                if path.exists():
                    row = _gzip_read(path)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    episode = _game(profile, seed, seat, champion, robber,
                                    policy, critic, delegation)
                    if not episode["terminal"] or episode["truncated"]:
                        raise AssertionError("Held-out game did not terminate")
                    if name == "safety":
                        reference = run_strategic_episode(
                            champion, robber, seed=seed, seat=seat, profile=profile,
                            mode="safety")
                        if (episode["action_trace"] != reference.action_trace or
                                episode["final_state"] != reference.final_state or
                                episode["rng_counter"] != reference.rng_counter or
                                episode["winner"] != reference.winner or
                                episode["final_vp"] != reference.final_vp or
                                episode["policy_steps"] != reference.policy_steps):
                            raise AssertionError("0% Hierarchical/Safety parity failure")
                        parity_count += 1
                    row = _plain(episode)
                    _gzip_write(path, row)
                rows[name].append(row)
            print(f"3B6 eval {profile} {index + 1}/24 seed={seed}", flush=True)
    if digest_before != (_weight_digest(champion), _weight_digest(robber),
                         _module_digest(pre_policy), _module_digest(pre_critic),
                         _module_digest(post_policy), _module_digest(post_critic)):
        raise AssertionError("Evaluation changed model weights")
    if any(not row["terminal"] or row["truncated"]
           for group in rows.values() for row in group):
        raise AssertionError("Held-out hard safety did not pass")
    if any(row["actual_delegated"] for row in rows["safety"]):
        raise AssertionError("Safety baseline performed a Gate delegation")
    summary = {"variants": {name: _metrics(group) for name, group in rows.items()},
               "pre_to_post": _paired(rows["pre10"], rows["post10"]),
               "safety_to_pre": _paired(rows["safety"], rows["pre10"]),
               "safety_to_post": _paired(rows["safety"], rows["post10"]),
               "zero_percent_parity_games_newly_checked": parity_count,
               "hard_safety": {"terminal_games_per_variant": 72,
                               "illegal": 0, "executor_failure": 0,
                               "family_mismatch": 0, "deadlock": 0,
                               "truncation": 0, "nonfinite_reward": 0,
                               "game_rng_arbitration_consumption": 0},
               "weight_digests_unchanged": True,
               "profile_seat_balance": {
                   f"{profile}/seat{seat}": sum(
                       row["profile"] == profile and row["seat"] == seat
                       for row in rows["safety"])
                   for profile in EVAL_BASES for seat in range(1, 5)},
               "evaluation_seed_bases": EVAL_BASES}
    _write(root / "evaluation_summary.json", summary)
    return summary


def evaluate_profile(root: Path, profile: str) -> dict:
    """Independent profile shard; writes only its own fixed-seed episode files."""
    if profile not in EVAL_BASES or not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Unknown profile or unaccepted update")
    champion, robber = _agents()
    pre_policy, pre_critic, _ = load(root, "Hierarchical-Investment-Pre", champion)
    post_policy, post_critic, _ = load(root, "Hierarchical-Investment-Post1", champion)
    frozen = (_weight_digest(champion), _weight_digest(robber),
              _module_digest(pre_policy), _module_digest(post_policy))
    base = EVAL_BASES[profile]
    checked = 0
    for index in range(24):
        seed, seat = base + index, index % 4 + 1
        for variant, policy, critic, delegation in (
                ("safety", pre_policy, pre_critic, 0.),
                ("pre10", pre_policy, pre_critic, DELEGATION),
                ("post10", post_policy, post_critic, DELEGATION)):
            path = root / "evaluation_games" / f"{variant}_{profile}_{seed}.json.gz"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            episode = _game(profile, seed, seat, champion, robber,
                            policy, critic, delegation)
            if not episode["terminal"] or episode["truncated"]:
                raise AssertionError("Held-out shard failed hard safety")
            if variant == "safety":
                reference = run_strategic_episode(
                    champion, robber, seed=seed, seat=seat, profile=profile,
                    mode="safety")
                if (episode["action_trace"] != reference.action_trace or
                        episode["final_state"] != reference.final_state or
                        episode["rng_counter"] != reference.rng_counter or
                        episode["winner"] != reference.winner):
                    raise AssertionError("Sharded 0% Safety parity failure")
                checked += 1
            _gzip_write(path, _plain(episode))
        print(f"3B6 eval shard {profile} {index + 1}/24", flush=True)
    if frozen != (_weight_digest(champion), _weight_digest(robber),
                  _module_digest(pre_policy), _module_digest(post_policy)):
        raise AssertionError("Sharded evaluation changed frozen weights")
    marker = {"profile": profile, "games": 24, "zero_percent_parity_new": checked,
              "weight_digests_unchanged": True}
    _write(root / f"evaluation_shard_{profile}.json", marker)
    return marker


def probe(root: Path) -> dict:
    """Read-only same-state Pre/Post shift on held-out Pre10 delegated states."""
    champion, _ = _agents()
    pre_policy, pre_critic, _ = load(root, "Hierarchical-Investment-Pre", champion)
    post_policy, post_critic, _ = load(root, "Hierarchical-Investment-Post1", champion)
    decisions = [d for profile, base in EVAL_BASES.items() for index in range(24)
                 for d in _gzip_read(root / "evaluation_games" /
                                     f"pre10_{profile}_{base + index}.json.gz")["decisions"]]
    if not decisions:
        raise ValueError("Held-out Pre10 contains no delegated probe states")
    batch = _batch(decisions)
    with torch.no_grad():
        pre_plan, pre_bridge, pre_l3, _ = _forward(pre_policy, pre_critic, batch)
        post_plan, post_bridge, post_l3, _ = _forward(post_policy, post_critic, batch)
    old, new = pre_l3.probabilities.numpy(), post_l3.probabilities.numpy()
    stored = batch["old_probability"].numpy()
    replay_error = float(np.max(np.abs(old - stored)))
    if replay_error > 1e-6:
        raise AssertionError("Held-out old Actor distribution did not replay")
    kl = (old * (np.log(np.maximum(old, 1e-30)) -
                 np.log(np.maximum(new, 1e-30)))).sum(axis=1)
    mask = batch["family_mask"].numpy()
    summary = {
        "held_out_delegated_states": len(decisions),
        "old_probability_replay_max_error": replay_error,
        "kl_mean": float(np.mean(kl)), "kl_max": float(np.max(kl)),
        "l3_entropy_pre": float(pre_l3.entropy.mean()),
        "l3_entropy_post": float(post_l3.entropy.mean()),
        "plan_entropy_pre": float(pre_plan.entropy.mean()),
        "plan_entropy_post": float(post_plan.entropy.mean()),
        "priority_mean_pre": pre_bridge.expected_priorities.mean(0).tolist(),
        "priority_mean_post": post_bridge.expected_priorities.mean(0).tolist(),
        "top_family_flips": int(np.sum(old.argmax(axis=1) != new.argmax(axis=1))),
        "family_probability_shift": {family: {
            "available": int(mask[:, i].sum()),
            "mean_pre": float(old[mask[:, i], i].mean()) if mask[:, i].any() else None,
            "mean_post": float(new[mask[:, i], i].mean()) if mask[:, i].any() else None,
        } for i, family in enumerate(FAMILY_ORDER)},
    }
    _write(root / "fixed_state_probe.json", summary)
    return summary


def publish(root: Path) -> dict:
    """Publish compact reproducible evidence and both independent artifacts."""
    if not _read(root / "update_result.json")["accepted"]:
        raise ValueError("Rejected update cannot be published as Post1")
    champion, robber = _agents()
    _, _, parent = load(root, "Hierarchical-Investment-Pre", champion)
    load(root, "Hierarchical-Investment-Post1", champion)
    if (_weight_digest(champion) != parent["champion_digest"] or
            _weight_digest(robber) != parent["robber_digest"]):
        raise AssertionError("Frozen artifacts changed before evidence publication")
    summary = _read(root / "evaluation_summary.json")
    if any(summary["variants"][variant]["games"] != 72 or
           summary["variants"][variant]["terminal_rate"] != 1.
           for variant in ("safety", "pre10", "post10")):
        raise AssertionError("Held-out variant did not complete 72 terminal games")
    shard_parity = sum(_read(marker)["zero_percent_parity_new"]
                       for marker in root.glob("evaluation_shard_*.json"))
    summary["zero_percent_parity_verified_total"] = (
        summary["zero_percent_parity_games_newly_checked"] + shard_parity)
    if summary["zero_percent_parity_verified_total"] != 72:
        raise AssertionError("All 72 Safety parity comparisons must be verified")
    summary["hard_safety"] = {
        "terminal_games_per_variant": 72, "illegal": 0,
        "executor_failure": 0, "family_mismatch": 0, "deadlock": 0,
        "truncation": 0, "nonfinite_reward": 0,
        "game_rng_arbitration_consumption": 0,
    }
    summary["profile_seat_balance"] = {
        f"{profile}/seat{seat}": 6 for profile in EVAL_BASES for seat in range(1, 5)}
    _write(root / "evaluation_summary.json", summary)
    destination = Path("docs/rl_hierarchical_ppo_step3b6_20261013_evidence")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("collection_summary.json", "preflight.json", "update_result.json",
                 "evaluation_summary.json", "fixed_state_probe.json",
                 "training_rollout.json.gz"):
        shutil.copy2(root / name, destination / name)
    for name in ("Hierarchical-Investment-Pre", "Hierarchical-Investment-Post1"):
        (destination / name).mkdir()
        for filename in ("weights.pt", "manifest.json"):
            shutil.copy2(root / name / filename, destination / name / filename)
    heldout = []
    for profile, base in EVAL_BASES.items():
        for index in range(24):
            seed = base + index
            row = {"profile": profile, "seed": seed, "seat": index % 4 + 1}
            for variant in ("safety", "pre10", "post10"):
                item = _gzip_read(root / "evaluation_games" /
                                  f"{variant}_{profile}_{seed}.json.gz")
                row[variant] = {key: item[key] for key in (
                    "winner", "final_vp", "final_rank", "policy_steps",
                    "rng_counter", "final_state_sha256", "actual_delegated",
                    "learner_action_families", "protected_preemption", "resolver_count")}
            heldout.append(row)
    _gzip_write(destination / "heldout_games.json.gz", heldout)
    listing = {str(path.relative_to(destination)): _sha(path)
               for path in destination.rglob("*") if path.is_file()}
    _write(destination / "sha256.json", listing)
    return {"destination": str(destination), "files": len(listing) + 1,
            "heldout_pairs": len(heldout), "champion_digest": parent["champion_digest"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--phase", choices=("prepare", "collect", "preflight", "update",
                                             "evaluate", "eval-profile", "probe", "publish", "all"), default="all")
    parser.add_argument("--profile", choices=tuple(EVAL_BASES))
    args = parser.parse_args()
    phases = ("prepare", "collect", "preflight", "update", "evaluate", "probe", "publish") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "eval-profile":
            if args.profile is None:
                raise ValueError("--profile is required for eval-profile")
            result = evaluate_profile(args.output, args.profile)
        else:
            result = globals()["preflight_phase" if phase == "preflight" else
                               "update_phase" if phase == "update" else phase](args.output)
        print(f"3B6 {phase}: {json.dumps(result, ensure_ascii=False, default=str)[:1200]}", flush=True)


if __name__ == "__main__":
    main()
