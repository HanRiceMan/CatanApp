"""Restartable, two-round, route-isolated on-policy City experiment.

Every round has a distinct seed range and a one-shot update marker. The final
held-out seeds are fixed here, never selected using observed outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json
import shutil

import numpy as np
import torch

from .city_value_ppo_step3b10 import CITY_FAMILIES
from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .hierarchical_ppo_step3b6 import _batch, one_guarded_update, preflight
from .hierarchical_rollout_step3b6 import run_hierarchical_episode
from .hierarchical_shadow_controller_step3b4 import HierarchicalRoute
from .run_city_value_ppo_step3b10 import _city_metrics
from .run_hierarchical_ppo_step3b6 import _metrics, _plain, _sha
from .run_hierarchical_ppo_step3b7 import _paired_profiles
from .run_route_isolated_city_step3b11 import (
    POST as PARENT_NAME, _load as load_parent, _investment_parity,
)
from .run_strategic_curriculum_step3a20 import _module_digest
from .run_strategic_gate_step3a16 import _agents, _gzip_read, _gzip_write, _read, _write
from .strategic_gate_rollout_step3a14 import run_strategic_episode


ROOT = Path("experiments/multi_round_city_step3b12_20261015")
EVIDENCE = Path("docs/rl_multi_round_city_step3b12_20261015_evidence")
PARENT_ROOT = Path("docs/rl_route_isolated_city_step3b11_20261014_evidence")
ROUTE = HierarchicalRoute.CITY_VALUE_COMPARISON
NAMES = ("Route-Isolated-City-Post1", "Multi-Round-City-Round1", "Multi-Round-City-Round2")
TRAIN_BASES = {
    1: {"rule": 36200000, "champion": 36201000, "mixed": 36202000},
    2: {"rule": 36300000, "champion": 36301000, "mixed": 36302000},
}
EVAL_BASES = {"rule": 36400000, "champion": 36401000, "mixed": 36402000}
VALIDATION_BASES = {"rule": 36500000, "champion": 36501000, "mixed": 36502000}
TRAIN_RATE = .50
EVAL_RATE = .10
TARGET = 256
MAX_TRANSITIONS = 384
MAX_BLOCKS_PER_ROUND = 50  # 600 games/round, 1200 training games total.
EVAL_PER_PROFILE = 96
VALIDATION_PER_PROFILE = 8
MINIBATCH_SEEDS = {1: 3912001, 2: 3912002}


def _load_version(root: Path, version: int, champion):
    if version == 0:
        return load_parent(PARENT_ROOT, PARENT_NAME, champion)
    folder = root / NAMES[version]
    manifest = _read(folder / "manifest.json")
    if manifest["champion_digest"] != _weight_digest(champion):
        raise AssertionError("Champion digest drift")
    policy, critic, _ = load_parent(PARENT_ROOT, PARENT_NAME, champion)
    weights = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    policy.load_state_dict(weights["policy"])
    critic.load_state_dict(weights["critic"])
    return policy.eval(), critic.eval(), manifest


def _game(profile, seed, seat, champion, robber, policy, critic, rate):
    return _plain(run_hierarchical_episode(
        champion, robber, policy, critic, seed=seed, seat=seat, profile=profile,
        delegation_probability=rate, delegation_seed=seed + 500000,
        family_seed=seed + 900000, multi_choice_only=True, delegated_route=ROUTE))


def prepare(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    champion, robber = _agents()
    policy, critic, parent = _load_version(root, 0, champion)
    frozen = {"champion": _weight_digest(champion), "robber": _weight_digest(robber),
              "plan_heads": _module_digest(policy.plan_heads),
              "base_actor": _module_digest(policy.l3_actor),
              "parent_sha256": _sha(PARENT_ROOT / PARENT_NAME / "weights.pt")}
    manifest = {"step": "3B-12", "parent": PARENT_NAME,
                "training_bases": {str(k): v for k, v in TRAIN_BASES.items()},
                "validation_bases": VALIDATION_BASES, "evaluation_bases": EVAL_BASES,
                "round_target_transitions": TARGET, "round_max_transitions": MAX_TRANSITIONS,
                "round_max_games": 12 * MAX_BLOCKS_PER_ROUND,
                "training_delegation": TRAIN_RATE, "evaluation_delegation": EVAL_RATE,
                "investment_delegation": 0., "route": ROUTE.value,
                "frozen": frozen, "parent_policy_digest": _module_digest(policy),
                "parent_critic_digest": _module_digest(critic),
                "optimizer": {"actor_lr": 1e-4, "critic_lr": 3e-4,
                              "batch_size": 64, "epochs": 2, "clip_range": .1,
                              "entropy_coef": .005},
                "reward": {"win": 1., "loss": -1., "vp_delta": .05,
                           "gamma": .99, "gae_lambda": .95},
                "minibatch_seeds": {str(k): v for k, v in MINIBATCH_SEEDS.items()},
                "family_order": list(CITY_FAMILIES),
                "executor": "step3a14_B_FROZEN_PPO",
                "player_trade": "protected_champion_preemption"}
    path = root / "experiment_manifest.json"
    if path.exists():
        if _read(path) != manifest:
            raise AssertionError("Fixed experiment manifest changed")
    else:
        _write(path, manifest)
    return {"manifest": str(path), "investment_parity": _investment_parity(champion, policy),
            "frozen": frozen}


def collect(root: Path, round_id: int):
    if round_id not in (1, 2):
        raise ValueError("Invalid round")
    if (root / f"round{round_id}_update_attempt.json").exists():
        raise ValueError("Round collection closed by optimizer attempt")
    if round_id == 2 and not _read(root / "round1_update_result.json")["accepted"]:
        raise ValueError("Round 1 update not accepted")
    champion, robber = _agents()
    policy, critic, _ = _load_version(root, round_id - 1, champion)
    before = (_weight_digest(champion), _weight_digest(robber),
              _module_digest(policy), _module_digest(critic))
    games = []
    bases = TRAIN_BASES[round_id]
    for block in range(MAX_BLOCKS_PER_ROUND):
        for profile, base in bases.items():
            for seat_index in range(4):
                seed, seat = base + block * 4 + seat_index, seat_index + 1
                path = root / f"round{round_id}_games" / f"{profile}_{seed}.json.gz"
                if path.exists():
                    row = _gzip_read(path)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    row = _game(profile, seed, seat, champion, robber, policy, critic,
                                TRAIN_RATE)
                    if not row["terminal"] or row["truncated"]:
                        raise AssertionError("Training episode not terminal")
                    _gzip_write(path, row)
                if any(d["route"] != ROUTE.value or sum(d["family_mask"]) < 2
                       for d in row["decisions"]):
                    raise AssertionError("Non-City or single-choice Actor sample")
                games.append(row)
        transitions = sum(item["actual_delegated"] for item in games)
        print(f"3B12 round={round_id} block={block+1} games={len(games)} "
              f"transitions={transitions}", flush=True)
        if transitions >= TARGET:
            break
    if before != (_weight_digest(champion), _weight_digest(robber),
                  _module_digest(policy), _module_digest(critic)):
        raise AssertionError("Collection mutated model")
    transitions = sum(item["actual_delegated"] for item in games)
    summary = {"round": round_id, "games": len(games), "transitions": transitions,
               "target_reached": transitions >= TARGET,
               "within_normal_upper_bound": transitions <= MAX_TRANSITIONS,
               "profile_games": dict(Counter(g["profile"] for g in games)),
               "seat_games": dict(Counter(g["seat"] for g in games)),
               "selected_families": dict(Counter(d["selected_family"] for g in games
                                               for d in g["decisions"])),
               "protected_preemptions": sum(g["protected_preemption"] for g in games),
               "parent_version": NAMES[round_id - 1]}
    _write(root / f"round{round_id}_collection.json", summary)
    if transitions >= TARGET:
        _gzip_write(root / f"round{round_id}_rollout.json.gz", games)
    return summary


def collect_profile(root: Path, round_id: int, profile: str, blocks: int,
                    start_block: int = 0):
    """Independent profile shard; balanced-block collector chooses the stop."""
    if round_id not in (1, 2) or profile not in TRAIN_BASES[round_id]:
        raise ValueError("Invalid round/profile")
    if not 0 <= start_block < blocks <= MAX_BLOCKS_PER_ROUND:
        raise ValueError("Outside fixed training game cap")
    if (root / f"round{round_id}_update_attempt.json").exists():
        raise ValueError("Round already closed")
    if round_id == 2 and not _read(root / "round1_update_result.json")["accepted"]:
        raise ValueError("Round 1 must be accepted")
    champion, robber = _agents()
    policy, critic, _ = _load_version(root, round_id - 1, champion)
    before = (_weight_digest(champion), _weight_digest(robber),
              _module_digest(policy), _module_digest(critic))
    transitions = 0
    base = TRAIN_BASES[round_id][profile]
    for block in range(start_block, blocks):
        for seat_index in range(4):
            seed, seat = base + block * 4 + seat_index, seat_index + 1
            path = root / f"round{round_id}_games" / f"{profile}_{seed}.json.gz"
            if path.exists():
                row = _gzip_read(path)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                row = _game(profile, seed, seat, champion, robber, policy, critic,
                            TRAIN_RATE)
                if not row["terminal"] or row["truncated"]:
                    raise AssertionError("Shard episode not terminal")
                _gzip_write(path, row)
            transitions += row["actual_delegated"]
        if block % 8 == 7:
            print(f"3B12 shard round={round_id} {profile} blocks={block+1} "
                  f"transitions={transitions}", flush=True)
    if before != (_weight_digest(champion), _weight_digest(robber),
                  _module_digest(policy), _module_digest(critic)):
        raise AssertionError("Shard changed weights")
    return {"round": round_id, "profile": profile,
            "games": 4 * (blocks - start_block),
            "transitions": transitions, "weights_unchanged": True}


def preflight_round(root: Path, round_id: int):
    summary = _read(root / f"round{round_id}_collection.json")
    if not summary["target_reached"] or not summary["within_normal_upper_bound"]:
        raise ValueError("On-policy City transitions outside precommitted 256–384 range")
    champion, _ = _agents()
    policy, critic, _ = _load_version(root, round_id - 1, champion)
    episodes = _gzip_read(root / f"round{round_id}_rollout.json.gz")
    report, _, advantages, returns = preflight(
        champion, policy, critic, episodes, minimum_transitions=TARGET,
        route=ROUTE, allowed_families=CITY_FAMILIES,
        delegation_probability=TRAIN_RATE)
    choices = np.asarray([d["selected_family"] for e in episodes for d in e["decisions"]])
    report["build_city_advantage"] = _stats(advantages[choices == "BUILD_CITY"])
    report["deferred_city_advantage"] = _stats(advantages[choices != "BUILD_CITY"])
    report["return_distribution"] = _stats(returns)
    report["status"] = "PASS"
    report["rollout_sha256"] = _sha(root / f"round{round_id}_rollout.json.gz")
    report["parent_sha256"] = (_sha(PARENT_ROOT / PARENT_NAME / "weights.pt")
                                if round_id == 1 else
                                _sha(root / NAMES[1] / "weights.pt"))
    _write(root / f"round{round_id}_preflight.json", report)
    return report


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p5": float(np.percentile(values, 5)) if len(values) else None,
            "p95": float(np.percentile(values, 95)) if len(values) else None,
            "mae": float(np.abs(values).mean()) if len(values) else None}


def _fixed_city_states(root: Path):
    episodes = _gzip_read(root / "validation_rollout.json.gz")
    return [d for e in episodes for d in e["decisions"]]


def validation(root: Path):
    """Fixed independent states and realized returns, never used for optimizer choice."""
    champion, robber = _agents()
    policy, critic, _ = _load_version(root, 0, champion)
    episodes = []
    for profile, base in VALIDATION_BASES.items():
        for index in range(VALIDATION_PER_PROFILE):
            seed = base + index
            episode = _game(profile, seed, index % 4 + 1, champion, robber,
                            policy, critic, TRAIN_RATE)
            if not episode["terminal"] or episode["truncated"]:
                raise AssertionError("Validation episode did not terminate")
            episodes.append(episode)
    if not any(e["decisions"] for e in episodes):
        raise AssertionError("No held-out City states")
    _gzip_write(root / "validation_rollout.json.gz", episodes)
    return {"games": len(episodes), "fixed_city_states": sum(len(e["decisions"])
                                                                for e in episodes),
            "sha256": _sha(root / "validation_rollout.json.gz")}


def fixed_diagnosis(root: Path, version: int):
    from .hierarchical_ppo_step3b6 import _forward
    champion, _ = _agents()
    policy, critic, _ = _load_version(root, version, champion)
    decisions = _fixed_city_states(root)
    batch = _batch(decisions)
    with torch.no_grad():
        plan, _, action, value = _forward(policy, critic, batch)
    probabilities = action.probabilities.numpy()
    available = batch["family_mask"].numpy()
    family = {name: {"mean_available_probability": float(
        probabilities[available[:, index], index].mean())
        if available[:, index].any() else None,
        "top_count": int((probabilities.argmax(axis=1) == index).sum())}
        for index, name in enumerate(("BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD",
                                      "BUY_DEVELOPMENT", "BANK_TRADE", "PLAYER_TRADE",
                                      "END_TURN"))}
    episodes = _gzip_read(root / "validation_rollout.json.gz")
    targets = []
    for episode in episodes:
        targets.extend(_realized_returns(episode["transitions"]))
    targets = np.asarray(targets, np.float64)
    predicted = value.numpy().astype(np.float64)
    error = predicted - targets
    order = np.argsort(predicted, kind="stable")
    calibration = []
    for indices in np.array_split(order, 4):
        if len(indices):
            calibration.append({"n": len(indices),
                                "mean_prediction": float(predicted[indices].mean()),
                                "mean_realized_return": float(targets[indices].mean()),
                                "mse": float(np.mean(error[indices] ** 2))})
    return {"version": version, "states": len(decisions),
            "policy": {"family": family, "entropy": _stats(action.entropy.numpy()),
                       "plan_entropy": _stats(plan.entropy.numpy()),
                       "probabilities_sha256": _tensor_sha(action.probabilities)},
            "critic": {"mse": float(np.mean(error ** 2)),
                       "mae": float(np.mean(np.abs(error))),
                       "mean_bias": float(error.mean()),
                       "prediction": _stats(predicted), "realized_return": _stats(targets),
                       "prediction_quartile_calibration": calibration,
                       "correlation": float(np.corrcoef(predicted, targets)[0, 1])
                       if len(targets) > 1 and predicted.std() > 0 and targets.std() > 0
                       else None}}


def _realized_returns(transitions):
    """Actual discounted macro return to terminal, independent of Critic GAE."""
    carry = 0.
    output = []
    for transition in reversed(transitions):
        if transition["truncated"]:
            raise ValueError("Truncated validation return cannot be treated as terminal")
        carry = transition["reward_accumulated"] + transition["discount"] * carry
        output.append(carry)
    return list(reversed(output))


def _tensor_sha(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def fixed_city_shift(root: Path, left_version: int, right_version: int):
    champion, _ = _agents()
    left, _, _ = _load_version(root, left_version, champion)
    right, _, _ = _load_version(root, right_version, champion)
    batch = _batch(_fixed_city_states(root))
    args = (batch["latent"], batch["runtime"], batch["global"],
            batch["candidates"], batch["plan_mask"], batch["facts"])
    with torch.no_grad():
        before = left(*args)[-1].probabilities.numpy()
        after = right(*args)[-1].probabilities.numpy()
    mask = batch["family_mask"].numpy()
    tv = np.abs(after - before).sum(-1) / 2
    return {"states": len(before), "mean_total_variation": float(tv.mean()),
            "max_total_variation": float(tv.max()),
            "top_family_flips": int((before.argmax(-1) != after.argmax(-1)).sum()),
            "mean_probability_change_by_family": {
                name: float((after[:, index] - before[:, index])[mask[:, index]].mean())
                if mask[:, index].any() else None
                for index, name in enumerate((
                    "BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "BUY_DEVELOPMENT",
                    "BANK_TRADE", "PLAYER_TRADE", "END_TURN"))}}


def update(root: Path, round_id: int):
    attempt = root / f"round{round_id}_update_attempt.json"
    if attempt.exists():
        raise ValueError("One optimizer attempt per round; no data reuse")
    ready = _read(root / f"round{round_id}_preflight.json")
    if ready["status"] != "PASS":
        raise ValueError("Round preflight failed")
    rollout = root / f"round{round_id}_rollout.json.gz"
    if _sha(rollout) != ready["rollout_sha256"]:
        raise AssertionError("Rollout SHA changed after preflight")
    champion, robber = _agents()
    policy, critic, parent = _load_version(root, round_id - 1, champion)
    parent_path = (PARENT_ROOT / PARENT_NAME / "weights.pt" if round_id == 1
                   else root / NAMES[1] / "weights.pt")
    if _sha(parent_path) != ready["parent_sha256"]:
        raise AssertionError("Policy changed after rollout")
    frozen = _read(root / "experiment_manifest.json")["frozen"]
    if (_weight_digest(champion) != frozen["champion"] or
            _weight_digest(robber) != frozen["robber"] or
            _module_digest(policy.plan_heads) != frozen["plan_heads"] or
            _module_digest(policy.l3_actor) != frozen["base_actor"]):
        raise AssertionError("Frozen model mismatch")
    prior_actor, prior_critic = None, None
    if round_id == 2:
        prior_states = torch.load(root / NAMES[1] / "optimizer.pt",
                                  map_location="cpu", weights_only=True)
        prior_actor, prior_critic = prior_states["actor"], prior_states["critic"]
    _write(attempt, {"round": round_id, "policy_version": NAMES[round_id - 1],
                     "data_sha256": ready["rollout_sha256"],
                     "minibatch_seed": MINIBATCH_SEEDS[round_id],
                     "prior_optimizer_sha256": _sha(root / NAMES[1] / "optimizer.pt")
                     if round_id == 2 else None})
    episodes = _gzip_read(rollout)
    result = one_guarded_update(
        champion, policy, critic, episodes,
        minibatch_seed=MINIBATCH_SEEDS[round_id], minimum_transitions=TARGET,
        route=ROUTE, allowed_families=CITY_FAMILIES,
        delegation_probability=TRAIN_RATE, actor_optimizer_state=prior_actor,
        critic_optimizer_state=prior_critic)
    if (result.metrics["city_adapter_delta"] <= 0 or
            result.metrics["l3_delta"] != 0 or
            result.metrics["plan_scorer_delta"] != 0 or
            result.metrics["priority_delta"] != 0 or
            _module_digest(policy.plan_heads) != frozen["plan_heads"] or
            _module_digest(policy.l3_actor) != frozen["base_actor"] or
            not _investment_parity(champion, policy)["exact"]):
        raise AssertionError("Route-isolation violation")
    if (_sha(parent_path) != ready["parent_sha256"] or
            _weight_digest(champion) != frozen["champion"] or
            _weight_digest(robber) != frozen["robber"]):
        raise AssertionError("Frozen parent changed during update")
    output = {"round": round_id, "accepted": result.accepted,
              "reason": result.reason, "metrics": result.metrics,
              "data_sha256": ready["rollout_sha256"],
              "parent_sha256": ready["parent_sha256"]}
    if result.accepted:
        folder = root / NAMES[round_id]
        folder.mkdir(parents=True, exist_ok=False)
        torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()},
                   folder / "weights.pt")
        torch.save({"actor": result.actor_optimizer_state,
                    "critic": result.critic_optimizer_state}, folder / "optimizer.pt")
        manifest = {"step": "3B-12", "name": NAMES[round_id],
                    "policy_version": round_id, "parent": NAMES[round_id - 1],
                    "parent_sha256": ready["parent_sha256"],
                    "training_seed_bases": TRAIN_BASES[round_id],
                    "training_games": len(episodes),
                    "training_transitions": ready["transitions"],
                    "training_data_sha256": ready["rollout_sha256"],
                    "training_delegation": TRAIN_RATE,
                    "minibatch_seed": MINIBATCH_SEEDS[round_id],
                    "optimizer_sha256": _sha(folder / "optimizer.pt"),
                    "champion_digest": frozen["champion"],
                    "robber_digest": frozen["robber"],
                    "plan_heads_digest": frozen["plan_heads"],
                    "base_actor_digest": frozen["base_actor"],
                    "city_adapter_digest": _module_digest(policy.city_adapter),
                    "city_critic_digest": _module_digest(critic)}
        _write(folder / "manifest.json", manifest)
        output["weights_sha256"] = _sha(folder / "weights.pt")
        output["optimizer_sha256"] = manifest["optimizer_sha256"]
    _write(root / f"round{round_id}_update_result.json", output)
    return {"round": round_id, "accepted": result.accepted,
            "reason": result.reason, "kl": result.metrics["kl"],
            "adapter_delta": result.metrics["city_adapter_delta"]}


def evaluate_profile(root: Path, profile: str, start: int, stop: int):
    if profile not in EVAL_BASES or not 0 <= start < stop <= EVAL_PER_PROFILE:
        raise ValueError("Outside precommitted held-out seed range")
    if not all(_read(root / f"round{r}_update_result.json")["accepted"] for r in (1, 2)):
        raise ValueError("Both rounds must be accepted before final evaluation")
    champion, robber = _agents()
    models = [_load_version(root, v, champion)[:2] for v in (0, 1, 2)]
    before = (_weight_digest(champion), _weight_digest(robber),
              tuple(_module_digest(p) for p, _ in models))
    variants = {"safety": (*models[0], 0.), "post1": (*models[0], EVAL_RATE),
                "round1": (*models[1], EVAL_RATE),
                "round2": (*models[2], EVAL_RATE)}
    parity = 0
    for index in range(start, stop):
        seed, seat = EVAL_BASES[profile] + index, index % 4 + 1
        for name, (policy, critic, rate) in variants.items():
            path = root / "evaluation_games" / f"{name}_{profile}_{seed}.json.gz"
            if path.exists():
                if name == "safety":
                    parity += 1
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            episode = run_hierarchical_episode(
                champion, robber, policy, critic, seed=seed, seat=seat,
                profile=profile, delegation_probability=rate,
                delegation_seed=seed + 500000, family_seed=seed + 900000,
                multi_choice_only=True, delegated_route=ROUTE)
            if not episode["terminal"] or episode["truncated"]:
                raise AssertionError("Held-out City episode not terminal")
            if name == "safety":
                reference = run_strategic_episode(champion, robber, seed=seed,
                    seat=seat, profile=profile, mode="safety")
                if not _raw_safety_parity(episode, reference):
                    raise AssertionError("0% Safety Action/State/RNG parity failure")
                parity += 1
            _gzip_write(path, _plain(episode))
        if index % 8 == 7:
            print(f"3B12 evaluation {profile} {index+1}/{EVAL_PER_PROFILE}", flush=True)
    if before != (_weight_digest(champion), _weight_digest(robber),
                  tuple(_module_digest(p) for p, _ in models)):
        raise AssertionError("Evaluation changed frozen weights")
    return {"profile": profile, "start": start, "stop": stop,
            "zero_percent_parity": parity, "weights_unchanged": True}


def _raw_safety_parity(episode, reference):
    """Compare typed Actions before the evidence serializer replaces them with repr."""
    return (episode["action_trace"] == reference.action_trace and
            episode["final_state"] == reference.final_state and
            episode["rng_counter"] == reference.rng_counter and
            episode["winner"] == reference.winner)


def parity_diagnose(root: Path, profile: str, index: int):
    if profile not in EVAL_BASES or not 0 <= index < EVAL_PER_PROFILE:
        raise ValueError("Outside fixed evaluation seeds")
    champion, robber = _agents()
    policy, critic, _ = _load_version(root, 0, champion)
    seed, seat = EVAL_BASES[profile] + index, index % 4 + 1
    candidate = run_hierarchical_episode(
        champion, robber, policy, critic, seed=seed, seat=seat,
        profile=profile, delegation_probability=0.,
        delegation_seed=seed + 500000, family_seed=seed + 900000,
        multi_choice_only=True, delegated_route=ROUTE)
    reference = run_strategic_episode(champion, robber, seed=seed, seat=seat,
                                      profile=profile, mode="safety")
    trace = candidate["action_trace"]
    expected = reference.action_trace
    first = next((i for i, (a, b) in enumerate(zip(trace, expected)) if a != b),
                 min(len(trace), len(expected)))
    return {"seed": seed, "profile": profile, "seat": seat,
            "first_divergence_index": first, "trace_lengths": [len(trace), len(expected)],
            "candidate_action": trace[first] if first < len(trace) else None,
            "reference_action": expected[first] if first < len(expected) else None,
            "winner": [candidate["winner"], reference.winner],
            "rng_counter": [candidate["rng_counter"], reference.rng_counter],
            "state_equal": candidate["final_state"] == reference.final_state,
            "trace_equal": trace == expected}


def summarize(root: Path):
    names = ("safety", "post1", "round1", "round2")
    rows = {name: [] for name in names}
    for profile, base in EVAL_BASES.items():
        for index in range(EVAL_PER_PROFILE):
            for name in names:
                path = root / "evaluation_games" / f"{name}_{profile}_{base+index}.json.gz"
                if not path.is_file():
                    raise ValueError(f"Missing precommitted held-out game: {path}")
                rows[name].append(_gzip_read(path))
    if any(not game["terminal"] or game["truncated"] for items in rows.values()
           for game in items):
        raise AssertionError("Held-out hard safety failure")
    frozen = _read(root / "experiment_manifest.json")["frozen"]
    champion, robber = _agents()
    policies = [_load_version(root, version, champion)[0] for version in (0, 1, 2)]
    if (_weight_digest(champion) != frozen["champion"] or
            _weight_digest(robber) != frozen["robber"] or
            any(_module_digest(p.plan_heads) != frozen["plan_heads"] or
                _module_digest(p.l3_actor) != frozen["base_actor"] for p in policies)):
        raise AssertionError("Frozen weights drift")
    comparison = {name: _paired_profiles(rows["safety"], rows[name])
                  for name in ("post1", "round1", "round2")}
    result = {"variants": {name: {**_metrics(items), **_city_metrics(items)}
                           for name, items in rows.items()},
              "safety_to_city": comparison,
              "post1_to_round1": _paired_profiles(rows["post1"], rows["round1"]),
              "round1_to_round2": _paired_profiles(rows["round1"], rows["round2"]),
              "post1_to_round2": _paired_profiles(rows["post1"], rows["round2"]),
              "action_trace_changes": {
                  f"{a}_to_{b}": sum(x["action_trace"] != y["action_trace"]
                                    for x, y in zip(rows[a], rows[b], strict=True))
                  for a, b in (("post1", "round1"), ("round1", "round2"),
                               ("post1", "round2"))},
              "profile_seat_balanced": True, "games_per_variant": 288,
              "hard_safety": "PASS", "zero_percent_parity": 288,
              "investment_fixed": [_investment_parity(champion, p) for p in policies],
              "fixed_city_diagnosis": [fixed_diagnosis(root, version)
                                       for version in (0, 1, 2)],
              "fixed_city_policy_shift": {
                  "post1_to_round1": fixed_city_shift(root, 0, 1),
                  "round1_to_round2": fixed_city_shift(root, 1, 2),
                  "post1_to_round2": fixed_city_shift(root, 0, 2)},
              "frozen": frozen}
    _write(root / "evaluation_summary.json", result)
    return result


def publish(root: Path):
    summary = _read(root / "evaluation_summary.json")
    if summary["games_per_variant"] != 288 or summary["hard_safety"] != "PASS":
        raise ValueError("Incomplete final evaluation")
    frozen = _read(root / "experiment_manifest.json")["frozen"]
    champion, robber = _agents()
    if (_sha(PARENT_ROOT / PARENT_NAME / "weights.pt") != frozen["parent_sha256"] or
            _weight_digest(champion) != frozen["champion"] or
            _weight_digest(robber) != frozen["robber"]):
        raise AssertionError("Parent or frozen Champion/Robber drift before publication")
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    names = ["experiment_manifest.json", "validation_rollout.json.gz",
             "evaluation_summary.json"]
    for round_id in (1, 2):
        names.extend(f"round{round_id}_{suffix}" for suffix in (
            "collection.json", "preflight.json", "update_attempt.json",
            "update_result.json", "rollout.json.gz"))
    for name in names:
        shutil.copy2(root / name, EVIDENCE / name)
    for version in (1, 2):
        source = root / NAMES[version]
        destination = EVIDENCE / NAMES[version]
        destination.mkdir()
        for name in ("weights.pt", "optimizer.pt", "manifest.json"):
            shutil.copy2(source / name, destination / name)
    games = []
    for profile, base in EVAL_BASES.items():
        for index in range(EVAL_PER_PROFILE):
            row = {"profile": profile, "seed": base + index, "seat": index % 4 + 1}
            for variant in ("safety", "post1", "round1", "round2"):
                episode = _gzip_read(root / "evaluation_games" /
                    f"{variant}_{profile}_{base+index}.json.gz")
                row[variant] = {key: episode[key] for key in (
                    "winner", "learner_id", "final_vp", "final_rank", "policy_steps",
                    "rng_counter", "final_state_sha256", "actual_delegated",
                    "learner_action_families", "city_opportunities")}
            games.append(row)
    _gzip_write(EVIDENCE / "heldout_games.json.gz", games)
    _write(EVIDENCE / "sha256.json", {
        str(path.relative_to(EVIDENCE)): _sha(path) for path in EVIDENCE.rglob("*")
        if path.is_file()})
    return {"destination": str(EVIDENCE), "heldout_games": len(games)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--phase", required=True, choices=(
        "prepare", "validation", "collect", "collect-profile", "preflight", "update",
        "eval-profile", "parity-diagnose", "summarize", "publish"))
    parser.add_argument("--round", type=int, choices=(1, 2))
    parser.add_argument("--profile", choices=tuple(EVAL_BASES))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=EVAL_PER_PROFILE)
    parser.add_argument("--blocks", type=int, default=40)
    parser.add_argument("--start-block", type=int, default=0)
    args = parser.parse_args()
    if args.phase in ("collect", "collect-profile", "preflight", "update") and args.round is None:
        parser.error("--round is required")
    if args.phase == "collect":
        result = collect(args.output, args.round)
    elif args.phase == "collect-profile":
        result = collect_profile(args.output, args.round, args.profile, args.blocks,
                                 args.start_block)
    elif args.phase == "preflight":
        result = preflight_round(args.output, args.round)
    elif args.phase == "update":
        result = update(args.output, args.round)
    elif args.phase == "eval-profile":
        result = evaluate_profile(args.output, args.profile, args.start, args.stop)
    elif args.phase == "parity-diagnose":
        result = parity_diagnose(args.output, args.profile, args.start)
    else:
        result = globals()[args.phase](args.output)
    print(f"3B12 {args.phase}: {json.dumps(result, ensure_ascii=False, default=str)[:1800]}",
          flush=True)


if __name__ == "__main__":
    main()
