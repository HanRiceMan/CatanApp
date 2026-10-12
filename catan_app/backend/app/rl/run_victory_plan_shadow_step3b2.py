"""Step 3B-2 read-only Plan/Priority-head shadow over the 36 fixed games."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from math import log
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
import torch

from app.agents.ppo import PPOAgent

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .run_strategic_gate_step3a14 import CHAMPION_ID, ROBBER_ID, _strict_parity
from .run_victory_plan_shadow_step3b1 import PROFILES, SEED_BASES
from .strategic_gate_rollout_step3a14 import run_strategic_episode
from .strategic_gate_step3a13 import frozen_snapshot
from .victory_plan_context import SCHEMA_VERSION as CONTEXT_SCHEMA
from .victory_plan_context import build_victory_plan_context
from .victory_plan_policy import (SCHEMA_VERSION as HEAD_SCHEMA,
                                  VictoryPlanShadowHeads,
                                  encode_victory_plan_features)


SCHEMA = "victory_plan_shadow_step3b2_v1"


def _digest(heads: VictoryPlanShadowHeads) -> str:
    digest = sha256()
    for name, tensor in sorted(heads.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _record(env, _action, _decision, _trace_index, *, seed: int, profile: str,
            seat: int, champion: PPOAgent, heads: VictoryPlanShadowHeads,
            records: list[dict]) -> None:
    game = env.game
    player_id = env.learning_player_id
    before = deepcopy(game)
    started = perf_counter()
    context = build_victory_plan_context(game, player_id)
    context_done = perf_counter()
    snapshot = frozen_snapshot(champion, game, player_id)
    snapshot_done = perf_counter()
    features = encode_victory_plan_features(context)
    with torch.no_grad():
        output = heads(
            snapshot.actor_latent,
            torch.tensor(snapshot.runtime_context, dtype=torch.float32).unsqueeze(0),
            torch.tensor(features.global_features).unsqueeze(0),
            torch.tensor(features.candidate_features).unsqueeze(0),
            torch.tensor(features.candidate_mask).unsqueeze(0),
        )
    finished = perf_counter()
    if game != before or game.random_source.counter != before.random_source.counter:
        raise AssertionError("Plan/Priority shadow mutated GameState or RNG")
    valid_count = int(features.candidate_mask.sum())
    probabilities = output.probabilities[0].numpy()
    priorities = output.priorities[0].numpy()
    if valid_count:
        if not np.array_equal(probabilities[~features.candidate_mask],
                              np.zeros((~features.candidate_mask).sum())):
            raise AssertionError("Invalid/padded candidate has nonzero probability")
        if abs(float(probabilities.sum()) - 1.0) > 1e-6:
            raise AssertionError("Valid Plan probability does not sum to 1")
        if abs(float(output.entropy.item()) - log(valid_count)) > 1e-5:
            raise AssertionError("Zero-init Plan entropy is not log(valid count)")
        if not np.allclose(probabilities[features.candidate_mask], 1.0/valid_count):
            raise AssertionError("Untrained Plan preferences are not neutral")
    elif probabilities.any() or output.plan_available.item():
        raise AssertionError("Unresolved state returned a Plan distribution")
    if not np.array_equal(priorities, np.full_like(priorities, 0.5)):
        raise AssertionError("Untrained priorities must all be 0.5")
    candidate_count = len(features.candidate_ids)
    records.append({
        "seed": seed, "profile": profile, "seat": seat,
        "turn": game.turn_number, "revision": game.revision,
        "policy_step": env.policy_steps, "current_score": context.current_score,
        "remaining_points": context.remaining_points,
        "title_holders": [context.longest_road.current_holder,
                          context.largest_army.current_holder],
        "owned_titles": [context.longest_road.currently_owned,
                         context.largest_army.currently_owned],
        "candidate_ids": [list(item) for item in features.candidate_ids],
        "candidate_mask": features.candidate_mask[:candidate_count].tolist(),
        "raw_plan_logits": output.raw_logits[0, :candidate_count].tolist(),
        "masked_plan_probabilities": probabilities[:candidate_count].tolist(),
        "entropy": float(output.entropy.item()),
        "valid_candidate_count": valid_count,
        "plan_available": bool(output.plan_available.item()),
        "priorities": priorities[:candidate_count].tolist(),
        "priority_valid_min": priorities[:candidate_count][features.candidate_mask[:candidate_count]].min(axis=0).tolist()
                              if valid_count else None,
        "priority_valid_max": priorities[:candidate_count][features.candidate_mask[:candidate_count]].max(axis=0).tolist()
                              if valid_count else None,
        "priority_valid_mean": priorities[:candidate_count][features.candidate_mask[:candidate_count]].mean(axis=0).tolist()
                               if valid_count else None,
        "context_digest": sha256(json.dumps(asdict(context), sort_keys=True,
                                            default=str).encode("utf-8")).hexdigest(),
        "timing_ms": {
            "plan_context": (context_done-started)*1000,
            "frozen_snapshot_and_runtime79": (snapshot_done-context_done)*1000,
            "head_features_and_forward": (finished-snapshot_done)*1000,
            "total": (finished-started)*1000,
        },
        "game_state_parity": True, "rng_parity": True,
    })


def _candidate_churn(records: list[dict]) -> dict:
    counts = Counter()
    for rows in (group for group in _group_by_game(records).values()):
        for prior, current in zip(rows, rows[1:]):
            score = prior["current_score"] != current["current_score"]
            title = prior["title_holders"] != current["title_holders"]
            candidates = prior["candidate_ids"] != current["candidate_ids"]
            before_mask = {tuple(key): valid for key, valid in zip(
                prior["candidate_ids"], prior["candidate_mask"], strict=True)}
            after_mask = {tuple(key): valid for key, valid in zip(
                current["candidate_ids"], current["candidate_mask"], strict=True)}
            mask = before_mask != after_mask
            counts["comparisons"] += 1
            counts["SCORE_CHANGED"] += int(score)
            counts["TITLE_CHANGED"] += int(title)
            counts["CANDIDATE_SET_CHANGED"] += int(candidates)
            counts["VALID_MASK_CHANGED"] += int(mask)
            counts["VALID_MASK_ONLY_CHANGED"] += int(mask and not candidates)
            counts["NO_STRUCTURAL_CHANGE"] += int(not (score or title or candidates or mask))
    return dict(counts)


def _group_by_game(records: list[dict]) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["profile"], row["seed"])].append(row)
    return grouped


def _timing_summary(records: list[dict]) -> dict:
    return {name: {
        "mean_ms": float(mean(row["timing_ms"][name] for row in records)),
        "median_ms": float(median(row["timing_ms"][name] for row in records)),
        "p95_ms": float(np.percentile([row["timing_ms"][name] for row in records], 95)),
    } for name in ("plan_context", "frozen_snapshot_and_runtime79",
                   "head_features_and_forward", "total")}


def collect(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    champion = PPOAgent(CHAMPION_ID, device="cpu")
    robber = PPOAgent(ROBBER_ID, device="cpu")
    champion.model.policy.requires_grad_(False)
    latent_dim = champion.model.policy.mlp_extractor.latent_dim_pi
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3302)
        heads = VictoryPlanShadowHeads(latent_dim)
    heads.eval()
    weights_before = (_weight_digest(champion), _weight_digest(robber), _digest(heads))
    checkpoint = output / "checkpoint.json.gz"
    records: list[dict] = []
    episodes: list[dict] = []
    if checkpoint.exists():
        with gzip.open(checkpoint, "rt", encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved["schema"] != SCHEMA or tuple(saved["weight_digests"]) != weights_before:
            raise AssertionError("Shadow checkpoint schema or weights changed")
        records = saved["records"]
        episodes = saved["episodes"]
    completed = {(row["profile"], row["seed"]) for row in episodes}
    for profile in PROFILES:
        for index in range(12):
            seed = SEED_BASES[profile] + index
            if (profile, seed) in completed:
                continue
            seat = index % 4 + 1
            baseline = run_strategic_episode(champion, robber, seed=seed, seat=seat,
                                             profile=profile, mode="legacy")
            observer = lambda *args, seed=seed, profile=profile, seat=seat: _record(
                *args, seed=seed, profile=profile, seat=seat,
                champion=champion, heads=heads, records=records)
            shadow = run_strategic_episode(
                champion, robber, seed=seed, seat=seat, profile=profile,
                mode="legacy", pre_action_audit_observer=observer)
            if not _strict_parity(baseline, shadow):
                raise AssertionError(f"Action/GameState/RNG parity failed: {profile}/{seed}/{seat}")
            if not baseline.terminal or baseline.truncated:
                raise AssertionError(f"Episode did not terminate: {profile}/{seed}")
            episodes.append({
                "profile": profile, "seed": seed, "seat": seat,
                "winner": baseline.winner, "final_vp": baseline.final_vp,
                "final_rank": baseline.final_rank, "policy_steps": baseline.policy_steps,
                "rng_counter": baseline.rng_counter,
                "action_count": len(baseline.action_trace), "parity": True,
            })
            temporary = output / "checkpoint.tmp.json.gz"
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump({"schema": SCHEMA, "weight_digests": weights_before,
                           "records": records, "episodes": episodes}, handle,
                          ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            temporary.replace(checkpoint)
            print(f"{profile} {index+1}/12 seed={seed} seat={seat} "
                  f"records={len(records)} parity=ok", flush=True)
    weights_after = (_weight_digest(champion), _weight_digest(robber), _digest(heads))
    if weights_before != weights_after:
        raise AssertionError("Shadow changed frozen or new head weights")
    examples = {}
    for row in records:
        key = f"score{row['current_score']}_titles{''.join(str(int(x)) for x in row['owned_titles'])}"
        if key not in examples:
            examples[key] = {k: row[k] for k in (
                "seed", "profile", "seat", "turn", "revision", "current_score",
                "remaining_points", "candidate_ids", "candidate_mask",
                "raw_plan_logits", "masked_plan_probabilities", "entropy",
                "priority_valid_mean")}
    valid_priorities = np.asarray([priority for row in records
                                   for priority, valid in zip(row["priorities"],
                                                              row["candidate_mask"], strict=True)
                                   if valid], dtype=np.float64)
    summary = {
        "schema": SCHEMA, "context_schema_version": CONTEXT_SCHEMA,
        "head_schema_version": HEAD_SCHEMA, "seed_bases": SEED_BASES,
        "games": len(episodes), "states": len(records),
        "profile_seat": dict(Counter(f"{row['profile']}/seat{row['seat']}" for row in episodes)),
        "candidate_count_distribution": dict(sorted(Counter(
            len(row["candidate_ids"]) for row in records).items())),
        "valid_candidate_count_distribution": dict(sorted(Counter(
            row["valid_candidate_count"] for row in records).items())),
        "unresolved_count": sum(not row["plan_available"] for row in records),
        "entropy": {
            "mean": float(mean(row["entropy"] for row in records)),
            "min": float(min(row["entropy"] for row in records)),
            "median": float(median(row["entropy"] for row in records)),
            "max": float(max(row["entropy"] for row in records)),
        },
        "priority_valid": {
            "min": valid_priorities.min(axis=0).tolist(),
            "max": valid_priorities.max(axis=0).tolist(),
            "mean": valid_priorities.mean(axis=0).tolist(),
        },
        "candidate_churn": _candidate_churn(records),
        "timing": _timing_summary(records),
        "parameters": {
            "feature_encoder": sum(p.numel() for p in heads.feature_encoder.parameters()),
            "plan_scorer": sum(p.numel() for p in heads.plan_scorer.parameters()),
            "priority_head": sum(p.numel() for p in heads.priority_head.parameters()),
            "total": sum(p.numel() for p in heads.parameters()),
        },
        "weight_digests_before_after": {
            "champion": [weights_before[0], weights_after[0]],
            "robber": [weights_before[1], weights_after[1]],
            "head": [weights_before[2], weights_after[2]],
        },
        "mutation_failures": 0, "rng_failures": 0,
        "episode_parity_failures": 0, "optimizer_created": False,
        "examples": examples, "episodes": episodes,
    }
    with gzip.open(output / "shadow_records.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = collect(args.output)
    print(json.dumps({k: summary[k] for k in (
        "games", "states", "candidate_churn", "timing", "parameters")},
        ensure_ascii=False, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
