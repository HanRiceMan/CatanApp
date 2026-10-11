"""Collect read-only Victory Plan candidates and Champion parity (no learning)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from app.agents.ppo import PPOAgent

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .run_strategic_gate_step3a14 import CHAMPION_ID, ROBBER_ID, _strict_parity
from .strategic_gate_rollout_step3a14 import run_strategic_episode
from .victory_plan_context import SCHEMA_VERSION, build_victory_plan_context


SCHEMA = "victory_plan_shadow_step3b1a_v2"
PROFILES = ("rule", "champion", "mixed")
SEED_BASES = {"rule": 33000000, "champion": 33001000, "mixed": 33002000}


def _json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _record(env, _action, _decision, _trace_index, *, seed: int, profile: str,
            seat: int, records: list[dict]) -> None:
    game = env.game
    player_id = env.learning_player_id
    before = deepcopy(game)
    context = build_victory_plan_context(game, player_id)
    if game != before or game.random_source.counter != before.random_source.counter:
        raise AssertionError("Victory Plan calculation mutated GameState or RNG")
    encoded = asdict(context)
    digest = sha256(_json(encoded).encode("utf-8")).hexdigest()
    counts = Counter(item.feasibility.value for item in context.candidates)
    titles = (int(context.longest_road.currently_owned),
              int(context.largest_army.currently_owned))
    records.append({
        "seed": seed, "profile": profile, "seat": seat, "turn": game.turn_number,
        "revision": game.revision, "policy_step": env.policy_steps,
        "current_score": context.current_score,
        "remaining_points": context.remaining_points,
        "owned_titles": list(titles),
        "candidate_list": [
            {**asdict(candidate), "candidate_id": list(candidate.candidate_id),
             "planned_point_gain": candidate.planned_point_gain,
             "overshoot_points": context.overshoot_points(candidate)}
            for candidate in context.candidates],
        "candidate_count": len(context.candidates),
        "valid_candidate_count": sum(context.candidate_mask),
        "feasibility_counts": {name: counts.get(name, 0)
                               for name in ("FEASIBLE", "UNKNOWN", "IMPOSSIBLE")},
        "unresolved": context.unresolved,
        "context_digest": digest,
        "game_state_parity": True, "rng_parity": True,
    })


def collect(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    champion = PPOAgent(CHAMPION_ID, device="cpu")
    robber = PPOAgent(ROBBER_ID, device="cpu")
    weights_before = (_weight_digest(champion), _weight_digest(robber))
    records: list[dict] = []
    episode_rows: list[dict] = []
    for profile in PROFILES:
        for index in range(12):
            seed = SEED_BASES[profile] + index
            seat = index % 4 + 1
            baseline = run_strategic_episode(champion, robber, seed=seed, seat=seat,
                                             profile=profile, mode="legacy")
            observer = lambda *args, seed=seed, profile=profile, seat=seat: _record(
                *args, seed=seed, profile=profile, seat=seat, records=records)
            shadow = run_strategic_episode(
                champion, robber, seed=seed, seat=seat, profile=profile,
                mode="legacy", pre_action_audit_observer=observer)
            if not _strict_parity(baseline, shadow):
                raise AssertionError(f"Champion parity failed: {profile}/{seed}/{seat}")
            if not baseline.terminal or baseline.truncated:
                raise AssertionError(f"Episode did not terminate: {profile}/{seed}")
            episode_rows.append({
                "profile": profile, "seed": seed, "seat": seat,
                "winner": baseline.winner, "final_vp": baseline.final_vp,
                "policy_steps": baseline.policy_steps, "rng_counter": baseline.rng_counter,
                "action_count": len(baseline.action_trace), "parity": True,
            })
            print(f"{profile} {index+1}/12 seed={seed} seat={seat} "
                  f"records={len(records)} parity=ok", flush=True)
    if weights_before != (_weight_digest(champion), _weight_digest(robber)):
        raise AssertionError("Model weights changed during read-only shadow")
    counts = Counter(row["candidate_count"] for row in records)
    valid_counts = Counter(row["valid_candidate_count"] for row in records)
    by_score: dict[int, list[int]] = defaultdict(list)
    by_title: dict[str, list[int]] = defaultdict(list)
    total_feasibility = Counter()
    examples: dict[str, dict] = {}
    for row in records:
        by_score[row["current_score"]].append(row["valid_candidate_count"])
        title = "".join(map(str, row["owned_titles"]))
        by_title[title].append(row["valid_candidate_count"])
        total_feasibility.update(row["feasibility_counts"])
        key = f"score{row['current_score']}_titles{title}"
        if key not in examples:
            examples[key] = {k: row[k] for k in (
                "seed", "profile", "seat", "turn", "revision", "current_score",
                "remaining_points", "owned_titles", "candidate_count",
                "valid_candidate_count", "feasibility_counts", "unresolved")}
            examples[key]["valid_plan_ids"] = [
                [c["settlement_gain_count"], c["city_upgrade_count"],
                 c["acquire_longest_road"], c["acquire_largest_army"]]
                for c in row["candidate_list"] if c["valid"]][:12]
    summary = {
        "schema": SCHEMA,
        "context_schema_version": SCHEMA_VERSION,
        "seed_bases": SEED_BASES,
        "games": len(episode_rows), "records": len(records),
        "profile_seat": dict(Counter(f"{r['profile']}/seat{r['seat']}" for r in episode_rows)),
        "candidate_count_distribution": dict(sorted(counts.items())),
        "valid_candidate_count_distribution": dict(sorted(valid_counts.items())),
        "maximum_enumerated": max(counts, default=0),
        "maximum_valid": max(valid_counts, default=0),
        "score_valid_candidate_count": {
            str(score): {"n": len(values), "mean": mean(values),
                         "min": min(values), "max": max(values)}
            for score, values in sorted(by_score.items())},
        "title_valid_candidate_count": {
            title: {"n": len(values), "mean": mean(values),
                    "min": min(values), "max": max(values)}
            for title, values in sorted(by_title.items())},
        "feasibility_total": dict(total_feasibility),
        "unresolved_records": sum(row["unresolved"] for row in records),
        "mutation_failures": 0, "rng_failures": 0, "episode_parity_failures": 0,
        "weights_before_after": {
            "champion": weights_before[0], "robber": weights_before[1]},
        "examples": examples, "episodes": episode_rows,
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
    print(_json({k: summary[k] for k in (
        "games", "records", "maximum_enumerated", "maximum_valid",
        "feasibility_total", "unresolved_records")}))


if __name__ == "__main__":
    main()
