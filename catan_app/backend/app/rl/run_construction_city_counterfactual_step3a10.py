"""Collect fresh CITY_ONLY paired counterfactuals; never update a policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.agents.heuristic import HeuristicAgent
from app.agents.ppo import PPOAgent

from .construction_city_counterfactual import (capture_boundary,
                                                create_profile_episode,
                                                paired_city_rollout,
                                                summarize_city_pairs)
from .construction_timing import timing_surface
from .macro_teacher_dataset import PROFILE_ROLES


CHAMPION_ID = "ppo_gnn_board_65k_s03_exp_v003"
ROBBER_ID = "ppo_robber_value_73k_s01_exp_v001"
SCHEMA_VERSION = "city_counterfactual_step3a10_v1"
PROFILES = tuple(PROFILE_ROLES)
SEED_START = 2631000


def seed_for(profile_index: int, seat: int, game_index: int) -> int:
    if profile_index not in range(len(PROFILES)) or seat not in range(1, 5):
        raise ValueError("Invalid profile/seat")
    if game_index not in range(10000):
        raise ValueError("Game index is outside the disjoint seed band")
    return SEED_START + profile_index * 100000 + seat * 10000 + game_index


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    ids = [record["state_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate state_id in recorded pairs")
    return records


def write_summary(root: Path, records: list[dict]) -> None:
    summary = summarize_city_pairs(records)
    summary["schema_version"] = SCHEMA_VERSION
    summary["champion_model_id"] = CHAMPION_ID
    summary["robber_model_id"] = ROBBER_ID
    summary["seed_start"] = SEED_START
    summary["game_split_key"] = "game_id"
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def collect(root: Path, *, per_stratum: int, max_games_per_stratum: int) -> None:
    if per_stratum < 1 or per_stratum > 25:
        raise ValueError("Use 1..25 CITY_ONLY states per profile × seat stratum")
    dataset = root / "pairs.jsonl"
    records = read_records(dataset)
    seen = {record["state_id"] for record in records}
    counts = Counter((record["profile"], record["seat"]) for record in records)
    champion_policy = PPOAgent(CHAMPION_ID, device="cpu")
    robber_policy = PPOAgent(ROBBER_ID, device="cpu")
    heuristic = HeuristicAgent()
    for profile_index, profile in enumerate(PROFILES):
        for seat in range(1, 5):
            if counts[(profile, seat)] >= per_stratum:
                continue
            for game_index in range(max_games_per_stratum):
                if counts[(profile, seat)] >= per_stratum:
                    break
                seed = seed_for(profile_index, seat, game_index)
                env, champion, learner_id = create_profile_episode(
                    champion_policy, robber_policy, seed=seed, seat=seat,
                    profile=profile)
                try:
                    done = truncated = False
                    while not (done or truncated):
                        game = env.game
                        if game.phase == "action":
                            surface = timing_surface(game, learner_id)
                            if surface.eligible and surface.subtype == "CITY_ONLY":
                                state_id = (f"city-cf-{profile}-{seed}-{learner_id}-"
                                            f"{env.policy_steps}")
                                if state_id not in seen and counts[(profile, seat)] < per_stratum:
                                    boundary = capture_boundary(env)
                                    pair = paired_city_rollout(
                                        env, champion, boundary, profile=profile,
                                        seat=seat)
                                    if pair["build"]["truncated"] or pair["defer"]["truncated"]:
                                        raise RuntimeError(f"Counterfactual truncated: {state_id}")
                                    with dataset.open("a", encoding="utf-8") as stream:
                                        stream.write(json.dumps(pair, ensure_ascii=False,
                                                                allow_nan=False) + "\n")
                                    seen.add(state_id)
                                    records.append(pair)
                                    counts[(profile, seat)] += 1
                                    print(f"pair {len(records)} {profile}/seat{seat} "
                                          f"champion={'BUILD' if pair['champion_build'] else 'DEFER'} "
                                          f"defer={pair['defer_action_family']} "
                                          f"ΔQ={pair['delta']['total']:.3f}", flush=True)
                                    if counts[(profile, seat)] >= per_stratum:
                                        break
                        action = (heuristic.select_action(game, learner_id)
                                  if game.phase == "rolling_order"
                                  else champion.select_action(game, learner_id))
                        _, _, done, truncated, _ = env.step_action(action)
                    if truncated:
                        raise RuntimeError(f"Baseline episode truncated: {seed}")
                finally:
                    env.close()
            else:
                raise RuntimeError(f"Insufficient CITY_ONLY states: {profile}/seat{seat} "
                                   f"{counts[(profile, seat)]}/{per_stratum}")
            write_summary(root, records)
    write_summary(root, records)
    print(f"COLLECTION COMPLETE pairs={len(records)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-stratum", type=int, default=17,
                        help="17 × 3 profiles × 4 seats = 204 paired states")
    parser.add_argument("--max-games-per-stratum", type=int, default=100)
    parser.add_argument("--experiment-id", default="construction_city_step3a10_20261004")
    parser.add_argument("--summary-only", action="store_true",
                        help="Rebuild summary from saved pairs without loading models")
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id:
        parser.error("Invalid experiment ID")
    root = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    root.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        dataset = root / "pairs.jsonl"
        if not dataset.exists():
            parser.error(f"Missing paired dataset: {dataset}")
        write_summary(root, read_records(dataset))
        return
    collect(root, per_stratum=args.per_stratum,
            max_games_per_stratum=args.max_games_per_stratum)


if __name__ == "__main__":
    main()
