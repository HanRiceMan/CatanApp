"""Collect paired multi-future CITY_ONLY audits; no PPO or Executor updates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.agents.heuristic import HeuristicAgent
from app.agents.ppo import PPOAgent

from .construction_city_counterfactual import (capture_boundary,
                                                create_profile_episode,
                                                paired_city_rollout)
from .construction_city_multifuture import (choose_phase2, future_seed,
                                            select_states, state_stats, summarize)
from .run_construction_city_counterfactual_step3a10 import (CHAMPION_ID, ROBBER_ID,
                                                             read_records)


SCHEMA_VERSION = "city_multifuture_step3a11_v1"
DEFAULT_EXPERIMENT = "construction_city_step3a11_20261004"
SOURCE_EXPERIMENT = "construction_city_step3a10_20261004"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def read_continuations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    keys = [(row["state_id"], row["future_seed_index"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate state/future-seed index in continuations")
    return rows


def _selection(root: Path, source: list[dict]) -> dict:
    path = root / "selection.json"
    if path.exists():
        payload = _read_json(path)
        if set(payload["state_categories"]) - {r["state_id"] for r in source}:
            raise ValueError("Saved selection does not match source dataset")
        return payload
    payload = select_states(source)
    _write_json(path, payload)
    return payload


def _phase2_ids(root: Path, selected: dict[str, str],
                source: list[dict], rows: list[dict]) -> list[str]:
    path = root / "phase2_selection.json"
    if path.exists():
        return _read_json(path)["state_ids"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["future_seed_index"] < 8:
            grouped[row["state_id"]].append(row)
    if set(grouped) != set(selected) or any(len(grouped[sid]) != 8 for sid in selected):
        raise ValueError("Phase 1 must complete before adaptive Phase 2")
    stats = {sid: state_stats(grouped[sid], sid) for sid in selected}
    source_by_id = {r["state_id"]: r for r in source}
    ids = choose_phase2(stats, source_by_id)
    _write_json(path, {"state_ids": ids, "rule":
                "SE>=0.02 and CI overlaps zero or sign consistency<0.75; family coverage; cap 12"})
    return ids


def _compact(pair: dict, source: dict, index: int) -> dict:
    build, defer, delta = pair["build"], pair["defer"], pair["delta"]
    return {
        "schema_version": SCHEMA_VERSION,
        "state_id": pair["state_id"], "game_id": pair["game_id"],
        "game_seed": pair["game_seed"], "profile": pair["profile"],
        "seat": pair["seat"], "pre_policy_steps": pair["pre_policy_steps"],
        "champion_actual_action": pair["champion_action"],
        "champion_build_or_defer": "BUILD" if pair["champion_build"] else "DEFER",
        "defer_executor_family": pair["defer_action_family"],
        "future_seed_index": index, "future_seed": pair["continuation_seed"],
        "original_rng_seed": pair["pre_rng_seed"],
        "original_rng_counter": pair["pre_rng_counter"],
        "runtime_context": source["context"],
        "q_build": build["discounted_environment_return"],
        "q_defer": defer["discounted_environment_return"],
        "delta_q": delta["total"],
        "discounted_vp_build": build["discounted_vp_component"],
        "discounted_vp_defer": defer["discounted_vp_component"],
        "delta_vp": delta["vp"],
        "discounted_terminal_build": build["discounted_terminal_component"],
        "discounted_terminal_defer": defer["discounted_terminal_component"],
        "delta_terminal": delta["terminal"],
        "build_win": build["win"], "defer_win": defer["win"],
        "build_final_vp": build["final_vp"],
        "defer_final_vp": defer["final_vp"],
        "build_rank": build["final_rank"],
        "defer_rank": defer["final_rank"],
        "build_truncated": build["truncated"],
        "defer_truncated": defer["truncated"],
    }


def collect(root: Path, source_rows: list[dict], selected: dict[str, str],
            targets: dict[str, range]) -> None:
    dataset = root / "continuations.jsonl"
    existing = read_continuations(dataset)
    keys = {(r["state_id"], r["future_seed_index"]) for r in existing}
    if len(keys) != len(existing):
        raise ValueError("Duplicate continuation record")
    if any(r["state_id"] not in selected for r in existing):
        raise ValueError("Continuation references an unselected state")
    pending = {sid: indices for sid, desired in targets.items()
               if (indices := [index for index in desired if (sid, index) not in keys])}
    if not pending:
        print("All requested continuations already collected", flush=True)
        return
    source_by_id = {r["state_id"]: r for r in source_rows}
    by_game: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for sid in pending:
        row = source_by_id[sid]
        by_game[(row["profile"], row["game_seed"], row["seat"])].append(row)
    champion_policy = PPOAgent(CHAMPION_ID, device="cpu")
    robber_policy = PPOAgent(ROBBER_ID, device="cpu")
    heuristic = HeuristicAgent()
    for (profile, seed, seat), states in sorted(by_game.items()):
        states_by_step = {r["pre_policy_steps"]: r for r in states}
        if len(states_by_step) != len(states):
            raise ValueError("Two states share one learner boundary")
        env, champion, learner_id = create_profile_episode(
            champion_policy, robber_policy, seed=seed, seat=seat,
            profile=profile)
        try:
            while env.policy_steps <= max(states_by_step):
                row = states_by_step.get(env.policy_steps)
                if row is not None:
                    sid = row["state_id"]
                    boundary = capture_boundary(env)
                    if (boundary.game.random_source.seed != row["pre_rng_seed"]
                            or boundary.game.random_source.counter != row["pre_rng_counter"]
                            or boundary.game.turn_number != row["pre_turn"]):
                        raise AssertionError(f"Baseline replay diverged: {sid}")
                    before = (env.policy_steps, env._request_number,
                              env._learning_score, env._learner_auxiliary_actions,
                              dict(env.controllers))
                    for index in pending[sid]:
                        pair = paired_city_rollout(
                            env, champion, boundary, profile=profile, seat=seat,
                            continuation_seed=future_seed(sid, index))
                        if (pair["champion_action"] != row["champion_action"]
                                or pair["defer_action_family"] != row["defer_action_family"]
                                or pair["context"] != row["context"]):
                            raise AssertionError(f"Source replay mismatch: {sid}")
                        if pair["build"]["truncated"] or pair["defer"]["truncated"]:
                            raise RuntimeError(f"Counterfactual did not finish: {sid}")
                        compact = _compact(pair, row, index)
                        with dataset.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(compact, ensure_ascii=False,
                                                    allow_nan=False) + "\n")
                        keys.add((sid, index))
                    if (env.game != boundary.game or
                            before != (env.policy_steps, env._request_number,
                                       env._learning_score,
                                       env._learner_auxiliary_actions,
                                       dict(env.controllers))):
                        raise AssertionError(f"Collection mutated source: {sid}")
                    print(f"{sid}: +{len(pending[sid])} paired seeds; total={len(keys)}",
                          flush=True)
                    del states_by_step[env.policy_steps]
                    if not states_by_step:
                        break
                game = env.game
                action = (heuristic.select_action(game, learner_id)
                          if game.phase == "rolling_order"
                          else champion.select_action(game, learner_id))
                _, _, done, truncated, _ = env.step_action(action)
                if done or truncated:
                    raise RuntimeError(f"Baseline ended before selected state: {seed}")
        finally:
            env.close()
    print(f"COLLECTION COMPLETE continuation records={len(keys)}", flush=True)


def write_summary(root: Path, source: list[dict], selected: dict[str, str]) -> None:
    rows = read_continuations(root / "continuations.jsonl")
    summary = summarize(rows, source, selected)
    summary["schema_version"] = SCHEMA_VERSION
    summary["reward_gamma"] = 0.99
    summary["near_tie_threshold"] = 0.01
    summary["champion_model_id"] = CHAMPION_ID
    summary["robber_model_id"] = ROBBER_ID
    summary["branch_completion"] = {
        "paired_records": len(rows),
        "build_complete": sum(not r["build_truncated"] for r in rows),
        "defer_complete": sum(not r["defer_truncated"] for r in rows),
    }
    _write_json(root / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("phase1", "phase2", "summary"),
                        required=True)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--max-states", type=int, default=None,
                        help="Smoke-only prefix; use a separate experiment ID")
    parser.add_argument("--phase1-seeds", type=int, default=8,
                        help="Smoke may use 2; full audit requires 8")
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id:
        parser.error("Invalid experiment ID")
    if args.max_states is not None and args.experiment_id == DEFAULT_EXPERIMENT:
        parser.error("A partial smoke must use a separate experiment ID")
    backend = Path(__file__).resolve().parents[2]
    source_path = backend / "experiments" / SOURCE_EXPERIMENT / "pairs.jsonl"
    source = read_records(source_path)
    if len(source) != 204:
        parser.error("Expected all 204 Step 3A-10 source states")
    root = backend / "experiments" / args.experiment_id
    root.mkdir(parents=True, exist_ok=True)
    selection = _selection(root, source)
    selected = selection["state_categories"]
    if args.max_states is not None:
        selected = dict(list(sorted(selected.items()))[:args.max_states])
    if args.phase == "phase1":
        if args.phase1_seeds not in range(2, 9):
            parser.error("Use 2..8 seeds for Phase 1")
        collect(root, source, selected,
                {sid: range(args.phase1_seeds) for sid in selected})
    elif args.phase == "phase2":
        if args.max_states is not None:
            parser.error("Adaptive Phase 2 is only for the full selected cohort")
        ids = _phase2_ids(root, selected, source,
                          read_continuations(root / "continuations.jsonl"))
        print(f"Adaptive Phase 2 states={len(ids)}", flush=True)
        collect(root, source, selected, {sid: range(8, 16) for sid in ids})
    else:
        write_summary(root, source, selected)


if __name__ == "__main__":
    main()
