"""Step 3A-8: 36-game candidate-only Shadow and fixed-seed parity audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from unittest.mock import patch

import numpy as np

from app.agents.heuristic import HeuristicAgent
from app.agents.ppo import PPOAgent
from app.domain.actions import BuildCityAction, BuildSettlementAction

from .construction_timing import (FrozenConstructionExecutors, timing_surface)
from .macro_teacher_dataset import OPPONENT_PROFILES, play_teacher_episode
from .expansion_planner import analyze_expansion_plan
from .runtime_context import encode_runtime_context
from .settlement_planning_agent import SettlementPlanningAgent
from .structured_context import build_structured_context_report


class CandidateShadowAgent(SettlementPlanningAgent):
    def __init__(self, champion, **planning):
        super().__init__(champion, **planning)
        self.executors = FrozenConstructionExecutors(self)
        self.records: list[dict] = []
        self.context_timings: list[float] = []
        self.paired_context_timings: list[float] = []
        self.legacy_context_timings: list[float] = []
        self.exclusion_counts: Counter[str] = Counter()
        self.game_seed = None
        self.profile = None

    def select_action(self, game, player_id):
        surface = timing_surface(game, player_id)
        if not surface.eligible:
            if surface.subtype is not None and surface.exclusion is not None:
                self.exclusion_counts[surface.exclusion] += 1
            return super().select_action(game, player_id)
        rng_before = game.random_source.counter
        state_before = deepcopy(game)
        candidates = self.executors.candidates(game, player_id, surface)
        start = perf_counter()
        context = encode_runtime_context(game, player_id,
                                         max_plan_roads=self.max_plan_roads)
        self.context_timings.append(perf_counter() - start)
        if not self.records:
            self.paired_context_timings.append(self.context_timings[-1])
            def legacy_report_and_plan(state, pid, *, max_plan_roads=3):
                report = build_structured_context_report(
                    state, pid, max_plan_roads=max_plan_roads)
                plan = analyze_expansion_plan(
                    deepcopy(state), pid, max_additional_roads=max_plan_roads)
                return report, plan

            start = perf_counter()
            with patch("app.rl.runtime_context.build_structured_context_report_with_plan",
                       side_effect=legacy_report_and_plan):
                legacy = encode_runtime_context(
                    game, player_id, max_plan_roads=self.max_plan_roads)
            self.legacy_context_timings.append(perf_counter() - start)
            if not np.array_equal(context, legacy):
                raise AssertionError("Optimized 79-d context differs from legacy")
        if not np.isfinite(context).all():
            raise AssertionError("Context contains NaN or inf")
        if game != state_before or game.random_source.counter != rng_before:
            raise AssertionError("Candidate generation mutated GameState or RNG")
        champion_action = super().select_action(game, player_id)
        self.records.append({
            "seed": self.game_seed, "profile": self.profile,
            "seat": player_id, "subtype": surface.subtype,
            "build_valid": candidates.build_action is not None,
            "defer_valid": candidates.defer_action is not None,
            "build_error": candidates.build_error,
            "defer_error": candidates.defer_error,
            "build_action": repr(candidates.build_action),
            "defer_action": repr(candidates.defer_action),
            "defer_family": (type(candidates.defer_action).__name__
                             if candidates.defer_action is not None else None),
            "build_vertex": (candidates.build_action.vertex_id
                             if isinstance(candidates.build_action,
                                           (BuildSettlementAction, BuildCityAction)) else None),
            "champion_action": repr(champion_action),
            "champion_is_build": isinstance(champion_action,
                                            (BuildSettlementAction, BuildCityAction)),
            "build_exact_match": candidates.build_action == champion_action,
            "defer_exact_match": candidates.defer_action == champion_action,
        })
        return champion_action


def _percentiles(samples):
    return {"n": len(samples), "mean_ms": 1000 * fmean(samples),
            "median_ms": 1000 * median(samples),
            "p95_ms": 1000 * float(np.percentile(samples, 95))} if samples else None


def collect(champion, robber, *, games_per_profile=12, seed_start=2370001):
    planning = champion.settlement_planning_config
    robber_planning = robber.settlement_planning_config
    if planning is None or robber_planning is None:
        raise ValueError("Both PPO agents require Planner configuration")
    factories = {
        "rule": HeuristicAgent,
        "champion": lambda: SettlementPlanningAgent(champion, **planning),
        "robber_ppo": lambda: SettlementPlanningAgent(robber, **robber_planning),
    }
    records, games, timings, paired_timings, legacy_timings = [], [], [], [], []
    exclusions: Counter[str] = Counter()
    for profile_index, profile in enumerate(OPPONENT_PROFILES):
        for index in range(games_per_profile):
            seed = seed_start + profile_index * 1000 + index
            seat = index % 4 + 1
            baseline = play_teacher_episode(
                SettlementPlanningAgent(champion, **planning), seed,
                champion_seat=seat, opponent_profile=profile,
                opponent_factories=factories, collect_reports=False)
            agent = CandidateShadowAgent(champion, **planning)
            agent.game_seed, agent.profile = seed, profile
            sampled = play_teacher_episode(
                agent, seed, champion_seat=seat, opponent_profile=profile,
                opponent_factories=factories, collect_reports=False)
            if (baseline.action_trace != sampled.action_trace
                    or baseline.final_state != sampled.final_state
                    or baseline.final_views != sampled.final_views):
                raise AssertionError(f"{profile}/{seed}: Action or GameState parity failed")
            for key in ("winner", "final_vp", "final_rank", "final_scores", "rng_counter"):
                if baseline.game[key] != sampled.game[key]:
                    raise AssertionError(f"{profile}/{seed}: {key} parity failed")
            records.extend(agent.records)
            timings.extend(agent.context_timings)
            paired_timings.extend(agent.paired_context_timings)
            legacy_timings.extend(agent.legacy_context_timings)
            exclusions.update(agent.exclusion_counts)
            games.append({"seed": seed, "seat": seat, "profile": profile,
                          "actions": len(baseline.action_trace),
                          "surface_count": len(agent.records), "parity": True})
            print(f"{profile} seed={seed} seat={seat} "
                  f"actions={len(baseline.action_trace)} "
                  f"surface={len(agent.records)}", flush=True)
    by_type = defaultdict(list)
    for record in records:
        by_type[record["subtype"]].append(record)
    summary = {
        "games": len(games), "surface_decisions": len(records),
        "all_games_parity": all(game["parity"] for game in games),
        "hard_exclusions": dict(exclusions),
        "per_subtype": {
            subtype: {
                "total": len(rows),
                "build_valid": sum(row["build_valid"] for row in rows),
                "defer_valid": sum(row["defer_valid"] for row in rows),
                "both_valid": sum(row["build_valid"] and row["defer_valid"]
                                  for row in rows),
                "build_exact_champion": sum(row["build_exact_match"] for row in rows),
                "defer_exact_champion_when_nonbuild": sum(
                    row["defer_exact_match"] for row in rows if not row["champion_is_build"]),
                "champion_nonbuild_count": sum(not row["champion_is_build"] for row in rows),
                "defer_family": dict(Counter(row["defer_family"] for row in rows)),
                "build_vertex": dict(Counter(str(row["build_vertex"]) for row in rows)),
                "invalid_reasons": dict(Counter(row["defer_error"] for row in rows
                                               if row["defer_error"])),
            } for subtype, rows in by_type.items()},
        "context_runtime": _percentiles(timings),
        "paired_optimized_context_runtime": _percentiles(paired_timings),
        "legacy_context_runtime": _percentiles(legacy_timings),
        "context_vector_exact_parity": True,
        "game_records": games,
    }
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games-per-profile", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=2370001)
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."} or args.games_per_profile < 1):
        parser.error("Invalid experiment id or games-per-profile")
    root = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if root.exists():
        parser.error("Experiment directory already exists")
    champion = PPOAgent("ppo_gnn_board_65k_s03_exp_v003", device="cpu")
    robber = PPOAgent("ppo_robber_value_73k_s01_exp_v001", device="cpu")
    records, summary = collect(champion, robber,
                               games_per_profile=args.games_per_profile,
                               seed_start=args.seed_start)
    root.mkdir(parents=True)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    with (root / "candidate_decisions.jsonl").open("w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"games": summary["games"],
                      "surface_decisions": summary["surface_decisions"],
                      "parity": summary["all_games_parity"]}), flush=True)


if __name__ == "__main__":
    main()
