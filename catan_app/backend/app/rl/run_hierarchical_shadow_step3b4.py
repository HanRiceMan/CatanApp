"""Step 3B-4: 300-game, action-inert hierarchical routing shadow.

Every game is paired with a fresh Legacy run. The saved Step 3B-3 records are
also checked for the action/pre-roll learner boundaries; no source is edited.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from app.agents.ppo import PPOAgent
from app.domain.game import legal_actions

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .hierarchical_shadow_controller_step3b4 import (
    DEVELOPMENT_USAGE_ORDER, FAMILY_ORDER, HierarchicalRoute,
    classify_hierarchical_route, soft_plan_representation, with_soft_plan,
)
from .run_immediate_construction_audit_step3b3 import (
    SCHEMA as SOURCE_SCHEMA, SEED_BASES, _action_family, _head_digest,
)
from .run_strategic_gate_step3a14 import CHAMPION_ID, ROBBER_ID, _strict_parity
from .strategic_gate_rollout_step3a14 import run_strategic_episode
from .strategic_gate_step3a13 import frozen_snapshot
from .victory_plan_context import build_victory_plan_context
from .victory_plan_policy import VictoryPlanShadowHeads, encode_victory_plan_features


SCHEMA = "hierarchical_shadow_step3b4_v1"
_STRATEGIC_ROUTES = {
    HierarchicalRoute.PRE_ROLL_DEV_USAGE,
    HierarchicalRoute.ACTION_DEV_USAGE,
    HierarchicalRoute.CONSTRUCTION_COMMIT,
    HierarchicalRoute.CITY_VALUE_COMPARISON,
    HierarchicalRoute.DUAL_CONSTRUCTION_COMPARISON,
    HierarchicalRoute.INVESTMENT,
}
_BUILD_ROUTE = {
    "SETTLEMENT_ONLY": HierarchicalRoute.CONSTRUCTION_COMMIT,
    "CITY_ONLY": HierarchicalRoute.CITY_VALUE_COMPARISON,
    "BOTH": HierarchicalRoute.DUAL_CONSTRUCTION_COMPARISON,
    "NONE": HierarchicalRoute.INVESTMENT,
}


def _read_gzip(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_gzip(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def _record(env, action, _decision, _trace_index, *, champion: PPOAgent,
            heads: VictoryPlanShadowHeads, seed: int, profile: str, seat: int,
            records: list[dict]) -> None:
    game = env.game
    player_id = env.learning_player_id
    before = deepcopy(game)
    legal = legal_actions(game, player_id)
    decision = classify_hierarchical_route(game, player_id, legal=legal)
    if decision.route in _STRATEGIC_ROUTES:
        plan_context = build_victory_plan_context(game, player_id)
        features = encode_victory_plan_features(plan_context)
        snapshot = frozen_snapshot(champion, game, player_id)
        with torch.no_grad():
            output = heads(
                snapshot.actor_latent,
                torch.tensor(snapshot.runtime_context.copy()).unsqueeze(0),
                torch.tensor(features.global_features.copy()).unsqueeze(0),
                torch.tensor(features.candidate_features.copy()).unsqueeze(0),
                torch.tensor(features.candidate_mask.copy()).unsqueeze(0),
            )
        soft = soft_plan_representation(plan_context, output)
        if soft.available:
            if not np.allclose(soft.expected_priorities, (0.5,) * 4, atol=1e-6):
                raise AssertionError("Zero-initialized Priority is not neutral")
            valid_count = sum(soft.candidate_mask)
            if not np.allclose(
                [probability for probability, valid in zip(
                    soft.candidate_probabilities, soft.candidate_mask, strict=True) if valid],
                [1 / valid_count] * valid_count, atol=1e-6,
            ):
                raise AssertionError("Zero-initialized Plan distribution is not uniform")
        decision = with_soft_plan(decision, soft)
    if game != before or game.random_source.counter != before.random_source.counter:
        raise AssertionError("Hierarchical Shadow mutated GameState or Game RNG")
    if (decision.phase == "action" and not game.free_road_remaining
            and decision.route not in {HierarchicalRoute.IMMEDIATE_WIN,
                                       HierarchicalRoute.ACTION_DEV_USAGE}):
        if decision.route is not _BUILD_ROUTE[decision.build_class]:
            raise AssertionError("Normal build class routed incorrectly")
    records.append({
        "seed": seed, "profile": profile, "seat": seat,
        "turn": game.turn_number, "revision": game.revision,
        "policy_step": env.policy_steps,
        "selected_action": repr(action), "selected_action_family": _action_family(action),
        "decision": asdict(decision),
        "game_state_parity": True, "rng_parity": True,
    })


def _source_records(source: Path, profile: str, seed: int) -> tuple[dict, list[dict]]:
    payload = _read_gzip(source / "episodes" / f"{profile}_{seed}.json.gz")
    if payload["schema"] != SOURCE_SCHEMA:
        raise AssertionError("Step 3B-3 source schema mismatch")
    return payload["episode"], payload["records"]


def _check_source_match(source_episode: dict, source_records: list[dict],
                        shadow_episode, shadow_rows: list[dict]) -> None:
    current = [row for row in shadow_rows
               if row["decision"]["phase"] in {"action", "turn_pre_roll"}
               and not (row["decision"]["phase"] == "action"
                        and row["decision"]["build_class"] is None)]
    if len(current) != len(source_records):
        raise AssertionError("Step 3B-3 learner boundary count mismatch")
    for original, row in zip(source_records, current, strict=True):
        decision = row["decision"]
        if (original["phase"] != decision["phase"] or
                original["turn"] != row["turn"] or
                original["revision"] != row["revision"] or
                original["policy_step"] != row["policy_step"] or
                original["action"] != row["selected_action"]):
            raise AssertionError("Step 3B-3 action/turn trace mismatch")
        if original["phase"] == "action":
            if original["build_class"] != decision["build_class"]:
                raise AssertionError("Step 3B-3 build-class mismatch")
            if original["immediate_win"] != (decision["route"] == "IMMEDIATE_WIN"):
                raise AssertionError("Step 3B-3 ImmediateWin mismatch")
    if (source_episode["winner"] != shadow_episode.winner or
            source_episode["final_vp"] != shadow_episode.final_vp or
            source_episode["final_rank"] != shadow_episode.final_rank or
            source_episode["policy_steps"] != shadow_episode.policy_steps or
            source_episode["rng_counter"] != shadow_episode.rng_counter or
            not source_episode["parity"]):
        raise AssertionError("Step 3B-3 episode outcome/RNG mismatch")


def summarize(output: Path, source: Path, weights: dict[str, str]) -> dict:
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if (source_summary["schema"] != SOURCE_SCHEMA or source_summary["games"] != 300
            or source_summary["mutation_failures"] or source_summary["rng_failures"]
            or source_summary["parity_failures"]):
        raise AssertionError("Step 3B-3 source invariant failed")
    routes, phases, fallbacks, forced, build_to_route, route_by_phase = (
        Counter(), Counter(), Counter(), Counter(), Counter(), Counter())
    profile_routes = {profile: Counter() for profile in SEED_BASES}
    comparison_available = {route.value: Counter() for route in HierarchicalRoute}
    legal_available = {route.value: Counter() for route in HierarchicalRoute}
    dev_available = {route.value: Counter() for route in HierarchicalRoute}
    selected_by_route = {route.value: Counter() for route in HierarchicalRoute}
    soft_available = soft_unavailable = 0
    soft_available_by_route = Counter()
    soft_unavailable_by_route = Counter()
    priority_sum = np.zeros(4, dtype=np.float64)
    normal_action_immediate_count = 0
    protected_trade_without_rule_availability = 0
    representative = {}
    episodes = 0
    for profile, base in SEED_BASES.items():
        for index in range(100):
            path = output / "episodes" / f"{profile}_{base + index}.json.gz"
            if not path.exists():
                continue
            payload = _read_gzip(path)
            if payload["schema"] != SCHEMA or payload["weight_digests"] != weights:
                raise AssertionError("Step 3B-4 checkpoint schema or weights changed")
            if not payload["episode"]["parity"]:
                raise AssertionError("Step 3B-4 episode parity failed")
            episodes += 1
            for row in payload["records"]:
                decision = row["decision"]
                route = decision["route"]
                routes[route] += 1
                route_by_phase[f"{decision['phase']}->{route}"] += 1
                normal_action_immediate_count += int(
                    decision["phase"] == "action" and decision["build_class"] is not None
                    and route == "IMMEDIATE_WIN")
                profile_routes[profile][route] += 1
                phases[decision["phase"]] += 1
                if decision["fallback_route"]:
                    fallbacks[decision["fallback_route"]] += 1
                if decision["forced_reason"]:
                    forced[decision["forced_reason"]] += 1
                if decision["build_class"] and route != "IMMEDIATE_WIN":
                    build_to_route[f"{decision['build_class']}->{route}"] += 1
                selected_by_route[route][row["selected_action_family"]] += 1
                protected_trade_without_rule_availability += int(
                    row["selected_action_family"] == "PLAYER_TRADE"
                    and not decision["player_trade_rule_available"])
                for family, available in zip(FAMILY_ORDER, decision["comparison_family_mask"],
                                             strict=True):
                    comparison_available[route][family] += int(available)
                for family, available in zip(FAMILY_ORDER, decision["legal_family_mask"],
                                             strict=True):
                    legal_available[route][family] += int(available)
                for option, available in zip(DEVELOPMENT_USAGE_ORDER,
                                             decision["development_usage_mask"], strict=True):
                    dev_available[route][option] += int(available)
                soft = decision["soft_plan"]
                if soft is not None:
                    soft_available += int(soft["available"])
                    soft_unavailable += int(not soft["available"])
                    if soft["available"]:
                        soft_available_by_route[route] += 1
                        priority_sum += np.asarray(soft["expected_priorities"], dtype=np.float64)
                    else:
                        soft_unavailable_by_route[route] += 1
                representative.setdefault(route, {
                    "profile": profile, "seed": row["seed"], "seat": row["seat"],
                    "turn": row["turn"], "revision": row["revision"],
                    "policy_step": row["policy_step"], "build_class": decision["build_class"],
                    "comparison_family_mask": decision["comparison_family_mask"],
                    "soft_plan_features": (soft["weighted_features"] if soft else None),
                    "expected_priorities": (soft["expected_priorities"] if soft else None),
                })
    if episodes == 300:
        for build_class, route in _BUILD_ROUTE.items():
            if build_to_route[f"{build_class}->{route.value}"] != source_summary["class_counts"][build_class]:
                raise AssertionError(f"Full-shadow routing coverage failed: {build_class}")
        if (sum(routes.values()) != sum(source_summary["learner_phase_counts"].values())
                or normal_action_immediate_count !=
                source_summary["immediate_win_excluded"]
                or protected_trade_without_rule_availability):
            raise AssertionError("Full-shadow learner boundary/ImmediateWin count drift")
    return {
        "schema": SCHEMA, "source_schema": SOURCE_SCHEMA,
        "games": episodes, "seed_bases": SEED_BASES,
        "profile_seat": {f"{profile}/seat{seat}": sum(
            (output / "episodes" / f"{profile}_{base + i}.json.gz").exists()
            for i in range(100) if i % 4 + 1 == seat)
            for profile, base in SEED_BASES.items() for seat in range(1, 5)},
        "route_counts": dict(routes),
        "route_by_phase": dict(route_by_phase),
        "normal_action_immediate_win_count": normal_action_immediate_count,
        "protected_trade_without_rule_availability": protected_trade_without_rule_availability,
        "route_by_profile": {profile: dict(value) for profile, value in profile_routes.items()},
        "phase_counts": dict(phases), "fallback_route_counts": dict(fallbacks),
        "forced_reasons": dict(forced), "build_class_to_route": dict(build_to_route),
        "comparison_family_available_by_route": {
            route: dict(value) for route, value in comparison_available.items()},
        "legal_family_available_by_route": {
            route: dict(value) for route, value in legal_available.items()},
        "development_usage_available_by_route": {
            route: dict(value) for route, value in dev_available.items()},
        "selected_action_family_by_route_diagnostic": {
            route: dict(value) for route, value in selected_by_route.items()},
        "soft_plan_available_states": soft_available,
        "soft_plan_unavailable_states": soft_unavailable,
        "soft_plan_available_by_route": dict(soft_available_by_route),
        "soft_plan_unavailable_by_route": dict(soft_unavailable_by_route),
        "expected_priority_mean": (priority_sum / soft_available).tolist()
        if soft_available else None,
        "representative": representative,
        "game_state_mutations": 0, "rng_mutations": 0,
        "behavior_parity_games": episodes, "optimizer_created": False,
        "weight_digests_before_and_after": weights,
    }


def collect(source: Path, output: Path, *, profile: str | None = None,
            start_index: int = 0, limit: int = 100) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)
    champion = PPOAgent(CHAMPION_ID, device="cpu")
    robber = PPOAgent(ROBBER_ID, device="cpu")
    champion.model.policy.requires_grad_(False)
    latent_dim = champion.model.policy.mlp_extractor.latent_dim_pi
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3302)
        heads = VictoryPlanShadowHeads(latent_dim)
    heads.eval()
    weights = {"champion": _weight_digest(champion),
               "robber": _weight_digest(robber),
               "plan_priority_head": _head_digest(heads)}
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if weights != source_summary["weight_digests_after"]:
        raise AssertionError("Parent Champion/Robber/Plan Head digest drift")
    names = (profile,) if profile is not None else tuple(SEED_BASES)
    for name in names:
        if name not in SEED_BASES:
            raise ValueError("Unknown profile")
        for index in range(start_index, min(limit, 100)):
            seed = SEED_BASES[name] + index
            path = output / "episodes" / f"{name}_{seed}.json.gz"
            if path.exists():
                continue
            seat = index % 4 + 1
            original_episode, original_records = _source_records(source, name, seed)
            baseline = run_strategic_episode(champion, robber, seed=seed, seat=seat,
                                             profile=name, mode="legacy")
            records: list[dict] = []
            observer = lambda *args: _record(
                *args, champion=champion, heads=heads, seed=seed, profile=name,
                seat=seat, records=records)
            shadow = run_strategic_episode(
                champion, robber, seed=seed, seat=seat, profile=name, mode="legacy",
                pre_action_audit_observer=observer)
            if not _strict_parity(baseline, shadow) or not baseline.terminal or baseline.truncated:
                raise AssertionError(f"Step 3B-4 behavior parity failed: {name}/{seed}")
            _check_source_match(original_episode, original_records, shadow, records)
            if weights != {"champion": _weight_digest(champion),
                           "robber": _weight_digest(robber),
                           "plan_priority_head": _head_digest(heads)}:
                raise AssertionError("Step 3B-4 weight digest drift")
            _write_gzip(path, {"schema": SCHEMA, "weight_digests": weights,
                               "episode": {"seed": seed, "profile": name, "seat": seat,
                                           "winner": baseline.winner,
                                           "final_vp": baseline.final_vp,
                                           "final_rank": baseline.final_rank,
                                           "policy_steps": baseline.policy_steps,
                                           "rng_counter": baseline.rng_counter,
                                           "parity": True, "terminal": baseline.terminal,
                                           "truncated": baseline.truncated},
                               "records": records})
            print(f"{name} {index + 1}/100 seed={seed} "
                  f"boundaries={len(records)} parity=ok", flush=True)
    if weights != {"champion": _weight_digest(champion),
                   "robber": _weight_digest(robber),
                   "plan_priority_head": _head_digest(heads)}:
        raise AssertionError("Step 3B-4 final weight digest drift")
    summary = summarize(output, source, weights)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
        indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(SEED_BASES))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    summary = collect(args.source, args.output, profile=args.profile,
                      start_index=args.start_index, limit=args.limit)
    print(json.dumps({key: summary[key] for key in
                      ("games", "route_counts", "build_class_to_route")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
