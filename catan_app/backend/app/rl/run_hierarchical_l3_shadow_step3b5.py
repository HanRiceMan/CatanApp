"""Step 3B-5 fixed 36-game, action-inert L1/L2/bridge/L3 shadow replay."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from app.agents.ppo import PPOAgent
from app.domain.game import legal_actions

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .hierarchical_l3_shadow_step3b5 import (
    L3ShadowActor, SCHEMA as POLICY_SCHEMA, differentiable_soft_plan,
    encode_l3_decision_facts,
)
from .hierarchical_shadow_controller_step3b4 import (
    HierarchicalRoute, classify_hierarchical_route, soft_plan_representation,
)
from .run_hierarchical_shadow_step3b4 import (
    SCHEMA as SOURCE_SCHEMA, _STRATEGIC_ROUTES, _read_gzip, _write_gzip,
)
from .run_immediate_construction_audit_step3b3 import SEED_BASES, _action_family, _head_digest
from .run_strategic_gate_step3a14 import CHAMPION_ID, ROBBER_ID, _strict_parity
from .strategic_gate_rollout_step3a14 import run_strategic_episode
from .strategic_gate_step3a13 import frozen_snapshot
from .victory_plan_context import build_victory_plan_context
from .victory_plan_policy import VictoryPlanShadowHeads, encode_victory_plan_features


SCHEMA = "hierarchical_l3_shadow_step3b5_v1"
SEEDS_PER_PROFILE = 12
_NORMAL_L3_ROUTES = {
    HierarchicalRoute.CONSTRUCTION_COMMIT,
    HierarchicalRoute.CITY_VALUE_COMPARISON,
    HierarchicalRoute.DUAL_CONSTRUCTION_COMPARISON,
    HierarchicalRoute.INVESTMENT,
}


def _record(env, action, _decision, _trace_index, *, champion: PPOAgent,
            heads: VictoryPlanShadowHeads, l3: L3ShadowActor, seed: int,
            profile: str, seat: int, records: list[dict]) -> None:
    game = env.game
    before = deepcopy(game)
    start = perf_counter()
    decision = classify_hierarchical_route(
        game, env.learning_player_id, legal=legal_actions(game, env.learning_player_id))
    shadow = None
    if decision.route in _STRATEGIC_ROUTES:
        plan_context = build_victory_plan_context(game, env.learning_player_id)
        encoded = encode_victory_plan_features(plan_context)
        snapshot = frozen_snapshot(champion, game, env.learning_player_id)
        actor_latent = snapshot.actor_latent
        runtime_context = torch.tensor(snapshot.runtime_context.copy()).unsqueeze(0)
        global_features = torch.tensor(encoded.global_features.copy()).unsqueeze(0)
        candidate_features = torch.tensor(encoded.candidate_features.copy()).unsqueeze(0)
        candidate_mask = torch.tensor(encoded.candidate_mask.copy()).unsqueeze(0)
        with torch.no_grad():
            plan_output = heads(actor_latent, runtime_context, global_features,
                                candidate_features, candidate_mask)
            bridge = differentiable_soft_plan(plan_output, candidate_features)
            diagnostic = soft_plan_representation(plan_context, plan_output)
            facts = encode_l3_decision_facts([decision])
            l3_output = l3(actor_latent, runtime_context, global_features, bridge, facts)
        if bridge.available.tolist() != [diagnostic.available]:
            raise AssertionError("3B-4 / 3B-5 Plan availability mismatch")
        if diagnostic.available:
            if not np.allclose(bridge.weighted_features[0],
                               diagnostic.weighted_features, atol=2e-6):
                raise AssertionError("3B-4 / 3B-5 soft Plan mismatch")
            if not np.allclose(bridge.expected_priorities[0],
                               diagnostic.expected_priorities, atol=2e-6):
                raise AssertionError("3B-4 / 3B-5 expected Priority mismatch")
        if bool(l3_output.available[0]) != (decision.route in _NORMAL_L3_ROUTES
                                            and any(decision.comparison_family_mask)):
            raise AssertionError("L3 route availability mismatch")
        probabilities = l3_output.probabilities[0]
        if torch.any(probabilities[~l3_output.comparison_mask[0]] != 0):
            raise AssertionError("Masked family has nonzero probability")
        if l3_output.available[0]:
            if not np.isclose(float(probabilities.sum()), 1, atol=1e-6):
                raise AssertionError("L3 family probabilities do not sum to one")
            valid_count = int(l3_output.comparison_mask[0].sum())
            if not torch.allclose(probabilities[l3_output.comparison_mask[0]],
                                  torch.full((valid_count,), 1 / valid_count), atol=1e-6):
                raise AssertionError("Untrained L3 is not neutral")
        elif torch.any(probabilities != 0):
            raise AssertionError("Unavailable L3 has probability mass")
        shadow = {
            "plan_available": bool(bridge.available[0]),
            "soft_plan_features": bridge.weighted_features[0].tolist()
            if bridge.available[0] else None,
            "expected_priorities": bridge.expected_priorities[0].tolist()
            if bridge.available[0] else None,
            "bridge_matches_3b4": True,
            "comparison_available": bool(l3_output.available[0]),
            "candidate_family_mask": l3_output.comparison_mask[0].tolist(),
            "raw_l3_logits": l3_output.raw_logits[0].tolist(),
            "masked_l3_probabilities": probabilities.tolist(),
            "l3_entropy": float(l3_output.entropy[0]),
            "plan_candidate_count": len(plan_context.candidates),
            "valid_plan_candidate_count": sum(plan_context.candidate_mask),
        }
    duration = perf_counter() - start
    if game != before or game.random_source.counter != before.random_source.counter:
        raise AssertionError("3B-5 Shadow mutated GameState / Game RNG")
    records.append({
        "profile": profile, "seed": seed, "seat": seat,
        "turn": game.turn_number, "revision": game.revision,
        "policy_step": env.policy_steps, "selected_action": repr(action),
        "selected_action_family": _action_family(action),
        "route": decision.route.value, "phase": decision.phase,
        "build_class": decision.build_class,
        "legal_family_mask": decision.legal_family_mask,
        "comparison_family_mask": decision.comparison_family_mask,
        "development_usage_mask": decision.development_usage_mask,
        "fallback_route": decision.fallback_route.value if decision.fallback_route else None,
        "construction_commit_active": decision.construction_commit_active,
        "shadow": shadow, "forward_seconds": duration,
        "game_state_parity": True, "rng_parity": True,
    })


def _check_source(source_rows: list[dict], rows: list[dict]) -> None:
    if len(source_rows) != len(rows):
        raise AssertionError("3B-4/3B-5 learner boundary count mismatch")
    for parent, row in zip(source_rows, rows, strict=True):
        old = parent["decision"]
        if (parent["selected_action"] != row["selected_action"]
                or parent["selected_action_family"] != row["selected_action_family"]
                or parent["turn"] != row["turn"]
                or parent["revision"] != row["revision"]
                or parent["policy_step"] != row["policy_step"]
                or old["route"] != row["route"]
                or old["phase"] != row["phase"]
                or old["build_class"] != row["build_class"]
                or old["legal_family_mask"] != list(row["legal_family_mask"])
                or old["comparison_family_mask"] != list(row["comparison_family_mask"])):
            raise AssertionError("3B-4/3B-5 boundary, route, or Action drift")


def summarize(output: Path, weights: dict[str, str], parameter_counts: dict[str, int]) -> dict:
    routes, l3_by_route, profile_routes = Counter(), Counter(), {
        profile: Counter() for profile in SEED_BASES}
    shadow_forwards = l3_available = plan_available = 0
    observer_seconds = shadow_forward_seconds = 0.0
    parity_games = 0
    representative = {}
    for profile, base in SEED_BASES.items():
        for index in range(SEEDS_PER_PROFILE):
            path = output / "episodes" / f"{profile}_{base + index}.json.gz"
            if not path.exists():
                continue
            payload = _read_gzip(path)
            if payload["schema"] != SCHEMA or payload["weight_digests"] != weights:
                raise AssertionError("3B-5 checkpoint schema or weights drift")
            if not payload["episode"]["parity"]:
                raise AssertionError("3B-5 behavior parity failure")
            parity_games += 1
            for row in payload["records"]:
                route = row["route"]
                routes[route] += 1
                profile_routes[profile][route] += 1
                observer_seconds += row["forward_seconds"]
                if row["shadow"] is not None:
                    shadow_forward_seconds += row["forward_seconds"]
                    shadow_forwards += 1
                    plan_available += int(row["shadow"]["plan_available"])
                    l3_available += int(row["shadow"]["comparison_available"])
                    l3_by_route[route] += int(row["shadow"]["comparison_available"])
                representative.setdefault(route, {
                    "profile": profile, "seed": row["seed"], "seat": row["seat"],
                    "turn": row["turn"], "revision": row["revision"],
                    "build_class": row["build_class"], "shadow": row["shadow"],
                })
    return {
        "schema": SCHEMA, "policy_schema": POLICY_SCHEMA,
        "games": parity_games, "seed_bases": SEED_BASES,
        "seeds_per_profile": SEEDS_PER_PROFILE,
        "profile_seat": {f"{profile}/seat{seat}": sum(
            (output / "episodes" / f"{profile}_{base + index}.json.gz").exists()
            for index in range(SEEDS_PER_PROFILE) if index % 4 + 1 == seat)
            for profile, base in SEED_BASES.items() for seat in range(1, 5)},
        "route_counts": dict(routes),
        "route_by_profile": {key: dict(value) for key, value in profile_routes.items()},
        "shadow_forward_states": shadow_forwards,
        "plan_available_states": plan_available,
        "l3_available_states": l3_available,
        "l3_available_by_route": dict(l3_by_route),
        "observer_seconds_total": observer_seconds,
        "shadow_forward_seconds_total": shadow_forward_seconds,
        "forward_seconds_per_shadow_state": shadow_forward_seconds / shadow_forwards
        if shadow_forwards else None,
        "parameter_counts": parameter_counts,
        "representative": representative,
        "behavior_parity_games": parity_games,
        "game_state_mutations": 0, "rng_mutations": 0,
        "bridge_diagnostic_mismatches": 0, "masked_probability_failures": 0,
        "optimizer_created": False,
        "weight_digests_before_and_after": weights,
    }


def collect(source: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)
    champion = PPOAgent(CHAMPION_ID, device="cpu")
    robber = PPOAgent(ROBBER_ID, device="cpu")
    champion.model.policy.requires_grad_(False)
    latent_dim = champion.model.policy.mlp_extractor.latent_dim_pi
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3302)
        heads = VictoryPlanShadowHeads(latent_dim).eval()
        torch.manual_seed(3505)
        l3 = L3ShadowActor(latent_dim).eval()
    weights = {
        "champion": _weight_digest(champion), "robber": _weight_digest(robber),
        "plan_priority_head": _head_digest(heads), "l3_head": _head_digest(l3),
    }
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if (source_summary["schema"] != SOURCE_SCHEMA
            or source_summary["games"] != 300
            or any(weights[key] != source_summary["weight_digests_before_and_after"][key]
                   for key in ("champion", "robber", "plan_priority_head"))):
        raise AssertionError("3B-4 source or frozen weight digest mismatch")
    parameter_counts = {
        "plan_priority_head": sum(parameter.numel() for parameter in heads.parameters()),
        "l3_encoder": sum(parameter.numel() for parameter in l3.encoder.parameters()),
        "l3_family_head": sum(parameter.numel() for parameter in l3.family_head.parameters()),
        "l3_total": sum(parameter.numel() for parameter in l3.parameters()),
        "bridge": 0,
    }
    for profile, base in SEED_BASES.items():
        for index in range(SEEDS_PER_PROFILE):
            seed = base + index
            path = output / "episodes" / f"{profile}_{seed}.json.gz"
            if path.exists():
                continue
            seat = index % 4 + 1
            source_payload = _read_gzip(source / "episodes" / f"{profile}_{seed}.json.gz")
            baseline = run_strategic_episode(champion, robber, seed=seed, seat=seat,
                                             profile=profile, mode="legacy")
            rows: list[dict] = []
            observer = lambda *args: _record(
                *args, champion=champion, heads=heads, l3=l3, seed=seed,
                profile=profile, seat=seat, records=rows)
            shadow = run_strategic_episode(
                champion, robber, seed=seed, seat=seat, profile=profile, mode="legacy",
                pre_action_audit_observer=observer)
            if not _strict_parity(baseline, shadow) or not baseline.terminal or baseline.truncated:
                raise AssertionError(f"3B-5 behavior parity failed: {profile}/{seed}")
            source_episode = source_payload["episode"]
            if (source_episode["winner"] != baseline.winner
                    or source_episode["final_vp"] != baseline.final_vp
                    or source_episode["final_rank"] != baseline.final_rank
                    or source_episode["policy_steps"] != baseline.policy_steps
                    or source_episode["rng_counter"] != baseline.rng_counter):
                raise AssertionError("3B-4/3B-5 final outcome or RNG drift")
            _check_source(source_payload["records"], rows)
            if weights != {
                "champion": _weight_digest(champion), "robber": _weight_digest(robber),
                "plan_priority_head": _head_digest(heads), "l3_head": _head_digest(l3),
            }:
                raise AssertionError("3B-5 frozen/Head weight drift")
            _write_gzip(path, {
                "schema": SCHEMA, "weight_digests": weights,
                "episode": {"seed": seed, "profile": profile, "seat": seat,
                            "winner": baseline.winner, "final_vp": baseline.final_vp,
                            "final_rank": baseline.final_rank,
                            "policy_steps": baseline.policy_steps,
                            "rng_counter": baseline.rng_counter,
                            "terminal": baseline.terminal, "truncated": baseline.truncated,
                            "parity": True},
                "records": rows,
            })
            print(f"{profile} {index + 1}/{SEEDS_PER_PROFILE} seed={seed} "
                  f"boundaries={len(rows)} parity=ok", flush=True)
    summary = summarize(output, weights, parameter_counts)
    if summary["games"] != SEEDS_PER_PROFILE * len(SEED_BASES):
        raise AssertionError("3B-5 fixed 36-game shadow incomplete")
    (output / "summary.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = collect(args.source, args.output)
    print(json.dumps({key: summary[key] for key in
                      ("games", "route_counts", "shadow_forward_states",
                       "l3_available_states")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
