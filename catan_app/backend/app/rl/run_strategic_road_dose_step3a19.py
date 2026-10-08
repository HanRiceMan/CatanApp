"""Read-only Step 3A-19 road-edge interventions and delegation dose audit.

The audited Gate is the frozen T=2 pre-update artifact. This module contains
no optimizer, and does not alter reward, masks, executors or model weights.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from app.domain.actions import BuildRoadAction
from app.domain.game import _longest_road_length, apply_action, legal_road_ids, legal_settlement_ids

from .action_space import action_to_id, get_action_mask
from .construction_city_counterfactual import capture_boundary
from .expansion_planner import _routes_from_network, _settlement_candidates
from .run_strategic_gate_step3a14 import _episode_record
from .run_strategic_gate_step3a16 import EVAL_SEED_START, _agents, _latent_dim
from .run_strategic_prior_step3a18 import _outcome, _read, _stats, _write
from .reward import actual_score
from .strategic_action_surface import _road_purposes, family_of
from .strategic_gate_rollout_step3a14 import run_strategic_episode
from .strategic_gate_ppo_step3a16 import load_strategic_artifact


SOURCE = Path("experiments/strategic_gate_step3a16_20261005")
OUTPUT = Path("experiments/strategic_road_dose_step3a19_20261006")
PROFILES = ("rule", "champion", "mixed")
RATES = (0.0, 0.1, 0.2, 0.5, 1.0)
EDGE_RE = re.compile(r"^BuildRoadAction\(edge_id=(\d+)\)$")


def _edge(value: str) -> int:
    match = EDGE_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"Not a Road Action: {value}")
    return int(match.group(1))


def _seed_rows():
    for pindex, profile in enumerate(PROFILES):
        for index in range(12):
            yield profile, EVAL_SEED_START + pindex * 1000 + index, index % 4 + 1


def _models():
    champion, robber = _agents()
    gate, critic, manifest = load_strategic_artifact(
        SOURCE / "Strategic-T2-preupdate", _latent_dim(champion))
    if manifest["temperature"] != 2.0:
        raise AssertionError("Road/dose audit requires frozen T=2 pre-update")
    return champion, robber, gate, critic


def _routes(game, player_id: int) -> dict[int, int]:
    routes = _routes_from_network(game, player_id)
    return {vertex: routes[vertex][0] for vertex in _settlement_candidates(game)
            if vertex in routes}


def edge_diagnostics(game, player_id: int, edge_id: int, candidate_logits) -> dict[str, Any]:
    """Structural facts only; expansion means a shortest path of <=3 roads."""
    mask = get_action_mask(game, player_id)
    action = BuildRoadAction(edge_id)
    legal = bool(mask[action_to_id(action)])
    if not legal:
        raise AssertionError("Compared Road edge is not legal at the shared state")
    legal_edges = [BuildRoadAction(edge) for edge in legal_road_ids(game, player_id, initial=False)
                   if mask[action_to_id(BuildRoadAction(edge))]]
    logits = {item.edge_id: float(candidate_logits[action_to_id(item)])
              for item in legal_edges}
    order = sorted(logits, key=lambda edge: (-logits[edge], edge))
    purposes = _road_purposes(game, player_id, legal_edges)
    before_routes = _routes(game, player_id)
    before_length = _longest_road_length(game, player_id)
    clone = deepcopy(game)
    rng_before = clone.random_source.counter
    apply_action(clone, player_id, action, clone.revision, "road-edge-diagnostic")
    after_routes = _routes(clone, player_id)
    if clone.random_source.counter != rng_before:
        raise AssertionError("Deterministic Road diagnostic consumed game RNG")
    newly_within_three = sorted(vertex for vertex, cost in after_routes.items()
                                if cost <= 3 and before_routes.get(vertex, 10_000) > 3)
    improved_vertices = sorted(vertex for vertex, cost in after_routes.items()
                               if cost < before_routes.get(vertex, 10_000))
    expansion = edge_id in purposes["expansion_edges"]
    longest = edge_id in purposes["longest_progress_edges"]
    return {
        "edge_id": edge_id, "legal": legal, "frozen_candidate_logit": logits[edge_id],
        "candidate_logit_rank": order.index(edge_id) + 1,
        "legal_road_candidates_before": len(order),
        "expansion_step": expansion, "longest_progress": longest,
        "structural_class": ("both" if expansion and longest else "expansion_only"
                             if expansion else "longest_only" if longest else "neither"),
        "longest_road_before": before_length,
        "longest_road_after": _longest_road_length(clone, player_id),
        "legal_road_count_after": len(legal_road_ids(clone, player_id, initial=False)),
        "legal_settlement_count_after": len(legal_settlement_ids(clone, player_id, initial=False)),
        "newly_reachable_empty_vertices_within_three_roads": newly_within_three,
        "shortest_path_improved_vertices": improved_vertices,
        "minimum_additional_roads_before": min(before_routes.values(), default=None),
        "minimum_additional_roads_after": min(after_routes.values(), default=None),
    }


def _window_observer(start_step: int, rows: list[dict[str, Any]]):
    def observe(env, action, _decision, _trace_index):
        relative = env.policy_steps - start_step
        if relative >= 30:
            return
        row = {"relative_policy_step": relative, "family": family_of(action)}
        # The action cascade needs all 30 learner steps; structural access is
        # audited only at the first five boundaries, which are reported below.
        if len(rows) < 5:
            player_id = env.learning_player_id
            routes = _routes(env.game, player_id)
            row.update({
                "legal_settlement_count": len(legal_settlement_ids(
                    env.game, player_id, initial=False)),
                "minimum_additional_roads": min(routes.values(), default=None),
                "reachable_empty_vertices_within_three_roads": sum(
                    value <= 3 for value in routes.values()),
            })
        rows.append(row)
    return observe


def _window_summary(rows):
    return {"action_counts": dict(Counter(row["family"] for row in rows)),
            "settlement_access": [
                {key: row[key] for key in ("relative_policy_step", "legal_settlement_count",
                                           "minimum_additional_roads",
                                           "reachable_empty_vertices_within_three_roads")}
                for row in rows[:5]]}


def road_audit(output: Path, *, max_states: int = 24, future_seeds: int = 8,
               full_aux_states: int = 0) -> dict[str, Any]:
    champion, robber, gate, critic = _models()
    all_rows: list[dict[str, Any]] = []
    captured: list[tuple[dict[str, Any], Any]] = []
    for profile, seed, seat in _seed_rows():
        rows = []

        def observe(env, _champion, snapshot, _mask, record):
            if (record["selected_family"] != "BUILD_ROAD" or
                    record["champion_counterfactual_family"] != "BUILD_ROAD"):
                return
            b_edge = _edge(record["concrete_action"])
            c_edge = _edge(record["champion_counterfactual_action"])
            game = env.game
            player_id = env.learning_player_id
            b = edge_diagnostics(game, player_id, b_edge, snapshot.candidate_logits)
            c = edge_diagnostics(game, player_id, c_edge, snapshot.candidate_logits)
            row = {"state_id": f"{profile}-{seed}-{record['pre_policy_step']}",
                   "profile": profile, "seed": seed, "seat": seat,
                   "policy_step": record["pre_policy_step"], "turn": record["turn"],
                   "score": actual_score(game, player_id),
                   "same_edge": b_edge == c_edge,
                   "b_edge": b, "champion_edge": c,
                   "gate_probability": record["selected_probability"],
                   "candidate_mask": record["candidate_mask"]}
            rows.append(row)
            if b_edge != c_edge:
                captured.append((row, capture_boundary(env)))

        episode = run_strategic_episode(
            champion, robber, seed=seed, seat=seat, profile=profile,
            mode="strategic", delegation_probability=1.0,
            delegation_seed=seed + 12345, gate_sampling_seed=seed + 54321,
            stochastic_gate=True, prior_temperature=2.0,
            strategic_gate=gate, strategic_critic=critic,
            decision_audit_observer=observe)
        source = _read(SOURCE / f"eval-pre100-{profile}-{seed}.json")
        if (episode.final_vp != source["final_vp"] or episode.winner != source["winner"]
                or episode.policy_steps != source["policy_steps"]
                or episode.rng_counter != source["rng_counter"]
                or len(episode.decisions) != source["actual_delegated_count"]):
            raise AssertionError(f"Frozen T=2 replay mismatch: {profile}/{seed}")
        all_rows.extend(rows)
        _write(output / "road_games" / f"{profile}-{seed}.json", rows)
        print(f"road {profile} {seed}: both={len(rows)} different={sum(not x['same_edge'] for x in rows)}", flush=True)
    _write(output / "road_disagreements.json", all_rows)
    different = [row for row in all_rows if not row["same_edge"]]
    # Balance profile and structural class, then spread within each stratum by
    # score. No outcome/return is inspected when selecting intervention states.
    selected: list[tuple[dict[str, Any], Any]] = []
    classes = ("expansion_only", "longest_only", "both", "neither")
    if max_states == 24:
        for profile in PROFILES:
            for structural_class in classes:
                pool = sorted((item for item in captured
                               if item[0]["profile"] == profile and
                               item[0]["b_edge"]["structural_class"] == structural_class),
                              key=lambda item: (item[0]["score"], item[0]["seed"],
                                                item[0]["policy_step"]))
                if len(pool) < 2:
                    raise AssertionError(f"Too few Road states in {profile}/{structural_class}")
                selected.extend((pool[len(pool) // 3], pool[2 * len(pool) // 3]))
    else:
        pools = {profile: sorted((item for item in captured if item[0]["profile"] == profile),
                                 key=lambda item: (item[0]["score"],
                                                   item[0]["b_edge"]["structural_class"],
                                                   item[0]["seed"], item[0]["policy_step"]))
                 for profile in PROFILES}
        while len(selected) < min(max_states, len(captured)) and any(pools.values()):
            for profile in PROFILES:
                if pools[profile] and len(selected) < min(max_states, len(captured)):
                    selected.append(pools[profile].pop(len(pools[profile]) // 2))
    if full_aux_states < 0 or full_aux_states > len(selected):
        raise ValueError("full_aux_states must be between zero and selected states")
    full_aux_indices = set(np.linspace(
        0, len(selected) - 1, full_aux_states, dtype=int).tolist()) if full_aux_states else set()
    pairs: list[dict[str, Any]] = []
    for state_index, (row, boundary) in enumerate(selected):
        for future_index in range(future_seeds):
            path = output / "road_pairs" / f"{row['state_id']}-{future_index}.json"
            future_seed = 319000000 + row["seed"] * 17 + future_index
            if path.exists():
                pair = _read(path)
            else:
                pair = {"state_id": row["state_id"], "profile": row["profile"],
                        "seat": row["seat"], "future_seed": future_seed,
                        "b_edge": row["b_edge"]["edge_id"],
                        "champion_edge": row["champion_edge"]["edge_id"],
                        "branches": {}}
            if (pair["state_id"] != row["state_id"] or pair["future_seed"] != future_seed
                    or pair["b_edge"] != row["b_edge"]["edge_id"] or
                    pair["champion_edge"] != row["champion_edge"]["edge_id"]):
                raise AssertionError("Road pair checkpoint does not match selected state")
            shared = dict(champion_policy=champion, robber_policy=robber,
                          seed=row["seed"], seat=row["seat"], profile=row["profile"],
                          initial_boundary=boundary, continuation_seed=future_seed,
                          delegation_seed=future_seed + 12345,
                          gate_sampling_seed=future_seed + 54321)
            desired = ("safety", "full") if state_index in full_aux_indices else ("safety",)
            missing = [controller for controller in desired if controller not in pair["branches"]]
            for controller in missing:
                outcomes = {}
                for edge_name, edge_id in (("b", row["b_edge"]["edge_id"]),
                                           ("champion", row["champion_edge"]["edge_id"])):
                    window: list[dict[str, Any]] = []
                    episode = run_strategic_episode(
                        **shared, mode=("safety" if controller == "safety" else "strategic"),
                        delegation_probability=(0.0 if controller == "safety" else 1.0),
                        stochastic_gate=True, prior_temperature=2.0,
                        strategic_gate=gate, strategic_critic=critic,
                        first_action_override=BuildRoadAction(edge_id),
                        pre_action_audit_observer=_window_observer(
                            boundary.policy_steps, window))
                    if not episode.terminal or episode.truncated:
                        raise AssertionError("Road paired continuation did not terminate")
                    outcomes[edge_name] = {**_outcome(episode),
                                           "window30": _window_summary(window)}
                b, c = outcomes["b"], outcomes["champion"]
                pair["branches"][controller] = {
                    **outcomes, "delta_return": b["return"] - c["return"],
                    "delta_vp": b["vp"] - c["vp"],
                    "delta_rank": b["rank"] - c["rank"],
                    "winner_flip": b["winner"] != c["winner"]}
            if missing:
                _write(path, pair)
            pairs.append(pair)
            print(f"road-pair {state_index+1}/{len(selected)} future={future_index+1}/{future_seeds}", flush=True)
    _write(output / "road_pairs.json", pairs)
    summary = summarize_road(all_rows, pairs)
    _write(output / "road_summary.json", summary)
    return summary


def summarize_road(rows: list[dict], pairs: list[dict]) -> dict[str, Any]:
    different = [row for row in rows if not row["same_edge"]]
    by_state: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_state[pair["state_id"]].append(pair)
    result: dict[str, Any] = {
        "both_road": len(rows), "same_edge": len(rows) - len(different),
        "different_edge": len(different),
        "by_profile_seat": {f"{profile}/{seat}": {"both": sum(
            row["profile"] == profile and row["seat"] == seat for row in rows),
            "different": sum(row["profile"] == profile and row["seat"] == seat
                             for row in different)}
            for profile in PROFILES for seat in range(1, 5)},
        "structural_class_all": {key: dict(Counter(
            row[key]["structural_class"] for row in rows))
            for key in ("b_edge", "champion_edge")},
        "structural_class_different": {key: dict(Counter(
            row[key]["structural_class"] for row in different))
            for key in ("b_edge", "champion_edge")},
        "logit_rank_different": {key: _stats([
            row[key]["candidate_logit_rank"] for row in different])
            for key in ("b_edge", "champion_edge")},
        "candidate_logit_different": {key: _stats([
            row[key]["frozen_candidate_logit"] for row in different])
            for key in ("b_edge", "champion_edge")},
        "longest_gain_different": {key: sum(
            row[key]["longest_road_after"] > row[key]["longest_road_before"]
            for row in different) for key in ("b_edge", "champion_edge")},
        "new_access_different": {key: sum(bool(
            row[key]["newly_reachable_empty_vertices_within_three_roads"])
            for row in different) for key in ("b_edge", "champion_edge")},
        "legal_settlement_after_different": {key: _stats([
            row[key]["legal_settlement_count_after"] for row in different])
            for key in ("b_edge", "champion_edge")},
        "legal_road_after_different": {key: _stats([
            row[key]["legal_road_count_after"] for row in different])
            for key in ("b_edge", "champion_edge")},
        "minimum_roads_reduced_different": {key: sum(
            row[key]["minimum_additional_roads_before"] is not None and
            row[key]["minimum_additional_roads_after"] is not None and
            row[key]["minimum_additional_roads_after"] < row[key]["minimum_additional_roads_before"]
            for row in different) for key in ("b_edge", "champion_edge")},
        "paired_states": len(by_state), "future_seeds_per_state": min(
            (len(group) for group in by_state.values()), default=0),
        "paired_futures": len(pairs),
    }
    for controller in ("safety", "full"):
        subset = [pair["branches"][controller] for pair in pairs
                  if controller in pair["branches"]]
        state_rows = []
        for state_id, group in by_state.items():
            values = [pair["branches"][controller]["delta_return"] for pair in group
                      if controller in pair["branches"]]
            if values:
                state_rows.append({"state_id": state_id, "delta_return": _stats(values),
                                   "b_wins": sum(value > .02 for value in values),
                                   "champion_edge_wins": sum(value < -.02 for value in values),
                                   "ties": sum(abs(value) <= .02 for value in values)})
        result[controller] = {
            "pairs": len(subset),
            "delta_return": _stats([row["delta_return"] for row in subset]),
            "state_mean_delta": _stats([row["delta_return"]["mean"] for row in state_rows]),
            "b_favored_states": sum(row["delta_return"]["mean"] > .02 for row in state_rows),
            "champion_edge_favored_states": sum(row["delta_return"]["mean"] < -.02 for row in state_rows),
            "near_tie_states": sum(abs(row["delta_return"]["mean"]) <= .02 for row in state_rows),
            "stable_7_of_8": sum(max(row["b_wins"], row["champion_edge_wins"]) >= 7
                                  for row in state_rows),
            "unstable_2_each": sum(row["b_wins"] >= 2 and row["champion_edge_wins"] >= 2
                                  for row in state_rows),
            "winner_flips": sum(row["winner_flip"] for row in subset),
            "learner_wins": {branch: sum(row[branch]["won"] for row in subset)
                             for branch in ("b", "champion")},
            "mean_within_state_sample_variance": (float(np.mean([
                np.var([pair["branches"][controller]["delta_return"] for pair in group
                        if controller in pair["branches"]], ddof=1)
                for group in by_state.values() if sum(
                    controller in pair["branches"] for pair in group) > 1]))
                if any(sum(controller in pair["branches"] for pair in group) > 1
                       for group in by_state.values()) else None),
            "delta_vp": _stats([row["delta_vp"] for row in subset]),
            "delta_rank": _stats([row["delta_rank"] for row in subset]),
            "delta_steps": _stats([
                row["b"]["policy_steps"] - row["champion"]["policy_steps"]
                for row in subset]),
            "window30_action_counts": {key: dict(sum((Counter(
                row[branch]["window30"]["action_counts"])
                for row in subset), Counter())) for key, branch in
                (("b", "b"), ("champion_edge", "champion"))},
            "next_boundary_settlement_access": {
                branch: {field: _stats([
                    item[branch]["window30"]["settlement_access"][1][field]
                    for item in subset
                    if len(item[branch]["window30"]["settlement_access"]) > 1 and
                    item[branch]["window30"]["settlement_access"][1][field] is not None])
                    for field in ("legal_settlement_count",
                                  "minimum_additional_roads",
                                  "reachable_empty_vertices_within_three_roads")}
                for branch in ("b", "champion")},
            "state_rows": state_rows,
        }
    return result


def dose_audit(output: Path) -> dict[str, Any]:
    champion, robber, gate, critic = _models()
    records: dict[str, list[dict[str, Any]]] = {f"{rate:g}": [] for rate in RATES}
    for profile, seed, seat in _seed_rows():
        base_path = output / "dose_games" / "0" / f"{profile}-{seed}.json"
        safety_trace: list[tuple[int, Any]] = []
        if base_path.exists():
            base = _read(base_path)
            safety_trace = [(player_id, repr_action) for player_id, repr_action in base["action_trace"]]
        else:
            episode = run_strategic_episode(
                champion, robber, seed=seed, seat=seat, profile=profile,
                mode="safety")
            source = _read(SOURCE / f"eval-safety-{profile}-{seed}.json")
            if (episode.final_vp != source["final_vp"] or episode.winner != source["winner"]
                    or episode.policy_steps != source["policy_steps"]
                    or episode.rng_counter != source["rng_counter"]):
                raise AssertionError("0% source replay mismatch")
            base = {**_episode_record(episode), "action_trace": [
                [player_id, repr(action)] for player_id, action in episode.action_trace],
                "first_divergence": None}
            safety_trace = [(pid, action) for pid, action in base["action_trace"]]
            _write(base_path, base)
        records["0"].append(base)
        for rate in RATES[1:]:
            label = f"{rate:g}"
            path = output / "dose_games" / label / f"{profile}-{seed}.json"
            if path.exists():
                row = _read(path)
            else:
                first: dict[str, Any] = {}

                def observe(env, action, decision, trace_index):
                    if first or trace_index >= len(safety_trace):
                        return
                    if (env.learning_player_id, repr(action)) == safety_trace[trace_index]:
                        return
                    if decision is None or safety_trace[trace_index][0] != env.learning_player_id:
                        raise AssertionError("First dose divergence was not delegated Gate Action")
                    safe_repr = safety_trace[trace_index][1]
                    safe_family = ("BUILD_ROAD" if safe_repr.startswith("BuildRoadAction")
                                   else family_of_repr(safe_repr))
                    first.update({"trace_index": trace_index,
                                  "policy_step": env.policy_steps,
                                  "safety_family": safe_family,
                                  "gate_family": family_of(action),
                                  "same_family_different_action": safe_family == family_of(action),
                                  "safety_action": safe_repr, "gate_action": repr(action)})

                episode = run_strategic_episode(
                    champion, robber, seed=seed, seat=seat, profile=profile,
                    mode="strategic", delegation_probability=rate,
                    delegation_seed=seed + 12345, gate_sampling_seed=seed + 54321,
                    stochastic_gate=True, prior_temperature=2.0,
                    strategic_gate=gate, strategic_critic=critic,
                    pre_action_audit_observer=observe)
                if not episode.terminal or episode.truncated:
                    raise AssertionError("Dose-response episode did not terminate")
                if rate == 1.0:
                    source = _read(SOURCE / f"eval-pre100-{profile}-{seed}.json")
                    if (episode.final_vp != source["final_vp"] or
                            episode.winner != source["winner"] or
                            episode.policy_steps != source["policy_steps"] or
                            episode.rng_counter != source["rng_counter"]):
                        raise AssertionError("100% source replay mismatch")
                actual_first = next((i for i, (pid, action) in enumerate(episode.action_trace)
                                     if i < len(safety_trace) and
                                     (pid, repr(action)) != safety_trace[i]), None)
                if actual_first != first.get("trace_index"):
                    raise AssertionError("Dose first divergence detector mismatch")
                row = {**_episode_record(episode), "first_divergence": first or None}
                _write(path, row)
            records[label].append(row)
            print(f"dose {label} {profile} {seed} delegated={row['actual_delegated_count']}", flush=True)
    summary = summarize_dose(records)
    summary["source_replay_parity"] = verify_source_replays(records)
    _write(output / "dose_summary.json", summary)
    return summary


def family_of_repr(value: str) -> str:
    prefixes = {"BuildSettlementAction": "BUILD_SETTLEMENT", "BuildCityAction": "BUILD_CITY",
                "BuildRoadAction": "BUILD_ROAD", "BuyDevelopmentAction": "BUY_DEVELOPMENT",
                "BankTradeAction": "BANK_TRADE", "EndTurnAction": "END_TURN",
                "ProposeTradeAction": "PROTECTED_PLAYER_TRADE"}
    for prefix, family in prefixes.items():
        if value.startswith(prefix + "("):
            return family
    return "PROTECTED_OTHER"


def summarize_dose(records: dict[str, list[dict]]) -> dict[str, Any]:
    safety = records["0"]
    if not safety:
        return {}
    result: dict[str, Any] = {}
    for rate in RATES:
        key = f"{rate:g}"
        rows = records[key]
        if len(rows) != len(safety):
            raise AssertionError("Dose rates have different paired game counts")
        delegates = sum(row["actual_delegated_count"] for row in rows)
        actions = sum((Counter(row["learner_action_families"]) for row in rows), Counter())
        safe_actions = sum((Counter(row["learner_action_families"]) for row in safety), Counter())
        construction = actions["BUILD_SETTLEMENT"] + actions["BUILD_CITY"]
        safe_construction = safe_actions["BUILD_SETTLEMENT"] + safe_actions["BUILD_CITY"]
        first = [row["first_divergence"] for row in rows if row["first_divergence"]]
        result[key] = {
            "games": len(rows), "profile": dict(Counter(row["profile"] for row in rows)),
            "seat": dict(Counter(str(row["seat"]) for row in rows)),
            "gate_surface": sum(row["gate_surface_count"] for row in rows),
            "protected_preemption": sum(row["protected_preemption_count"] for row in rows),
            "protected_available": sum(row["protected_trade_available_count"] for row in rows),
            "delegation_lottery_target": sum(row["delegation_lottery_count"] for row in rows),
            "actual_delegated": delegates, "delegated_per_game": delegates / len(rows),
            "action_family": dict(actions),
            "construction": construction, "development": actions["BUY_DEVELOPMENT"],
            "wins": sum(row["won"] for row in rows),
            "vp": _stats([row["final_vp"] for row in rows]),
            "rank": _stats([row["final_rank"] for row in rows]),
            "policy_steps": _stats([row["policy_steps"] for row in rows]),
            "paired_vs_safety": {
                "vp_delta": _stats([row["final_vp"] - base["final_vp"]
                                    for row, base in zip(rows, safety, strict=True)]),
                "rank_delta": _stats([row["final_rank"] - base["final_rank"]
                                      for row, base in zip(rows, safety, strict=True)]),
                "steps_delta": _stats([row["policy_steps"] - base["policy_steps"]
                                       for row, base in zip(rows, safety, strict=True)]),
                "learner_win_delta": sum(int(row["won"]) - int(base["won"])
                                         for row, base in zip(rows, safety, strict=True)),
            },
            "terminal": sum(row["terminal"] for row in rows),
            "truncated": sum(row["truncated"] for row in rows),
            "first_divergence": dict(Counter(
                f"{item['safety_family']} -> {item['gate_family']}" for item in first)),
            "first_road_road": sum(item["safety_family"] == "BUILD_ROAD" and
                                   item["gate_family"] == "BUILD_ROAD" for item in first),
            "first_divergence_games": len(first),
            "per_delegation_reference": ({
                "construction": (construction - safe_construction) / delegates,
                "development": (actions["BUY_DEVELOPMENT"] - safe_actions["BUY_DEVELOPMENT"]) / delegates,
                "vp": sum(row["final_vp"] - base["final_vp"]
                          for row, base in zip(rows, safety, strict=True)) / delegates,
            } if delegates else None),
        }
    return result


def verify_source_replays(records: dict[str, list[dict]]) -> dict[str, int]:
    """Check Safety/Full endpoints and all delegated concrete decisions."""
    verified = {"safety": 0, "full_t2": 0, "full_decisions": 0}
    for label, source_prefix, result_key in (("0", "eval-safety", "safety"),
                                             ("1", "eval-pre100", "full_t2")):
        for row in records[label]:
            source = _read(SOURCE / f"{source_prefix}-{row['profile']}-{row['seed']}.json")
            for key in ("winner", "final_vp", "final_rank", "policy_steps",
                        "rng_counter", "learner_action_families", "actual_delegated_count"):
                if row[key] != source[key]:
                    raise AssertionError(f"Source replay differs at {label}/{row['seed']}/{key}")
            if label == "1":
                if [(item["selected_family"], item["concrete_action"])
                    for item in row["decisions"]] != [
                    (item["selected_family"], item["concrete_action"])
                    for item in source["decisions"]]:
                    raise AssertionError(f"Full concrete Action replay differs: {row['seed']}")
                verified["full_decisions"] += len(row["decisions"])
            verified[result_key] += 1
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("road", "dose", "summary", "all"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-states", type=int, default=24)
    parser.add_argument("--future-seeds", type=int, default=8)
    parser.add_argument("--full-aux-states", type=int, default=0)
    args = parser.parse_args()
    if args.phase in {"road", "all"}:
        road_audit(args.output, max_states=args.max_states,
                   future_seeds=args.future_seeds, full_aux_states=args.full_aux_states)
    if args.phase in {"dose", "all"}:
        dose_audit(args.output)
    if args.phase == "summary":
        _write(args.output / "road_summary.json", summarize_road(
            _read(args.output / "road_disagreements.json"),
            _read(args.output / "road_pairs.json")))
        records = {f"{rate:g}": [_read(args.output / "dose_games" / f"{rate:g}" /
                                      f"{profile}-{seed}.json") for profile, seed, _ in _seed_rows()]
                   for rate in RATES}
        summary = summarize_dose(records)
        summary["source_replay_parity"] = verify_source_replays(records)
        _write(args.output / "dose_summary.json", summary)


if __name__ == "__main__":
    main()
