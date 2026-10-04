"""Run the no-update Step 3A-14 parity, safety, forced, and Gate smoke gates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from app.agents.ppo import PPOAgent

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .strategic_action_surface import FAMILIES
from .strategic_gate_rollout_step3a14 import StrategicEpisode, run_strategic_episode


SCHEMA = "strategic_gate_integration_step3a14_v1"
CHAMPION_ID = "ppo_gnn_board_65k_s03_exp_v003"
ROBBER_ID = "ppo_robber_value_73k_s01_exp_v001"
PROFILES = ("rule", "champion", "mixed")


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def _strict_parity(left: StrategicEpisode, right: StrategicEpisode) -> bool:
    return (left.action_trace == right.action_trace
            and left.winner == right.winner
            and left.final_vp == right.final_vp
            and left.final_rank == right.final_rank
            and left.final_state == right.final_state
            and left.rng_counter == right.rng_counter
            and left.policy_steps == right.policy_steps
            and left.terminal == right.terminal
            and left.truncated == right.truncated)


def _safety_difference(legacy: StrategicEpisode,
                       safety: StrategicEpisode) -> dict[str, Any]:
    a, b = legacy.action_trace, safety.action_trace
    index = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    if index is None:
        if not _strict_parity(legacy, safety):
            raise AssertionError("Safety state differs without a resolver action difference")
        return {"changed": False, "first_difference_index": None,
                "resolver_changed_events": 0}
    matches = [event for event in safety.resolver_events
               if event["changed_action"] and event["action_trace_index"] == index]
    if len(matches) != 1 or not matches[0]["immediate_terminal"]:
        raise AssertionError("Non-resolver or nonterminal Safety difference")
    if safety.winner != safety.learner_id or not safety.terminal:
        raise AssertionError("Resolver difference did not give learner immediate win")
    if len(b) != index + 1:
        raise AssertionError("Safety episode continued after immediate win")
    return {"changed": True, "first_difference_index": index,
            "resolver_changed_events": len(matches),
            "champion_counterfactual": matches[0]["champion_counterfactual_action"],
            "resolver_action": matches[0]["resolver_action"],
            "winner": safety.winner}


def _episode_record(episode: StrategicEpisode) -> dict[str, Any]:
    return {
        "seed": episode.seed, "profile": episode.profile, "seat": episode.seat,
        "mode": episode.mode,
        "delegation_probability": episode.delegation_probability,
        "delegation_seed": episode.delegation_seed,
        "gate_sampling_seed": episode.gate_sampling_seed,
        "learner_id": episode.learner_id,
        "winner": episode.winner, "won": episode.winner == episode.learner_id,
        "final_vp": episode.final_vp, "final_rank": episode.final_rank,
        "policy_steps": episode.policy_steps,
        "terminal": episode.terminal, "truncated": episode.truncated,
        "rng_counter": episode.rng_counter,
        "gate_surface_count": episode.gate_surface_count,
        "delegation_lottery_count": episode.delegation_lottery_count,
        "protected_trade_available_count": episode.protected_available_count,
        "protected_preemption_count": episode.protected_preemption_count,
        "actual_delegated_count": len(episode.decisions),
        "resolver_events": list(episode.resolver_events),
        "decisions": list(episode.decisions),
        "transitions": list(episode.transitions),
        "learner_action_families": episode.learner_action_families,
        "forced_applied": episode.forced_applied,
    }


def _run_pair(champion, robber, *, seed: int, seat: int,
              profile: str) -> tuple[dict, dict, dict]:
    legacy = run_strategic_episode(champion, robber, seed=seed, seat=seat,
                                   profile=profile, mode="legacy")
    parity = run_strategic_episode(champion, robber, seed=seed, seat=seat,
                                   profile=profile, mode="strategic",
                                   delegation_probability=0.0,
                                   delegation_seed=seed + 12345,
                                   gate_sampling_seed=seed + 54321,
                                   resolver_enabled=False)
    if not _strict_parity(legacy, parity) or parity.decisions:
        raise AssertionError(f"Legacy 0% parity failed: {profile}/{seed}")
    safety = run_strategic_episode(champion, robber, seed=seed, seat=seat,
                                   profile=profile, mode="safety")
    difference = _safety_difference(legacy, safety)
    return _episode_record(legacy), _episode_record(safety), difference


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None,
                "min": None, "max": None}
    sorted_values = sorted(values)
    return {"n": len(values), "mean": mean(values),
            "median": median(values),
            "p95": sorted_values[int(.95 * (len(values) - 1))],
            "min": sorted_values[0], "max": sorted_values[-1]}


def _variant_stats(rows: list[dict]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row["learner_action_families"])
    return {
        "games": len(rows),
        "wins": sum(row["won"] for row in rows),
        "win_rate": mean([row["won"] for row in rows]) if rows else None,
        "final_vp": _stats([row["final_vp"] for row in rows]),
        "rank": _stats([row["final_rank"] for row in rows]),
        "policy_steps": _stats([row["policy_steps"] for row in rows]),
        "action_phase_family_counts": dict(counts),
        "terminal": sum(row["terminal"] for row in rows),
        "truncated": sum(row["truncated"] for row in rows),
        "resolver_fires": sum(len(row["resolver_events"]) for row in rows),
    }


def _gate_stats(rows: list[dict], gamma: float) -> dict[str, Any]:
    decisions = [item for row in rows for item in row["decisions"]]
    transitions = [item for row in rows for item in row["transitions"]]
    if len(decisions) != len(transitions):
        raise AssertionError("Each delegated Gate decision needs one transition")
    for row in rows:
        for item in row["transitions"]:
            if item["macro_duration"] < 1 or item["close_policy_step"] - item[
                    "start_policy_step"] != item["macro_duration"]:
                raise AssertionError("Macro duration mismatch")
            if not item["done"] and not item["next_actual_delegated"]:
                raise AssertionError("Nonterminal bootstrap not at next delegated Gate")
            if item["done"] and item["discount"] != 0:
                raise AssertionError("Terminal bootstrap must be zero")
            recomputed = sum(gamma ** i * reward for i, reward in enumerate(
                item["reward_steps"]))
            if abs(recomputed - item["reward_accumulated"]) > 1e-8:
                raise AssertionError("Macro reward accumulation mismatch")
    chosen = Counter(item["selected_family"] for item in decisions)
    selected_probability = {family: _stats([item["selected_probability"]
        for item in decisions if item["selected_family"] == family])
        for family in FAMILIES}
    available_probability = {family: _stats([
        item["masked_probability"][index] for item in decisions
        if item["candidate_mask"][index]])
        for index, family in enumerate(FAMILIES)}
    top_rate = {family: (sum(
        item["selected_family"] == family and item["masked_probability"][
            FAMILIES.index(family)] == max(item["masked_probability"])
        for item in decisions) / chosen[family] if chosen[family] else None)
        for family in FAMILIES}
    entropy_by_count = {bucket: _stats([
        item["entropy"] for item in decisions
        if (str(item["candidate_count"]) if item["candidate_count"] <= 3 else "4+")
        == bucket]) for bucket in ("2", "3", "4+")}
    return {
        "gate_surface_total": sum(row["gate_surface_count"] for row in rows),
        "protected_trade_available": sum(row["protected_trade_available_count"]
                                         for row in rows),
        "protected_preemption": sum(row["protected_preemption_count"] for row in rows),
        "delegation_lottery_target": sum(row["delegation_lottery_count"] for row in rows),
        "actual_delegated": len(decisions),
        "family_selected": dict(chosen),
        "selected_probability": selected_probability,
        "available_probability": available_probability,
        "selected_top1_rate": top_rate,
        "entropy_by_candidate_count": entropy_by_count,
        "macro_duration": _stats([item["macro_duration"] for item in transitions]),
        "macro_reward": _stats([item["reward_accumulated"] for item in transitions]),
        "next_delegated": sum(item["next_actual_delegated"] for item in transitions),
        "terminal_transitions": sum(item["done"] for item in transitions),
        "protected_trades_crossed": sum(item["protected_trades_crossed"]
                                         for item in transitions),
        "nondelegated_surfaces_crossed": sum(item["nondelegated_surfaces_crossed"]
                                            for item in transitions),
        "family_agreement": (mean([item["selected_family"] ==
             item["champion_counterfactual_family"] for item in decisions])
             if decisions else None),
        "concrete_action_agreement": (mean([item["concrete_action"] ==
             item["champion_counterfactual_action"] for item in decisions])
             if decisions else None),
        "executor_failure": 0,
        "illegal_action": 0,
        "family_outside": 0,
    }


def _read_checkpoint(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def collect(root: Path, *, parity_seed: int = 2950000,
            deterministic_seed: int = 2960000,
            stochastic_seed: int = 2970000) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    champion = PPOAgent(CHAMPION_ID, device="cpu")
    robber = PPOAgent(ROBBER_ID, device="cpu")
    digest_before = _weight_digest(champion)
    parity_rows: list[dict] = []
    safety_differences: list[dict] = []
    for p_index, profile in enumerate(PROFILES):
        for index in range(4):
            seed = parity_seed + p_index * 1000 + index
            seat = index + 1
            checkpoint = root / f"parity-{profile}-{seed}.json"
            row = _read_checkpoint(checkpoint)
            if row is None:
                legacy, safety, difference = _run_pair(
                    champion, robber, seed=seed, seat=seat, profile=profile)
                row = {"legacy": legacy, "safety": safety,
                       "safety_difference": difference, "parity": True}
                _write(checkpoint, row)
            parity_rows.append(row)
            safety_differences.append(row["safety_difference"])
            print(f"parity {profile} {seed} seat={seat} "
                  f"resolver_change={row['safety_difference']['changed']}", flush=True)
    known_path = root / "known-missed-win-2941006.json"
    known = _read_checkpoint(known_path)
    if known is None:
        legacy = run_strategic_episode(champion, robber, seed=2941006, seat=3,
                                       profile="champion", mode="legacy")
        safety = run_strategic_episode(champion, robber, seed=2941006, seat=3,
                                       profile="champion", mode="safety")
        difference = _safety_difference(legacy, safety)
        if not difference["changed"] or "BuildSettlementAction(vertex_id=0)" not in difference[
                "resolver_action"]:
            raise AssertionError("Known missed-win state was not resolved")
        known = {"legacy": _episode_record(legacy), "safety": _episode_record(safety),
                 "safety_difference": difference}
        _write(known_path, known)
    print("known missed win resolved", flush=True)

    forced_rows: list[dict] = []
    for family in FAMILIES:
        checkpoint = root / f"forced-{family.lower()}.json"
        row = _read_checkpoint(checkpoint)
        if row is None:
            episode = run_strategic_episode(
                champion, robber, seed=2940000, seat=1, profile="rule",
                mode="strategic", force_family=family,
                delegation_probability=0, delegation_seed=401,
                gate_sampling_seed=402)
            if not episode.forced_applied or not episode.terminal or episode.truncated:
                raise AssertionError(f"Forced family integration failed: {family}")
            if len(episode.decisions) != 1 or len(episode.transitions) != 1:
                raise AssertionError("Forced integration must be isolated")
            row = _episode_record(episode)
            _write(checkpoint, row)
        forced_rows.append(row)
        print(f"forced {family} applied", flush=True)

    deterministic_rows: list[dict] = []
    for p_index, profile in enumerate(PROFILES):
        for index in range(4):
            seed = deterministic_seed + p_index * 1000 + index
            seat = index + 1
            checkpoint = root / f"deterministic-{profile}-{seed}.json"
            row = _read_checkpoint(checkpoint)
            if row is None:
                episode = run_strategic_episode(
                    champion, robber, seed=seed, seat=seat, profile=profile,
                    mode="strategic", delegation_probability=.05,
                    delegation_seed=seed + 12345,
                    gate_sampling_seed=seed + 54321,
                    stochastic_gate=False)
                if not episode.terminal or episode.truncated:
                    raise AssertionError("Deterministic smoke did not finish cleanly")
                row = _episode_record(episode)
                _write(checkpoint, row)
            deterministic_rows.append(row)
            print(f"deterministic {profile} {seed} gate={row['actual_delegated_count']}",
                  flush=True)
    if not any(row["actual_delegated_count"] for row in deterministic_rows):
        raise AssertionError("Deterministic smoke had no actual Gate delegation")

    stochastic_rows: list[dict] = []
    stochastic_baseline: list[dict] = []
    stochastic_safety: list[dict] = []
    stochastic_safety_diff: list[dict] = []
    for p_index, profile in enumerate(PROFILES):
        for index in range(12):
            seed = stochastic_seed + p_index * 1000 + index
            seat = index % 4 + 1
            checkpoint = root / f"stochastic-{profile}-{seed}.json"
            row = _read_checkpoint(checkpoint)
            if row is None:
                legacy, safety, difference = _run_pair(
                    champion, robber, seed=seed, seat=seat, profile=profile)
                episode = run_strategic_episode(
                    champion, robber, seed=seed, seat=seat, profile=profile,
                    mode="strategic", delegation_probability=.10,
                    delegation_seed=seed + 12345,
                    gate_sampling_seed=seed + 54321,
                    stochastic_gate=True)
                if not episode.terminal or episode.truncated:
                    raise AssertionError("Stochastic smoke did not finish cleanly")
                row = {"legacy": legacy, "safety": safety,
                       "safety_difference": difference,
                       "strategic": _episode_record(episode), "parity": True}
                _write(checkpoint, row)
            stochastic_baseline.append(row["legacy"])
            stochastic_safety.append(row["safety"])
            stochastic_safety_diff.append(row["safety_difference"])
            stochastic_rows.append(row["strategic"])
            print(f"stochastic {profile} {seed} seat={seat} "
                  f"gate={row['strategic']['actual_delegated_count']}", flush=True)
    digest_after = _weight_digest(champion)
    if digest_before != digest_after:
        raise AssertionError("Frozen Champion weights changed")
    summary = {
        "schema_version": SCHEMA,
        "champion_model_id": CHAMPION_ID,
        "robber_model_id": ROBBER_ID,
        "champion_weight_digest_before": digest_before,
        "champion_weight_digest_after": digest_after,
        "legacy_zero_percent_parity": len(parity_rows),
        "stochastic_paired_legacy_zero_percent_parity": len(stochastic_rows),
        "safety_change_in_parity_set": sum(row["changed"] for row in safety_differences),
        "safety_change_in_stochastic_set": sum(row["changed"]
                                                for row in stochastic_safety_diff),
        "known_missed_win": known["safety_difference"],
        "forced_families": {family: {
            "terminal": row["terminal"], "truncated": row["truncated"],
            "concrete_action": row["decisions"][0]["concrete_action"],
            "application_probe": row["decisions"][0]["application_probe"],
        } for family, row in zip(FAMILIES, forced_rows, strict=True)},
        "deterministic": {
            "outcome": _variant_stats(deterministic_rows),
            "gate": _gate_stats(deterministic_rows, float(champion.model.gamma)),
        },
        "stochastic": {
            "legacy": _variant_stats(stochastic_baseline),
            "safety": _variant_stats(stochastic_safety),
            "strategic": _variant_stats(stochastic_rows),
            "gate": _gate_stats(stochastic_rows, float(champion.model.gamma)),
            "profile_games": dict(Counter(row["profile"] for row in stochastic_rows)),
            "seat_games": dict(Counter(row["seat"] for row in stochastic_rows)),
        },
        "seed_ranges": {"parity": parity_seed,
                        "deterministic": deterministic_seed,
                        "stochastic": stochastic_seed},
    }
    _write(root / "summary.json", summary)
    _write(root / "games.json", {"parity": parity_rows,
                                  "stochastic": [{"legacy": legacy,
                                                  "safety": safety,
                                                  "strategic": strategic,
                                                  "safety_difference": difference}
                                                 for legacy, safety, strategic, difference
                                                 in zip(stochastic_baseline,
                                                        stochastic_safety,
                                                        stochastic_rows,
                                                        stochastic_safety_diff,
                                                        strict=True)]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="strategic_gate_step3a14_20261004")
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("Invalid experiment ID")
    root = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if (root / "summary.json").exists():
        parser.error("Experiment already complete; use a fresh ID")
    summary = collect(root)
    print(json.dumps({"legacy_parity": summary["legacy_zero_percent_parity"],
                      "forced": len(summary["forced_families"]),
                      "deterministic_gate": summary["deterministic"]["gate"]["actual_delegated"],
                      "stochastic_gate": summary["stochastic"]["gate"]["actual_delegated"]}),
          flush=True)


if __name__ == "__main__":
    main()
