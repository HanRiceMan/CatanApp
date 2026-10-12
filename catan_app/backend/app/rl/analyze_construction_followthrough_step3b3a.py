"""Step 3B-3A: read-only same-turn follow-through from Step 3B-3 records.

The source records contain every learner pre-action normal-action boundary,
including the return after an opponent responds to a proposed trade. Opponent
response details are not inferred from gaps in revision numbers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re

from .run_immediate_construction_audit_step3b3 import SCHEMA as SOURCE_SCHEMA
from .run_immediate_construction_audit_step3b3 import SEED_BASES


SCHEMA = "construction_followthrough_step3b3a_v1"
BUILD_CLASSES = ("SETTLEMENT_ONLY", "CITY_ONLY", "BOTH")
OUTCOMES = (
    "BUILT_TARGET_IMMEDIATELY", "BUILT_TARGET_LATER_SAME_TURN",
    "BUILT_OTHER_SCORE_SOURCE", "ENDED_TURN_WITHOUT_BUILD",
    "BUILDABILITY_LOST_WITHOUT_BUILD", "TERMINAL_OTHER",
)
RESOURCE_ORDER = ("wood", "brick", "sheep", "wheat", "ore")
_BANK_TRADE = re.compile(
    r"BankTradeAction\(give_resource='([^']+)', receive_resource='([^']+)'\)")


def _target_available(row: dict, kind: str) -> bool:
    return (bool(row["settlement_vertices"]) if kind == "SETTLEMENT_ONLY" else
            bool(row["city_vertices"]) if kind == "CITY_ONLY" else
            bool(row["settlement_vertices"] or row["city_vertices"]))


def _target_build(family: str, kind: str) -> bool:
    if kind == "SETTLEMENT_ONLY":
        return family == "BUILD_SETTLEMENT"
    if kind == "CITY_ONLY":
        return family == "BUILD_CITY"
    return family in {"BUILD_SETTLEMENT", "BUILD_CITY"}


def _resource_delta(before: dict, after: dict | None) -> dict | None:
    return ({name: after[name] - before[name] for name in RESOURCE_ORDER}
            if after is not None else None)


def _normal_rows_same_turn(rows: list[dict], start: int) -> list[tuple[int, dict]]:
    turn = rows[start]["turn"]
    return [(index, row) for index, row in enumerate(rows[start:], start)
            if row["turn"] == turn and row["phase"] == "action"]


def follow_one(rows: list[dict], start: int, episode: dict) -> dict:
    """Follow a target build until the end of its learner turn.

    Temporary loss of buildability and an intervening score do not close the
    opportunity: the same target can still be built later in the turn.
    """
    origin = rows[start]
    kind = origin["build_class"]
    if origin["phase"] != "action" or origin["immediate_win"] or kind not in BUILD_CLASSES:
        raise ValueError("Origin must be an ordinary buildable action state")
    turn_rows = _normal_rows_same_turn(rows, start)
    if not turn_rows or turn_rows[0][0] != start:
        raise AssertionError("Origin missing from turn trajectory")
    initial_resources = origin["resource_vector"]
    before: list[str] = []
    trade_events: list[dict] = []
    road_events: list[dict] = []
    loss_seen = False
    other_score_seen = False
    result: dict | None = None
    last_row = origin
    for position, (index, row) in enumerate(turn_rows):
        family = row["action_family"]
        last_row = row
        next_item = turn_rows[position + 1] if position + 1 < len(turn_rows) else None
        next_row = next_item[1] if next_item else None
        if family == "BANK_TRADE":
            match = _BANK_TRADE.fullmatch(row["action"])
            if match is None:
                raise AssertionError("Unparseable recorded BankTrade Action")
            trade_events.append({
                "revision": row["revision"], "give": match.group(1),
                "receive": match.group(2), "pre_buildable": _target_available(row, kind),
                "post_buildable": (_target_available(next_row, kind) if next_row else None),
                "resources_before": row["resource_vector"],
                "resources_after": next_row["resource_vector"] if next_row else None,
            })
        elif family == "PLAYER_TRADE":
            trade_events.append({
                "revision": row["revision"], "family": family,
                "returned_to_action_same_turn": next_row is not None,
                "next_action_revision": next_row["revision"] if next_row else None,
                "post_buildable": (_target_available(next_row, kind) if next_row else None),
                "resources_before": row["resource_vector"],
                "resources_after": next_row["resource_vector"] if next_row else None,
                "response_counter_details_available_in_source": False,
            })
        elif family == "BUILD_ROAD":
            road_events.append({"revision": row["revision"],
                                "post_buildable": (_target_available(next_row, kind)
                                                   if next_row else None)})

        # An actual build resolves the opportunity before checking later masks.
        if _target_build(family, kind):
            outcome = ("BUILT_TARGET_IMMEDIATELY" if position == 0
                       else "BUILT_TARGET_LATER_SAME_TURN")
            result = {"outcome": outcome, "built_family": family,
                      "actions_before_build": before.copy(),
                      "actions_before_build_count": len(before),
                      "endpoint_revision": row["revision"],
                      "endpoint_policy_step": row["policy_step"],
                      "endpoint_resource_pre_action": row["resource_vector"]}
            break
        if family in {"BUILD_SETTLEMENT", "BUILD_CITY"}:
            other_score_seen = True
        if family == "END_TURN":
            break
        before.append(family)
        if next_row is not None:
            if next_row["current_score"] > row["current_score"]:
                other_score_seen = True
            if not _target_available(next_row, kind):
                loss_seen = True
    if result is None:
        terminal_here = rows[-1] is last_row and bool(episode["terminal"])
        outcome = ("BUILT_OTHER_SCORE_SOURCE" if other_score_seen else
                   "BUILDABILITY_LOST_WITHOUT_BUILD" if loss_seen else
                   "ENDED_TURN_WITHOUT_BUILD" if last_row["action_family"] == "END_TURN" else
                   "TERMINAL_OTHER" if terminal_here else
                   "ENDED_TURN_WITHOUT_BUILD")
        result = {"outcome": outcome, "built_family": None,
                  "endpoint_revision": last_row["revision"],
                  "endpoint_policy_step": last_row["policy_step"],
                  "endpoint_resource_pre_action": last_row["resource_vector"]}
    if result is None or result["outcome"] not in OUTCOMES:
        raise AssertionError("Unclosed build opportunity")
    all_later_builds = [row["action_family"] for _, row in turn_rows
                        if row["action_family"] in {"BUILD_SETTLEMENT", "BUILD_CITY"}]
    last_turn_action = turn_rows[-1][1]["action_family"]
    record = {
        "profile": origin["profile"], "seed": origin["seed"], "seat": origin["seat"],
        "turn": origin["turn"], "initial_revision": origin["revision"],
        "initial_policy_step": origin["policy_step"], "initial_build_class": kind,
        "initial_action_family": origin["action_family"],
        "initial_action": origin["action"],
        "initial_resources": initial_resources,
        "initial_settlement_legal_count": len(origin["settlement_vertices"]),
        "initial_city_legal_count": len(origin["city_vertices"]),
        "trade_events_before_close": trade_events,
        "road_events_before_close": road_events,
        "buildability_lost_before_close": loss_seen,
        "other_score_before_close": other_score_seen,
        "ended_turn_without_target": (
            not result["outcome"].startswith("BUILT_TARGET")
            and last_turn_action == "END_TURN"),
        "last_turn_action_family": last_turn_action,
        "settlement_built_after_origin_same_turn": "BUILD_SETTLEMENT" in all_later_builds,
        "city_built_after_origin_same_turn": "BUILD_CITY" in all_later_builds,
        "all_build_actions_after_origin_same_turn": all_later_builds,
        **result,
    }
    record["resource_delta_to_endpoint_pre_action"] = _resource_delta(
        initial_resources, record["endpoint_resource_pre_action"])
    return record


def load_source(source: Path) -> list[tuple[dict, list[dict]]]:
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if (source_summary["schema"] != SOURCE_SCHEMA or source_summary["games"] != 300
            or source_summary["mutation_failures"] or source_summary["rng_failures"]
            or source_summary["parity_failures"] or source_summary["optimizer_created"]):
        raise AssertionError("Step 3B-3 source invariants did not pass")
    episodes = []
    for profile, base in SEED_BASES.items():
        for index in range(100):
            path = source / "episodes" / f"{profile}_{base + index}.json.gz"
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            episode = payload["episode"]
            if (payload["schema"] != SOURCE_SCHEMA or episode["profile"] != profile
                    or episode["seed"] != base + index or episode["seat"] != index % 4 + 1
                    or not episode["parity"] or not episode["terminal"] or episode["truncated"]):
                raise AssertionError(f"Invalid source episode: {path}")
            rows = payload["records"]
            for a, b in zip(rows, rows[1:]):
                if b["policy_step"] <= a["policy_step"] or b["revision"] <= a["revision"]:
                    raise AssertionError(f"Nonmonotone learner boundary: {path}")
            episodes.append((episode, rows))
    return episodes


def analyze_episodes(episodes: list[tuple[dict, list[dict]]]) -> tuple[list[dict], list[dict], dict]:
    state_results, opportunities = [], []
    excluded = Counter()
    for episode, rows in episodes:
        first_in_turn: set[tuple[int, str]] = set()
        for index, row in enumerate(rows):
            if row["phase"] != "action" or row["build_class"] not in BUILD_CLASSES:
                continue
            if row["immediate_win"]:
                excluded[row["build_class"]] += 1
                continue
            result = follow_one(rows, index, episode)
            state_results.append(result)
            identity = (row["turn"], row["build_class"])
            if identity not in first_in_turn:
                first_in_turn.add(identity)
                opportunities.append(result)
    summary = summarize(state_results, opportunities, episodes, excluded)
    return state_results, opportunities, summary


def _outcome_counts(rows: list[dict]) -> dict:
    counts = Counter(row["outcome"] for row in rows)
    return {name: counts[name] for name in OUTCOMES}


def _bucket(n: int) -> str:
    return str(n) if n < 3 else "3+"


def summarize(states: list[dict], opportunities: list[dict],
              episodes: list[tuple[dict, list[dict]]], excluded: Counter) -> dict:
    class_summary = {}
    for kind in BUILD_CLASSES:
        origin = [row for row in opportunities if row["initial_build_class"] == kind]
        state = [row for row in states if row["initial_build_class"] == kind]
        outcomes = _outcome_counts(origin)
        same_turn = (outcomes["BUILT_TARGET_IMMEDIATELY"]
                     + outcomes["BUILT_TARGET_LATER_SAME_TURN"])
        profile = {name: {
            "opportunities": len(subset),
            "same_turn_target_build": sum(row["outcome"].startswith("BUILT_TARGET")
                                          for row in subset),
            "rate": (sum(row["outcome"].startswith("BUILT_TARGET") for row in subset)
                     / len(subset) if subset else None),
        } for name in SEED_BASES
            for subset in [[row for row in origin if row["profile"] == name]]}
        class_summary[kind] = {
            "opportunities": len(origin), "state_level": len(state),
            "outcomes": outcomes, "same_turn_target_build": same_turn,
            "same_turn_target_build_rate": same_turn / len(origin) if origin else None,
            "no_target_build_same_turn": len(origin) - same_turn,
            "ended_turn_without_target": sum(
                row["ended_turn_without_target"] for row in origin),
            "buildability_lost_before_close": sum(
                row["buildability_lost_before_close"] for row in origin),
            "other_score_before_close": sum(
                row["other_score_before_close"] for row in origin),
            "profile": profile,
            "later_actions_before_count": dict(Counter(_bucket(row["actions_before_build_count"])
                                                 for row in origin
                                                 if row["outcome"] == "BUILT_TARGET_LATER_SAME_TURN")),
            "all_build_actions_before_count": dict(Counter(
                "0" if row["outcome"] == "BUILT_TARGET_IMMEDIATELY" else
                _bucket(row["actions_before_build_count"])
                for row in origin if row["outcome"].startswith("BUILT_TARGET"))),
            "later_actions_before_family": dict(Counter(family for row in origin
                                                         if row["outcome"] == "BUILT_TARGET_LATER_SAME_TURN"
                                                         for family in row["actions_before_build"])),
            "first_action_distribution": dict(Counter(row["initial_action_family"]
                                                      for row in origin)),
            "state_nonimmediate_count": sum(not _target_build(
                row["initial_action_family"], kind) for row in state),
            "state_nonimmediate_outcomes": _outcome_counts([
                row for row in state if row["initial_action_family"] not in
                ({"BUILD_SETTLEMENT"} if kind == "SETTLEMENT_ONLY" else
                 {"BUILD_CITY"} if kind == "CITY_ONLY" else
                 {"BUILD_SETTLEMENT", "BUILD_CITY"})]),
        }
    later = [row for row in opportunities
             if row["outcome"] == "BUILT_TARGET_LATER_SAME_TURN"]
    first_nonimmediate = [row for row in states if not _target_build(
        row["initial_action_family"], row["initial_build_class"])]
    return {
        "schema": SCHEMA, "source_schema": SOURCE_SCHEMA,
        "source_games": len(episodes), "source_seed_bases": SEED_BASES,
        "profile_seat": dict(Counter(f"{episode['profile']}/seat{episode['seat']}"
                                     for episode, _ in episodes)),
        "immediate_win_excluded_by_class": dict(excluded),
        "state_level_count": len(states), "opportunity_count": len(opportunities),
        "state_to_opportunity_reduction": len(states) - len(opportunities),
        "class_summary": class_summary,
        "state_level_nonimmediate": {
            "count": len(first_nonimmediate),
            "by_class_and_first_action": {
                kind: {family: _outcome_counts([
                    row for row in first_nonimmediate
                    if row["initial_build_class"] == kind and
                    row["initial_action_family"] == family])
                    for family in sorted({row["initial_action_family"] for row in first_nonimmediate
                                          if row["initial_build_class"] == kind})}
                for kind in BUILD_CLASSES},
        },
        "later_build_actions_before_family": dict(Counter(
            family for row in later for family in row["actions_before_build"])),
        "later_build_actions_before_count": dict(Counter(
            _bucket(row["actions_before_build_count"]) for row in later)),
        "bank_trade_to_build": sum(row["initial_action_family"] == "BANK_TRADE" and
                                   row["outcome"] == "BUILT_TARGET_LATER_SAME_TURN"
                                   for row in opportunities),
        "player_trade_to_build": sum(row["initial_action_family"] == "PLAYER_TRADE" and
                                     row["outcome"] == "BUILT_TARGET_LATER_SAME_TURN"
                                     for row in opportunities),
        "road_to_build": sum(row["initial_action_family"] == "BUILD_ROAD" and
                             row["outcome"] == "BUILT_TARGET_LATER_SAME_TURN"
                             for row in opportunities),
        "play_development_to_build": sum(row["initial_action_family"] == "PLAY_DEVELOPMENT" and
                                         row["outcome"] == "BUILT_TARGET_LATER_SAME_TURN"
                                         for row in opportunities),
        "trade_events": {
            "bank": sum(event.get("give") is not None for row in opportunities
                        for event in row["trade_events_before_close"]),
            "player": sum(event.get("family") == "PLAYER_TRADE" for row in opportunities
                          for event in row["trade_events_before_close"]),
            "player_returned_to_action_same_turn": sum(
                event.get("family") == "PLAYER_TRADE" and
                event["returned_to_action_same_turn"]
                for row in opportunities for event in row["trade_events_before_close"]),
        },
        "resource_delta_to_endpoint_by_outcome": {
            outcome: {name: (sum(row["resource_delta_to_endpoint_pre_action"][name]
                                 for row in opportunities if row["outcome"] == outcome)
                            / max(1, sum(row["outcome"] == outcome for row in opportunities)))
                      for name in RESOURCE_ORDER}
            for outcome in OUTCOMES},
        "same_turn_multiple_build_opportunities": sum(
            len(row["all_build_actions_after_origin_same_turn"]) >= 2
            for row in opportunities),
        "source_parity_failures": 0,
        "source_observer_mutation_and_rng_parity_inherited": True,
        "optimizer_created": False, "game_reruns": 0,
    }


def run(source: Path, output: Path) -> dict:
    episodes = load_source(source)
    states, opportunities, summary = analyze_episodes(episodes)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("state_followthrough", states), ("opportunities", opportunities)):
        with gzip.open(output / f"{name}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
        indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.source, args.output)
    print(json.dumps({key: summary[key] for key in
                      ("source_games", "state_level_count", "opportunity_count",
                       "class_summary", "trade_events")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
