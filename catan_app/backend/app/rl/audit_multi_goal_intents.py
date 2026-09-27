"""既存Macro Teacher Datasetをtri-state Multi-Goalとして保守的に監査する。

保存済みDatasetにはGameStateやPlannerの全中間値がないため、この監査のactiveは
再構成できるsource evidenceだけに限定した下限値である。unknownをFalseへ変換しない。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable

import numpy as np

from .macro_goal import ObjectiveGoal
from .multi_goal_intent import IntentStatus, MULTI_GOALS


def _own_counts(observation: np.ndarray) -> dict[str, int]:
    """Observation v2先頭の自プレイヤー公開値を復号する。"""
    return {
        "actual_score": int(round(float(observation[16]) * 10)),
        "settlements": int(round(float(observation[17]) * 5)),
        "cities": int(round(float(observation[18]) * 4)),
        "roads": int(round(float(observation[19]) * 15)),
    }


def backfill_multi_goal_signals(
    record: dict[str, Any], observation: np.ndarray, planning: dict[str, Any],
) -> dict[str, str]:
    """保存済み情報だけで根拠のあるactive/inactiveを復元する。

    DEVELOPMENT/称号は購入・建設前のassessmentが保存されていないため、既存の
    action-localラベル以外はunknownとする。この制約自体が再収集要否の診断になる。
    """
    counts = _own_counts(observation)
    mask = record.get("available_goal_mask", {})
    turns = record.get("estimated_turns_to_goal", {})
    goal = record.get("objective_goal")
    execution = record.get("execution_mode")
    target_sites = int(planning.get("target_sites", 5))
    max_wait = float(planning.get("max_wait_rounds", 6.0))
    max_city_wait = float(planning.get("max_city_wait_rounds", 9.0))
    endgame_reserve = int(planning.get("endgame_point_reserve_score", 0))

    settlement_wait = turns.get(ObjectiveGoal.SETTLEMENT.value)
    city_wait = turns.get(ObjectiveGoal.CITY.value)
    settlement_plan = bool(
        goal == ObjectiveGoal.SETTLEMENT.value
        or (
        mask.get(ObjectiveGoal.SETTLEMENT.value)
        and settlement_wait is not None
        and float(settlement_wait) <= max_wait
        and (
            counts["settlements"] + counts["cities"] < target_sites
            or (
                endgame_reserve
                and counts["actual_score"] >= endgame_reserve
                and city_wait is not None
                and float(settlement_wait) < float(city_wait)
            )
        )
        )
    )
    signals = {goal_name.value: IntentStatus.UNKNOWN.value for goal_name in MULTI_GOALS}
    if settlement_plan:
        signals[ObjectiveGoal.SETTLEMENT.value] = IntentStatus.ACTIVE.value
    elif counts["settlements"] >= 5 and counts["cities"] >= 4:
        signals[ObjectiveGoal.SETTLEMENT.value] = IntentStatus.INACTIVE.value

    # 5拠点到達後の都市待ち、または既存Teacherが都市目的をsource-levelで
    # 確定した局面だけをactiveとする。単なる都市候補の存在はactiveにしない。
    city_plan = bool(
        mask.get(ObjectiveGoal.CITY.value)
        and (
            not settlement_plan
            or counts["settlements"] + counts["cities"] >= target_sites
            or (city_wait is not None and float(city_wait) <= max_city_wait
                and counts["actual_score"] >= 9)
        )
    )
    if goal == ObjectiveGoal.CITY.value or city_plan:
        signals[ObjectiveGoal.CITY.value] = IntentStatus.ACTIVE.value
    elif counts["cities"] >= 4:
        signals[ObjectiveGoal.CITY.value] = IntentStatus.INACTIVE.value

    # KNIGHT_TITLEでの購入も、発展カード購入Intentを同時activeとする。
    if (execution == "BUY_DEVELOPMENT"
            and goal in {ObjectiveGoal.DEVELOPMENT.value,
                         ObjectiveGoal.KNIGHT_TITLE.value}):
        signals[ObjectiveGoal.DEVELOPMENT.value] = IntentStatus.ACTIVE.value
    elif mask.get(ObjectiveGoal.DEVELOPMENT.value) is False:
        signals[ObjectiveGoal.DEVELOPMENT.value] = IntentStatus.INACTIVE.value

    if goal == ObjectiveGoal.ROAD_TITLE.value:
        signals[ObjectiveGoal.ROAD_TITLE.value] = IntentStatus.ACTIVE.value
    elif counts["roads"] >= 15:
        signals[ObjectiveGoal.ROAD_TITLE.value] = IntentStatus.INACTIVE.value

    if goal == ObjectiveGoal.KNIGHT_TITLE.value:
        signals[ObjectiveGoal.KNIGHT_TITLE.value] = IntentStatus.ACTIVE.value

    return signals


def _distribution(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    return {
        status.value: {
            "count": counter[status.value],
            "rate": counter[status.value] / total if total else 0.0,
        }
        for status in IntentStatus
    }


def _run_lengths(rows: list[tuple[tuple[str, int], int, dict[str, str]]],
                 goal: str) -> list[int]:
    lengths: list[int] = []
    active_key: tuple[str, int] | None = None
    active_length = 0
    for key, _, signals in rows:
        if key != active_key:
            if active_length:
                lengths.append(active_length)
            active_key = key
            active_length = 0
        if signals[goal] == IntentStatus.ACTIVE.value:
            active_length += 1
        elif active_length:
            lengths.append(active_length)
            active_length = 0
    if active_length:
        lengths.append(active_length)
    return lengths


def _persistence(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {
            "run_count": 0, "mean": 0.0, "median": 0.0, "max": 0,
            "one_decision_rate": 0.0, "two_plus_rate": 0.0,
            "five_plus_rate": 0.0,
        }
    return {
        "run_count": len(lengths),
        "mean": fmean(lengths),
        "median": median(lengths),
        "max": max(lengths),
        "one_decision_rate": sum(length == 1 for length in lengths) / len(lengths),
        "two_plus_rate": sum(length >= 2 for length in lengths) / len(lengths),
        "five_plus_rate": sum(length >= 5 for length in lengths) / len(lengths),
    }


def summarize_multi_goal_backfill(
    records: Iterable[dict[str, Any]], observations: np.ndarray,
    planning: dict[str, Any],
) -> dict[str, Any]:
    records = list(records)
    if len(records) != len(observations):
        raise ValueError("Decision数とObservation数が一致しません。")
    statuses = {goal.value: Counter() for goal in MULTI_GOALS}
    by_profile: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {goal.value: Counter() for goal in MULTI_GOALS}
    )
    cooccurrence = {
        first.value: Counter({second.value: 0 for second in MULTI_GOALS})
        for first in MULTI_GOALS
    }
    active_counts = Counter()
    ordered: list[tuple[tuple[str, int], int, dict[str, str]]] = []
    for index, (record, observation) in enumerate(zip(records, observations, strict=True)):
        signals = backfill_multi_goal_signals(record, observation, planning)
        profile = str(record.get("opponent_profile", "unknown"))
        for goal, status in signals.items():
            statuses[goal][status] += 1
            by_profile[profile][goal][status] += 1
        active = [goal for goal, status in signals.items()
                  if status == IntentStatus.ACTIVE.value]
        active_counts[str(min(len(active), 3)) + ("+" if len(active) >= 3 else "")] += 1
        for first in active:
            for second in active:
                cooccurrence[first][second] += 1
        ordered.append((
            (str(record["game_id"]), int(record["player_id"])),
            int(record.get("observation_index", index)),
            signals,
        ))
    ordered.sort(key=lambda item: (item[0][0], item[0][1], item[1]))
    total = len(records)
    return {
        "analysis_kind": "conservative_partial_backfill",
        "decision_count": total,
        "signal_distribution": {
            goal.value: _distribution(statuses[goal.value], total)
            for goal in MULTI_GOALS
        },
        "signal_distribution_by_profile": {
            profile: {
                goal.value: _distribution(counters[goal.value], sum(counters[goal.value].values()))
                for goal in MULTI_GOALS
            }
            for profile, counters in sorted(by_profile.items())
        },
        "active_goal_count_distribution": {
            key: {"count": active_counts[key],
                  "rate": active_counts[key] / total if total else 0.0}
            for key in ("0", "1", "2", "3+")
        },
        "active_cooccurrence_counts": {
            goal: dict(counter) for goal, counter in cooccurrence.items()
        },
        "active_cooccurrence_rates": {
            first: {
                second: count / total if total else 0.0
                for second, count in counter.items()
            }
            for first, counter in cooccurrence.items()
        },
        "persistence": {
            goal.value: _persistence(_run_lengths(ordered, goal.value))
            for goal in MULTI_GOALS
        },
        "limitations": [
            "active率は保存済み情報から再構成できる下限値であり、unknownをinactiveへ変換していない",
            "DEVELOPMENTのassessmentとROAD_TITLE/KNIGHT_TITLE候補は全Decisionに保存されていない",
            "完全なmulti-goal Teacher labelにはsource-level診断を付けた再収集が必要",
        ],
    }


def load_dataset(dataset_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    records = [json.loads(line) for line in (dataset_dir / "decisions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines() if line]
    observations = np.load(dataset_dir / "observations.npz")["observations"]
    definition = json.loads((dataset_dir / "definition.json").read_text(encoding="utf-8"))
    return records, observations, dict(definition.get("planning", {}))


def main() -> None:
    parser = argparse.ArgumentParser(description="既存DatasetのMulti-Goal Intent監査")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, observations, planning = load_dataset(args.dataset)
    result = summarize_multi_goal_backfill(records, observations, planning)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "decision_count": result["decision_count"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
