"""ChampionのMacro診断を、行動へ介入せず収集・集計するStep 2A基盤。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

import numpy as np

from app.agents.heuristic import HeuristicAgent
from app.domain.actions import Action
from app.domain.game import (GameState, apply_action, create_game, game_view,
                             pending_ai_player_id)

from .diagnostics_metrics import endgame_rank
from .macro_goal import ObjectiveGoal, PlannerDecisionReport
from .observation import encode_observation, get_observation
from .reward import actual_score
from .settlement_planning_agent import SettlementPlanningAgent


DATASET_SCHEMA_VERSION = "macro_teacher_v1"
NONE_LABEL = "None"


@dataclass(frozen=True)
class TeacherEpisode:
    records: tuple[dict[str, Any], ...]
    observations: np.ndarray
    game: dict[str, Any]
    action_trace: tuple[tuple[int, Action], ...]
    final_views: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TeacherDataset:
    records: tuple[dict[str, Any], ...]
    observations: np.ndarray
    games: tuple[dict[str, Any], ...]


def _turn_band(turn: int) -> str:
    """複雑な局面定義を持ち込まない、診断専用の簡易区分。"""
    if turn <= 25:
        return "early"
    if turn <= 50:
        return "mid"
    return "late"


def _goal_label(value: str | None) -> str:
    return value if value is not None else NONE_LABEL


def _normalized_final_views(game: GameState) -> tuple[dict[str, Any], ...]:
    views = []
    for viewer_id in range(1, 5):
        view = game_view(game, viewer_id)
        # create_gameごとに採番されるIDは対局内容ではないためparity対象外。
        view.pop("id", None)
        views.append(view)
    return tuple(views)


def observe_teacher_decision(
    agent: SettlementPlanningAgent, game: GameState, player_id: int,
) -> tuple[Action, PlannerDecisionReport, np.ndarray]:
    """Action・診断・同じAction直前の既存Observationを副作用なしで得る。"""
    observation = encode_observation(get_observation(
        game, player_id, version=agent.policy.manifest.observation_version,
    ))
    action, report = agent.select_action_with_report(game, player_id)
    return action, report, observation


def play_teacher_episode(
    agent: SettlementPlanningAgent,
    seed: int,
    champion_id: int,
    *,
    collect_reports: bool = True,
) -> TeacherEpisode:
    """Champion 1人対ルールAI 3人を最後まで実行する。"""
    if champion_id not in range(1, 5):
        raise ValueError("champion_idは1〜4で指定してください。")
    game = create_game(seed, ai_player_ids=(1, 2, 3, 4))
    opponents = {
        player_id: HeuristicAgent() for player_id in range(1, 5)
        if player_id != champion_id
    }
    raw_records: list[dict[str, Any]] = []
    observations: list[np.ndarray] = []
    action_trace: list[tuple[int, Action]] = []
    for step in range(100_000):
        if game.phase == "game_over":
            break
        actor = pending_ai_player_id(game)
        if actor is None:
            raise RuntimeError(f"AI担当を決められません: {game.phase}")
        if actor == champion_id and collect_reports:
            action, report, observation = observe_teacher_decision(
                agent, game, actor,
            )
            # 非Macroフェーズは学習対象から分離する。通常道路等のGoal=Noneは
            # is_macro_decision=Trueなので、そのまま重要な診断例として残る。
            if report.is_macro_decision:
                record = report.to_dict()
                record.update({
                    "dataset_schema_version": DATASET_SCHEMA_VERSION,
                    "game_seed": seed,
                    "game_id": game.id,
                    "seat": game.seat_order.index(actor) + 1,
                    "observation_index": len(observations),
                    "observation_version": agent.policy.manifest.observation_version,
                    "turn_band": _turn_band(game.turn_number),
                })
                raw_records.append(record)
                observations.append(observation.copy())
        elif actor == champion_id:
            action = agent.select_action(game, actor)
        else:
            action = opponents[actor].select_action(game, actor)
        action_trace.append((actor, action))
        apply_action(game, actor, action, game.revision, f"macro-teacher-{step:06d}")
    else:
        raise RuntimeError(f"seed {seed}: 100000操作で終了しませんでした。")

    scores = {player.id: actual_score(game, player.id) for player in game.players}
    rank, _ = endgame_rank(scores, champion_id, game.winner_id)
    for record in raw_records:
        record.update({
            "winner": game.winner_id,
            "won": game.winner_id == champion_id,
            "final_vp": scores[champion_id],
            "final_rank": rank,
        })
    observation_size = agent.policy.manifest.observation_size
    encoded = (np.stack(observations).astype(np.float32, copy=False)
               if observations else np.empty((0, observation_size), dtype=np.float32))
    game_record = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "game_seed": seed,
        "game_id": game.id,
        "champion_id": champion_id,
        "seat": game.seat_order.index(champion_id) + 1,
        "winner": game.winner_id,
        "won": game.winner_id == champion_id,
        "final_vp": scores[champion_id],
        "final_rank": rank,
        "final_scores": {str(key): value for key, value in scores.items()},
        "turn": game.turn_number,
        "all_action_count": len(action_trace),
        "macro_decision_count": len(raw_records),
    }
    return TeacherEpisode(
        records=tuple(raw_records), observations=encoded, game=game_record,
        action_trace=tuple(action_trace), final_views=_normalized_final_views(game),
    )


def collect_teacher_dataset(
    agent: SettlementPlanningAgent,
    seeds: Iterable[int],
) -> TeacherDataset:
    """席を1→4でローテーションし、複数ゲームの行番号を連結する。"""
    records: list[dict[str, Any]] = []
    observations: list[np.ndarray] = []
    games: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        episode = play_teacher_episode(
            agent, int(seed), (index % 4) + 1, collect_reports=True,
        )
        offset = len(observations)
        for record in episode.records:
            copied = dict(record)
            copied["observation_index"] = offset + int(record["observation_index"])
            records.append(copied)
        observations.extend(episode.observations)
        games.append(episode.game)
    observation_size = agent.policy.manifest.observation_size
    encoded = (np.stack(observations).astype(np.float32, copy=False)
               if observations else np.empty((0, observation_size), dtype=np.float32))
    return TeacherDataset(tuple(records), encoded, tuple(games))


def _count_summary(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    return {
        name: {"count": count, "rate": count / total if total else 0.0}
        for name, count in sorted(counter.items())
    }


def summarize_taxonomy(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Goal taxonomyの偏りと未分類理由を、異種scoreを混ぜずに集計する。"""
    rows = list(records)
    total = len(rows)
    known_goals = [goal.value for goal in ObjectiveGoal] + [NONE_LABEL]
    goals = Counter({goal: 0 for goal in known_goals})
    cross: dict[str, Counter[str]] = defaultdict(Counter)
    none_action = Counter()
    none_execution = Counter()
    none_reason = Counter()
    none_phase = Counter()
    none_turn_band = Counter()
    goal_reason: dict[str, Counter[str]] = defaultdict(Counter)
    progress: dict[str, Counter[str]] = defaultdict(Counter)
    outcome: dict[str, Counter[str]] = defaultdict(Counter)
    semantic_values: dict[str, list[float]] = defaultdict(list)
    build_road_total = 0
    build_road_none = 0

    for row in rows:
        goal = _goal_label(row.get("objective_goal"))
        execution = str(row["execution_mode"])
        goals[goal] += 1
        cross[goal][execution] += 1
        turn_band = str(row.get("turn_band") or _turn_band(int(row["turn"])))
        progress[turn_band][goal] += 1
        outcome["winner" if row.get("won") else "non_winner"][goal] += 1
        for reason in row.get("reason_codes", []):
            goal_reason[goal][str(reason)] += 1
        if execution == "BUILD_ROAD":
            build_road_total += 1
        if goal == NONE_LABEL:
            action_family = str(row["selected_action"].get("type", "Unknown"))
            none_action[action_family] += 1
            none_execution[execution] += 1
            none_phase[str(row["phase"])] += 1
            none_turn_band[turn_band] += 1
            for reason in row.get("reason_codes", []):
                none_reason[str(reason)] += 1
            if execution == "BUILD_ROAD":
                build_road_none += 1
        semantics = row.get("goal_score_semantics", {})
        values = row.get("goal_scores", {})
        for score_goal, semantic in semantics.items():
            value = values.get(score_goal)
            if value is not None and semantic != "not_applicable":
                semantic_values[str(semantic)].append(float(value))

    score_summary = {}
    for semantic, values in sorted(semantic_values.items()):
        score_summary[semantic] = {
            "count": len(values), "min": min(values), "max": max(values),
            "mean": fmean(values), "std": pstdev(values),
        }
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "decision_count": total,
        "objective_goal_distribution": _count_summary(goals, total),
        "objective_goal_by_execution_mode": {
            goal: _count_summary(counter, sum(counter.values()))
            for goal, counter in sorted(cross.items())
        },
        "none_analysis": {
            "count": goals[NONE_LABEL],
            "rate": goals[NONE_LABEL] / total if total else 0.0,
            "selected_action_family": _count_summary(none_action, goals[NONE_LABEL]),
            "execution_mode": _count_summary(none_execution, goals[NONE_LABEL]),
            "reason_code": _count_summary(none_reason, sum(none_reason.values())),
            "phase": _count_summary(none_phase, goals[NONE_LABEL]),
            "turn_band": _count_summary(none_turn_band, goals[NONE_LABEL]),
            "build_road": {
                "none_count": build_road_none,
                "all_build_road_count": build_road_total,
                "none_rate": build_road_none / build_road_total if build_road_total else 0.0,
            },
        },
        "objective_goal_by_reason_code": {
            goal: dict(sorted(counter.items()))
            for goal, counter in sorted(goal_reason.items())
        },
        "objective_goal_by_turn_band": {
            band: _count_summary(counter, sum(counter.values()))
            for band, counter in sorted(progress.items())
        },
        "objective_goal_by_outcome": {
            result: _count_summary(counter, sum(counter.values()))
            for result, counter in sorted(outcome.items())
        },
        "goal_score_semantics": score_summary,
        "turn_band_definition": {
            "early": "turn <= 25", "mid": "26 <= turn <= 50", "late": "turn >= 51",
        },
    }


def save_teacher_dataset(
    output_dir: Path,
    dataset: TeacherDataset,
    *,
    definition: dict[str, Any],
) -> dict[str, Any]:
    """JSONL・圧縮NPZ・診断JSONを新規ディレクトリへ保存する。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = summarize_taxonomy(dataset.records)
    files = {
        "records": "decisions.jsonl",
        "observations": "observations.npz",
        "games": "games.jsonl",
        "summary": "taxonomy_summary.json",
        "definition": "definition.json",
    }
    with (output_dir / files["records"]).open("w", encoding="utf-8") as handle:
        for record in dataset.records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / files["games"]).open("w", encoding="utf-8") as handle:
        for game in dataset.games:
            handle.write(json.dumps(game, ensure_ascii=False) + "\n")
    np.savez_compressed(output_dir / files["observations"],
                        observations=dataset.observations)
    (output_dir / files["summary"]).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    manifest = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "game_count": len(dataset.games),
        "decision_count": len(dataset.records),
        "observation_shape": list(dataset.observations.shape),
        "files": files,
        **definition,
    }
    (output_dir / files["definition"]).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return {"manifest": manifest, "summary": summary}


def verify_action_parity(
    plain_agent: SettlementPlanningAgent,
    diagnostic_agent: SettlementPlanningAgent,
    seeds: Iterable[int],
) -> dict[str, Any]:
    """診断なし/ありで全Action列・結果・最終表示が一致するか確認する。"""
    checked = 0
    all_actions = 0
    for index, seed in enumerate(seeds):
        champion_id = (index % 4) + 1
        plain = play_teacher_episode(
            plain_agent, int(seed), champion_id, collect_reports=False,
        )
        diagnostic = play_teacher_episode(
            diagnostic_agent, int(seed), champion_id, collect_reports=True,
        )
        if plain.action_trace != diagnostic.action_trace:
            raise AssertionError(f"seed {seed}: Action列が一致しません。")
        result_keys = ("winner", "final_vp", "final_rank", "final_scores", "turn")
        if any(plain.game[key] != diagnostic.game[key] for key in result_keys):
            raise AssertionError(f"seed {seed}: 最終結果が一致しません。")
        if plain.final_views != diagnostic.final_views:
            raise AssertionError(f"seed {seed}: 最終GameState表示が一致しません。")
        checked += 1
        all_actions += len(plain.action_trace)
    return {
        "games": checked, "all_actions": all_actions,
        "action_trace_equal": True, "winner_equal": True,
        "final_score_equal": True, "final_rank_equal": True,
        "final_game_state_equal": True,
    }
