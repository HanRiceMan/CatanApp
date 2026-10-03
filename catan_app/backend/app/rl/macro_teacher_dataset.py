"""ChampionのMacro診断を、行動へ介入せず収集・集計するStep 2A基盤。"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from app.agents.base import Agent
from app.agents.heuristic import HeuristicAgent
from app.domain.actions import Action
from app.domain.game import (GameState, apply_action, create_game, game_view,
                             pending_ai_player_id)

from .diagnostics_metrics import endgame_rank
from .macro_goal import ObjectiveGoal, PlannerDecisionReport
from .observation import encode_observation, get_observation
from .reward import actual_score
from .settlement_planning_agent import SettlementPlanningAgent


DATASET_SCHEMA_VERSION = "macro_teacher_v4_strategic_intent"
NONE_LABEL = "None"
OPPONENT_PROFILES = ("rule", "champion", "mixed")
PROFILE_ROLES = {
    "rule": ("rule", "rule", "rule"),
    "champion": ("champion", "champion", "champion"),
    "mixed": ("champion", "robber_ppo", "rule"),
}
AgentFactory = Callable[[], Agent]


@dataclass(frozen=True)
class TeacherEpisode:
    records: tuple[dict[str, Any], ...]
    observations: np.ndarray
    game: dict[str, Any]
    action_trace: tuple[tuple[int, Action], ...]
    final_views: tuple[dict[str, Any], ...]
    final_state: GameState


@dataclass(frozen=True)
class TeacherDataset:
    records: tuple[dict[str, Any], ...]
    observations: np.ndarray
    games: tuple[dict[str, Any], ...]


def _champion_id_for_seat(seed: int, seat: int) -> int:
    """同じseedの順番決めだけを再現し、収集対象を実seatへ割り当てる。"""
    if seat not in range(1, 5):
        raise ValueError("seatは1〜4で指定してください。")
    game = create_game(seed, ai_player_ids=(1, 2, 3, 4))
    roller = HeuristicAgent()
    for step in range(16):
        if game.phase != "rolling_order":
            return game.seat_order[seat - 1]
        actor = pending_ai_player_id(game)
        if actor is None:
            raise RuntimeError("順番決めのAI担当を解決できません。")
        action = roller.select_action(game, actor)
        apply_action(game, actor, action, game.revision, f"seat-probe-{step:02d}")
    raise RuntimeError("順番決めが16操作以内に完了しませんでした。")


def _profile_assignments(
    game: GameState,
    champion_id: int,
    opponent_profile: str,
    opponent_factories: Mapping[str, AgentFactory] | None,
    *,
    champion_model_id: str,
    robber_model_id: str,
) -> tuple[dict[int, Agent], tuple[dict[str, Any], ...]]:
    if opponent_profile not in OPPONENT_PROFILES:
        raise ValueError(f"未対応のopponent profileです: {opponent_profile}")
    factories: dict[str, AgentFactory] = {"rule": HeuristicAgent}
    if opponent_factories is not None:
        factories.update(opponent_factories)
    roles = PROFILE_ROLES[opponent_profile]
    opponent_ids = [
        player_id for player_id in game.seat_order if player_id != champion_id
    ]
    opponents: dict[int, Agent] = {}
    role_by_player = {champion_id: "teacher_champion"}
    for player_id, role in zip(opponent_ids, roles, strict=True):
        try:
            opponents[player_id] = factories[role]()
        except KeyError as error:
            raise ValueError(f"{role}用Agent factoryがありません。") from error
        role_by_player[player_id] = role
    model_by_role = {
        "teacher_champion": champion_model_id,
        "champion": champion_model_id,
        "robber_ppo": robber_model_id,
        "rule": None,
    }
    seat_agents = tuple({
        "seat": seat,
        "player_id": player_id,
        "agent_type": role_by_player[player_id],
        "model_id": model_by_role[role_by_player[player_id]],
        "uses_planner": role_by_player[player_id] != "rule",
    } for seat, player_id in enumerate(game.seat_order, start=1))
    return opponents, seat_agents


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


def _normalized_final_state(game: GameState) -> GameState:
    """対局ごとにランダム発行される識別子だけを除いた全GameState。"""
    state = deepcopy(game)
    state.id = ""
    state.player_tokens = {}
    return state


def observe_teacher_decision(
    agent: SettlementPlanningAgent, game: GameState, player_id: int,
) -> tuple[Action, PlannerDecisionReport, np.ndarray]:
    """Action・診断・同じAction直前の既存Observationを副作用なしで得る。"""
    observation = encode_observation(get_observation(
        game, player_id, version=agent.policy.manifest.observation_version,
    ))
    action, report = agent.select_action_with_report(game, player_id)
    return action, report, observation


def observe_teacher_decision_with_intent(
    agent: SettlementPlanningAgent, game: GameState, player_id: int,
):
    """旧Planner診断と新Strategic Intent診断を、同じAction直前状態で得る。"""
    observation = encode_observation(get_observation(
        game, player_id, version=agent.policy.manifest.observation_version,
    ))
    action, report, intent_report = agent.select_action_with_intent_report(
        game, player_id,
    )
    return action, report, intent_report, observation


def play_teacher_episode(
    agent: SettlementPlanningAgent,
    seed: int,
    champion_id: int | None = None,
    *,
    champion_seat: int | None = None,
    opponent_profile: str = "rule",
    opponent_factories: Mapping[str, AgentFactory] | None = None,
    champion_model_id: str = "ppo_gnn_board_65k_s03_exp_v003",
    robber_model_id: str = "ppo_robber_value_73k_s01_exp_v001",
    collect_reports: bool = True,
) -> TeacherEpisode:
    """指定profileの相手3人と最後まで実行し、1席だけTeacher収集する。"""
    if (champion_id is None) == (champion_seat is None):
        raise ValueError("champion_idまたはchampion_seatを一方だけ指定してください。")
    if champion_seat is not None:
        champion_id = _champion_id_for_seat(seed, champion_seat)
    if champion_id not in range(1, 5):
        raise ValueError("champion_idは1〜4で指定してください。")
    game = create_game(seed, ai_player_ids=(1, 2, 3, 4))
    opponents: dict[int, Agent] | None = None
    seat_agents: tuple[dict[str, Any], ...] | None = None
    raw_records: list[dict[str, Any]] = []
    observations: list[np.ndarray] = []
    action_trace: list[tuple[int, Action]] = []
    for step in range(100_000):
        if game.phase == "game_over":
            break
        actor = pending_ai_player_id(game)
        if actor is None:
            raise RuntimeError(f"AI担当を決められません: {game.phase}")
        if game.phase != "rolling_order" and opponents is None:
            opponents, seat_agents = _profile_assignments(
                game, champion_id, opponent_profile, opponent_factories,
                champion_model_id=champion_model_id,
                robber_model_id=robber_model_id,
            )
        if actor == champion_id and collect_reports:
            action, report, intent_report, observation = observe_teacher_decision_with_intent(
                agent, game, actor,
            )
            # 非Macroフェーズは学習対象から分離する。通常道路等のGoal=Noneは
            # is_macro_decision=Trueなので、そのまま重要な診断例として残る。
            if report.is_macro_decision:
                record = report.to_dict()
                record.update(intent_report.to_dict())
                record.update({
                    "dataset_schema_version": DATASET_SCHEMA_VERSION,
                    "game_seed": seed,
                    "game_id": game.id,
                    "opponent_profile": opponent_profile,
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
            # 順番決め中はprofile配置前だが、全Agentの合法手は同じRollOrder。
            action = (opponents[actor] if opponents is not None
                      else HeuristicAgent()).select_action(game, actor)
        action_trace.append((actor, action))
        apply_action(game, actor, action, game.revision, f"macro-teacher-{step:06d}")
    else:
        raise RuntimeError(f"seed {seed}: 100000操作で終了しませんでした。")

    if seat_agents is None:
        raise RuntimeError("対局終了までopponent profileを配置できませんでした。")
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
        "opponent_profile": opponent_profile,
        "seat_agents": list(seat_agents),
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
        "rng_counter": game.random_source.counter,
    }
    return TeacherEpisode(
        records=tuple(raw_records), observations=encoded, game=game_record,
        action_trace=tuple(action_trace), final_views=_normalized_final_views(game),
        final_state=_normalized_final_state(game),
    )


def collect_teacher_dataset(
    agent: SettlementPlanningAgent,
    seeds: Iterable[int],
    *,
    opponent_profile: str = "rule",
    opponent_factories: Mapping[str, AgentFactory] | None = None,
    champion_model_id: str = "ppo_gnn_board_65k_s03_exp_v003",
    robber_model_id: str = "ppo_robber_value_73k_s01_exp_v001",
) -> TeacherDataset:
    """実seatを1→4でローテーションし、複数ゲームの行番号を連結する。"""
    records: list[dict[str, Any]] = []
    observations: list[np.ndarray] = []
    games: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        episode = play_teacher_episode(
            agent, int(seed), champion_seat=(index % 4) + 1,
            opponent_profile=opponent_profile,
            opponent_factories=opponent_factories,
            champion_model_id=champion_model_id,
            robber_model_id=robber_model_id,
            collect_reports=True,
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


def merge_teacher_datasets(
    datasets: Iterable[TeacherDataset],
    *,
    observation_size: int,
) -> TeacherDataset:
    """profile別Datasetを、Observation indexを保って一つに連結する。"""
    records: list[dict[str, Any]] = []
    observations: list[np.ndarray] = []
    games: list[dict[str, Any]] = []
    for dataset in datasets:
        offset = len(observations)
        for record in dataset.records:
            copied = dict(record)
            copied["observation_index"] = offset + int(record["observation_index"])
            records.append(copied)
        observations.extend(dataset.observations)
        games.extend(dataset.games)
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
    seat_distribution: dict[str, Counter[str]] = defaultdict(Counter)
    semantic_values: dict[str, list[float]] = defaultdict(list)
    source_trade_by_execution: dict[str, Counter[str]] = defaultdict(Counter)
    source_trade_objectives = Counter()
    source_trade_reasons = Counter()
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
        seat_distribution[str(row.get("seat", "unknown"))][goal] += 1
        for reason in row.get("reason_codes", []):
            goal_reason[goal][str(reason)] += 1
        if execution == "BUILD_ROAD":
            build_road_total += 1
        if execution in {"TRADE_PLAYER", "TRADE_BANK"}:
            available = bool(row.get("source_level_objective_available", False))
            source_trade_by_execution[execution][
                "available" if available else "unavailable"
            ] += 1
            if available:
                source_trade_objectives[_goal_label(row.get("trade_objective"))] += 1
            for trade_reason in row.get("trade_reason_codes", []):
                source_trade_reasons[str(trade_reason)] += 1
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
        "objective_goal_by_seat": {
            seat: _count_summary(counter, sum(counter.values()))
            for seat, counter in sorted(seat_distribution.items())
        },
        "source_trade_analysis": {
            "by_execution_mode": {
                execution: {
                    "total": sum(counter.values()),
                    "source_objective_available": counter["available"],
                    "source_objective_rate": (
                        counter["available"] / sum(counter.values())
                        if sum(counter.values()) else 0.0
                    ),
                }
                for execution, counter in sorted(source_trade_by_execution.items())
            },
            "objective_distribution": dict(sorted(source_trade_objectives.items())),
            "reason_codes": dict(sorted(source_trade_reasons.items())),
        },
        "goal_score_semantics": score_summary,
        "turn_band_definition": {
            "early": "turn <= 25", "mid": "26 <= turn <= 50", "late": "turn >= 51",
        },
    }


def summarize_profile_stability(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """profile別taxonomyと、Goal比率の最大差・TV距離を返す。"""
    rows = list(records)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("opponent_profile", "unspecified"))].append(row)
    profiles = {
        profile: summarize_taxonomy(profile_rows)
        for profile, profile_rows in sorted(grouped.items())
    }
    goals = [goal.value for goal in ObjectiveGoal] + [NONE_LABEL]
    goal_differences = {}
    for goal in goals:
        rates = {
            profile: float(summary["objective_goal_distribution"][goal]["rate"])
            for profile, summary in profiles.items()
        }
        counts = {
            profile: int(summary["objective_goal_distribution"][goal]["count"])
            for profile, summary in profiles.items()
        }
        goal_differences[goal] = {
            "rates": rates,
            "counts": counts,
            "max_rate_difference": (
                max(rates.values()) - min(rates.values()) if rates else 0.0
            ),
        }
    pairwise = {}
    names = sorted(profiles)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            absolute = [
                abs(goal_differences[goal]["rates"][left]
                    - goal_differences[goal]["rates"][right])
                for goal in goals
            ]
            pairwise[f"{left}_vs_{right}"] = {
                "total_variation_distance": 0.5 * sum(absolute),
                "maximum_goal_rate_difference": max(absolute, default=0.0),
            }
    rare_goals = (
        ObjectiveGoal.DEVELOPMENT.value,
        ObjectiveGoal.ROAD_TITLE.value,
        ObjectiveGoal.KNIGHT_TITLE.value,
        ObjectiveGoal.HOLD.value,
    )
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "profiles": profiles,
        "goal_distribution_difference": goal_differences,
        "pairwise_distribution_distance": pairwise,
        "rare_goal_counts": {
            profile: {
                goal: summary["objective_goal_distribution"][goal]["count"]
                for goal in rare_goals
            }
            for profile, summary in profiles.items()
        },
    }


def _intent_run_lengths(rows: list[dict[str, Any]], intent: str) -> list[int]:
    ordered = sorted(rows, key=lambda row: (
        str(row.get("game_id")), int(row.get("player_id", 0)),
        int(row.get("observation_index", 0)),
    ))
    lengths: list[int] = []
    current_key: tuple[str, int] | None = None
    run = 0
    for row in ordered:
        key = (str(row.get("game_id")), int(row.get("player_id", 0)))
        if key != current_key:
            if run:
                lengths.append(run)
            current_key = key
            run = 0
        state = row.get("strategic_intent_states", {}).get(intent, "unknown")
        if state == "active":
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def _persistence_summary(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {
            "run_count": 0, "mean": 0.0, "median": 0.0, "max": 0,
            "one_decision_rate": 0.0, "two_plus_rate": 0.0,
            "five_plus_rate": 0.0,
        }
    return {
        "run_count": len(lengths), "mean": fmean(lengths),
        "median": median(lengths), "max": max(lengths),
        "one_decision_rate": sum(item == 1 for item in lengths) / len(lengths),
        "two_plus_rate": sum(item >= 2 for item in lengths) / len(lengths),
        "five_plus_rate": sum(item >= 5 for item in lengths) / len(lengths),
    }


def summarize_strategic_intents(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """新5 Intentのcoverage・共起・持続性・Action attributionを集計する。"""
    from .strategic_intent import STRATEGIC_INTENTS

    rows = list(records)
    names = [item.value for item in STRATEGIC_INTENTS]
    total = len(rows)
    states = {name: Counter() for name in names}
    profile_states: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {name: Counter() for name in names}
    )
    cooccurrence = {name: Counter({other: 0 for other in names}) for name in names}
    active_count = Counter()
    sources = Counter()
    execution_sources: dict[str, Counter[str]] = defaultdict(Counter)
    intent_attribution: dict[str, Counter[str]] = defaultdict(Counter)
    source_explanation: dict[str, Counter[str]] = defaultdict(Counter)
    buy_development = Counter()
    build_road = Counter()
    active_not_contributed = Counter()
    contributed_without_active = Counter()
    new_cards_dependency = 0
    new_cards_purchase_dependency = 0
    new_knight_use_dependency = 0

    for row in rows:
        intent_states = row.get("strategic_intent_states", {})
        contribution = row.get("reason_contributed_to_selection", {})
        profile = str(row.get("opponent_profile", "unknown"))
        for name in names:
            state = str(intent_states.get(name, "unknown"))
            states[name][state] += 1
            profile_states[profile][name][state] += 1
            if state == "active" and contribution.get(name) is not True:
                active_not_contributed[name] += 1
            if contribution.get(name) is True and state != "active":
                contributed_without_active[name] += 1
        active = [name for name in names if intent_states.get(name) == "active"]
        active_count[str(min(len(active), 3)) + ("+" if len(active) >= 3 else "")] += 1
        for left in active:
            for right in active:
                cooccurrence[left][right] += 1

        execution = str(row.get("selected_execution", "OTHER"))
        source = str(row.get("execution_source", "UNKNOWN"))
        sources[source] += 1
        execution_sources[execution][source] += 1
        any_contribution = any(value is True for value in contribution.values())
        intent_attribution[execution][
            "attributed" if any_contribution else "unattributed"
        ] += 1
        source_explanation[execution][
            "planner_source" if source not in {"BASE_PPO", "UNKNOWN"}
            else "base_ppo_or_unknown"
        ] += 1
        if execution == "BUY_DEVELOPMENT":
            knight = contribution.get("KNIGHT_TITLE") is True
            robber = contribution.get("ROBBER_RELIEF") is True
            if knight and robber:
                buy_development["KNIGHT_TITLE+ROBBER_RELIEF"] += 1
            elif knight:
                buy_development["KNIGHT_TITLE"] += 1
            elif robber:
                buy_development["ROBBER_RELIEF"] += 1
            elif source == "BASE_PPO":
                buy_development["BASE_PPO"] += 1
            else:
                buy_development["INTENT_UNKNOWN_OR_OTHER"] += 1
        if execution == "BUILD_ROAD":
            settlement = contribution.get("SETTLEMENT") is True
            road_title = contribution.get("ROAD_TITLE") is True
            if settlement and road_title:
                build_road["SETTLEMENT+ROAD_TITLE"] += 1
            elif settlement:
                build_road["SETTLEMENT"] += 1
            elif road_title:
                build_road["ROAD_TITLE"] += 1
            else:
                build_road["OTHER"] += 1
        if row.get("new_development_cards_blocked") is True:
            new_cards_dependency += 1
        if row.get("new_development_cards_blocked_purchase") is True:
            new_cards_purchase_dependency += 1
        if row.get("new_knight_blocked_use") is True:
            new_knight_use_dependency += 1

    coverage = {}
    for name in names:
        active = states[name]["active"]
        inactive = states[name]["inactive"]
        unknown = states[name]["unknown"]
        known = active + inactive
        coverage[name] = {
            "active": active, "inactive": inactive, "unknown": unknown,
            "known_rate": known / total if total else 0.0,
            "positive_rate_within_known": active / known if known else 0.0,
            "negative_rate_within_known": inactive / known if known else 0.0,
        }

    profile_coverage = {}
    for profile, intent_counters in sorted(profile_states.items()):
        profile_total = sum(intent_counters[names[0]].values()) if names else 0
        profile_coverage[profile] = {}
        for name in names:
            active = intent_counters[name]["active"]
            inactive = intent_counters[name]["inactive"]
            unknown = intent_counters[name]["unknown"]
            known = active + inactive
            profile_coverage[profile][name] = {
                "active": active, "inactive": inactive, "unknown": unknown,
                "known_rate": known / profile_total if profile_total else 0.0,
                "positive_rate_within_known": active / known if known else 0.0,
                "negative_rate_within_known": inactive / known if known else 0.0,
            }
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "decision_count": total,
        "intent_coverage": coverage,
        "intent_coverage_by_profile": profile_coverage,
        "active_cooccurrence_counts": {
            name: dict(counter) for name, counter in cooccurrence.items()
        },
        "active_goal_count_distribution": {
            key: {"count": active_count[key],
                  "rate": active_count[key] / total if total else 0.0}
            for key in ("0", "1", "2", "3+")
        },
        "persistence": {
            name: _persistence_summary(_intent_run_lengths(rows, name))
            for name in names
        },
        "execution_source_distribution": dict(sources),
        "execution_source_by_execution": {
            execution: dict(counter)
            for execution, counter in sorted(execution_sources.items())
        },
        "intent_attribution_coverage": {
            execution: {
                **dict(counter),
                "attributed_rate": (
                    counter["attributed"] / sum(counter.values())
                    if sum(counter.values()) else 0.0
                ),
            }
            for execution, counter in sorted(intent_attribution.items())
        },
        "source_level_explanation_coverage": {
            execution: {
                **dict(counter),
                "planner_source_rate": (
                    counter["planner_source"] / sum(counter.values())
                    if sum(counter.values()) else 0.0
                ),
            }
            for execution, counter in sorted(source_explanation.items())
        },
        "buy_development_attribution": dict(buy_development),
        "build_road_attribution": dict(build_road),
        "active_but_not_contributed": dict(active_not_contributed),
        "contributed_without_active": dict(contributed_without_active),
        "new_development_cards_dependency_count": new_cards_dependency,
        "new_development_cards_blocked_purchase_count": (
            new_cards_purchase_dependency
        ),
        "new_knight_blocked_use_count": new_knight_use_dependency,
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
    stability = summarize_profile_stability(dataset.records)
    strategic_summary = summarize_strategic_intents(dataset.records)
    files = {
        "records": "decisions.jsonl",
        "observations": "observations.npz",
        "games": "games.jsonl",
        "summary": "taxonomy_summary.json",
        "profile_stability": "profile_stability.json",
        "strategic_intents": "strategic_intent_summary.json",
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
    (output_dir / files["profile_stability"]).write_text(
        json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output_dir / files["strategic_intents"]).write_text(
        json.dumps(strategic_summary, ensure_ascii=False, indent=2), encoding="utf-8",
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
    return {
        "manifest": manifest, "summary": summary,
        "profile_stability": stability,
        "strategic_intents": strategic_summary,
    }


def verify_action_parity(
    plain_agent: SettlementPlanningAgent,
    diagnostic_agent: SettlementPlanningAgent,
    seeds: Iterable[int],
    *,
    opponent_profile: str = "rule",
    opponent_factories: Mapping[str, AgentFactory] | None = None,
    champion_model_id: str = "ppo_gnn_board_65k_s03_exp_v003",
    robber_model_id: str = "ppo_robber_value_73k_s01_exp_v001",
) -> dict[str, Any]:
    """診断なし/ありで全Action列・結果・最終GameStateが一致するか確認する。"""
    checked = 0
    all_actions = 0
    for index, seed in enumerate(seeds):
        champion_seat = (index % 4) + 1
        plain = play_teacher_episode(
            plain_agent, int(seed), champion_seat=champion_seat,
            opponent_profile=opponent_profile,
            opponent_factories=opponent_factories,
            champion_model_id=champion_model_id,
            robber_model_id=robber_model_id,
            collect_reports=False,
        )
        diagnostic = play_teacher_episode(
            diagnostic_agent, int(seed), champion_seat=champion_seat,
            opponent_profile=opponent_profile,
            opponent_factories=opponent_factories,
            champion_model_id=champion_model_id,
            robber_model_id=robber_model_id,
            collect_reports=True,
        )
        if plain.action_trace != diagnostic.action_trace:
            raise AssertionError(f"seed {seed}: Action列が一致しません。")
        result_keys = ("winner", "final_vp", "final_rank", "final_scores", "turn")
        if any(plain.game[key] != diagnostic.game[key] for key in result_keys):
            raise AssertionError(f"seed {seed}: 最終結果が一致しません。")
        if plain.final_views != diagnostic.final_views:
            raise AssertionError(f"seed {seed}: 最終GameState表示が一致しません。")
        if plain.final_state != diagnostic.final_state:
            raise AssertionError(f"seed {seed}: 全GameStateが一致しません。")
        if plain.game["rng_counter"] != diagnostic.game["rng_counter"]:
            raise AssertionError(f"seed {seed}: RNG counterが一致しません。")
        checked += 1
        all_actions += len(plain.action_trace)
    return {
        "opponent_profile": opponent_profile,
        "games": checked, "all_actions": all_actions,
        "action_trace_equal": True, "winner_equal": True,
        "final_score_equal": True, "final_rank_equal": True,
        "final_game_state_equal": True, "rng_counter_equal": True,
    }
