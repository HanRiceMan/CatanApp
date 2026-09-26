"""5拠点計画のAction局面を収集し、短期建設結果と終局結果を付ける。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from app.agents.ppo import PPOAgent
from app.domain.actions import (Action, BankTradeAction, BuildCityAction,
                                BuildRoadAction, BuildSettlementAction,
                                BuyDevelopmentAction, EndTurnAction)
from app.domain.game import (BUILD_COSTS, RESOURCES, GameState,
                             _port_ratio, legal_city_ids, legal_road_ids,
                             legal_settlement_ids, player_for)

from .action_space import action_to_id, get_action_mask
from .construction_metrics import construction_summary
from .evaluate import EvaluationConfig, evaluate_agent
from .evaluate_construction_priority import VALIDATED_PLANNING
from .expansion_planner import (analyze_expansion_plan, estimated_build_wait,
                                player_production)
from .observation import encode_observation, get_observation
from .reward import actual_score
from .settlement_planning_agent import SettlementPlanningAgent
from .train_development_budget import summary, write
from .victory_race import public_score


def _family(action: Action) -> str:
    if isinstance(action, BuildSettlementAction):
        return "settlement"
    if isinstance(action, BuildCityAction):
        return "city"
    if isinstance(action, BuildRoadAction):
        return "road"
    if isinstance(action, BuyDevelopmentAction):
        return "development"
    if isinstance(action, BankTradeAction):
        return "bank_trade"
    if isinstance(action, EndTurnAction):
        return "end_turn"
    return type(action).__name__.removesuffix("Action").lower()


def _deficit(resources: dict[str, int], cost: dict[str, int]) -> int:
    return sum(max(0, amount - resources[resource])
               for resource, amount in cost.items())


class ConstructionDecisionRecorder:
    def __init__(self, delegate: SettlementPlanningAgent, base: PPOAgent):
        self.delegate = delegate
        self.base = base
        self.decisions: list[dict] = []
        self.observations: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self._turns: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._decisions_in_turn: Counter[tuple[int, int, int]] = Counter()

    def select_action(self, game: GameState, player_id: int) -> Action:
        planned = self.delegate.select_action(game, player_id)
        if game.phase != "action":
            return planned
        base = self.base.select_action(game, player_id)
        key = (game.seed, player_id)
        if not self._turns[key] or self._turns[key][-1] != game.turn_number:
            self._turns[key].append(game.turn_number)
        own_turn = len(self._turns[key])
        turn_key = (game.seed, player_id, game.turn_number)
        decision_in_turn = self._decisions_in_turn[turn_key]
        self._decisions_in_turn[turn_key] += 1
        mask = get_action_mask(game, player_id)
        player = player_for(game, player_id)
        resources = {resource: player.resources[resource] for resource in RESOURCES}
        production = player_production(game, player_id)
        plan = analyze_expansion_plan(game, player_id)
        target = plan.selected_target
        legal_settlements = legal_settlement_ids(game, player_id, initial=False)
        legal_cities = legal_city_ids(game, player_id)
        legal_roads = legal_road_ids(game, player_id, initial=False)
        opponents = [other for other in game.players if other.id != player_id]
        row = {
            "seed": game.seed,
            "player_id": player_id,
            "seat": game.seat_order.index(player_id) + 1,
            "revision": game.revision,
            "turn_number": game.turn_number,
            "own_turn": own_turn,
            "decision_in_turn": decision_in_turn,
            "is_first_action_decision": decision_in_turn == 0,
            "score": actual_score(game, player_id),
            "public_score": public_score(game, player_id),
            "rival_max_score": max(public_score(game, other.id) for other in opponents),
            "settlements": player.settlements,
            "cities": player.cities,
            "sites": player.settlements + player.cities,
            "building_points": player.settlements + 2 * player.cities,
            "resources": resources,
            "resource_count": sum(resources.values()),
            "production": production,
            "production_total": sum(production.values()),
            "port_ratios": {resource: _port_ratio(game, player_id, resource)
                            for resource in RESOURCES},
            "settlement_deficit": _deficit(resources, BUILD_COSTS["settlement"]),
            "city_deficit": _deficit(resources, BUILD_COSTS["city"]),
            "settlement_wait": (target.wait_rounds if target is not None else None),
            "city_wait": (estimated_build_wait(game, player_id, BUILD_COSTS["city"])
                          if player.settlements and player.cities < 4 else None),
            "target_vertex_id": target.vertex_id if target is not None else None,
            "target_additional_roads": (target.additional_roads if target is not None else None),
            "target_pips": target.total_pips if target is not None else None,
            "target_site_value": target.site_value if target is not None else None,
            "target_plan_value": target.plan_value if target is not None else None,
            "has_connected_target": plan.best_connected_target is not None,
            "should_wait_for_settlement": plan.should_wait_for_settlement,
            "legal_settlement_count": len(legal_settlements),
            "legal_city_count": len(legal_cities),
            "legal_road_count": len(legal_roads),
            "chosen_action_id": action_to_id(planned),
            "chosen_family": _family(planned),
            "base_action_id": action_to_id(base),
            "base_family": _family(base),
            "planner_changed_action": action_to_id(planned) != action_to_id(base),
            "paid_road": isinstance(planned, BuildRoadAction) and not game.free_road_remaining,
        }
        self.decisions.append(row)
        observation = encode_observation(get_observation(
            game, player_id, version=self.base.manifest.observation_version,
        ))
        self.observations.append(observation.astype(np.float32, copy=False))
        self.masks.append(mask.astype(np.bool_, copy=False))
        return planned


def _add_labels(decisions: list[dict], report: dict) -> None:
    outcomes = {episode["seed"]: episode for episode in report["episodes"]}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in decisions:
        grouped[row["seed"]].append(row)
    for seed, rows in grouped.items():
        episode = outcomes[seed]
        for index, row in enumerate(rows):
            future = rows[index:]
            builds = [candidate for candidate in future
                      if candidate["chosen_family"] in {"settlement", "city"}]
            settlements = [candidate for candidate in future
                           if candidate["chosen_family"] == "settlement"]
            cities = [candidate for candidate in future
                      if candidate["chosen_family"] == "city"]
            row["next_build_own_turns"] = (
                builds[0]["own_turn"] - row["own_turn"] if builds else None
            )
            row["next_settlement_own_turns"] = (
                settlements[0]["own_turn"] - row["own_turn"] if settlements else None
            )
            row["next_city_own_turns"] = (
                cities[0]["own_turn"] - row["own_turn"] if cities else None
            )
            row["construction_points_next_4_turns"] = sum(
                candidate["chosen_family"] in {"settlement", "city"}
                for candidate in future
                if candidate["own_turn"] <= row["own_turn"] + 4
            )
            row["build_within_2_turns"] = (
                row["next_build_own_turns"] is not None
                and row["next_build_own_turns"] <= 2
            )
            row["learner_won"] = episode["learner_won"]
            row["final_score"] = episode["learner_score"]
            row["final_rank"] = episode["learner_rank"]
            row["final_sites"] = (episode["learner_final_pieces"]["settlements"]
                                  + episode["learner_final_pieces"]["cities"])
            row["final_cities"] = episode["learner_final_pieces"]["cities"]


def _group(rows: list[dict], field: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    result = {}
    for key, values in sorted(groups.items()):
        delays = [row["next_build_own_turns"] for row in values
                  if row["next_build_own_turns"] is not None]
        result[key] = {
            "decisions": len(values),
            "episodes": len({row["seed"] for row in values}),
            "win_rate": fmean(row["learner_won"] for row in values),
            "average_final_score": fmean(row["final_score"] for row in values),
            "build_within_2_rate": fmean(row["build_within_2_turns"] for row in values),
            "average_next_build_turns_if_reached": fmean(delays) if delays else None,
            "average_construction_points_next_4": fmean(
                row["construction_points_next_4_turns"] for row in values
            ),
        }
    return result


def _analysis(decisions: list[dict], report: dict) -> dict:
    first = [row for row in decisions if row["is_first_action_decision"]]
    transitions = [{**row, "base_to_chosen":
                    f"{row['base_family']}->{row['chosen_family']}"}
                   for row in first]
    spending = [row for row in decisions
                if row["chosen_family"] in {"road", "development"}]
    connected = [row for row in first if row["has_connected_target"]]
    post_target_connected = [
        row for row in first
        if (row["sites"] >= 5 and row["settlements"] < 5 and row["cities"] > 0
            and row["should_wait_for_settlement"])
    ]
    post_target_no_settlement_piece = [
        row for row in first
        if (row["sites"] >= 5 and row["settlements"] >= 5
            and row["should_wait_for_settlement"])
    ]
    distance_three = [row for row in first if row["target_additional_roads"] == 3]
    bottleneck_rows = []
    for row in first:
        if row["target_vertex_id"] is None:
            category = "no_target"
        elif row["target_additional_roads"] == 0:
            missing = tuple(resource for resource in RESOURCES
                            if row["resources"][resource]
                            < BUILD_COSTS["settlement"].get(resource, 0))
            category = "connected_missing_" + ("_".join(missing) if missing else "none")
        else:
            category = f"road_distance_{row['target_additional_roads']}"
        bottleneck_rows.append({**row, "bottleneck": category})
    avoidable = {
        "paid_road_with_connected_target": [
            row for row in decisions if row["paid_road"] and row["has_connected_target"]
        ],
        "development_when_build_wait_at_most_2": [
            row for row in decisions
            if row["chosen_family"] == "development"
            and min(value for value in (row["settlement_wait"], row["city_wait"], 99)
                    if value is not None) <= 2
        ],
        "end_turn_with_one_card_construction_deficit": [
            row for row in decisions if row["chosen_family"] == "end_turn"
            and min(row["settlement_deficit"], row["city_deficit"]) <= 1
        ],
    }
    negative_post_target_roads = [
        row for row in post_target_connected
        if (row["chosen_family"] == "road"
            and row["construction_points_next_4_turns"] == 0)
    ]
    positive_post_target_paths = [
        row for row in post_target_connected
        if (row["chosen_family"] in {"bank_trade", "end_turn", "settlement", "city"}
            and row["construction_points_next_4_turns"] >= 1)
    ]
    score_bands = Counter()
    for episode in report["episodes"]:
        score = episode["learner_score"]
        band = "win" if episode["learner_won"] else (
            "loss_8_9" if score >= 8 else "loss_0_7"
        )
        score_bands[band] += 1
    return {
        "evaluation": {**summary(report), "construction": construction_summary(report)},
        "decision_count": len(decisions),
        "first_action_decision_count": len(first),
        "planner_changed_action_rate": fmean(
            row["planner_changed_action"] for row in decisions
        ),
        "first_decision_by_chosen_family": _group(first, "chosen_family"),
        "first_decision_by_base_family": _group(first, "base_family"),
        "first_decision_by_base_to_chosen": _group(transitions, "base_to_chosen"),
        "first_decision_by_bottleneck": _group(bottleneck_rows, "bottleneck"),
        "connected_first_decisions": _group(connected, "chosen_family"),
        "post_target_connected_first_decisions": _group(
            post_target_connected, "chosen_family"
        ),
        "post_target_no_settlement_piece_first_decisions": _group(
            post_target_no_settlement_piece, "chosen_family"
        ),
        "distance_three_first_decisions": _group(distance_three, "chosen_family"),
        "spending_decisions": _group(spending, "chosen_family"),
        "episode_score_bands": dict(score_bands),
        "proposed_training_subsets": {
            "negative_post_target_road_without_build_in_4_turns": {
                "decisions": len(negative_post_target_roads),
                "episodes": len({row["seed"] for row in negative_post_target_roads}),
            },
            "positive_post_target_construction_path": {
                "decisions": len(positive_post_target_paths),
                "episodes": len({row["seed"] for row in positive_post_target_paths}),
            },
        },
        "diagnostic_patterns": {
            name: {
                "decisions": len(rows),
                "episodes": len({row["seed"] for row in rows}),
                "loss_episode_rate": (fmean(not row["learner_won"] for row in rows)
                                      if rows else 0.0),
                "build_within_2_rate": (fmean(row["build_within_2_turns"] for row in rows)
                                        if rows else 0.0),
            }
            for name, rows in avoidable.items()
        },
        "information_boundary": (
            "入力は自分の手札と公開盤面から作るv2観測・合法手・公開計画特徴。"
            "相手手札内訳、発展山札順、未来の出目は保存しない。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=920001)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.games <= 0:
        parser.error("実験名・ゲーム数を確認してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent("ppo_gnn_board_65k_s03_exp_v003", device="cpu")
    planner = SettlementPlanningAgent(base, **VALIDATED_PLANNING)
    recorder = ConstructionDecisionRecorder(planner, base)
    config = EvaluationConfig(
        heuristic_opponents=3, rotate_player_ids=True,
        policy_compatible_opponents=True, observation_version="v2",
    )
    report = evaluate_agent(
        recorder, range(args.seed_start, args.seed_start + args.games),
        config=config, agent_name="construction_decision_collection",
    ).to_dict()
    if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
        raise AssertionError("データ収集対局が正常に完了していません。")
    _add_labels(recorder.decisions, report)
    observations = np.stack(recorder.observations).astype(np.float32)
    masks = np.stack(recorder.masks).astype(np.bool_)
    np.savez_compressed(
        output / "decision_tensors.npz", observations=observations, masks=masks,
        chosen_action_ids=np.asarray(
            [row["chosen_action_id"] for row in recorder.decisions], dtype=np.int16,
        ),
        base_action_ids=np.asarray(
            [row["base_action_id"] for row in recorder.decisions], dtype=np.int16,
        ),
        next_build_own_turns=np.asarray([
            -1 if row["next_build_own_turns"] is None else row["next_build_own_turns"]
            for row in recorder.decisions
        ], dtype=np.int16),
        construction_points_next_4_turns=np.asarray([
            row["construction_points_next_4_turns"] for row in recorder.decisions
        ], dtype=np.int8),
        wins=np.asarray([row["learner_won"] for row in recorder.decisions], dtype=np.bool_),
        final_scores=np.asarray(
            [row["final_score"] for row in recorder.decisions], dtype=np.int8,
        ),
    )
    write(output / "definition.json", {"arguments": vars(args),
                                        "planning": VALIDATED_PLANNING})
    write(output / "evaluation.json", report)
    write(output / "construction_decisions.json", recorder.decisions)
    analysis = _analysis(recorder.decisions, report)
    write(output / "analysis.json", analysis)
    print(json.dumps({"stage": "complete", "decisions": len(recorder.decisions),
                      "games": args.games, "evaluation": analysis["evaluation"],
                      "patterns": analysis["diagnostic_patterns"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
