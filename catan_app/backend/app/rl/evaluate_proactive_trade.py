"""プレイヤー間交渉を有効にした直接対局で、保守的な自発交渉を比較する。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from app.agents.heuristic import HeuristicAgent
from app.agents.ppo import PPOAgent
from app.domain.actions import (BuildCityAction, BuildRoadAction,
                                BuildSettlementAction, BuyDevelopmentAction,
                                PlaceInitialRoadAction, PlaceInitialSettlementAction,
                                ProposeTradeAction)
from app.domain.game import (apply_action, create_game, pending_ai_player_id,
                             player_for)

from .diagnostics_metrics import endgame_rank
from .decision_diagnostics import diagnose_action
from .evaluate_construction_priority import VALIDATED_PLANNING
from .expansion_planner import player_production
from .initial_road_planner import assess_initial_road
from .reward import actual_score
from .settlement_planning_agent import SettlementPlanningAgent
from .train_development_budget import write


def play_episode(agent, seed: int, learner_id: int) -> dict:
    game = create_game(seed, ai_player_ids=(1, 2, 3, 4))
    opponents = {player_id: HeuristicAgent() for player_id in range(1, 5)
                 if player_id != learner_id}
    actions: Counter[str] = Counter()
    initial_vertices: list[int] = []
    initial_road_assessments: list[dict] = []
    development_buy_sites: list[int] = []
    development_buy_scores: list[int] = []
    for step in range(100_000):
        if game.phase == "game_over":
            break
        actor = pending_ai_player_id(game)
        if actor is None:
            raise RuntimeError(f"AI担当を決められません: {game.phase}")
        selected = (agent if actor == learner_id else opponents[actor]).select_action(
            game, actor
        )
        if actor == learner_id:
            if isinstance(selected, BuildSettlementAction):
                actions["build_settlement"] += 1
            elif isinstance(selected, BuildCityAction):
                actions["build_city"] += 1
            elif isinstance(selected, BuildRoadAction) and not game.free_road_remaining:
                actions["build_road_paid"] += 1
                road_view = diagnose_action(game, actor, selected, agent)
                if (road_view is not None
                        and road_view["longest_after"] == road_view["longest_before"]
                        and not road_view["expansion_recommended"]):
                    actions["unproductive_paid_road"] += 1
                if road_view is not None and road_view["blocked_endpoints"]:
                    actions["blocked_endpoint_paid_road"] += 1
                if road_view is not None and road_view["contested_endpoints"]:
                    actions["contested_endpoint_paid_road"] += 1
            elif isinstance(selected, BuyDevelopmentAction):
                actions["buy_development"] += 1
                development_buy_sites.append(
                    player_for(game, actor).settlements + player_for(game, actor).cities
                )
                development_buy_scores.append(actual_score(game, actor))
            elif isinstance(selected, ProposeTradeAction):
                actions["propose_trade"] += 1
                if any(sum(offer.give.values()) >= 2 * sum(offer.want.values())
                       for offer in selected.offers):
                    actions["propose_trade_2_to_1"] += 1
                if sum(player_for(game, actor).resources.values()) >= 8:
                    actions["propose_trade_at_burst_risk"] += 1
            elif isinstance(selected, PlaceInitialSettlementAction):
                initial_vertices.append(selected.vertex_id)
            elif isinstance(selected, PlaceInitialRoadAction):
                initial_road_assessments.append(
                    assess_initial_road(game, actor, selected.edge_id)
                )
        apply_action(game, actor, selected, game.revision,
                     f"trade-eval-{step:06d}")
        game.ai_action_number += 1
    else:
        raise RuntimeError(f"seed {seed}: 100000操作で終了しませんでした。")

    scores = {player.id: actual_score(game, player.id) for player in game.players}
    rank, _ = endgame_rank(scores, learner_id, game.winner_id)
    player = player_for(game, learner_id)
    proposed = [entry for entry in game.activity_log
                if entry["kind"] == "trade"
                and entry.get("trade", {}).get("proposer_id") == learner_id
                and not entry.get("trade", {}).get("is_counter", False)]
    accepted = [entry for entry in game.activity_log
                if entry["kind"] == "trade_accepted"
                and entry.get("trade", {}).get("proposer_id") == learner_id]
    accepted = [entry for entry in accepted
                if not entry.get("trade", {}).get("is_counter", False)]
    return {
        "seed": seed,
        "learner_id": learner_id,
        "winner_id": game.winner_id,
        "learner_won": game.winner_id == learner_id,
        "learner_score": scores[learner_id],
        "learner_rank": rank,
        "seat_order": list(game.seat_order),
        "learner_seat": game.seat_order.index(learner_id) + 1,
        "player_scores": {str(key): value for key, value in scores.items()},
        "turn_number": game.turn_number,
        "settlements": player.settlements,
        "cities": player.cities,
        "roads": sum(owner == learner_id for owner in game.roads.values()),
        "unproductive_paid_roads": actions["unproductive_paid_road"],
        "blocked_endpoint_paid_roads": actions["blocked_endpoint_paid_road"],
        "contested_endpoint_paid_roads": actions["contested_endpoint_paid_road"],
        "actions": dict(actions),
        "initial_vertices": initial_vertices,
        "initial_road_assessments": initial_road_assessments,
        "initial_road_zero_targets": sum(
            item["target_count"] == 0 for item in initial_road_assessments
        ),
        "initial_road_contested": sum(
            item["opponent_road_count"] > 0 for item in initial_road_assessments
        ),
        "initial_road_sealed": sum(
            item["opponent_road_count"] > 0 and item["open_exit_count"] == 0
            for item in initial_road_assessments
        ),
        "initial_road_target_value": float(np.mean([
            item["best_target_value"] for item in initial_road_assessments
        ])),
        "final_production": player_production(game, learner_id),
        "development_cards_remaining": len(player.development_cards),
        "development_buys": actions["buy_development"],
        "development_buy_sites": development_buy_sites,
        "development_buy_scores": development_buy_scores,
        "development_buys_at_two_sites": sum(
            site_count == 2 for site_count in development_buy_sites
        ),
        "development_buys_at_nine_points": sum(
            score == 9 for score in development_buy_scores
        ),
        "played_knights": player.played_knights,
        "has_longest_road": game.longest_road_holder_id == learner_id,
        "has_largest_army": game.largest_army_holder_id == learner_id,
        "trade_proposals": len(proposed),
        "trade_acceptances": len(accepted),
        "two_for_one_proposals": actions["propose_trade_2_to_1"],
        "burst_risk_trade_proposals": actions["propose_trade_at_burst_risk"],
    }


def summarize(episodes: list[dict]) -> dict:
    mean = lambda key: float(np.mean([episode[key] for episode in episodes]))
    proposals = sum(episode["trade_proposals"] for episode in episodes)
    acceptances = sum(episode["trade_acceptances"] for episode in episodes)
    return {
        "games": len(episodes),
        "win_rate": mean("learner_won"),
        "score": mean("learner_score"),
        "rank": mean("learner_rank"),
        "settlements": mean("settlements"),
        "cities": mean("cities"),
        "roads": mean("roads"),
        "unproductive_paid_roads": mean("unproductive_paid_roads"),
        "blocked_endpoint_paid_roads": mean("blocked_endpoint_paid_roads"),
        "contested_endpoint_paid_roads": mean("contested_endpoint_paid_roads"),
        "trade_proposals": proposals,
        "trade_acceptances": acceptances,
        "trade_acceptance_rate": acceptances / proposals if proposals else 0.0,
        "two_for_one_proposals": mean("two_for_one_proposals"),
        "burst_risk_trade_proposals": mean("burst_risk_trade_proposals"),
        "development_buys": mean("development_buys"),
        "zero_development_buy_rate": float(np.mean([
            episode["development_buys"] == 0 for episode in episodes
        ])),
        "development_buys_at_two_sites": mean("development_buys_at_two_sites"),
        "development_buys_at_nine_points": mean(
            "development_buys_at_nine_points"
        ),
        "played_knights": mean("played_knights"),
        "largest_army_rate": mean("has_largest_army"),
        "initial_road_zero_targets": mean("initial_road_zero_targets"),
        "initial_road_contested": mean("initial_road_contested"),
        "initial_road_sealed": mean("initial_road_sealed"),
        "initial_road_target_value": mean("initial_road_target_value"),
    }


def paired(reference: list[dict], candidate: list[dict]) -> dict:
    rng = np.random.default_rng(20260930)
    indices = rng.integers(0, len(reference), size=(4000, len(reference)))
    extractors = {
        "win_rate": lambda episode: float(episode["learner_won"]),
        "score": lambda episode: episode["learner_score"],
        "rank": lambda episode: episode["learner_rank"],
        "settlements": lambda episode: episode["settlements"],
        "cities": lambda episode: episode["cities"],
        "unproductive_paid_roads": lambda episode: episode[
            "unproductive_paid_roads"
        ],
        "development_buys": lambda episode: episode["development_buys"],
        "development_buys_at_two_sites": lambda episode: episode[
            "development_buys_at_two_sites"
        ],
        "development_buys_at_nine_points": lambda episode: episode[
            "development_buys_at_nine_points"
        ],
        "played_knights": lambda episode: episode["played_knights"],
        "two_for_one_proposals": lambda episode: episode["two_for_one_proposals"],
        "burst_risk_trade_proposals": lambda episode: episode[
            "burst_risk_trade_proposals"
        ],
        "initial_road_zero_targets": lambda episode: episode["initial_road_zero_targets"],
        "initial_road_contested": lambda episode: episode["initial_road_contested"],
        "initial_road_target_value": lambda episode: episode["initial_road_target_value"],
    }
    result = {}
    for name, extract in extractors.items():
        delta = np.array([extract(b) - extract(a)
                          for a, b in zip(reference, candidate)])
        interval = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
        result[name] = {
            "difference": float(delta.mean()),
            "paired_bootstrap_95_ci": interval.tolist(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=1100001)
    parser.add_argument("--only", choices=("baseline", "strategic_response",
                                            "proactive_trade", "scarcity_trade",
                                            "endgame_development", "threat_trade",
                                            "setup_ore_trade", "initial_road_planning",
                                            "adaptive_development",
                                            "adaptive_development_conservative",
                                            "adaptive_development_balanced",
                                            "surplus_trade",
                                            "purposeful_roads",
                                            "current_vs_surplus",
                                            "current_vs_purposeful",
                                            "city_reservation",
                                            "current_vs_city_reservation"))
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."} or args.games <= 0):
        parser.error("実験名・ゲーム数が不正です。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent("ppo_gnn_board_65k_s03_exp_v003", device="cpu")
    configs = {
        "baseline": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": False,
            "strategic_trade_response": False,
        },
        "strategic_response": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": False,
            "strategic_trade_response": True,
        },
        "proactive_trade": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": True,
            "strategic_trade_response": True,
        },
        "scarcity_trade": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": True,
            "strategic_trade_response": True,
            "proactive_trade_max_scarcity_deficit": 3,
        },
        "endgame_development": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": True,
            "strategic_trade_response": True,
            "proactive_trade_max_scarcity_deficit": 3,
            "endgame_development_fallback": True,
        },
        "threat_trade": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": True,
            "strategic_trade_response": True,
            "proactive_trade_max_scarcity_deficit": 3,
            "proactive_trade_opponent_score_limit": 7,
            "strategic_trade_opponent_score_limit": 8,
        },
        "setup_ore_trade": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": True,
            "strategic_trade_response": True,
            "proactive_trade_max_scarcity_deficit": 3,
            "setup_min_ore_pips": 1,
        },
        "initial_road_planning": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": True,
            "strategic_trade_response": True,
            "proactive_trade_max_scarcity_deficit": 3,
            "collision_aware_initial_roads": True,
        },
        "adaptive_development": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": True,
            "strategic_trade_response": True,
            "proactive_trade_max_scarcity_deficit": 3,
            "adaptive_development_purchase": True,
        },
        "adaptive_development_conservative": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": True,
            "strategic_trade_response": True,
            "proactive_trade_max_scarcity_deficit": 3,
            "adaptive_development_purchase": True,
            "development_max_build_delay": 1.0,
        },
        "adaptive_development_balanced": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": True,
            "strategic_trade_response": True,
            "proactive_trade_max_scarcity_deficit": 3,
            "adaptive_development_purchase": True,
            "development_stall_wait_rounds": 4.0,
            "development_max_build_delay": 1.0,
        },
        "surplus_trade": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": True,
            "strategic_trade_response": True,
            "proactive_trade_max_scarcity_deficit": 3,
            "strategic_surplus_trade": True,
        },
        "purposeful_roads": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": True,
            "strategic_trade_response": True,
            "proactive_trade_max_scarcity_deficit": 3,
            "strategic_surplus_trade": True,
            "purposeful_paid_roads": True,
        },
        "city_reservation": {
            **VALIDATED_PLANNING,
            "proactive_player_trade": True,
            "strategic_trade_response": True,
            "proactive_trade_max_scarcity_deficit": 3,
            "strategic_surplus_trade": True,
            "purposeful_paid_roads": True,
            "protect_city_bank_trade_reserve": True,
        },
    }
    if args.only == "current_vs_surplus":
        configs = {
            "baseline": configs["adaptive_development_balanced"],
            "surplus_trade": configs["surplus_trade"],
        }
    elif args.only == "current_vs_purposeful":
        configs = {
            "baseline": configs["surplus_trade"],
            "purposeful_roads": configs["purposeful_roads"],
        }
    elif args.only == "current_vs_city_reservation":
        configs = {
            "baseline": configs["purposeful_roads"],
            "city_reservation": configs["city_reservation"],
        }
    elif args.only:
        configs = {args.only: configs[args.only]}
    write(output / "definition.json", {"arguments": vars(args), "planning": configs,
                                        "player_trades_enabled": True})
    reports = {}
    for name, planning in configs.items():
        agent = SettlementPlanningAgent(base, **planning)
        episodes = [play_episode(agent, seed, (index % 4) + 1)
                    for index, seed in enumerate(
                        range(args.seed_start, args.seed_start + args.games)
                    )]
        reports[name] = episodes
        write(output / f"evaluation_{name}.json", {
            "agent_name": name, "episodes": episodes,
            "summary": summarize(episodes),
        })
        print(json.dumps({"stage": name, **summarize(episodes)},
                         ensure_ascii=False), flush=True)
    result = {"summaries": {name: summarize(episodes)
                            for name, episodes in reports.items()}}
    if "baseline" in reports:
        result["paired_vs_baseline"] = {
            name: paired(reports["baseline"], episodes)
            for name, episodes in reports.items() if name != "baseline"
        }
    write(output / "evaluation_summary.json", result)


if __name__ == "__main__":
    main()
