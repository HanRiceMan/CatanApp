"""同じ盤面で中盤戦略を分岐し、目標価値学習用の終局結果を収集する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .evaluate import EvaluationConfig, evaluate_agent
from .goal_value import GOAL_CONTEXT_VERSION, GoalContextRecordingAgent
from .settlement_planning_agent import SettlementPlanningAgent
from .train_development_budget import summary, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=530001)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.games <= 0:
        parser.error("実験名・ゲーム数を確認してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent(args.model_id, device="cpu")
    adopted = {
        "target_sites": 4, "max_wait_rounds": 9,
        "first_city_max_deficit": 2, "max_city_wait_rounds": 9,
        "prioritize_affordable_city": True, "prioritize_immediate_win": True,
        "prioritize_immediate_titles": True, "title_horizon": 2,
    }
    snapshots: dict[int, dict] = {}
    agents = {
        "balanced": GoalContextRecordingAgent(
            SettlementPlanningAgent(base, **adopted), snapshots,
        ),
        "fifth_site": SettlementPlanningAgent(base, **{
            **adopted, "target_sites": 5,
        }),
        "first_city": SettlementPlanningAgent(base, **{
            **adopted, "target_cities": 1,
        }),
        "early_titles": SettlementPlanningAgent(base, **{
            **adopted, "title_horizon": 3, "title_min_score": 4,
        }),
        "army_purchase": SettlementPlanningAgent(base, **{
            **adopted, "buy_for_largest_army": True,
        }),
    }
    config = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                              policy_compatible_opponents=True, observation_version="v2")
    reports, summaries = {}, {}
    seeds = list(range(args.seed_start, args.seed_start + args.games))
    for name, candidate in agents.items():
        report = evaluate_agent(candidate, seeds, config=config, agent_name=name).to_dict()
        if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
            raise AssertionError(f"{name}: 評価が正常に完了していません。")
        write(output / f"evaluation_{name}.json", report)
        reports[name], summaries[name] = report, summary(report)
        print(json.dumps({"stage": name, **summaries[name]}, ensure_ascii=False), flush=True)

    episodes_by_strategy = {
        name: {episode["seed"]: episode for episode in report["episodes"]}
        for name, report in reports.items()
    }
    rows = []
    for seed in seeds:
        if seed not in snapshots:
            continue
        outcomes = {
            name: {
                "won": episodes[seed]["learner_won"],
                "score": episodes[seed]["learner_score"],
                "rank": episodes[seed]["learner_rank"],
                "turn_number": episodes[seed]["turn_number"],
            }
            for name, episodes in episodes_by_strategy.items()
        }
        rows.append({"seed": seed, "context": snapshots[seed], "outcomes": outcomes})
    oracle_wins = sum(any(value["won"] for value in row["outcomes"].values()) for row in rows)
    total_oracle_wins = sum(any(episodes[seed]["learner_won"]
                                for episodes in episodes_by_strategy.values())
                            for seed in seeds)
    write(output / "goal_outcomes.json", {
        "definition": vars(args),
        "context_version": GOAL_CONTEXT_VERSION,
        "strategies": list(agents),
        "rows": rows,
        "summary": {
            "recorded_contexts": len(rows),
            "context_oracle_wins": oracle_wins,
            "context_oracle_win_rate": oracle_wins / len(rows) if rows else 0.0,
            "total_games": len(seeds),
            "total_oracle_wins": total_oracle_wins,
            "total_oracle_win_rate": total_oracle_wins / len(seeds),
            "strategies": summaries,
        },
    })
    print(json.dumps({"stage": "oracle", "contexts": len(rows),
                      "context_wins": oracle_wins,
                      "context_win_rate": oracle_wins / len(rows) if rows else 0.0,
                      "total_games": len(seeds), "total_wins": total_oracle_wins,
                      "total_win_rate": total_oracle_wins / len(seeds)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
