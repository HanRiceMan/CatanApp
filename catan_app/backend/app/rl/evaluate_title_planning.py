"""4拠点後の2点称号を、1手・2手先まで追う候補を比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .evaluate import EvaluationConfig, evaluate_agent
from .settlement_planning_agent import SettlementPlanningAgent
from .train_development_budget import paired_comparison, summary, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=430001)
    parser.add_argument("--candidates", nargs="+",
                        choices=["title_h1", "title_h2", "title_h2_buy", "title_h2_city"],
                        default=["title_h1", "title_h2", "title_h2_buy", "title_h2_city"])
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.games <= 0:
        parser.error("実験名・ゲーム数を確認してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent(args.model_id, device="cpu")
    common = {"target_sites": 4, "max_wait_rounds": 9}
    choices = {
        "title_h1": SettlementPlanningAgent(
            base, **common, prioritize_immediate_titles=True, title_horizon=1,
        ),
        "title_h2": SettlementPlanningAgent(
            base, **common, prioritize_immediate_titles=True, title_horizon=2,
        ),
        "title_h2_buy": SettlementPlanningAgent(
            base, **common, prioritize_immediate_titles=True, title_horizon=2,
            buy_for_largest_army=True,
        ),
        "title_h2_city": SettlementPlanningAgent(
            base, **common, prioritize_immediate_titles=True, title_horizon=2,
            prioritize_affordable_city=True,
        ),
    }
    agents = {"settlement_plan": SettlementPlanningAgent(base, **common)}
    agents.update((name, choices[name]) for name in args.candidates)
    config = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                              policy_compatible_opponents=True, observation_version="v2")
    reports, summaries = {}, {}
    seeds = range(args.seed_start, args.seed_start + args.games)
    for name, agent in agents.items():
        report = evaluate_agent(agent, seeds, config=config, agent_name=name).to_dict()
        if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
            raise AssertionError(f"{name}: 評価が正常に完了していません。")
        write(output / f"evaluation_{name}.json", report)
        reports[name], summaries[name] = report, summary(report)
        print(json.dumps({"stage": name, **summaries[name]}, ensure_ascii=False), flush=True)
    comparisons = {name: paired_comparison(reports["settlement_plan"], report)
                   for name, report in reports.items() if name != "settlement_plan"}
    write(output / "evaluation_summary.json", {
        "definition": vars(args), "summaries": summaries,
        "paired_vs_settlement_plan": comparisons,
    })


if __name__ == "__main__":
    main()
