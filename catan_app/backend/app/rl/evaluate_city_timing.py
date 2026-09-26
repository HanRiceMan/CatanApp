"""4拠点後、最初の1都市だけを予約する時間条件を比較する。"""

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
    parser.add_argument("--seed-start", type=int, default=470001)
    parser.add_argument("--candidates", nargs="+", type=int, default=[3, 5, 7])
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id or args.games <= 0
            or any(wait <= 0 for wait in args.candidates)):
        parser.error("実験名・ゲーム数・都市待ち時間を確認してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent(args.model_id, device="cpu")
    common = {
        "target_sites": 4, "max_wait_rounds": 9,
        "prioritize_affordable_city": True,
        "prioritize_immediate_titles": True, "title_horizon": 2,
    }
    agents = {"victory_plan": SettlementPlanningAgent(base, **common)}
    agents.update({
        f"first_city_w{wait}": SettlementPlanningAgent(
            base, **common, target_cities=1, max_city_wait_rounds=wait,
        )
        for wait in args.candidates
    })
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
    comparisons = {name: paired_comparison(reports["victory_plan"], report)
                   for name, report in reports.items() if name != "victory_plan"}
    write(output / "evaluation_summary.json", {
        "definition": vars(args), "summaries": summaries,
        "paired_vs_victory_plan": comparisons,
    })


if __name__ == "__main__":
    main()
