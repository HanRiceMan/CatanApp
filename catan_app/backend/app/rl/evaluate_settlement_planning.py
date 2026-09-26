"""資源予約付き開拓地計画層を、現行PPOと同一盤面で比較する。"""

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
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=390001)
    parser.add_argument("--target-sites", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--max-wait-rounds", type=float, nargs="+", default=[6.0, 9.0])
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id or args.games <= 0
            or min(args.max_wait_rounds) <= 0
            or any(sites not in range(3, 6) for sites in args.target_sites)):
        parser.error("実験名・ゲーム数・待ち時間を確認してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent(args.model_id, device="cpu")
    agents = {"original": base}
    for sites in args.target_sites:
        for wait in args.max_wait_rounds:
            label = str(wait).rstrip("0").rstrip(".").replace(".", "p")
            agents[f"plan_s{sites}_w{label}"] = SettlementPlanningAgent(
                base, target_sites=sites, max_wait_rounds=wait
            )
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
    comparisons = {name: paired_comparison(reports["original"], report)
                   for name, report in reports.items() if name != "original"}
    write(output / "evaluation_summary.json", {
        "definition": vars(args), "summaries": summaries,
        "paired_vs_original": comparisons,
    })


if __name__ == "__main__":
    main()
