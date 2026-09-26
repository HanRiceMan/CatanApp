"""現行の勝利計画を保ったまま、4拠点と5拠点目標を比較する。"""

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
    parser.add_argument("--seed-start", type=int, default=480001)
    parser.add_argument("--waits", nargs="+", type=float, default=[6.0, 9.0])
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id or args.games <= 0
            or any(wait <= 0 for wait in args.waits)):
        parser.error("実験名・ゲーム数・待ち時間を確認してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent(args.model_id, device="cpu")
    common = {
        "prioritize_affordable_city": True,
        "prioritize_immediate_titles": True, "title_horizon": 2,
    }
    agents = {
        "sites4_wait9": SettlementPlanningAgent(
            base, target_sites=4, max_wait_rounds=9, **common,
        ),
    }
    agents.update({
        f"sites5_wait{str(wait).rstrip('0').rstrip('.')}": SettlementPlanningAgent(
            base, target_sites=5, max_wait_rounds=wait, **common,
        )
        for wait in args.waits
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
    comparisons = {name: paired_comparison(reports["sites4_wait9"], report)
                   for name, report in reports.items() if name != "sites4_wait9"}
    write(output / "evaluation_summary.json", {
        "definition": vars(args), "summaries": summaries,
        "paired_vs_sites4_wait9": comparisons,
    })


if __name__ == "__main__":
    main()
