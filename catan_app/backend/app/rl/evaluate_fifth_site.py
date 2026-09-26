"""4拠点で止めず、5拠点目まで拡張計画を続ける条件を比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .evaluate import EvaluationConfig, evaluate_agent
from .evaluate_construction_priority import VALIDATED_PLANNING
from .settlement_planning_agent import SettlementPlanningAgent
from .train_development_budget import paired_comparison, summary, write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=880001)
    parser.add_argument("--waits", type=float, nargs="+", default=[6, 9])
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id or args.games <= 0
            or any(value <= 0 for value in args.waits)):
        parser.error("実験名・ゲーム数を確認してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent("ppo_gnn_board_65k_s03_exp_v003", device="cpu")
    agents = {"sites_4_wait_9": SettlementPlanningAgent(base, **VALIDATED_PLANNING)}
    agents.update({
        f"sites_5_wait_{value:g}": SettlementPlanningAgent(
            base, **{**VALIDATED_PLANNING, "target_sites": 5,
                     "max_wait_rounds": value},
        )
        for value in args.waits
    })
    config = EvaluationConfig(
        heuristic_opponents=3, rotate_player_ids=True,
        policy_compatible_opponents=True, observation_version="v2",
    )
    seeds = range(args.seed_start, args.seed_start + args.games)
    reports, summaries = {}, {}
    for name, agent in agents.items():
        report = evaluate_agent(agent, seeds, config=config, agent_name=name).to_dict()
        if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
            raise AssertionError(f"{name}: 評価が正常に完了していません。")
        write(output / f"evaluation_{name}.json", report)
        reports[name], summaries[name] = report, summary(report)
        print(json.dumps({"stage": name, **summaries[name]}, ensure_ascii=False), flush=True)
    baseline = reports["sites_4_wait_9"]
    write(output / "evaluation_summary.json", {
        "definition": vars(args), "summaries": summaries,
        "paired_vs_sites_4": {
            name: paired_comparison(baseline, report)
            for name, report in reports.items() if report is not baseline
        },
    })


if __name__ == "__main__":
    main()
