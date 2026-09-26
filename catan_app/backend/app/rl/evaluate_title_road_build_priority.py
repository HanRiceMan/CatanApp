"""近い建設がある時に称号用道路を延期する閾値を比較する。"""

import argparse
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .construction_metrics import construction_summary
from .evaluate import EvaluationConfig, evaluate_agent
from .evaluate_construction_priority import VALIDATED_PLANNING
from .settlement_planning_agent import SettlementPlanningAgent
from .train_development_budget import paired_comparison, summary, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=780001)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[1.0, 2.0])
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id or args.games <= 0
            or any(not 0 < value <= 6 for value in args.thresholds)):
        parser.error("実験名・ゲーム数・閾値が不正です。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent("ppo_gnn_board_65k_s03_exp_v003", device="cpu")
    configs = {"current": dict(VALIDATED_PLANNING)}
    configs.update({
        f"defer_title_wait_{value:g}": {
            **VALIDATED_PLANNING, "title_road_build_wait_threshold": value,
        }
        for value in args.thresholds
    })
    write(output / "definition.json", {"arguments": vars(args), "planning": configs})
    evaluation = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                                  policy_compatible_opponents=True, observation_version="v2")
    reports, summaries = {}, {}
    seeds = range(args.seed_start, args.seed_start + args.games)
    for name, planning in configs.items():
        report = evaluate_agent(
            SettlementPlanningAgent(base, **planning), seeds,
            config=evaluation, agent_name=name,
        ).to_dict()
        if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
            raise AssertionError(f"{name}: 評価が正常終了していません。")
        reports[name] = report
        summaries[name] = {**summary(report), "construction": construction_summary(report)}
        write(output / f"evaluation_{name}.json", report)
        print(json.dumps({"stage": name, **summaries[name]}, ensure_ascii=False), flush=True)
    write(output / "evaluation_summary.json", {
        "summaries": summaries,
        "paired_vs_current": {
            name: paired_comparison(reports["current"], report)
            for name, report in reports.items() if name != "current"
        },
    })


if __name__ == "__main__":
    main()
