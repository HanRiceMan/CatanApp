"""4拠点以上・7点以上で、近い3都市目だけを予約する条件を比較する。"""

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
    parser.add_argument("--seed-start", type=int, default=790001)
    parser.add_argument("--deficits", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id or args.games <= 0
            or any(value not in range(1, 6) for value in args.deficits)):
        parser.error("実験名・ゲーム数・不足枚数が不正です。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent("ppo_gnn_board_65k_s03_exp_v003", device="cpu")
    configs = {"current": dict(VALIDATED_PLANNING)}
    configs.update({
        f"third_city_deficit_{deficit}": {
            **VALIDATED_PLANNING,
            "third_city_max_deficit": deficit,
            "third_city_min_score": 7,
        }
        for deficit in args.deficits
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
