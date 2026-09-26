"""都市化する地点へ、次の都市資源を補う価値を加えて比較する。"""

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
    parser.add_argument("--seed-start", type=int, default=840001)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.5, 1.0])
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id or args.games <= 0
            or any(not 0 < value <= 2 for value in args.weights)):
        parser.error("実験名・ゲーム数・重みを確認してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent("ppo_gnn_board_65k_s03_exp_v003", device="cpu")
    agents = {"weight_0": SettlementPlanningAgent(base, **VALIDATED_PLANNING)}
    agents.update({
        f"weight_{value:g}": SettlementPlanningAgent(
            base, **VALIDATED_PLANNING, city_growth_value_weight=value,
        )
        for value in args.weights
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
    write(output / "evaluation_summary.json", {
        "definition": vars(args), "summaries": summaries,
        "paired_vs_weight_0": {
            name: paired_comparison(reports["weight_0"], report)
            for name, report in reports.items() if name != "weight_0"
        },
    })


if __name__ == "__main__":
    main()
