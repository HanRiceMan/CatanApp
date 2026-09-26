"""最長路まで3本以内の計画を開始する公開得点を比較する。"""

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
    parser.add_argument("--seed-start", type=int, default=610001)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.games <= 0:
        parser.error("実験名・ゲーム数を確認してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent(args.model_id, device="cpu")
    common = {
        "target_sites": 4, "max_wait_rounds": 9,
        "first_city_max_deficit": 2, "max_city_wait_rounds": 9,
        "expansion_aware_setup": True, "endgame_point_reserve_score": 8,
        "prioritize_affordable_city": True, "prioritize_immediate_win": True,
        "prioritize_immediate_titles": True,
    }
    agents = {
        "horizon_2": SettlementPlanningAgent(base, **common, title_horizon=2),
    }
    agents.update({
        f"horizon_3_from_{score}": SettlementPlanningAgent(
            base, **common, title_horizon=3, longest_road_min_score=score,
        )
        for score in (4, 6, 8)
    })
    config = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                              policy_compatible_opponents=True, observation_version="v2")
    reports, summaries = {}, {}
    seeds = range(args.seed_start, args.seed_start + args.games)
    for name, candidate in agents.items():
        report = evaluate_agent(candidate, seeds, config=config, agent_name=name).to_dict()
        if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
            raise AssertionError(f"{name}: 評価が正常に完了していません。")
        write(output / f"evaluation_{name}.json", report)
        reports[name], summaries[name] = report, summary(report)
        print(json.dumps({"stage": name, **summaries[name]}, ensure_ascii=False), flush=True)
    comparisons = {name: paired_comparison(reports["horizon_2"], report)
                   for name, report in reports.items() if name != "horizon_2"}
    write(output / "evaluation_summary.json", {
        "definition": vars(args), "summaries": summaries,
        "paired_vs_horizon_2": comparisons,
    })


if __name__ == "__main__":
    main()
