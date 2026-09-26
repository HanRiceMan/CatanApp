"""2個目の初期配置で木材・レンガへ要求する最低pipを比較する。"""

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
    parser.add_argument("--seed-start", type=int, default=640001)
    parser.add_argument(
        "--pips", nargs="+", type=int, default=[1, 2, 3],
        help="比較する最低pip。基準の1は自動的に含めます。",
    )
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id or args.games <= 0
            or any(pips < 1 or pips > 5 for pips in args.pips)):
        parser.error("実験名・ゲーム数を確認してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent(args.model_id, device="cpu")
    common = {
        "target_sites": 4, "max_wait_rounds": 9,
        "first_city_max_deficit": 2, "max_city_wait_rounds": 9,
        "expansion_aware_setup": True, "endgame_point_reserve_score": 8,
        "prioritize_affordable_city": True, "prioritize_immediate_win": True,
        "prioritize_immediate_titles": True, "title_horizon": 2,
    }
    pips_values = list(dict.fromkeys([1, *args.pips]))
    agents = {
        f"min_pips_{pips}": SettlementPlanningAgent(
            base, **common, expansion_setup_min_pips=pips,
        )
        for pips in pips_values
    }
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
    comparisons = {name: paired_comparison(reports["min_pips_1"], report)
                   for name, report in reports.items() if name != "min_pips_1"}
    write(output / "evaluation_summary.json", {
        "definition": vars(args), "summaries": summaries,
        "paired_vs_min_pips_1": comparisons,
    })


if __name__ == "__main__":
    main()
