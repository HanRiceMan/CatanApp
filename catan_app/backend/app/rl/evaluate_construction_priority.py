"""固定した直近検証構成に対し、購入可能な建設を称号より先にする案を比較。"""

import argparse
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .construction_metrics import construction_summary
from .evaluate import EvaluationConfig, evaluate_agent
from .settlement_planning_agent import SettlementPlanningAgent
from .train_development_budget import paired_comparison, summary, write


# 本番の暗黙デフォルトに依存させず、比較する構成を固定・記録する。
VALIDATED_PLANNING = {
    "target_sites": 5, "max_wait_rounds": 6, "max_city_wait_rounds": 9,
    "first_city_max_deficit": 2, "second_city_max_deficit": 2,
    "second_city_min_score": 6,
    "post_target_connected_reserve_max_wait": 2,
    "post_target_connected_reserve_min_score": 6,
    "post_target_connected_reserve_max_score": 7,
    "expansion_aware_setup": True, "expansion_setup_min_pips": 3,
    "setup_min_wheat_pips": 1, "setup_best_coverage_fallback": True,
    "collision_aware_initial_roads": True,
    "adaptive_development_purchase": True,
    "development_stall_wait_rounds": 4.0,
    "development_max_build_delay": 1.0,
    "development_tactical_score": 8,
    "belief_aware_robber": True,
    "endgame_point_reserve_score": 8,
    "prioritize_affordable_city": True, "prioritize_immediate_win": True,
    "prioritize_immediate_titles": True, "title_horizon": 2,
    "early_affordable_city_min_sites": 3,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=750001)
    parser.add_argument("--compare-production", action="store_true",
                        help="保存済み本番設定と直近検証構成を比較する")
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."} or args.games <= 0):
        parser.error("実験名・ゲーム数が不正です。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent("ppo_gnn_board_65k_s03_exp_v003", device="cpu")
    configs = {
        "validated_baseline": dict(VALIDATED_PLANNING),
        "construction_first": {**VALIDATED_PLANNING, "construction_before_titles": True},
    }
    if args.compare_production:
        if base.settlement_planning_config is None:
            raise ValueError("保存済み本番計画設定がありません。")
        configs = {
            "production_before": dict(base.settlement_planning_config),
            "validated_baseline": dict(VALIDATED_PLANNING),
        }
    write(output / "definition.json", {"arguments": vars(args), "planning": configs})
    evaluation = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                                  policy_compatible_opponents=True, observation_version="v2")
    reports, summaries = {}, {}
    for name, planning in configs.items():
        candidate = SettlementPlanningAgent(base, **planning)
        report = evaluate_agent(candidate, range(args.seed_start, args.seed_start + args.games),
                                config=evaluation, agent_name=name).to_dict()
        if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
            raise AssertionError("評価が正常終了していません。")
        reports[name] = report
        summaries[name] = {**summary(report), "construction": construction_summary(report)}
        write(output / f"evaluation_{name}.json", report)
        print(json.dumps({"stage": name, **summaries[name]}, ensure_ascii=False), flush=True)
    write(output / "evaluation_summary.json", {
        "summaries": summaries,
        "paired": paired_comparison(*reports.values()),
    })


if __name__ == "__main__":
    main()
