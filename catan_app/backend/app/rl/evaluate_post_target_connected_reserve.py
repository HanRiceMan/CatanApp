"""5拠点後、再利用可能な開拓地コマを接続済み候補へ予約する案を比較する。"""

import argparse
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .construction_metrics import construction_summary
from .evaluate import EvaluationConfig, evaluate_agent
from .evaluate_construction_priority import VALIDATED_PLANNING
from .settlement_planning_agent import SettlementPlanningAgent
from .train_development_budget import paired_comparison, summary, write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=930001)
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."} or args.games <= 0):
        parser.error("実験名・ゲーム数が不正です。")

    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent("ppo_gnn_board_65k_s03_exp_v003", device="cpu")
    configs = {
        "validated_baseline": dict(VALIDATED_PLANNING),
        "connected_reserve_s6_7_w2": {
            **VALIDATED_PLANNING,
            "post_target_connected_reserve_max_wait": 2,
            "post_target_connected_reserve_min_score": 6,
            "post_target_connected_reserve_max_score": 7,
        },
    }
    write(output / "definition.json", {"arguments": vars(args), "planning": configs})
    evaluation = EvaluationConfig(
        heuristic_opponents=3, rotate_player_ids=True,
        policy_compatible_opponents=True, observation_version="v2",
    )
    reports, summaries = {}, {}
    for name, planning in configs.items():
        report = evaluate_agent(
            SettlementPlanningAgent(base, **planning),
            range(args.seed_start, args.seed_start + args.games),
            config=evaluation, agent_name=name,
        ).to_dict()
        if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
            raise AssertionError("評価が正常終了していません。")
        reports[name] = report
        summaries[name] = {
            **summary(report), "construction": construction_summary(report),
        }
        write(output / f"evaluation_{name}.json", report)
        print(json.dumps({"stage": name, **summaries[name]}, ensure_ascii=False), flush=True)
    write(output / "evaluation_summary.json", {
        "summaries": summaries,
        "paired": paired_comparison(
            reports["validated_baseline"], reports["connected_reserve_s6_7_w2"]
        ),
    })


if __name__ == "__main__":
    main()
