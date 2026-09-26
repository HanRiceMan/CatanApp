"""保存済み拡張方策を現行PPOと同じ未使用盤面で比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from app.agents.ppo import PPOAgent

from .evaluate import EvaluationConfig, evaluate_agent


def _view(summary: dict) -> dict[str, float]:
    actions = summary.get("average_actions", {})
    opportunities = summary.get("average_opportunities", {})
    strategy = summary.get("average_strategy", {})
    roads = float(strategy.get("road_builds", 0))
    opened = float(strategy.get("road_opens_settlement_site", 0))
    return {
        "win_rate": summary["win_rate"],
        "average_score": summary["average_score"],
        "average_rank": summary["average_rank"],
        "settlement": actions.get("build_settlement", 0),
        "city": actions.get("build_city", 0),
        "paid_road": actions.get("build_road_paid", 0),
        "development": actions.get("buy_development", 0),
        "road_while_planner_waits": opportunities.get(
            "build_road_while_planner_waits", 0
        ),
        "development_while_build_near": opportunities.get(
            "buy_development_while_build_near", 0
        ),
        "development_during_planned_stall": opportunities.get(
            "buy_development_during_planned_stall", 0
        ),
        "planned_road_selected": opportunities.get("planned_road_selected", 0),
        "off_plan_road_selected": opportunities.get("off_plan_road_selected", 0),
        "road_opens_site_rate": opened / roads if roads else 0.0,
        "largest_army_rate": summary.get("largest_army_acquisition_rate", 0),
        "longest_road_rate": summary.get("longest_road_acquisition_rate", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--games", type=int, default=80)
    parser.add_argument("--seed-start", type=int, default=330_001)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("gamesは正の整数にしてください。")
    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpointが見つかりません: {checkpoint}")
    output.mkdir(parents=True)

    source = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    candidate = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    candidate.model = MaskablePPO.load(str(checkpoint), device="cpu")
    seeds = tuple(range(args.seed_start, args.seed_start + args.games))
    config = EvaluationConfig(
        heuristic_opponents=3,
        rotate_player_ids=True,
        policy_compatible_opponents=True,
        observation_version=source.manifest.observation_version,
    )
    reports = {
        "source": evaluate_agent(source, seeds, config=config,
                                 agent_name=args.model_id).to_dict(),
        "candidate": evaluate_agent(candidate, seeds, config=config,
                                    agent_name=f"{args.model_id}:expansion-candidate").to_dict(),
    }
    for name, report in reports.items():
        summary = report["summary"]
        if summary["illegal_action_count"] or summary["truncated"]:
            raise AssertionError(f"{name}: 非法Actionまたは打ち切りを検出しました。")
        (output / f"evaluation_{name}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    views = {name: _view(report["summary"]) for name, report in reports.items()}
    differences = {
        key: views["candidate"][key] - views["source"][key]
        for key in views["source"]
    }
    result = {
        "definition": {
            "source_model": args.model_id,
            "checkpoint": str(checkpoint),
            "games": args.games,
            "seed_start": args.seed_start,
            "seeds": seeds,
        },
        "source": views["source"],
        "candidate": views["candidate"],
        "candidate_minus_source": differences,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
