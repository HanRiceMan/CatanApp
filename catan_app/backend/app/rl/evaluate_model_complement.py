"""同じ計画層を重ねた保存モデル群の勝敗補完性を評価する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .evaluate import EvaluationConfig, evaluate_agent
from .evaluate_construction_priority import VALIDATED_PLANNING
from .settlement_planning_agent import SettlementPlanningAgent
from .train_development_budget import summary, write


DEFAULT_MODELS = (
    "ppo_gnn_board_65k_s03_exp_v003",
    "ppo_gnn_long_96k_s02_exp_v002",
    "ppo_gnn_setup_n1024_s01_v001",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=860001)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id or args.games <= 0
            or len(set(args.models)) != len(args.models)):
        parser.error("実験名・ゲーム数・モデルを確認してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    config = EvaluationConfig(
        heuristic_opponents=3, rotate_player_ids=True,
        policy_compatible_opponents=True, observation_version="v2",
    )
    seeds = range(args.seed_start, args.seed_start + args.games)
    reports, summaries = {}, {}
    for model_id in args.models:
        base = PPOAgent(model_id, device="cpu")
        if base.manifest.observation_version != "v2":
            raise AssertionError(f"{model_id}: v2観測のモデルではありません。")
        agent = SettlementPlanningAgent(base, **VALIDATED_PLANNING)
        report = evaluate_agent(agent, seeds, config=config, agent_name=model_id).to_dict()
        if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
            raise AssertionError(f"{model_id}: 評価が正常に完了していません。")
        write(output / f"evaluation_{model_id}.json", report)
        reports[model_id], summaries[model_id] = report, summary(report)
        print(json.dumps({"stage": model_id, **summaries[model_id]},
                         ensure_ascii=False), flush=True)
    episode_rows = [reports[model_id]["episodes"] for model_id in args.models]
    for rows in episode_rows[1:]:
        if [(row["seed"], row["learner_id"]) for row in rows] != [
                (row["seed"], row["learner_id"]) for row in episode_rows[0]]:
            raise AssertionError("モデル間でseed・席順が一致しません。")
    win_matrix = [
        [bool(rows[index]["learner_won"]) for rows in episode_rows]
        for index in range(args.games)
    ]
    complement = {
        "any_model_win_rate": sum(any(row) for row in win_matrix) / args.games,
        "all_model_loss_rate": sum(not any(row) for row in win_matrix) / args.games,
        "current_loss_rescued_rate": sum(
            not row[0] and any(row[1:]) for row in win_matrix
        ) / args.games,
        "current_win_lost_by_all_alternatives_rate": sum(
            row[0] and not any(row[1:]) for row in win_matrix
        ) / args.games,
        "win_patterns": {},
    }
    for row in win_matrix:
        key = "".join("1" if value else "0" for value in row)
        complement["win_patterns"][key] = complement["win_patterns"].get(key, 0) + 1
    write(output / "evaluation_summary.json", {
        "definition": vars(args), "summaries": summaries, "complement": complement,
    })
    print(json.dumps({"stage": "complement", **complement}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
