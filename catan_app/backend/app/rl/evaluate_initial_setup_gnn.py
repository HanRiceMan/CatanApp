"""保存済み初期配置MLP/GNNを50k PPOと組み合わせて対局評価する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.agents.ppo import PPOAgent

from .analyze_supervised_setup import paired_effect
from .compare_initial_setup_adapter import IndependentSetupAgent
from .evaluate import EvaluationConfig, evaluate_agent
from .gnn_setup_policy import GraphInitialSetupPolicy
from .initial_setup_policy import InitialSetupPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--trained-experiment", required=True)
    parser.add_argument("--model-id", default="ppo_hierarchical_50k_v001")
    parser.add_argument("--models-root", type=Path, default=None)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=108_001)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    trained = root / "experiments" / args.trained_experiment
    output = root / "experiments" / args.experiment_id
    if not trained.is_dir():
        parser.error("学習済み実験フォルダがありません。")
    if output.exists():
        parser.error("評価実験フォルダが既にあります。")
    output.mkdir(parents=True)

    baseline = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    mlp_base = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    mlp = InitialSetupPolicy(mlp_base.model.policy)
    mlp.load_state_dict(torch.load(trained / "mlp_policy.pt", map_location="cpu", weights_only=True))
    mlp.eval()
    gnn_base = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    gnn = GraphInitialSetupPolicy(gnn_base.model.policy)
    gnn.load_state_dict(torch.load(trained / "gnn_policy.pt", map_location="cpu", weights_only=True))
    gnn.eval()
    agents = {"heuristic_setup": baseline,
              "ranked_mlp": IndependentSetupAgent(mlp_base, mlp),
              "ranked_gnn": IndependentSetupAgent(gnn_base, gnn)}
    seeds = tuple(range(args.seed_start, args.seed_start + args.games))
    config = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                              policy_compatible_opponents=True, observation_version="v2")
    reports = {}
    summaries = {}
    for name, agent in agents.items():
        report = evaluate_agent(agent, seeds, config=config, agent_name=name).to_dict()
        reports[name] = report
        summary = report["summary"]
        summary["average_initial_resource_kinds"] = sum(
            row["learner_initial_resource_kinds"] for row in report["episodes"]) / args.games
        summaries[name] = summary
        (output / f"{name}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"condition": name, "win_rate": summary["win_rate"],
                          "average_score": summary["average_score"],
                          "average_rank": summary["average_rank"],
                          "initial_pips": summary["average_initial_pips"],
                          "resource_kinds": summary["average_initial_resource_kinds"]},
                         ensure_ascii=False), flush=True)
    effects = {name: {
        "win_rate": paired_effect(reports[name], reports["heuristic_setup"], "learner_won"),
        "average_score": paired_effect(reports[name], reports["heuristic_setup"], "learner_score")}
        for name in ("ranked_mlp", "ranked_gnn")}
    (output / "paired_effects.json").write_text(
        json.dumps(effects, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
