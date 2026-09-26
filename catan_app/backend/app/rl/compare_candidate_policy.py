"""同一seed・step数で固定ID型MLPと候補共有型PPOを小規模比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .analyze_supervised_setup import paired_effect
from .audit import runtime_versions, source_fingerprint
from .evaluate import EvaluationConfig, evaluate_agent
from .setup_diagnostics import probe_setup, summarize
from .train import TrainConfig, train_maskable_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--training-seed", type=int, default=20260923)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=40001)
    parser.add_argument("--heuristic-initial-placement", action="store_true",
                        help="学習者の初期配置を両方とも既存Heuristicに固定して通常ターンを比較する")
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    if args.steps <= 0 or args.eval_games <= 0:
        parser.error("stepsとeval-gamesは正の整数です。")
    root = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if root.exists():
        parser.error("実験フォルダが既にあります。別IDを指定してください。")
    root.mkdir(parents=True)
    seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    definition = {"experiment_id": args.experiment_id, "training_steps": args.steps,
                  "training_seed": args.training_seed, "evaluation_seeds": seeds,
                  "observation_version": "v2", "action_space_version": "v1",
                  "n_steps": 256, "batch_size": 64, "n_epochs": 4, "learning_rate": 3e-4,
                  "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01,
                  "heuristic_initial_placement": args.heuristic_initial_placement,
                  "opponents_training": "heuristic_x3", "opponents_evaluation": ["heuristic_x3", "heuristic_x1_random_x2"],
                  "seat_rotation_evaluation": True, "source_sha256": source_fingerprint(),
                  "runtime_versions": runtime_versions()}
    (root / "definition.json").write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = {}
    reports = {}
    for architecture in ("mlp", "candidate"):
        model_id = f"{architecture}_{args.steps}_s{args.training_seed}"
        config = TrainConfig(model_id=model_id, models_root=root / "models",
                             total_timesteps=args.steps, seed=args.training_seed,
                             observation_version="v2", policy_architecture=architecture,
                             heuristic_initial_placement=args.heuristic_initial_placement,
                             n_steps=256, batch_size=64, n_epochs=4, learning_rate=3e-4,
                             gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                             checkpoint_interval_steps=max(256, args.steps // 2),
                             evaluation_seeds=tuple(range(args.eval_seed_start + 1000,
                                                          args.eval_seed_start + 1004)))
        print(f"training {model_id}", flush=True)
        train_maskable_ppo(config)
        agent = PPOAgent(model_id, models_root=root / "models", device="cpu")
        setup_rows = [probe_setup(agent, seed, index, observation_version="v2")
                      for index, seed in enumerate(seeds)]
        setup_summary = summarize(setup_rows)
        (root / f"{architecture}_setup.json").write_text(
            json.dumps({"summary": setup_summary, "episodes": setup_rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        summaries[f"{architecture}_setup"] = setup_summary
        print(json.dumps({"architecture": architecture, "setup_pips": setup_summary["total_pips"],
                          "first_vertex_share": setup_summary["settlement_most_common"][0]["share"]},
                         ensure_ascii=False), flush=True)
        for heuristic_count in (3, 1):
            evaluation = evaluate_agent(
                agent, seeds,
                config=EvaluationConfig(heuristic_opponents=heuristic_count, rotate_player_ids=True,
                                        policy_compatible_opponents=True, observation_version="v2"),
                agent_name=model_id,
            ).to_dict()
            key = f"{architecture}_H{heuristic_count}_R{3-heuristic_count}"
            (root / f"{key}.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
            reports[key] = evaluation
            summaries[key] = evaluation["summary"]
            (root / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
            print(json.dumps({"cell": key, "win_rate": evaluation["summary"]["win_rate"],
                              "average_score": evaluation["summary"]["average_score"],
                              "illegal_action_count": evaluation["summary"]["illegal_action_count"],
                              "truncated": evaluation["summary"]["truncated"]}, ensure_ascii=False), flush=True)
    paired = {}
    for heuristic_count in (3, 1):
        suffix = f"H{heuristic_count}_R{3-heuristic_count}"
        mlp = reports[f"mlp_{suffix}"]
        candidate = reports[f"candidate_{suffix}"]
        initial_vertices_equal = all(
            left["learner_initial_vertices"] == right["learner_initial_vertices"]
            for left, right in zip(mlp["episodes"], candidate["episodes"])
        )
        if args.heuristic_initial_placement and not initial_vertices_equal:
            raise AssertionError("固定した初期配置が比較対象間で一致しません。")
        def effect(field: str) -> dict:
            result = paired_effect(candidate, mlp, field)
            return {"candidate_minus_mlp": result["candidate_minus_heuristic"],
                    "bootstrap_95_percent_interval": result["bootstrap_95_percent_interval"]}
        paired[suffix] = {"games": len(mlp["episodes"]),
                          "initial_vertices_equal": initial_vertices_equal,
                          "win_rate_difference": effect("learner_won"),
                          "average_score_difference": effect("learner_score")}
    (root / "paired_comparison.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
