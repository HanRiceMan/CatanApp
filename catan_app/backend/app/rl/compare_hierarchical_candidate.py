"""通常の候補共有PPOと階層候補PPOを、固定初期配置で比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .analyze_supervised_setup import paired_effect
from .audit import runtime_versions, source_fingerprint
from .evaluate import EvaluationConfig, evaluate_agent
from .train import TrainConfig, train_maskable_ppo


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[20260927])
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=60001)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    if args.steps <= 0 or args.eval_games <= 0 or len(set(args.training_seeds)) != len(args.training_seeds):
        parser.error("steps・eval-gamesは正の整数、training-seedsは重複なしです。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if output.exists():
        parser.error("同名実験フォルダが既にあります。別IDを指定してください。")
    output.mkdir(parents=True)
    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    _write(output / "definition.json", {
        "experiment_id": args.experiment_id,
        "training_seeds": args.training_seeds,
        "training_steps_requested": args.steps,
        "evaluation_seeds": seeds,
        "architectures": ["candidate", "hierarchical_candidate"],
        "heuristic_initial_placement": True,
        "training_opponents": "heuristic_x3",
        "evaluation_opponents": "heuristic_x3",
        "observation_version": "v2",
        "action_space_version": "v1",
        "n_steps": 256, "batch_size": 64, "n_epochs": 4,
        "learning_rate": 3e-4, "gamma": .99, "gae_lambda": .95, "ent_coef": .01,
        "source_sha256": source_fingerprint(), "runtime_versions": runtime_versions(),
    })
    summaries = {}
    for training_seed in args.training_seeds:
        reports = {}
        for architecture in ("candidate", "hierarchical_candidate"):
            model_id = f"{architecture}_{args.steps}_s{training_seed}"
            train_maskable_ppo(TrainConfig(
                model_id=model_id, models_root=output / "models",
                total_timesteps=args.steps, seed=training_seed,
                observation_version="v2", policy_architecture=architecture,
                heuristic_initial_placement=True,
                n_steps=256, batch_size=64, n_epochs=4, learning_rate=3e-4,
                gamma=.99, gae_lambda=.95, ent_coef=.01,
                checkpoint_interval_steps=max(256, args.steps // 2),
                evaluation_seeds=tuple(range(args.eval_seed_start + 1000,
                                             args.eval_seed_start + 1004)),
            ))
            agent = PPOAgent(model_id, models_root=output / "models", device="cpu")
            report = evaluate_agent(
                agent, seeds,
                config=EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                                        policy_compatible_opponents=True,
                                        observation_version="v2"),
                agent_name=model_id,
            ).to_dict()
            summary = report["summary"]
            if summary["illegal_action_count"] or summary["truncated"]:
                raise AssertionError(f"非法Actionまたは打ち切り: {model_id}")
            reports[architecture] = report
            key = f"s{training_seed}_{architecture}"
            _write(output / f"{key}.json", report)
            summaries[key] = summary
            _write(output / "summary.json", summaries)
            print(json.dumps({"cell": key, "win_rate": summary["win_rate"],
                              "average_score": summary["average_score"],
                              "paid_roads": summary["average_actions"].get("build_road_paid", 0.0),
                              "settlements": summary["average_actions"].get("build_settlement", 0.0)},
                             ensure_ascii=False), flush=True)
        baseline, hierarchical = reports["candidate"], reports["hierarchical_candidate"]
        paired = {}
        for name, field in (("win_rate", "learner_won"), ("average_score", "learner_score")):
            effect = paired_effect(hierarchical, baseline, field)
            paired[name] = {"hierarchical_minus_candidate": effect["candidate_minus_heuristic"],
                            "bootstrap_95_percent_interval": effect["bootstrap_95_percent_interval"]}
        summaries[f"s{training_seed}_paired"] = {"games": len(seeds), **paired}
        _write(output / "summary.json", summaries)


if __name__ == "__main__":
    main()
