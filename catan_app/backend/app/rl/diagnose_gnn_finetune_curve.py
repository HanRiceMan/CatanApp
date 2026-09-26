"""GNN初期配置を固定したPPO継続学習を短い間隔で保存・評価する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import FloatSchedule

from app.agents.ppo import PPOAgent

from .compare_initial_setup_adapter import IndependentSetupAgent
from .env import CatanEnv, CatanEnvConfig
from .evaluate import EvaluationConfig, evaluate_agent
from .gnn_setup_policy import GraphInitialSetupAgent, GraphInitialSetupPolicy


TRAIN_METRICS = (
    "train/approx_kl",
    "train/clip_fraction",
    "train/entropy_loss",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/explained_variance",
    "train/loss",
    "train/learning_rate",
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _training_metrics(model: MaskablePPO) -> dict[str, float]:
    values = model.logger.name_to_value
    return {
        name.removeprefix("train/"): float(values[name])
        for name in TRAIN_METRICS
        if name in values
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--gnn-experiment", required=True)
    parser.add_argument("--model-id", default="ppo_hierarchical_50k_v001")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--additional-steps", type=int, default=5_120)
    parser.add_argument("--checkpoint-steps", type=int, default=1_024)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="PPOが1回の更新前に集めるstep数。省略時は元モデルの値を維持する。",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="継続学習時の割引率。省略時は元モデルの値を維持する。",
    )
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=130_001)
    args = parser.parse_args()

    if args.additional_steps <= 0 or args.checkpoint_steps <= 0:
        parser.error("step数は0より大きくしてください。")
    if args.additional_steps % args.checkpoint_steps:
        parser.error("additional-stepsはcheckpoint-stepsの倍数にしてください。")
    if args.learning_rate <= 0:
        parser.error("learning-rateは0より大きくしてください。")
    if args.gamma is not None and not 0 < args.gamma <= 1:
        parser.error("gammaは0より大きく1以下にしてください。")
    if args.rollout_steps is not None and args.rollout_steps <= 0:
        parser.error("rollout-stepsは0より大きくしてください。")

    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。")
    output.mkdir(parents=True)

    source = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    gnn = GraphInitialSetupPolicy(source.model.policy)
    gnn_path = root / "experiments" / args.gnn_experiment / "gnn_policy.pt"
    gnn.load_state_dict(torch.load(gnn_path, map_location="cpu", weights_only=True))
    gnn.eval()
    setup_agent = GraphInitialSetupAgent(gnn)

    environment = CatanEnv(
        CatanEnvConfig(
            learning_player_id=1,
            opponent_controller="heuristic",
            heuristic_initial_placement=False,
            observation_version="v2",
        ),
        initial_placement_agent=setup_agent,
    )
    artifact = source.registry.models_root / args.model_id / source.manifest.artifact_filename
    load_options = ({"n_steps": args.rollout_steps}
                    if args.rollout_steps is not None else {})
    if args.gamma is not None:
        load_options["gamma"] = args.gamma
    model = MaskablePPO.load(
        str(artifact), env=environment, device="cpu", **load_options
    )
    source_steps = model.num_timesteps
    source_learning_rate = float(model.lr_schedule(1.0))
    if args.checkpoint_steps % model.n_steps:
        parser.error(
            f"checkpoint-stepsはPPOのrollout長 {model.n_steps} の倍数にしてください。"
        )
    model.learning_rate = args.learning_rate
    model.lr_schedule = FloatSchedule(args.learning_rate)
    model.set_random_seed(args.seed)

    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    evaluation_config = EvaluationConfig(
        heuristic_opponents=3,
        rotate_player_ids=True,
        policy_compatible_opponents=True,
        observation_version="v2",
    )
    definition = {
        "source_model": args.model_id,
        "gnn_experiment": args.gnn_experiment,
        "training_seed": args.seed,
        "source_steps": source_steps,
        "additional_steps": args.additional_steps,
        "checkpoint_steps": args.checkpoint_steps,
        "source_rollout_steps": source.model.n_steps,
        "finetune_rollout_steps": model.n_steps,
        "source_gamma": source.model.gamma,
        "finetune_gamma": model.gamma,
        "source_learning_rate": source_learning_rate,
        "finetune_learning_rate": args.learning_rate,
        "evaluation_seeds": seeds,
    }
    _write(output / "definition.json", definition)

    base_agent = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    points: list[dict[str, object]] = []

    for added_steps in range(0, args.additional_steps + 1, args.checkpoint_steps):
        if added_steps:
            model.learn(
                total_timesteps=args.checkpoint_steps,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            model.save(str(output / f"checkpoint_{added_steps}_steps.zip"))
            policy_agent = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
            policy_agent.model = model
        else:
            policy_agent = base_agent

        agent = IndependentSetupAgent(policy_agent, gnn)
        report = evaluate_agent(
            agent,
            seeds,
            config=evaluation_config,
            agent_name=f"{args.model_id}:gnn_setup:+{added_steps}",
        ).to_dict()
        summary = report["summary"]
        if summary["illegal_action_count"] or summary["truncated"]:
            raise AssertionError(f"非法Actionまたは打ち切り: +{added_steps} steps")
        point = {
            "added_steps": added_steps,
            "total_steps": source_steps + added_steps,
            "training_metrics": _training_metrics(model) if added_steps else {},
            "summary": summary,
        }
        points.append(point)
        _write(output / f"evaluation_{added_steps}_steps.json", report)
        _write(output / "curve.json", {"definition": definition, "points": points})
        print(
            json.dumps(
                {
                    "added_steps": added_steps,
                    "win_rate": summary["win_rate"],
                    "average_score": summary["average_score"],
                    "average_rank": summary["average_rank"],
                    **point["training_metrics"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    environment.close()


if __name__ == "__main__":
    main()
