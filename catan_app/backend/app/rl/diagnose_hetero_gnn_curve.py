"""現行GNNから異種GNN v2をゼロ残差で追加し、PPO学習曲線を測る。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from app.agents.ppo import PPOAgent

from .candidate_policy import (
    GraphHierarchicalCandidateMaskablePolicy,
    HeteroGraphHierarchicalCandidateMaskablePolicy,
    configure_hetero_graph_finetune,
    initialize_hetero_policy_from_graph,
)
from .env import CatanEnv, CatanEnvConfig
from .evaluate import EvaluationConfig, evaluate_agent
from .gnn_setup_policy import GraphInitialSetupAgent


TRAIN_METRICS = (
    "train/approx_kl", "train/clip_fraction", "train/entropy_loss",
    "train/value_loss", "train/policy_gradient_loss", "train/explained_variance",
    "train/loss", "train/learning_rate",
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _training_metrics(model: MaskablePPO) -> dict[str, float]:
    return {
        name.removeprefix("train/"): float(model.logger.name_to_value[name])
        for name in TRAIN_METRICS if name in model.logger.name_to_value
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--additional-steps", type=int, default=10_240)
    parser.add_argument("--checkpoint-steps", type=int, default=5_120)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20261901)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=370_001)
    args = parser.parse_args()
    if args.additional_steps <= 0 or args.checkpoint_steps <= 0:
        parser.error("step数は正の整数にしてください。")
    if args.additional_steps % args.checkpoint_steps:
        parser.error("additional-stepsはcheckpoint-stepsの倍数にしてください。")
    if args.learning_rate <= 0:
        parser.error("learning-rateは0より大きくしてください。")
    if args.gamma is not None and not 0 < args.gamma <= 1:
        parser.error("gammaは0より大きく1以下にしてください。")

    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。")
    output.mkdir(parents=True)
    source = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    if type(source.model.policy) is not GraphHierarchicalCandidateMaskablePolicy:
        raise TypeError("移行元はGraphHierarchicalCandidateMaskablePolicyである必要があります。")
    if source.initial_setup_policy is None:
        raise TypeError("移行元には初期配置GNNが必要です。")
    setup_agent = GraphInitialSetupAgent(source.initial_setup_policy)
    environment = CatanEnv(CatanEnvConfig(
        learning_player_id=1,
        opponent_controller="heuristic",
        policy_compatible_opponents=True,
        observation_version="v2",
    ), initial_placement_agent=setup_agent)
    gamma = source.model.gamma if args.gamma is None else args.gamma
    model = MaskablePPO(
        HeteroGraphHierarchicalCandidateMaskablePolicy,
        environment,
        learning_rate=args.learning_rate,
        n_steps=source.model.n_steps,
        batch_size=source.model.batch_size,
        n_epochs=source.model.n_epochs,
        gamma=gamma,
        gae_lambda=source.model.gae_lambda,
        ent_coef=source.model.ent_coef,
        vf_coef=source.model.vf_coef,
        max_grad_norm=source.model.max_grad_norm,
        normalize_advantage=source.model.normalize_advantage,
        policy_kwargs={"net_arch": [], "ortho_init": False},
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    added = initialize_hetero_policy_from_graph(model.policy, source.model.policy)
    selection = configure_hetero_graph_finetune(model.policy)
    model.num_timesteps = source.model.num_timesteps
    model.set_random_seed(args.seed)
    if args.checkpoint_steps % model.n_steps:
        parser.error(f"checkpoint-stepsはrollout長 {model.n_steps} の倍数にしてください。")

    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    config = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                              policy_compatible_opponents=True, observation_version="v2")
    definition = {
        "source_model": args.model_id,
        "training_seed": args.seed,
        "source_steps": source.model.num_timesteps,
        "additional_steps": args.additional_steps,
        "checkpoint_steps": args.checkpoint_steps,
        "rollout_steps": model.n_steps,
        "gamma": gamma,
        "learning_rate": args.learning_rate,
        "hetero_message_layers": 2,
        "node_types": {"tile": 19, "vertex": 54, "edge": 72},
        "relations": ["tile_to_vertex", "vertex_to_tile",
                      "vertex_to_edge", "edge_to_vertex"],
        **selection,
        "added_parameter_keys": added,
        "evaluation_seeds": seeds,
    }
    _write(output / "definition.json", definition)
    points: list[dict[str, object]] = []
    for added_steps in range(0, args.additional_steps + 1, args.checkpoint_steps):
        if added_steps:
            model.learn(total_timesteps=args.checkpoint_steps, reset_num_timesteps=False,
                        progress_bar=False)
            model.save(str(output / f"checkpoint_{added_steps}_steps.zip"))
        candidate = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
        candidate.model = model
        report = evaluate_agent(
            candidate, seeds, config=config,
            agent_name=f"{args.model_id}:hetero_gnn:+{added_steps}",
        ).to_dict()
        summary = report["summary"]
        if summary["illegal_action_count"] or summary["truncated"]:
            raise AssertionError(f"非法Actionまたは打ち切り: +{added_steps} steps")
        point = {
            "added_steps": added_steps,
            "total_steps": source.model.num_timesteps + added_steps,
            "training_metrics": _training_metrics(model) if added_steps else {},
            "summary": summary,
        }
        points.append(point)
        _write(output / f"evaluation_{added_steps}_steps.json", report)
        _write(output / "curve.json", {"definition": definition, "points": points})
        print(json.dumps({
            "added_steps": added_steps,
            "win_rate": summary["win_rate"],
            "average_score": summary["average_score"],
            "average_rank": summary["average_rank"],
            **point["training_metrics"],
        }, ensure_ascii=False), flush=True)
    environment.close()


if __name__ == "__main__":
    main()
