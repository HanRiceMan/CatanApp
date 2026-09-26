"""現行GNNへ拡張戦略残差を追加し、カリキュラム学習後に通常ルールで評価する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from app.agents.ppo import PPOAgent

from .candidate_policy import (
    ExpansionGraphHierarchicalCandidateMaskablePolicy,
    GraphHierarchicalCandidateMaskablePolicy,
    configure_expansion_graph_finetune,
    initialize_expansion_policy_from_graph,
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
    values = model.logger.name_to_value
    return {name.removeprefix("train/"): float(values[name])
            for name in TRAIN_METRICS if name in values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--additional-steps", type=int, default=5_120)
    parser.add_argument("--checkpoint-steps", type=int, default=1_024)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=20261601)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--eval-seed-start", type=int, default=240_001)
    parser.add_argument("--no-curriculum", action="store_true")
    args = parser.parse_args()
    if args.additional_steps <= 0 or args.checkpoint_steps <= 0:
        parser.error("step数は正の整数にしてください。")
    if args.additional_steps % args.checkpoint_steps:
        parser.error("additional-stepsはcheckpoint-stepsの倍数にしてください。")
    if args.learning_rate <= 0:
        parser.error("learning-rateは0より大きくしてください。")

    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。")
    output.mkdir(parents=True)

    source = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    if type(source.model.policy) is not GraphHierarchicalCandidateMaskablePolicy:
        raise TypeError("移行元は現行GraphHierarchicalCandidate方策である必要があります。")
    if source.initial_setup_policy is None:
        raise TypeError("移行元には初期配置GNNが必要です。")
    setup_agent = GraphInitialSetupAgent(source.initial_setup_policy)
    environment = CatanEnv(
        CatanEnvConfig(
            learning_player_id=1,
            opponent_controller="heuristic",
            policy_compatible_opponents=True,
            observation_version=source.manifest.observation_version,
            expansion_curriculum=not args.no_curriculum,
        ),
        initial_placement_agent=setup_agent,
    )
    model = MaskablePPO(
        ExpansionGraphHierarchicalCandidateMaskablePolicy,
        environment,
        learning_rate=args.learning_rate,
        n_steps=source.model.n_steps,
        batch_size=source.model.batch_size,
        n_epochs=source.model.n_epochs,
        gamma=source.model.gamma,
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
    added = initialize_expansion_policy_from_graph(model.policy, source.model.policy)
    selection = configure_expansion_graph_finetune(model.policy)
    model.num_timesteps = source.model.num_timesteps
    model.set_random_seed(args.seed)
    if args.checkpoint_steps % model.n_steps:
        parser.error(f"checkpoint-stepsはrollout長 {model.n_steps} の倍数にしてください。")

    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    evaluation_config = EvaluationConfig(
        heuristic_opponents=3,
        rotate_player_ids=True,
        policy_compatible_opponents=True,
        observation_version=source.manifest.observation_version,
    )
    definition = {
        "source_model": args.model_id,
        "training_seed": args.seed,
        "source_steps": source.model.num_timesteps,
        "additional_steps": args.additional_steps,
        "checkpoint_steps": args.checkpoint_steps,
        "rollout_steps": model.n_steps,
        "gamma": model.gamma,
        "learning_rate": args.learning_rate,
        "expansion_curriculum": not args.no_curriculum,
        "normal_rules_restored_during_evaluation": True,
        "strategic_inputs": [
            "vertex_resource_pips", "self_resource_inventory",
            "self_production_profile", "port_type_and_ratio",
            "road_endpoints", "site_counts",
        ],
        **selection,
        "added_parameter_keys": added,
        "evaluation_seeds": seeds,
    }
    _write(output / "definition.json", definition)

    points: list[dict[str, object]] = []
    for added_steps in range(0, args.additional_steps + 1, args.checkpoint_steps):
        if added_steps:
            model.learn(total_timesteps=args.checkpoint_steps,
                        reset_num_timesteps=False, progress_bar=False)
            model.save(str(output / f"checkpoint_{added_steps}_steps.zip"))
        candidate = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
        candidate.model = model
        report = evaluate_agent(
            candidate, seeds, config=evaluation_config,
            agent_name=f"{args.model_id}:expansion_gnn:+{added_steps}",
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
            "settlements": summary["average_actions"].get("build_settlement", 0),
            "roads": summary["average_actions"].get("build_road_paid", 0),
            "development": summary["average_actions"].get("buy_development", 0),
            **point["training_metrics"],
        }, ensure_ascii=False), flush=True)
    environment.close()


if __name__ == "__main__":
    main()
