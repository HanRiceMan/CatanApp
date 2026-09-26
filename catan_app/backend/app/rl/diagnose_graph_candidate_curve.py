"""既存階層PPOを保ったまま通常ターンGraph補正だけを学習・評価する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from app.agents.ppo import PPOAgent

from .candidate_policy import (
    GraphFamilyHierarchicalCandidateMaskablePolicy,
    GraphHierarchicalCandidateMaskablePolicy,
    HierarchicalCandidateMaskablePolicy,
    configure_graph_residual_finetune,
    initialize_graph_policy_from_hierarchical,
)
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
    return {
        name.removeprefix("train/"): float(model.logger.name_to_value[name])
        for name in TRAIN_METRICS
        if name in model.logger.name_to_value
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--gnn-experiment", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_setup_n1024_s01_v001")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--additional-steps", type=int, default=10_240)
    parser.add_argument("--checkpoint-steps", type=int, default=5_120)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--freeze-base-actor", action="store_true",
                        help="既存Actorを固定し、Graph補正とValue推定だけを更新する。")
    parser.add_argument("--graph-family-head", action="store_true",
                        help="盤面Graphの要約をAction Family選択にも残差入力する。")
    parser.add_argument("--family-pretrain-artifact", type=Path, default=None,
                        help="教師あり事前学習済みGraph Family方策のstate_dict。")
    parser.add_argument("--family-residual-scale", type=float, default=1.0,
                        help="教師ありGraph Family残差を元Family logitへ加える比率。")
    parser.add_argument("--seed", type=int, default=20261501)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=180_001)
    args = parser.parse_args()

    if args.additional_steps < 0 or args.checkpoint_steps <= 0:
        parser.error("additional-stepsは0以上、checkpoint-stepsは0より大きくしてください。")
    if args.additional_steps % args.checkpoint_steps:
        parser.error("additional-stepsはcheckpoint-stepsの倍数にしてください。")
    if args.learning_rate <= 0:
        parser.error("learning-rateは0より大きくしてください。")
    if args.gamma is not None and not 0 < args.gamma <= 1:
        parser.error("gammaは0より大きく1以下にしてください。")
    if not 0 <= args.family_residual_scale <= 1:
        parser.error("family-residual-scaleは0〜1で指定してください。")

    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。")
    output.mkdir(parents=True)

    source = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    if type(source.model.policy) is not HierarchicalCandidateMaskablePolicy:
        raise TypeError("移行元はHierarchicalCandidateMaskablePolicyである必要があります。")

    setup = GraphInitialSetupPolicy(source.model.policy)
    setup_path = root / "experiments" / args.gnn_experiment / "gnn_policy.pt"
    setup.load_state_dict(torch.load(setup_path, map_location="cpu", weights_only=True))
    setup.eval()
    setup_agent = GraphInitialSetupAgent(setup)
    environment = CatanEnv(
        CatanEnvConfig(
            learning_player_id=1,
            opponent_controller="heuristic",
            observation_version="v2",
        ),
        initial_placement_agent=setup_agent,
    )

    gamma = source.model.gamma if args.gamma is None else args.gamma
    policy_class = (GraphFamilyHierarchicalCandidateMaskablePolicy
                    if args.graph_family_head else GraphHierarchicalCandidateMaskablePolicy)
    model = MaskablePPO(
        policy_class,
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
    added_parameters = initialize_graph_policy_from_hierarchical(model.policy, source.model.policy)
    if args.family_pretrain_artifact is not None:
        if not args.graph_family_head:
            parser.error("Family事前学習モデルには--graph-family-headが必要です。")
        artifact = args.family_pretrain_artifact.resolve()
        if not artifact.is_file():
            parser.error(f"Family事前学習モデルがありません: {artifact}")
        model.policy.load_state_dict(
            torch.load(artifact, map_location="cpu", weights_only=True)
        )
    if args.graph_family_head:
        model.policy.mlp_extractor.graph_family_scale = args.family_residual_scale
    parameter_selection = (
        configure_graph_residual_finetune(model.policy)
        if args.freeze_base_actor
        else {
            "trainable_parameters": sum(parameter.numel() for parameter in model.policy.parameters()),
            "frozen_parameters": 0,
        }
    )
    source_steps = source.model.num_timesteps
    model.num_timesteps = source_steps
    model.set_random_seed(args.seed)
    if args.checkpoint_steps % model.n_steps:
        parser.error(f"checkpoint-stepsはrollout長 {model.n_steps} の倍数にしてください。")

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
        "rollout_steps": model.n_steps,
        "gamma": gamma,
        "learning_rate": args.learning_rate,
        "graph_message_layers": 2,
        "graph_family_head": args.graph_family_head,
        "family_pretrain_artifact": (
            str(args.family_pretrain_artifact.resolve())
            if args.family_pretrain_artifact is not None else None
        ),
        "family_residual_scale": args.family_residual_scale,
        "freeze_base_actor": args.freeze_base_actor,
        **parameter_selection,
        "transferred_parameters": len(source.model.policy.state_dict()),
        "added_parameter_keys": added_parameters,
        "evaluation_seeds": seeds,
    }
    _write(output / "definition.json", definition)

    baseline = IndependentSetupAgent(source, setup)
    points: list[dict[str, object]] = []
    for added_steps in range(0, args.additional_steps + 1, args.checkpoint_steps):
        if added_steps:
            model.learn(total_timesteps=args.checkpoint_steps, reset_num_timesteps=False,
                        progress_bar=False)
            model.save(str(output / f"checkpoint_{added_steps}_steps.zip"))
            policy_agent = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
            policy_agent.model = model
            agent = IndependentSetupAgent(policy_agent, setup)
        else:
            if args.family_pretrain_artifact is None:
                agent = baseline
            else:
                policy_agent = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
                policy_agent.model = model
                agent = IndependentSetupAgent(policy_agent, setup)
        report = evaluate_agent(
            agent,
            seeds,
            config=evaluation_config,
            agent_name=f"{args.model_id}:normal_turn_gnn:+{added_steps}",
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
