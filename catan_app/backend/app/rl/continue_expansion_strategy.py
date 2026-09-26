"""事前調整済み拡張GNNを、制限付きPPOから通常ルールPPOへ段階移行する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
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
from .pretrain_expansion_target import (collect_target_examples,
                                        train_expansion_target)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _environment(source: PPOAgent, *, curriculum: bool,
                 early_development_penalty: float = 0.0,
                 unproductive_road_penalty: float = 0.0) -> CatanEnv:
    if source.initial_setup_policy is None:
        raise ValueError("初期配置GNNを持つ現行PPOモデルが必要です。")
    return CatanEnv(
        CatanEnvConfig(
            learning_player_id=1,
            opponent_controller="heuristic",
            policy_compatible_opponents=True,
            observation_version=source.manifest.observation_version,
            expansion_curriculum=curriculum,
            early_development_penalty=early_development_penalty,
            unproductive_road_penalty=unproductive_road_penalty,
        ),
        initial_placement_agent=GraphInitialSetupAgent(source.initial_setup_policy),
    )


def _summary_view(summary: dict) -> dict[str, float | int | None]:
    actions = summary.get("average_actions", {})
    opportunities = summary.get("average_opportunities", {})
    strategy = summary.get("average_strategy", {})
    road_builds = float(strategy.get("road_builds", 0))
    opened = float(strategy.get("road_opens_settlement_site", 0))
    settlement_builds = float(strategy.get("settlement_builds", 0))
    return {
        "win_rate": summary["win_rate"],
        "average_score": summary["average_score"],
        "average_rank": summary["average_rank"],
        "build_settlement": actions.get("build_settlement", 0),
        "build_city": actions.get("build_city", 0),
        "build_road_paid": actions.get("build_road_paid", 0),
        "buy_development": actions.get("buy_development", 0),
        "build_road_with_settlement_available": opportunities.get(
            "build_road_with_settlement_available", 0
        ),
        "buy_development_before_third_site": opportunities.get(
            "buy_development_before_third_site", 0
        ),
        "buy_development_while_builds_unavailable": opportunities.get(
            "buy_development_while_builds_unavailable", 0
        ),
        "build_road_while_planner_waits": opportunities.get(
            "build_road_while_planner_waits", 0
        ),
        "buy_development_while_build_near": opportunities.get(
            "buy_development_while_build_near", 0
        ),
        "buy_development_during_planned_stall": opportunities.get(
            "buy_development_during_planned_stall", 0
        ),
        "planned_road_selected": opportunities.get("planned_road_selected", 0),
        "off_plan_road_selected": opportunities.get("off_plan_road_selected", 0),
        "planned_settlement_selected": opportunities.get(
            "planned_settlement_selected", 0
        ),
        "road_opens_settlement_site_rate": opened / road_builds if road_builds else 0.0,
        "road_without_immediate_settlement_site": strategy.get(
            "road_without_immediate_settlement_site", 0
        ),
        "settlement_average_pips": (
            float(strategy.get("settlement_pips_total", 0)) / settlement_builds
            if settlement_builds else 0.0
        ),
        "settlement_new_resource_kinds": strategy.get(
            "settlement_new_resource_kinds_total", 0
        ),
        "largest_army_rate": summary.get("largest_army_acquisition_rate", 0),
        "longest_road_rate": summary.get("longest_road_acquisition_rate", 0),
    }


def _evaluate(model: MaskablePPO, source: PPOAgent, seeds: tuple[int, ...],
              stage: str, output: Path) -> dict:
    candidate = PPOAgent(source.manifest.model_id, models_root=source.registry.models_root,
                         device="cpu")
    candidate.model = model
    report = evaluate_agent(
        candidate,
        seeds,
        config=EvaluationConfig(
            heuristic_opponents=3,
            rotate_player_ids=True,
            policy_compatible_opponents=True,
            observation_version=source.manifest.observation_version,
        ),
        agent_name=f"{source.manifest.model_id}:expansion:{stage}",
    ).to_dict()
    summary = report["summary"]
    if summary["illegal_action_count"] or summary["truncated"]:
        raise AssertionError(f"{stage}: 非法Actionまたは打ち切りを検出しました。")
    _write(output / f"evaluation_{stage}.json", report)
    return _summary_view(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--pretrained-policy", type=Path, default=Path(
        "experiments/expansion_target_pretrain_s04_20260924/expansion_target_policy.pt"
    ))
    parser.add_argument("--curriculum-steps", type=int, default=2_048)
    parser.add_argument("--normal-steps", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--training-seed", type=int, default=20261621)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-seed-start", type=int, default=260_001)
    parser.add_argument("--anchor-games", type=int, default=0)
    parser.add_argument("--anchor-seed-start", type=int, default=280_001)
    parser.add_argument("--anchor-learning-rate", type=float, default=1e-5)
    parser.add_argument("--anchor-restricted-weight", type=float, default=4.0)
    parser.add_argument("--anchor-max-epochs", type=int, default=5)
    parser.add_argument("--anchor-patience", type=int, default=2)
    parser.add_argument("--parameter-anchor-strength", type=float, default=0.0)
    parser.add_argument("--curriculum-mode", choices=("mask", "penalty"), default="mask")
    parser.add_argument("--early-development-penalty", type=float, default=0.03)
    parser.add_argument("--unproductive-road-penalty", type=float, default=0.02)
    args = parser.parse_args()
    if min(args.curriculum_steps, args.normal_steps, args.eval_games) <= 0 or args.anchor_games < 0:
        parser.error("step数と評価局数は正の整数にしてください。")
    if min(args.learning_rate, args.anchor_learning_rate,
           args.anchor_max_epochs, args.anchor_patience) <= 0:
        parser.error("学習率・anchorのepoch数・patienceは0より大きくしてください。")
    if not 0.0 <= args.parameter_anchor_strength <= 1.0:
        parser.error("parameter-anchor-strengthは0以上1以下にしてください。")
    if min(args.early_development_penalty, args.unproductive_road_penalty) < 0:
        parser.error("strategy penaltyは0以上にしてください。")
    if args.anchor_games and args.parameter_anchor_strength:
        parser.error("局面KL anchorとparameter anchorは同時に指定できません。")

    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。")
    output.mkdir(parents=True)

    source = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    if type(source.model.policy) is not GraphHierarchicalCandidateMaskablePolicy:
        raise TypeError("移行元は現行GraphHierarchicalCandidate方策である必要があります。")
    use_penalty_curriculum = args.curriculum_mode == "penalty"
    curriculum_environment = _environment(
        source,
        curriculum=not use_penalty_curriculum,
        early_development_penalty=(args.early_development_penalty
                                   if use_penalty_curriculum else 0.0),
        unproductive_road_penalty=(args.unproductive_road_penalty
                                   if use_penalty_curriculum else 0.0),
    )
    model = MaskablePPO(
        ExpansionGraphHierarchicalCandidateMaskablePolicy,
        curriculum_environment,
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
        seed=args.training_seed,
        device="cpu",
        verbose=0,
    )
    initialize_expansion_policy_from_graph(model.policy, source.model.policy)
    pretrained_path = args.pretrained_policy.resolve()
    model.policy.load_state_dict(torch.load(pretrained_path, map_location="cpu",
                                            weights_only=True))
    parameter_anchor_prefixes = (
        "mlp_extractor.expansion_vertex_encoder.",
        "mlp_extractor.expansion_graph_layers.",
        "mlp_extractor.expansion_edge_encoder.",
        "mlp_extractor.expansion_settlement_head.",
        "mlp_extractor.expansion_road_head.",
        "mlp_extractor.expansion_family_head.",
    )
    parameter_anchor_reference = {
        name: parameter.detach().clone()
        for name, parameter in model.policy.named_parameters()
        if name.startswith(parameter_anchor_prefixes)
    }
    if not parameter_anchor_reference:
        raise AssertionError("parameter anchor対象の拡張残差が見つかりません。")
    selection = configure_expansion_graph_finetune(model.policy)
    model.num_timesteps = source.model.num_timesteps
    model.set_random_seed(args.training_seed)
    if args.curriculum_steps % model.n_steps or args.normal_steps % model.n_steps:
        parser.error(f"各step数はrollout長 {model.n_steps} の倍数にしてください。")

    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    definition = {
        "source_model": args.model_id,
        "pretrained_policy": str(pretrained_path),
        "training_seed": args.training_seed,
        "source_steps": source.model.num_timesteps,
        "curriculum_steps": args.curriculum_steps,
        "normal_steps": args.normal_steps,
        "learning_rate": args.learning_rate,
        "rollout_steps": model.n_steps,
        "evaluation_seeds": seeds,
        "normal_rules_during_all_evaluations": True,
        "curriculum_mode": args.curriculum_mode,
        "early_development_penalty": (args.early_development_penalty
                                        if use_penalty_curriculum else None),
        "unproductive_road_penalty": (args.unproductive_road_penalty
                                        if use_penalty_curriculum else None),
        "anchor_games": args.anchor_games,
        "anchor_seed_start": args.anchor_seed_start if args.anchor_games else None,
        "anchor_learning_rate": args.anchor_learning_rate if args.anchor_games else None,
        "anchor_restricted_weight": args.anchor_restricted_weight if args.anchor_games else None,
        "anchor_max_epochs": args.anchor_max_epochs if args.anchor_games else None,
        "anchor_patience": args.anchor_patience if args.anchor_games else None,
        "parameter_anchor_strength": args.parameter_anchor_strength,
        **selection,
    }
    _write(output / "definition.json", definition)

    anchor_sets = None
    if args.anchor_games:
        validation_games = max(4, args.anchor_games // 4)
        test_games = max(4, args.anchor_games // 4)
        train_start = args.anchor_seed_start
        validation_start = train_start + args.anchor_games
        test_start = validation_start + validation_games
        anchor_sets = (
            collect_target_examples(source, range(train_start, validation_start)),
            collect_target_examples(source, range(validation_start, test_start)),
            collect_target_examples(source, range(test_start, test_start + test_games)),
        )

    anchor_history: list[dict[str, object]] = []

    def apply_parameter_anchor(stage: str, cycle: int) -> None:
        strength = args.parameter_anchor_strength
        if not strength:
            return
        squared_before = 0.0
        squared_after = 0.0
        with torch.no_grad():
            for name, parameter in model.policy.named_parameters():
                reference = parameter_anchor_reference.get(name)
                if reference is None:
                    continue
                delta = parameter - reference
                squared_before += float(delta.square().sum())
                parameter.mul_(1.0 - strength).add_(reference, alpha=strength)
                squared_after += float((parameter - reference).square().sum())
        anchor_history.append({
            "stage": stage,
            "cycle": cycle,
            "method": "parameter_proximal",
            "strength": strength,
            "distance_before": math.sqrt(squared_before),
            "distance_after": math.sqrt(squared_after),
        })

    def learn_phase(steps: int, stage: str) -> None:
        cycles = steps // model.n_steps
        for cycle in range(1, cycles + 1):
            model.learn(total_timesteps=model.n_steps,
                        reset_num_timesteps=False, progress_bar=False)
            apply_parameter_anchor(stage, cycle)
            if anchor_sets is None:
                continue
            report = train_expansion_target(
                model, *anchor_sets,
                learning_rate=args.anchor_learning_rate,
                batch_size=256,
                max_epochs=args.anchor_max_epochs,
                patience=args.anchor_patience,
                restricted_weight=args.anchor_restricted_weight,
                seed=args.training_seed + cycle,
            )
            anchor_history.append({
                "stage": stage,
                "cycle": cycle,
                "method": "sampled_kl",
                "best_epoch": report["best_epoch"],
                "initial_validation_weighted_kl": report["initial_validation_weighted_kl"],
                "best_validation_weighted_kl": report["best_validation_kl"],
                "epochs": report["history"],
            })
            configure_expansion_graph_finetune(model.policy)

    summaries: dict[str, dict] = {}
    summaries["pretrained"] = _evaluate(model, source, seeds, "pretrained", output)

    learn_phase(args.curriculum_steps, "curriculum")
    model.save(str(output / "checkpoint_curriculum.zip"))
    summaries["curriculum"] = _evaluate(model, source, seeds, "curriculum", output)

    normal_environment = _environment(source, curriculum=False)
    model.set_env(normal_environment)
    learn_phase(args.normal_steps, "normal")
    model.save(str(output / "checkpoint_normal.zip"))
    summaries["normal"] = _evaluate(model, source, seeds, "normal", output)

    result = {"definition": definition, "summaries": summaries,
              "anchor_history": anchor_history}
    _write(output / "result.json", result)
    curriculum_environment.close()
    normal_environment.close()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
