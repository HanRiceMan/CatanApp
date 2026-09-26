"""初期配置GNNを固定し、50k階層PPOの通常手番だけを追加学習する。"""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--gnn-experiment", required=True)
    parser.add_argument("--model-id", default="ppo_hierarchical_50k_v001")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--additional-steps", type=int, default=5_000)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="継続学習時だけ使用する学習率。省略時は元モデルの設定を維持する。",
    )
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=110_001)
    args = parser.parse_args()
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
    environment = CatanEnv(CatanEnvConfig(
        learning_player_id=1, opponent_controller="heuristic",
        heuristic_initial_placement=False, observation_version="v2"),
        initial_placement_agent=setup_agent)
    artifact = source.registry.models_root / args.model_id / source.manifest.artifact_filename
    model = MaskablePPO.load(str(artifact), env=environment, device="cpu")
    start_steps = model.num_timesteps
    source_learning_rate = float(model.lr_schedule(1.0))
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            parser.error("--learning-rate は0より大きい値を指定してください。")
        model.learning_rate = args.learning_rate
        model.lr_schedule = FloatSchedule(args.learning_rate)
    model.set_random_seed(args.seed)
    model.learn(total_timesteps=args.additional_steps, reset_num_timesteps=False,
                progress_bar=False)
    model.save(str(output / "model.zip"))
    environment.close()

    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    config = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                              policy_compatible_opponents=True, observation_version="v2")
    candidates = {}
    baseline_base = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    candidates["before_finetune"] = IndependentSetupAgent(baseline_base, gnn)
    after_base = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    after_base.model = model
    candidates["after_finetune"] = IndependentSetupAgent(after_base, gnn)
    reports = {name: evaluate_agent(agent, seeds, config=config, agent_name=name).to_dict()
               for name, agent in candidates.items()}
    result = {
        "definition": {"source_model": args.model_id, "gnn_experiment": args.gnn_experiment,
                       "training_seed": args.seed, "start_steps": start_steps,
                       "additional_steps": args.additional_steps,
                       "source_learning_rate": source_learning_rate,
                       "finetune_learning_rate": (
                           args.learning_rate
                           if args.learning_rate is not None
                           else source_learning_rate
                       ),
                       "final_steps": model.num_timesteps, "evaluation_seeds": seeds},
        "summaries": {name: report["summary"] for name, report in reports.items()},
        "reports": reports,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summaries"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
