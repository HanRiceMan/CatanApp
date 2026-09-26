"""保存済み候補方策の行動種別選択を、通常ターンだけ差し替えて比較する。"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .analyze_supervised_setup import paired_effect
from .audit import runtime_versions, source_fingerprint
from .evaluate import EvaluationConfig, evaluate_agent
from .policy_selection import FamilySelectionAgent


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[20260923, 20260925])
    parser.add_argument("--checkpoint-steps", type=int, default=None,
                        help="省略時は学習完了後model.zip。指定時だけ途中保存チェックポイントを使う")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=40001)
    parser.add_argument("--methods", nargs="+", choices=("global_argmax", "family_mass", "family_l2"),
                        default=["global_argmax", "family_mass", "family_l2"])
    parser.add_argument("--heuristic-opponents", type=int, choices=range(4), default=3)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    if ((args.checkpoint_steps is not None and args.checkpoint_steps <= 0)
            or args.eval_games <= 0 or len(set(args.training_seeds)) != len(args.training_seeds)):
        parser.error("step数・評価対局数は正の整数、学習seedは重複なしで指定してください。")
    if len(set(args.methods)) != len(args.methods) or "global_argmax" not in args.methods:
        parser.error("methodsは重複なしで、global_argmaxを必ず含めてください。")
    experiments_root = Path(__file__).resolve().parents[2] / "experiments"
    output = experiments_root / args.experiment_id
    if output.exists():
        parser.error("同名実験フォルダが既にあります。別IDを指定してください。")

    from sb3_contrib import MaskablePPO

    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    checkpoints = {}
    for training_seed in args.training_seeds:
        models_root = experiments_root / f"candidate_fixed_setup_10k_s{training_seed}" / "models"
        model_id = f"candidate_10000_s{training_seed}"
        artifact = "model.zip" if args.checkpoint_steps is None else f"checkpoint_{args.checkpoint_steps}_steps.zip"
        path = models_root / model_id / artifact
        if not path.is_file():
            parser.error(f"保存済みチェックポイントが見つかりません: {path}")
        checkpoints[training_seed] = (models_root, model_id, path)

    output.mkdir(parents=True)
    _write_json(output / "definition.json", {
        "experiment_id": args.experiment_id,
        "training_seeds": args.training_seeds,
        "model_artifact": "final_model" if args.checkpoint_steps is None else "checkpoint",
        "checkpoint_steps": args.checkpoint_steps,
        "evaluation_seeds": seeds,
        "methods": args.methods,
        "within_family_selection": "highest_individual_probability",
        "application_phase": "action_only",
        "opponents": f"heuristic_x{args.heuristic_opponents}_random_x{3-args.heuristic_opponents}",
        "rotate_player_ids": True,
        "policy_compatible_opponents": True,
        "source_sha256": source_fingerprint(),
        "runtime_versions": runtime_versions(),
        "checkpoint_sha256": {str(seed): sha256(path.read_bytes()).hexdigest()
                              for seed, (_, _, path) in checkpoints.items()},
    })
    summaries = {}
    for training_seed, (models_root, model_id, checkpoint) in checkpoints.items():
        reports = {}
        for method in args.methods:
            policy = PPOAgent(model_id, models_root=models_root, deterministic=True, device="cpu")
            policy.model = MaskablePPO.load(str(checkpoint), device="cpu")
            expected_steps = (policy.manifest.training_steps
                              if args.checkpoint_steps is None else args.checkpoint_steps)
            if policy.model.num_timesteps != expected_steps:
                raise ValueError(f"チェックポイント内のstep数が不一致です: {checkpoint}")
            agent = policy if method == "global_argmax" else FamilySelectionAgent(policy, method)
            report = evaluate_agent(
                agent, seeds,
                config=EvaluationConfig(heuristic_opponents=args.heuristic_opponents, rotate_player_ids=True,
                                        policy_compatible_opponents=True,
                                        observation_version=policy.manifest.observation_version),
                agent_name=f"{model_id}:{'final' if args.checkpoint_steps is None else args.checkpoint_steps}:{method}",
            ).to_dict()
            summary = report["summary"]
            if summary["illegal_action_count"] or summary["truncated"]:
                raise AssertionError(f"非法Actionまたは打ち切りが発生しました: {training_seed}:{method}")
            reports[method] = report
            key = f"s{training_seed}_{method}"
            _write_json(output / f"{key}.json", report)
            summaries[key] = summary
            _write_json(output / "summary.json", summaries)
            print(json.dumps({"cell": key, "win_rate": summary["win_rate"],
                              "average_score": summary["average_score"],
                              "paid_roads": summary["average_actions"].get("build_road_paid", 0.0),
                              "settlements": summary["average_actions"].get("build_settlement", 0.0),
                              "bank_trades": summary["average_actions"].get("bank_trade", 0.0)},
                             ensure_ascii=False), flush=True)

        baseline = reports["global_argmax"]
        paired = {}
        for method in (method for method in args.methods if method != "global_argmax"):
            current = reports[method]
            if any(a["learner_initial_vertices"] != b["learner_initial_vertices"]
                   for a, b in zip(current["episodes"], baseline["episodes"])):
                raise AssertionError(f"初期配置が選択方式間で一致しません: {training_seed}:{method}")
            effects = {}
            for name, field in (("win_rate", "learner_won"), ("average_score", "learner_score")):
                result = paired_effect(current, baseline, field)
                effects[name] = {
                    "method_minus_global_argmax": result["candidate_minus_heuristic"],
                    "bootstrap_95_percent_interval": result["bootstrap_95_percent_interval"],
                }
            paired[method] = {"games": len(seeds), "identical_initial_placements": True, **effects}
        summaries[f"s{training_seed}_paired"] = paired
        _write_json(output / "summary.json", summaries)


if __name__ == "__main__":
    main()
