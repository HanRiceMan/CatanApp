"""保存済み候補方策のargmaxと確率抽出を、同一盤面で評価する。"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .analyze_supervised_setup import paired_effect
from .audit import runtime_versions, source_fingerprint
from .evaluate import EvaluationConfig, evaluate_agent


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _paired_difference(stochastic: dict, deterministic: dict, field: str) -> dict:
    effect = paired_effect(stochastic, deterministic, field)
    return {
        "stochastic_minus_deterministic": effect["candidate_minus_heuristic"],
        "bootstrap_95_percent_interval": effect["bootstrap_95_percent_interval"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[20260923, 20260925])
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=[5000, 10000])
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=40001)
    parser.add_argument("--policy-seed", type=int, default=20260926)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    if args.eval_games <= 0 or any(step <= 0 for step in args.checkpoint_steps):
        parser.error("eval-gamesとcheckpoint-stepsは正の整数です。")
    if len(set(args.training_seeds)) != len(args.training_seeds) or len(set(args.checkpoint_steps)) != len(args.checkpoint_steps):
        parser.error("学習seedとチェックポイントstep数に重複は指定できません。")
    root = Path(__file__).resolve().parents[2] / "experiments"
    output = root / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。別IDを指定してください。")

    from sb3_contrib import MaskablePPO

    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    checkpoints = {}
    for training_seed in args.training_seeds:
        models_root = root / f"candidate_fixed_setup_10k_s{training_seed}" / "models"
        model_id = f"candidate_10000_s{training_seed}"
        for steps in args.checkpoint_steps:
            path = models_root / model_id / f"checkpoint_{steps}_steps.zip"
            if not path.is_file():
                parser.error(f"チェックポイントが見つかりません: {path}")
            checkpoints[f"s{training_seed}_{steps}"] = {
                "path": path, "models_root": models_root, "model_id": model_id,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }

    output.mkdir(parents=True)
    _write_json(output / "definition.json", {
        "experiment_id": args.experiment_id,
        "training_seeds": args.training_seeds,
        "checkpoint_steps": args.checkpoint_steps,
        "evaluation_seeds": seeds,
        "stochastic_policy_seed": args.policy_seed,
        "evaluation_opponents": "heuristic_x3",
        "rotate_player_ids": True,
        "policy_compatible_opponents": True,
        "source_sha256": source_fingerprint(),
        "runtime_versions": runtime_versions(),
        "checkpoint_sha256": {key: value["sha256"] for key, value in checkpoints.items()},
    })
    summaries = {}
    for key, paths in checkpoints.items():
        reports = {}
        for mode in ("deterministic", "stochastic"):
            agent = PPOAgent(paths["model_id"], models_root=paths["models_root"],
                             deterministic=mode == "deterministic", device="cpu")
            agent.model = MaskablePPO.load(str(paths["path"]), device="cpu")
            if agent.model.num_timesteps != int(key.rsplit("_", 1)[1]):
                raise ValueError(f"チェックポイント内のstep数が不一致です: {paths['path']}")
            if mode == "stochastic":
                agent.model.set_random_seed(args.policy_seed)
            report = evaluate_agent(
                agent, seeds,
                config=EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                                        policy_compatible_opponents=True,
                                        observation_version=agent.manifest.observation_version),
                agent_name=f"{paths['model_id']}:{key}:{mode}",
            ).to_dict()
            if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
                raise AssertionError(f"非法Actionまたは打ち切りが発生しました: {key}:{mode}")
            reports[mode] = report
            _write_json(output / f"{key}_{mode}.json", report)
            summaries[f"{key}_{mode}"] = report["summary"]
            _write_json(output / "summary.json", summaries)
            print(json.dumps({"cell": f"{key}_{mode}",
                              "win_rate": report["summary"]["win_rate"],
                              "average_score": report["summary"]["average_score"],
                              "paid_roads": report["summary"]["average_actions"].get("build_road_paid", 0.0),
                              "settlements": report["summary"]["average_actions"].get("build_settlement", 0.0)},
                             ensure_ascii=False), flush=True)
        deterministic, stochastic = reports["deterministic"], reports["stochastic"]
        if any(a["learner_initial_vertices"] != b["learner_initial_vertices"]
               for a, b in zip(deterministic["episodes"], stochastic["episodes"])):
            raise AssertionError(f"初期配置が選択方式間で一致しません: {key}")
        summaries[f"{key}_paired"] = {
            "games": len(seeds),
            "identical_initial_placements": True,
            "win_rate_difference": _paired_difference(stochastic, deterministic, "learner_won"),
            "average_score_difference": _paired_difference(stochastic, deterministic, "learner_score"),
        }
        _write_json(output / "summary.json", summaries)


if __name__ == "__main__":
    main()
