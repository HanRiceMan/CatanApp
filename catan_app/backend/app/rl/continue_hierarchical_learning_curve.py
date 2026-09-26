"""階層候補PPOを10kから25k・50kへ継続し、同一盤面で学習曲線を測る。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from app.agents.ppo import PPOAgent

from .audit import runtime_versions, source_fingerprint
from .evaluate import EvaluationConfig, evaluate_agent
from .train import TrainConfig, train_maskable_ppo


CHECKPOINTS = (10_000, 25_000, 50_000)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_source(value: str) -> tuple[int, Path]:
    seed_text, separator, directory_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("sourceは TRAINING_SEED=MODEL_DIRECTORY 形式です。")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("sourceの学習seedは整数です。") from error
    directory = Path(directory_text).resolve()
    required = (directory / "checkpoint_10000_steps.zip", directory / "metadata.json")
    if not all(path.is_file() for path in required):
        raise argparse.ArgumentTypeError(f"10k checkpointまたはmetadataがありません: {directory}")
    return seed, directory


def _checkpoint_agent(model_directory: Path, steps: int) -> PPOAgent:
    from sb3_contrib import MaskablePPO

    agent = PPOAgent(model_directory.name, models_root=model_directory.parent, device="cpu")
    checkpoint = model_directory / f"checkpoint_{steps}_steps.zip"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"評価checkpointがありません: {checkpoint}")
    model = MaskablePPO.load(str(checkpoint), device="cpu")
    if model.num_timesteps != steps:
        raise ValueError(f"checkpoint内部のstep数が一致しません: {checkpoint}")
    agent.model = model
    return agent


def _evaluate(model_directory: Path, steps: int, seeds: tuple[int, ...]) -> dict:
    report = evaluate_agent(
        _checkpoint_agent(model_directory, steps),
        seeds,
        config=EvaluationConfig(
            heuristic_opponents=3,
            rotate_player_ids=True,
            policy_compatible_opponents=True,
            observation_version="v2",
        ),
        agent_name=f"{model_directory.name}:checkpoint_{steps}",
    ).to_dict()
    summary = report["summary"]
    if summary["illegal_action_count"] or summary["truncated"]:
        raise AssertionError(f"非法Actionまたは打ち切り: {model_directory.name}@{steps}")
    return report


def _train_stage(*, model_id: str, models_root: Path, source_id: str,
                 source_steps: int, target_steps: int, seed: int) -> Path:
    directory = models_root / model_id
    if (directory / "metadata.json").is_file() and (directory / "model.zip").is_file():
        return directory
    if directory.exists():
        raise FileExistsError(f"未完了のモデルフォルダがあります: {directory}")
    train_maskable_ppo(TrainConfig(
        model_id=model_id,
        models_root=models_root,
        total_timesteps=target_steps,
        seed=seed,
        observation_version="v2",
        policy_architecture="hierarchical_candidate",
        heuristic_initial_placement=True,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=.99,
        gae_lambda=.95,
        ent_coef=.01,
        checkpoint_interval_steps=target_steps - source_steps,
        resume_checkpoint_steps=source_steps,
        resume_from_model_id=source_id,
        evaluation_seeds=tuple(range(61_001, 61_005)),
        device="cpu",
    ))
    expected = directory / f"checkpoint_{target_steps}_steps.zip"
    if not expected.is_file():
        raise FileNotFoundError(f"継続学習後のcheckpointがありません: {expected}")
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source", action="append", type=_parse_source, required=True,
                        metavar="SEED=MODEL_DIRECTORY")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=62_001)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    if args.eval_games <= 0:
        parser.error("eval-gamesは正の整数です。")
    sources = dict(args.source)
    if len(sources) != len(args.source):
        parser.error("学習seedが重複しています。")

    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    models_root = output / "models"
    output.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(exist_ok=True)
    evaluation_seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    definition = {
        "experiment_id": args.experiment_id,
        "sources": {str(seed): str(directory) for seed, directory in sources.items()},
        "checkpoints": CHECKPOINTS,
        "evaluation_seeds": evaluation_seeds,
        "policy_architecture": "hierarchical_candidate",
        "heuristic_initial_placement": True,
        "training_opponents": "heuristic_x3",
        "evaluation_opponents": "heuristic_x3",
        "observation_version": "v2",
        "action_space_version": "v1",
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 4,
        "learning_rate": 3e-4,
        "gamma": .99,
        "gae_lambda": .95,
        "ent_coef": .01,
        "source_sha256": source_fingerprint(),
        "runtime_versions": runtime_versions(),
    }
    definition_path = output / "definition.json"
    if definition_path.is_file():
        existing = json.loads(definition_path.read_text(encoding="utf-8"))
        stable_keys = tuple(key for key in definition if key not in {"source_sha256", "runtime_versions"})
        if any(existing.get(key) != definition[key] for key in stable_keys):
            parser.error("既存実験の定義が今回の指定と一致しません。")
    else:
        _write(definition_path, definition)

    summaries = {}
    summary_path = output / "summary.json"
    if summary_path.is_file():
        summaries.update(json.loads(summary_path.read_text(encoding="utf-8")))

    for seed, source_directory in sorted(sources.items()):
        staging_id = f"source_10000_s{seed}"
        staging = models_root / staging_id
        staging.mkdir(exist_ok=True)
        staged_checkpoint = staging / "checkpoint_10000_steps.zip"
        if not staged_checkpoint.is_file():
            shutil.copy2(source_directory / "checkpoint_10000_steps.zip", staged_checkpoint)

        model_25 = _train_stage(
            model_id=f"hierarchical_25000_s{seed}", models_root=models_root,
            source_id=staging_id, source_steps=10_000, target_steps=25_000, seed=seed,
        )
        model_50 = _train_stage(
            model_id=f"hierarchical_50000_s{seed}", models_root=models_root,
            source_id=model_25.name, source_steps=25_000, target_steps=50_000, seed=seed,
        )
        directories = {10_000: source_directory, 25_000: model_25, 50_000: model_50}
        for steps, directory in directories.items():
            key = f"s{seed}_{steps}"
            report_path = output / f"{key}.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                report = _evaluate(directory, steps, evaluation_seeds)
                _write(report_path, report)
            summaries[key] = report["summary"]
            _write(summary_path, summaries)
            summary = report["summary"]
            print(json.dumps({
                "cell": key,
                "win_rate": summary["win_rate"],
                "average_score": summary["average_score"],
                "paid_roads": summary["average_actions"].get("build_road_paid", 0.0),
                "settlements": summary["average_actions"].get("build_settlement", 0.0),
            }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
