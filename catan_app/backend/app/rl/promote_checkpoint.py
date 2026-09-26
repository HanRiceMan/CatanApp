"""実験checkpointを、評価結果付きでUI用Model Registryへ昇格する。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .model_registry import ModelManifest, ModelRegistry


def promote_checkpoint(*, source_model_directory: Path, checkpoint_steps: int,
                       target_model_id: str, evaluation_report: Path,
                       models_root: Path | None = None) -> Path:
    if Path(target_model_id).name != target_model_id:
        raise ValueError("target-model-idは安全なフォルダ名にしてください。")
    source_model_directory = source_model_directory.resolve()
    checkpoint = source_model_directory / f"checkpoint_{checkpoint_steps}_steps.zip"
    metadata_path = source_model_directory / "metadata.json"
    config_path = source_model_directory / "config.json"
    if not checkpoint.is_file() or not metadata_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("checkpoint、metadata、configのいずれかがありません。")
    report = json.loads(evaluation_report.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or "summary" not in report:
        raise ValueError("evaluation-reportの形式が不正です。")
    summary = report["summary"]
    if summary.get("illegal_action_count") or summary.get("truncated"):
        raise ValueError("非法Actionまたは打ち切りを含むモデルは昇格できません。")

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(checkpoint), device="cpu")
    if model.num_timesteps != checkpoint_steps:
        raise ValueError("checkpoint内部のstep数が指定値と一致しません。")

    source = ModelManifest.from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))
    root = models_root or ModelRegistry().models_root
    target = root / target_model_id
    if target.exists():
        raise FileExistsError(f"昇格先モデルは既に存在します: {target}")
    target.mkdir(parents=True)
    try:
        shutil.copy2(checkpoint, target / "model.zip")
        training_config = json.loads(config_path.read_text(encoding="utf-8"))
        training_config.update({
            "model_id": target_model_id,
            "total_timesteps": checkpoint_steps,
            "promoted_from": str(source_model_directory),
            "promoted_checkpoint_steps": checkpoint_steps,
        })
        (target / "config.json").write_text(
            json.dumps(training_config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        manifest = ModelManifest(
            model_id=target_model_id,
            algorithm=source.algorithm,
            artifact_filename="model.zip",
            observation_version=source.observation_version,
            observation_size=source.observation_size,
            action_space_version=source.action_space_version,
            action_space_size=source.action_space_size,
            training_steps=checkpoint_steps,
            training_seed=source.training_seed,
            created_at=datetime.now(timezone.utc).isoformat(),
            evaluation=report,
        )
        (target / "metadata.json").write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        ModelRegistry(root).load(target_model_id)
    except Exception:
        shutil.rmtree(target)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-directory", type=Path, required=True)
    parser.add_argument("--checkpoint-steps", type=int, required=True)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, default=None)
    args = parser.parse_args()
    if args.checkpoint_steps <= 0:
        parser.error("checkpoint-stepsは正の整数です。")
    target = promote_checkpoint(
        source_model_directory=args.source_model_directory,
        checkpoint_steps=args.checkpoint_steps,
        target_model_id=args.target_model_id,
        evaluation_report=args.evaluation_report,
        models_root=args.models_root,
    )
    print(json.dumps({"promoted_model": args.target_model_id, "directory": str(target)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
