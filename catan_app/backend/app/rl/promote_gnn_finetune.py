"""PPO checkpointと初期配置GNNを1つのUI用モデルとして登録する。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch
from sb3_contrib import MaskablePPO

from .gnn_setup_policy import GraphInitialSetupPolicy
from .model_registry import ModelManifest, ModelRegistry


SETUP_ARTIFACT = "gnn_setup_policy.pt"


def promote_gnn_finetune(
    *,
    source_model_id: str,
    checkpoint: Path,
    gnn_policy: Path,
    evaluation_report: Path,
    target_model_id: str,
    training_seed: int,
    models_root: Path | None = None,
) -> Path:
    if Path(target_model_id).name != target_model_id:
        raise ValueError("target-model-idは安全なフォルダ名にしてください。")
    registry = ModelRegistry(models_root)
    source = registry.load(source_model_id)
    checkpoint = checkpoint.resolve()
    gnn_policy = gnn_policy.resolve()
    if not checkpoint.is_file() or not gnn_policy.is_file():
        raise FileNotFoundError("PPO checkpointまたはGNN初期配置モデルがありません。")

    report = json.loads(evaluation_report.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or "summary" not in report:
        raise ValueError("evaluation-reportの形式が不正です。")
    summary = report["summary"]
    if summary.get("illegal_action_count") or summary.get("truncated"):
        raise ValueError("非法Actionまたは打ち切りを含むモデルは登録できません。")

    model = MaskablePPO.load(str(checkpoint), device="cpu")
    from .candidate_policy import GraphHierarchicalCandidateMaskablePolicy
    policy_architecture = (
        "gnn_hierarchical_candidate"
        if isinstance(model.policy, GraphHierarchicalCandidateMaskablePolicy)
        else None
    )
    setup = GraphInitialSetupPolicy(model.policy)
    setup.load_state_dict(torch.load(gnn_policy, map_location="cpu", weights_only=True))
    setup.eval()

    target = registry.models_root / target_model_id
    if target.exists():
        raise FileExistsError(f"登録先モデルは既に存在します: {target}")
    source_directory = registry.models_root / source_model_id
    target.mkdir(parents=True)
    try:
        shutil.copy2(checkpoint, target / "model.zip")
        shutil.copy2(gnn_policy, target / SETUP_ARTIFACT)
        try:
            config = json.loads((source_directory / "config.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            config = {}
        config.update({
            "model_id": target_model_id,
            "total_timesteps": model.num_timesteps,
            "training_seed": training_seed,
            "n_steps": model.n_steps,
            "learning_rate": float(model.lr_schedule(1.0)),
            "heuristic_initial_placement": False,
            "initial_setup_policy": {
                "type": "gnn",
                "artifact_filename": SETUP_ARTIFACT,
            },
            "promoted_from": str(checkpoint),
        })
        if policy_architecture is not None:
            config["policy_architecture"] = policy_architecture
        (target / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        manifest = ModelManifest(
            model_id=target_model_id,
            algorithm=source.algorithm,
            artifact_filename="model.zip",
            observation_version=source.observation_version,
            observation_size=source.observation_size,
            action_space_version=source.action_space_version,
            action_space_size=source.action_space_size,
            training_steps=model.num_timesteps,
            training_seed=training_seed,
            created_at=datetime.now(timezone.utc).isoformat(),
            evaluation=report,
            initial_setup={
                "type": "gnn",
                "artifact_filename": SETUP_ARTIFACT,
                "architecture": "two_layer_vertex_message_passing",
            },
        )
        (target / "metadata.json").write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        registry.load(target_model_id)
    except Exception:
        shutil.rmtree(target)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-id", default="ppo_hierarchical_50k_v001")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gnn-policy", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--models-root", type=Path, default=None)
    args = parser.parse_args()
    if args.training_seed < 0:
        parser.error("training-seedは0以上にしてください。")
    target = promote_gnn_finetune(
        source_model_id=args.source_model_id,
        checkpoint=args.checkpoint,
        gnn_policy=args.gnn_policy,
        evaluation_report=args.evaluation_report,
        target_model_id=args.target_model_id,
        training_seed=args.training_seed,
        models_root=args.models_root,
    )
    print(json.dumps({"registered_model": args.target_model_id, "directory": str(target)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
