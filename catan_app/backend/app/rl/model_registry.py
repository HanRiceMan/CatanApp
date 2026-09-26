"""学習済みモデルをUI・APIから安全に選択するためのメタデータRegistry。"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Final, Mapping

from .action_space import ACTION_SPACE_SIZE, ACTION_SPACE_VERSION
from .observation import OBSERVATION_VECTOR_SIZE, OBSERVATION_VECTOR_SIZES, OBSERVATION_VERSION

MODEL_MANIFEST_FILENAME: Final[str] = "metadata.json"
MODEL_REGISTRY_VERSION: Final[str] = "v1"


@dataclass(frozen=True)
class ModelManifest:
    """モデル本体と推論時の入出力仕様を結び付ける不変メタデータ。"""

    model_id: str
    algorithm: str
    artifact_filename: str
    observation_version: str
    observation_size: int
    action_space_version: str
    action_space_size: int
    training_steps: int
    training_seed: int
    created_at: str
    evaluation: Mapping[str, Any]
    registry_version: str = MODEL_REGISTRY_VERSION
    initial_setup: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelManifest":
        required = {
            "model_id", "algorithm", "artifact_filename", "observation_version", "observation_size",
            "action_space_version", "action_space_size", "training_steps", "training_seed", "created_at",
        }
        missing = required - set(values)
        if missing:
            raise ValueError(f"モデルmetadataに必須項目がありません: {sorted(missing)}")
        manifest = cls(
            model_id=values["model_id"], algorithm=values["algorithm"], artifact_filename=values["artifact_filename"],
            observation_version=values["observation_version"], observation_size=values["observation_size"],
            action_space_version=values["action_space_version"], action_space_size=values["action_space_size"],
            training_steps=values["training_steps"], training_seed=values["training_seed"],
            created_at=values["created_at"], evaluation=values.get("evaluation", {}),
            registry_version=values.get("registry_version", MODEL_REGISTRY_VERSION),
            initial_setup=values.get("initial_setup"),
        )
        if not all(isinstance(value, str) and value for value in (manifest.model_id, manifest.algorithm,
                                                                    manifest.artifact_filename, manifest.created_at)):
            raise ValueError("モデルmetadataの文字列項目が不正です。")
        if Path(manifest.model_id).name != manifest.model_id or Path(manifest.artifact_filename).name != manifest.artifact_filename:
            raise ValueError("モデルmetadataのパス指定が不正です。")
        if any(type(value) is not int or value < 0 for value in (manifest.observation_size, manifest.action_space_size,
                                                                  manifest.training_steps, manifest.training_seed)):
            raise ValueError("モデルmetadataの数値項目が不正です。")
        if manifest.initial_setup is not None:
            if not isinstance(manifest.initial_setup, Mapping):
                raise ValueError("initial_setupの形式が不正です。")
            setup_type = manifest.initial_setup.get("type")
            if setup_type not in {"gnn"}:
                raise ValueError("未対応のinitial_setup形式です。")
            filename = manifest.initial_setup.get("artifact_filename")
            if not isinstance(filename, str) or not filename or Path(filename).name != filename:
                raise ValueError("initial_setupのartifact_filenameが不正です。")
        return manifest

    @property
    def is_runtime_compatible(self) -> bool:
        return (self.registry_version == MODEL_REGISTRY_VERSION
                and self.observation_version in OBSERVATION_VECTOR_SIZES
                and self.observation_size == OBSERVATION_VECTOR_SIZES[self.observation_version]
                and self.action_space_version == ACTION_SPACE_VERSION
                and self.action_space_size == ACTION_SPACE_SIZE)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"is_runtime_compatible": self.is_runtime_compatible}


class ModelRegistry:
    """``backend/models/<model_id>/`` を読むだけのRegistry。

    モデルの推論・Torch・FastAPIには依存しないため、UI接続前から互換性を検証できる。
    """

    def __init__(self, models_root: Path | str | None = None):
        self.models_root = Path(models_root) if models_root is not None else Path(__file__).resolve().parents[2] / "models"

    def load(self, model_id: str) -> ModelManifest:
        if Path(model_id).name != model_id:
            raise ValueError("model_idが不正です。")
        manifest_path = self.models_root / model_id / MODEL_MANIFEST_FILENAME
        try:
            values = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise KeyError(f"モデルが見つかりません: {model_id}") from error
        if not isinstance(values, dict):
            raise ValueError("モデルmetadataの形式が不正です。")
        manifest = ModelManifest.from_dict(values)
        if manifest.model_id != model_id:
            raise ValueError("metadataのmodel_idとディレクトリ名が一致しません。")
        if not manifest.is_runtime_compatible:
            raise ValueError("現在のObservation / Action Spaceと互換性がないモデルです。")
        if not (manifest_path.parent / manifest.artifact_filename).is_file():
            raise ValueError("モデル本体が見つかりません。")
        if manifest.initial_setup is not None:
            setup_artifact = manifest.initial_setup["artifact_filename"]
            if not (manifest_path.parent / setup_artifact).is_file():
                raise ValueError("初期配置モデル本体が見つかりません。")
        return manifest

    def list_available(self) -> tuple[ModelManifest, ...]:
        if not self.models_root.is_dir():
            return ()
        manifests = []
        for manifest_path in sorted(self.models_root.glob(f"*/{MODEL_MANIFEST_FILENAME}")):
            try:
                manifests.append(self.load(manifest_path.parent.name))
            except (KeyError, ValueError):
                # UIには不完全・互換性なしのモデルを出さない。
                continue
        return tuple(manifests)
