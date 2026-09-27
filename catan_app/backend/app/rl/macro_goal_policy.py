"""PlannerのObjective Goalを模倣する、実行Actionへ未接続のMacro Head。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .action_space import ACTION_SPACE_SIZE
from .hierarchical_distribution import FAMILY_COUNT
from .macro_goal import ObjectiveGoal
from .observation import OBSERVATION_VECTOR_SIZES


MACRO_POLICY_VERSION = "macro_goal_head_v1"
TRAINABLE_MACRO_GOALS = (
    ObjectiveGoal.SETTLEMENT.value,
    ObjectiveGoal.CITY.value,
    ObjectiveGoal.DEVELOPMENT.value,
    ObjectiveGoal.ROAD_TITLE.value,
    ObjectiveGoal.KNIGHT_TITLE.value,
)
GOAL_TO_INDEX = {goal: index for index, goal in enumerate(TRAINABLE_MACRO_GOALS)}
MACRO_GRAPH_SUMMARY_SIZE = 448
MACRO_CRITIC_FEATURE_SIZE = 128
MACRO_RAW_OBSERVATION_SIZE = OBSERVATION_VECTOR_SIZES["v2"]
MACRO_ACTOR_LOGIT_SIZE = ACTION_SPACE_SIZE + FAMILY_COUNT
MACRO_FEATURE_SIZE = (
    MACRO_RAW_OBSERVATION_SIZE + MACRO_GRAPH_SUMMARY_SIZE
    + MACRO_ACTOR_LOGIT_SIZE + MACRO_CRITIC_FEATURE_SIZE
)


@dataclass(frozen=True)
class MacroTeacherExamples:
    observations: np.ndarray
    labels: np.ndarray
    records: tuple[dict[str, Any], ...]
    excluded_counts: Mapping[str, int]
    definition: Mapping[str, Any]


@dataclass(frozen=True)
class MacroGameSplit:
    train_game_ids: tuple[str, ...]
    validation_game_ids: tuple[str, ...]
    test_game_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "train": list(self.train_game_ids),
            "validation": list(self.validation_game_ids),
            "test": list(self.test_game_ids),
        }


class MacroGoalHead(nn.Module):
    """凍結済みChampion表現からObjective Goalだけを予測するHead。"""

    def __init__(
        self,
        input_size: int = MACRO_FEATURE_SIZE,
        goal_count: int = len(TRAINABLE_MACRO_GOALS),
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.goal_count = goal_count
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, goal_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise ValueError(
                f"Macro特徴量は(batch, {self.input_size})である必要があります。"
            )
        return self.network(features)


def load_macro_teacher_examples(dataset_dir: Path) -> MacroTeacherExamples:
    """Step 2A Datasetから明示Goalだけを読み、None/HOLDを除外する。"""
    definition = json.loads(
        (dataset_dir / "definition.json").read_text(encoding="utf-8")
    )
    files = definition.get("files", {})
    observation_path = dataset_dir / files.get("observations", "observations.npz")
    record_path = dataset_dir / files.get("records", "decisions.jsonl")
    with np.load(observation_path) as archive:
        all_observations = np.asarray(archive["observations"], dtype=np.float32)

    observations: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    with record_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            goal = record.get("objective_goal")
            if goal not in GOAL_TO_INDEX:
                excluded["None" if goal is None else str(goal)] += 1
                continue
            observation_index = int(record["observation_index"])
            if observation_index < 0 or observation_index >= len(all_observations):
                raise ValueError("Decisionのobservation_indexがDataset範囲外です。")
            observations.append(all_observations[observation_index])
            labels.append(GOAL_TO_INDEX[goal])
            records.append(record)

    if not observations:
        raise ValueError("学習可能なMacro Goalラベルがありません。")
    encoded = np.stack(observations).astype(np.float32, copy=False)
    expected_size = int(definition["observation_shape"][1])
    if encoded.shape[1] != expected_size:
        raise ValueError("Dataset definitionとObservation次元が一致しません。")
    return MacroTeacherExamples(
        observations=encoded,
        labels=np.asarray(labels, dtype=np.int64),
        records=tuple(records),
        excluded_counts=dict(sorted(excluded.items())),
        definition=definition,
    )


def _stratified_game_groups(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], list[str]]:
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for record in records:
        groups[(str(record["opponent_profile"]), int(record["seat"]))].add(
            str(record["game_id"])
        )
    return {key: sorted(values) for key, values in groups.items()}


def make_macro_game_split(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> MacroGameSplit:
    """profile×seatを層化し、同一gameを一つのsplitにだけ割り当てる。"""
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation/test比率は0より大きい必要があります。")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validationとtestの合計比率は1未満です。")
    train: list[str] = []
    validation: list[str] = []
    test: list[str] = []
    for stratum_index, (_, games) in enumerate(sorted(
        _stratified_game_groups(records).items()
    )):
        if len(games) < 3:
            raise ValueError("各profile×seat層には最低3 game必要です。")
        shuffled = list(games)
        random.Random(seed + stratum_index * 1_000_003).shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        test_count = max(1, round(len(shuffled) * test_fraction))
        if validation_count + test_count >= len(shuffled):
            test_count = 1
            validation_count = 1
        validation.extend(shuffled[:validation_count])
        test.extend(shuffled[validation_count:validation_count + test_count])
        train.extend(shuffled[validation_count + test_count:])

    split = MacroGameSplit(
        train_game_ids=tuple(sorted(train)),
        validation_game_ids=tuple(sorted(validation)),
        test_game_ids=tuple(sorted(test)),
    )
    sets = [set(split.train_game_ids), set(split.validation_game_ids),
            set(split.test_game_ids)]
    if any(sets[left] & sets[right] for left in range(3)
           for right in range(left + 1, 3)):
        raise AssertionError("同一gameが複数splitへ入りました。")
    return split


def split_example_indices(
    records: Sequence[Mapping[str, Any]], split: MacroGameSplit,
) -> dict[str, np.ndarray]:
    game_sets = {
        "train": set(split.train_game_ids),
        "validation": set(split.validation_game_ids),
        "test": set(split.test_game_ids),
    }
    result = {}
    assigned: set[int] = set()
    for name, game_ids in game_sets.items():
        indices = np.asarray([
            index for index, record in enumerate(records)
            if str(record["game_id"]) in game_ids
        ], dtype=np.int64)
        if not len(indices):
            raise ValueError(f"{name} splitにDecisionがありません。")
        if assigned & set(indices.tolist()):
            raise AssertionError("Decisionが複数splitへ入りました。")
        assigned.update(indices.tolist())
        result[name] = indices
    if len(assigned) != len(records):
        raise AssertionError("splitへ割り当てられていないDecisionがあります。")
    return result


def module_state_digest(module: nn.Module) -> str:
    """凍結元が学習前後で変わっていないことをbitwiseに確認する。"""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def extract_macro_features(
    actor: nn.Module,
    observations: np.ndarray,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    """Championの凍結GNN・Family・Critic表現をMacro Head入力へ変換する。"""
    required = ("_encoded_spatial_states", "family_head", "critic")
    if any(not hasattr(actor, attribute) for attribute in required):
        raise TypeError("現行Graph ChampionのActor表現が必要です。")
    if observations.ndim != 2:
        raise ValueError("Observationは2次元配列である必要があります。")
    try:
        device = next(actor.parameters()).device
    except StopIteration as error:
        raise TypeError("Actorにparameterがありません。") from error
    was_training = actor.training
    actor.eval()
    encoded: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, len(observations), batch_size):
                batch = torch.as_tensor(
                    observations[start:start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                vertex, edge, tile = actor._encoded_spatial_states(batch)
                graph_summary = torch.cat((
                    vertex.mean(dim=1),
                    vertex.amax(dim=1),
                    edge.mean(dim=1),
                    tile.mean(dim=1),
                ), dim=1)
                # Plannerは既存PPOが選んだ具体Actionを入口に判断するため、
                # 同じObservationから得る候補377＋Family15 logitsも保持する。
                actor_logits = actor.forward_actor(batch)
                critic = actor.critic(batch[:, :actor.observation_size])
                # Graph poolingだけでは、少数Goalを分ける個別の資源・称号進捗が
                # 平均化される。Observation v2と既存Actor出力を併記し情報を落とさない。
                features = torch.cat(
                    (batch, graph_summary, actor_logits, critic), dim=1
                )
                if features.shape[1] != MACRO_FEATURE_SIZE:
                    raise AssertionError(
                        f"Macro特徴量が{MACRO_FEATURE_SIZE}次元ではありません。"
                    )
                encoded.append(features.cpu().numpy().astype(np.float32, copy=False))
    finally:
        actor.train(was_training)
    return np.concatenate(encoded, axis=0)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """Goalごとの総損失寄与を揃え、rare classを多数Goalへ埋没させない。"""
    counts = np.bincount(labels, minlength=len(TRAINABLE_MACRO_GOALS)).astype(np.float64)
    if np.any(counts == 0):
        missing = [TRAINABLE_MACRO_GOALS[index]
                   for index, count in enumerate(counts) if count == 0]
        raise ValueError(f"train splitに存在しないGoalがあります: {missing}")
    weights = counts.sum() / counts
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def classification_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    goal_count = len(TRAINABLE_MACRO_GOALS)
    confusion = np.zeros((goal_count, goal_count), dtype=np.int64)
    for actual, estimate in zip(truth.tolist(), predicted.tolist(), strict=True):
        confusion[int(actual), int(estimate)] += 1
    per_goal: dict[str, Any] = {}
    recalls = []
    for index, goal in enumerate(TRAINABLE_MACRO_GOALS):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted_count if predicted_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_goal[goal] = {
            "support": support,
            "predicted": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if support:
            recalls.append(recall)
    result: dict[str, Any] = {
        "samples": int(len(truth)),
        "accuracy": float((truth == predicted).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_goal": per_goal,
        "confusion_matrix": confusion.tolist(),
        "confusion_labels": list(TRAINABLE_MACRO_GOALS),
    }
    if probabilities is not None:
        confidence = probabilities.max(axis=1)
        result["mean_confidence"] = float(confidence.mean())
        result["high_confidence_rate"] = float((confidence >= 0.80).mean())
        high = confidence >= 0.80
        result["high_confidence_accuracy"] = (
            float((truth[high] == predicted[high]).mean()) if high.any() else None
        )
    return result


def grouped_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    key: str,
) -> dict[str, Any]:
    positions: dict[str, list[int]] = defaultdict(list)
    for local_position, source_index in enumerate(indices.tolist()):
        positions[str(records[source_index][key])].append(local_position)
    return {
        value: classification_metrics(
            truth[group], predicted[group], probabilities=probabilities[group]
        )
        for value, members in sorted(positions.items())
        for group in [np.asarray(members, dtype=np.int64)]
    }


def iter_minibatches(
    indices: np.ndarray, *, batch_size: int, generator: np.random.Generator,
) -> Iterable[np.ndarray]:
    shuffled = generator.permutation(indices)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start:start + batch_size]


def predict_macro_head(
    head: MacroGoalHead,
    features: np.ndarray,
    *,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(head.parameters()).device
    was_training = head.training
    head.eval()
    probabilities: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                tensor = torch.as_tensor(
                    features[start:start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                probabilities.append(
                    head(tensor).softmax(dim=1).cpu().numpy()
                )
    finally:
        head.train(was_training)
    combined = np.concatenate(probabilities, axis=0)
    return combined.argmax(axis=1).astype(np.int64), combined


def save_macro_head(
    output_dir: Path,
    head: MacroGoalHead,
    *,
    metadata: Mapping[str, Any],
    split: MacroGameSplit,
    metrics: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(head.state_dict(), output_dir / "macro_head.pt")
    (output_dir / "metadata.json").write_text(
        json.dumps({
            "macro_policy_version": MACRO_POLICY_VERSION,
            "goal_labels": list(TRAINABLE_MACRO_GOALS),
            "feature_size": MACRO_FEATURE_SIZE,
            "artifact_filename": "macro_head.pt",
            **dict(metadata),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "split.json").write_text(
        json.dumps(split.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "history.json").write_text(
        json.dumps(list(history), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_macro_head(artifact_dir: Path, *, device: str = "cpu") -> MacroGoalHead:
    metadata = json.loads(
        (artifact_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("macro_policy_version") != MACRO_POLICY_VERSION:
        raise ValueError("未対応のMacro Policy artifactです。")
    if tuple(metadata.get("goal_labels", ())) != TRAINABLE_MACRO_GOALS:
        raise ValueError("Macro Goal label順が現在の実装と一致しません。")
    head = MacroGoalHead(input_size=int(metadata["feature_size"]))
    head.load_state_dict(torch.load(
        artifact_dir / metadata["artifact_filename"],
        map_location=device,
        weights_only=True,
    ))
    head.to(device).eval()
    return head
