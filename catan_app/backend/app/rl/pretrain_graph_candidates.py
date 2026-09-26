"""PPOと教師AIが同じFamilyを選んだ局面だけでGraph候補Headを事前学習する。"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from app.agents.ppo import PPOAgent

from .action_families import ACTION_FAMILIES, ACTION_FAMILY_INDEX, action_family
from .action_space import ACTION_SPACE_SIZE, action_to_id, id_to_action
from .candidate_policy import (
    GraphHierarchicalCandidateMaskablePolicy,
    configure_graph_candidate_pretrain,
)
from .compare import PolicyCompatibleHeuristicAgent
from .env import CatanEnv, CatanEnvConfig
from .evaluate import EvaluationConfig, evaluate_agent
from .hierarchical_distribution import ACTION_FAMILY_IDS


TARGET_FAMILIES = frozenset(("robber_move", "road", "settlement", "city"))
_FAMILY_IDS = np.asarray(ACTION_FAMILY_IDS, dtype=np.int64)


@dataclass(frozen=True)
class CandidateExamples:
    observations: np.ndarray
    action_masks: np.ndarray
    labels: np.ndarray
    families: np.ndarray
    collection: dict[str, object]

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def save_example_sets(
    path: Path,
    train: CandidateExamples,
    validation: CandidateExamples,
    test: CandidateExamples,
) -> None:
    values: dict[str, np.ndarray] = {}
    for name, examples in (("train", train), ("validation", validation), ("test", test)):
        values[f"{name}_observations"] = examples.observations
        values[f"{name}_action_masks"] = examples.action_masks
        values[f"{name}_labels"] = examples.labels
        values[f"{name}_families"] = examples.families
        values[f"{name}_collection"] = np.asarray(
            json.dumps(examples.collection, ensure_ascii=False)
        )
    np.savez_compressed(path, **values)


def load_example_sets(path: Path) -> tuple[CandidateExamples, CandidateExamples,
                                            CandidateExamples]:
    with np.load(path, allow_pickle=False) as values:
        result = []
        for name in ("train", "validation", "test"):
            result.append(CandidateExamples(
                observations=values[f"{name}_observations"],
                action_masks=values[f"{name}_action_masks"],
                labels=values[f"{name}_labels"],
                families=values[f"{name}_families"],
                collection=json.loads(str(values[f"{name}_collection"].item())),
            ))
    return tuple(result)  # type: ignore[return-value]


def collect_common_family_examples(
    policy_agent: PPOAgent,
    seeds: Iterable[int],
) -> CandidateExamples:
    """PPOと教師が同じ空間Familyを選んだ局面の教師候補を集める。

    初期配置は保存済みGNNで進める。通常手番は教師AIで進めるが、教師と
    PPOのFamilyが一致した場合に限り候補選択を教師ラベルとして保存する。
    """
    environment = CatanEnv(CatanEnvConfig(
        learning_player_id=1,
        opponent_controller="heuristic",
        policy_compatible_opponents=True,
        observation_version="v2",
    ))
    teacher = PolicyCompatibleHeuristicAgent()
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    families: list[int] = []
    compared = 0
    common = 0
    teacher_counts: Counter[str] = Counter()
    common_counts: Counter[str] = Counter()
    try:
        for seed in seeds:
            observation, info = environment.reset(seed=int(seed))
            terminated = truncated = False
            while not terminated and not truncated:
                game = environment.game
                if game is None:
                    raise RuntimeError("Environmentにゲームがありません。")
                action_mask = np.asarray(info["action_mask"], dtype=np.bool_)
                if game.phase == "setup_ready":
                    action = policy_agent.select_action(game, environment.learning_player_id)
                else:
                    action = teacher.select_action(game, environment.learning_player_id)
                    family = action_family(action)
                    if family in TARGET_FAMILIES:
                        compared += 1
                        teacher_counts[family] += 1
                        predicted, _ = policy_agent.model.predict(
                            observation, deterministic=True, action_masks=action_mask,
                        )
                        predicted_id = int(np.asarray(predicted).item())
                        predicted_family = action_family(id_to_action(predicted_id))
                        label = action_to_id(action)
                        family_index = ACTION_FAMILY_INDEX[family]
                        family_mask = action_mask & (_FAMILY_IDS == family_index)
                        if predicted_family == family and family_mask.sum() >= 2:
                            common += 1
                            common_counts[family] += 1
                            observations.append(np.asarray(observation, dtype=np.float32).copy())
                            masks.append(family_mask)
                            labels.append(label)
                            families.append(family_index)
                action_id = action_to_id(action)
                if not action_mask[action_id]:
                    raise AssertionError("進行ActionがAction Mask外です。")
                observation, _, terminated, truncated, info = environment.step(action_id)
            if truncated:
                raise AssertionError(f"教師データ収集が打ち切られました: seed={seed}")
    finally:
        environment.close()
    if not observations:
        raise AssertionError("共通Familyの候補教師データを収集できませんでした。")
    return CandidateExamples(
        observations=np.stack(observations),
        action_masks=np.stack(masks),
        labels=np.asarray(labels, dtype=np.int64),
        families=np.asarray(families, dtype=np.int64),
        collection={
            "compared_spatial_decisions": compared,
            "common_family_decisions": common,
            "common_family_rate": common / compared if compared else 0.0,
            "teacher_family_counts": dict(teacher_counts),
            "common_family_counts": dict(common_counts),
        },
    )


def _candidate_logits(policy, observations: torch.Tensor) -> torch.Tensor:
    return policy.mlp_extractor.forward_actor(observations)[:, :ACTION_SPACE_SIZE]


def _metrics(policy, examples: CandidateExamples, device: torch.device) -> dict[str, object]:
    policy.eval()
    with torch.no_grad():
        observations = torch.as_tensor(examples.observations, device=device)
        masks = torch.as_tensor(examples.action_masks, dtype=torch.bool, device=device)
        logits = _candidate_logits(policy, observations).masked_fill(~masks, -1e8)
        predictions = logits.argmax(dim=1).cpu().numpy()
    correct = predictions == examples.labels
    family_counts = Counter(int(value) for value in examples.families)
    family_correct = Counter(int(family) for family, matched in zip(examples.families, correct)
                             if matched)
    return {
        "examples": len(examples),
        "teacher_candidate_agreement": float(correct.mean()),
        "agreement_by_family": {
            ACTION_FAMILIES[index]: (
                family_correct.get(index, 0) / family_counts[index]
                if family_counts.get(index, 0) else None
            )
            for index in range(len(ACTION_FAMILIES))
            if ACTION_FAMILIES[index] in TARGET_FAMILIES
        },
        "collection": examples.collection,
    }


def train_graph_candidate_policy(
    policy: GraphHierarchicalCandidateMaskablePolicy,
    train: CandidateExamples,
    validation: CandidateExamples,
    test: CandidateExamples,
    *,
    learning_rate: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 20,
    patience: int = 4,
    anchor_weight: float = 0.0,
    seed: int = 42,
) -> dict[str, object]:
    """Family Headを固定したままGraph候補Headを教師候補へ近づける。"""
    if not isinstance(policy, GraphHierarchicalCandidateMaskablePolicy):
        raise TypeError("GraphHierarchicalCandidateMaskablePolicyが必要です。")
    if (learning_rate <= 0 or batch_size <= 0 or max_epochs <= 0 or patience <= 0
            or anchor_weight < 0):
        raise ValueError("学習設定は正の値にしてください。")
    device = next(policy.parameters()).device
    anchor_policy = deepcopy(policy).to(device).eval()
    for parameter in anchor_policy.parameters():
        parameter.requires_grad = False
    parameter_report = configure_graph_candidate_pretrain(policy)
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    counts = Counter(int(value) for value in train.families)
    example_weights = np.asarray([
        1.0 / np.sqrt(counts[int(family)]) for family in train.families
    ], dtype=np.float32)
    example_weights /= example_weights.mean()
    criterion = nn.CrossEntropyLoss(reduction="none")
    initial = {
        "validation": _metrics(policy, validation, device),
        "test": _metrics(policy, test, device),
    }
    best_state = {key: value.detach().cpu().clone()
                  for key, value in policy.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    train_observations = torch.as_tensor(train.observations)
    train_masks = torch.as_tensor(train.action_masks, dtype=torch.bool)
    train_labels = torch.as_tensor(train.labels, dtype=torch.long)
    train_weights = torch.as_tensor(example_weights)

    for epoch in range(1, max_epochs + 1):
        policy.train()
        permutation = torch.randperm(len(train), generator=generator)
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(train), batch_size):
            indices = permutation[start:start + batch_size]
            observations = train_observations[indices].to(device)
            masks = train_masks[indices].to(device)
            labels = train_labels[indices].to(device)
            weights = train_weights[indices].to(device)
            logits = _candidate_logits(policy, observations).masked_fill(~masks, -1e8)
            losses = criterion(logits, labels)
            teacher_loss = (losses * weights).sum() / weights.sum()
            with torch.no_grad():
                anchor_logits = _candidate_logits(
                    anchor_policy, observations
                ).masked_fill(~masks, -1e8)
                anchor_probabilities = torch.softmax(anchor_logits, dim=1)
            anchor_loss = F.kl_div(
                torch.log_softmax(logits, dim=1), anchor_probabilities,
                reduction="batchmean",
            )
            loss = teacher_loss + anchor_weight * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total_loss += float((losses.detach() * weights).sum())
            total_weight += float(weights.sum())

        policy.eval()
        with torch.no_grad():
            observations = torch.as_tensor(validation.observations, device=device)
            masks = torch.as_tensor(validation.action_masks, dtype=torch.bool, device=device)
            labels = torch.as_tensor(validation.labels, dtype=torch.long, device=device)
            validation_loss = float(criterion(
                _candidate_logits(policy, observations).masked_fill(~masks, -1e8), labels
            ).mean())
        history.append({"epoch": epoch, "train_loss": total_loss / total_weight,
                        "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in policy.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    policy.load_state_dict(best_state)
    return {
        "parameter_selection": parameter_report,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "anchor_weight": anchor_weight,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "before": initial,
        "after": {
            "validation": _metrics(policy, validation, device),
            "test": _metrics(policy, test, device),
        },
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--dataset-artifact", type=Path, default=None,
                        help="既に収集済みのcandidate_examples.npzを再利用する。")
    parser.add_argument("--train-games", type=int, default=200)
    parser.add_argument("--validation-games", type=int, default=50)
    parser.add_argument("--test-games", type=int, default=50)
    parser.add_argument("--evaluation-games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=230_001)
    parser.add_argument("--evaluation-seed-start", type=int, default=240_001)
    parser.add_argument("--training-seed", type=int, default=20261701)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--anchor-weight", type=float, default=0.0,
                        help="元PPOの合法候補分布を維持するKL損失の係数。")
    args = parser.parse_args()
    if min(args.train_games, args.validation_games, args.test_games,
           args.evaluation_games) <= 0:
        parser.error("ゲーム数は正の整数にしてください。")

    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。")
    output.mkdir(parents=True)
    agent = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    if type(agent.model.policy) is not GraphHierarchicalCandidateMaskablePolicy:
        raise TypeError("移行元はGraphHierarchicalCandidateMaskablePolicyである必要があります。")

    train_start = args.seed_start
    validation_start = train_start + args.train_games
    test_start = validation_start + args.validation_games
    if args.dataset_artifact is None:
        train = collect_common_family_examples(agent, range(train_start, validation_start))
        validation = collect_common_family_examples(agent, range(validation_start, test_start))
        test = collect_common_family_examples(
            agent, range(test_start, test_start + args.test_games)
        )
        save_example_sets(output / "candidate_examples.npz", train, validation, test)
    else:
        dataset_path = args.dataset_artifact.resolve()
        if not dataset_path.is_file():
            parser.error(f"教師データがありません: {dataset_path}")
        train, validation, test = load_example_sets(dataset_path)
    evaluation_seeds = tuple(range(
        args.evaluation_seed_start, args.evaluation_seed_start + args.evaluation_games
    ))
    evaluation_config = EvaluationConfig(
        heuristic_opponents=3,
        rotate_player_ids=True,
        policy_compatible_opponents=True,
        observation_version="v2",
    )
    before = evaluate_agent(
        agent, evaluation_seeds, config=evaluation_config,
        agent_name=f"{args.model_id}:before_candidate_pretrain",
    ).to_dict()
    report = train_graph_candidate_policy(
        agent.model.policy, train, validation, test,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        anchor_weight=args.anchor_weight,
        seed=args.training_seed,
    )
    after = evaluate_agent(
        agent, evaluation_seeds, config=evaluation_config,
        agent_name=f"{args.model_id}:after_candidate_pretrain",
    ).to_dict()
    report["datasets"] = {
        "train_games": args.train_games,
        "validation_games": args.validation_games,
        "test_games": args.test_games,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "test_examples": len(test),
        "seed_start": args.seed_start,
        "dataset_artifact": (
            str(args.dataset_artifact.resolve())
            if args.dataset_artifact is not None
            else str(output / "candidate_examples.npz")
        ),
    }
    report["gameplay_evaluation"] = {
        "seeds": evaluation_seeds,
        "before": before["summary"],
        "after": after["summary"],
    }
    torch.save(agent.model.policy.state_dict(), output / "graph_candidate_policy.pt")
    _write(output / "report.json", report)
    _write(output / "evaluation_before.json", before)
    _write(output / "evaluation_after.json", after)
    print(json.dumps({
        "artifact": str(output / "graph_candidate_policy.pt"),
        "datasets": report["datasets"],
        "before_test": report["before"]["test"],
        "after_test": report["after"]["test"],
        "gameplay_before": before["summary"],
        "gameplay_after": after["summary"],
        "best_epoch": report["best_epoch"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
