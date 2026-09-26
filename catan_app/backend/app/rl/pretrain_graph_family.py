"""ルールAIの通常行動を教師に、Graph Family残差Headを事前学習する。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from torch import nn

from app.agents.ppo import PPOAgent

from .action_families import ACTION_FAMILIES, ACTION_FAMILY_INDEX, action_family
from .action_space import ACTION_SPACE_SIZE, action_to_id
from .candidate_policy import (
    GraphFamilyHierarchicalCandidateMaskablePolicy,
    HierarchicalCandidateMaskablePolicy,
    configure_graph_family_pretrain,
    initialize_graph_policy_from_hierarchical,
)
from .compare import PolicyCompatibleHeuristicAgent
from .env import CatanEnv, CatanEnvConfig
from .gnn_setup_policy import GraphInitialSetupAgent, GraphInitialSetupPolicy
from .hierarchical_distribution import ACTION_FAMILY_IDS, FAMILY_COUNT


@dataclass(frozen=True)
class FamilyExamples:
    observations: np.ndarray
    family_masks: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def _family_mask(action_mask: np.ndarray) -> np.ndarray:
    result = np.zeros(FAMILY_COUNT, dtype=np.bool_)
    family_ids = np.asarray(ACTION_FAMILY_IDS, dtype=np.int64)
    for family in range(FAMILY_COUNT):
        result[family] = bool(action_mask[family_ids == family].any())
    return result


def collect_family_examples(
    setup_agent: GraphInitialSetupAgent,
    seeds: Iterable[int],
) -> FamilyExamples:
    """学習者席もルールAIで進め、複数Familyが合法な公開Observationだけを集める。"""
    environment = CatanEnv(
        CatanEnvConfig(
            learning_player_id=1,
            opponent_controller="heuristic",
            policy_compatible_opponents=True,
            observation_version="v2",
        ),
        initial_placement_agent=setup_agent,
    )
    teacher = PolicyCompatibleHeuristicAgent()
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    try:
        for seed in seeds:
            observation, info = environment.reset(seed=int(seed))
            terminated = truncated = False
            while not terminated and not truncated:
                game = environment.game
                if game is None:
                    raise RuntimeError("Environmentにゲームがありません。")
                action = teacher.select_action(game, environment.learning_player_id)
                action_id = action_to_id(action)
                action_mask = np.asarray(info["action_mask"], dtype=np.bool_)
                if not action_mask[action_id]:
                    raise AssertionError("教師AIがAction Mask外のActionを選びました。")
                family_mask = _family_mask(action_mask)
                label = ACTION_FAMILY_INDEX[action_family(action)]
                if family_mask.sum() >= 2:
                    observations.append(np.asarray(observation, dtype=np.float32).copy())
                    masks.append(family_mask)
                    labels.append(label)
                observation, _, terminated, truncated, info = environment.step(action_id)
            if truncated:
                raise AssertionError(f"教師データ収集が打ち切られました: seed={seed}")
    finally:
        environment.close()
    if not observations:
        raise AssertionError("複数Familyを比較できる教師データを収集できませんでした。")
    return FamilyExamples(
        observations=np.stack(observations),
        family_masks=np.stack(masks),
        labels=np.asarray(labels, dtype=np.int64),
    )


def _family_logits(policy, observations: torch.Tensor) -> torch.Tensor:
    return policy.mlp_extractor.forward_actor(observations)[:, ACTION_SPACE_SIZE:]


def _metrics(policy, examples: FamilyExamples, device: torch.device) -> dict[str, object]:
    policy.eval()
    with torch.no_grad():
        observations = torch.as_tensor(examples.observations, device=device)
        masks = torch.as_tensor(examples.family_masks, dtype=torch.bool, device=device)
        logits = _family_logits(policy, observations).masked_fill(~masks, -1e8)
        predictions = logits.argmax(dim=1).cpu().numpy()
    labels = examples.labels
    counts = Counter(int(value) for value in labels)
    correct = Counter(int(label) for label, prediction in zip(labels, predictions)
                      if label == prediction)
    return {
        "examples": len(examples),
        "teacher_agreement": float((predictions == labels).mean()),
        "label_counts": {ACTION_FAMILIES[index]: counts.get(index, 0)
                         for index in range(FAMILY_COUNT)},
        "agreement_by_family": {
            ACTION_FAMILIES[index]: (
                correct.get(index, 0) / counts[index] if counts.get(index, 0) else None
            )
            for index in range(FAMILY_COUNT)
        },
    }


def train_graph_family_policy(
    model: MaskablePPO,
    train: FamilyExamples,
    validation: FamilyExamples,
    test: FamilyExamples,
    *,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    max_epochs: int = 30,
    patience: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Graph Family残差だけを学習し、validation lossで早期停止する。"""
    if not isinstance(model.policy, GraphFamilyHierarchicalCandidateMaskablePolicy):
        raise TypeError("GraphFamilyHierarchicalCandidateMaskablePolicyが必要です。")
    if learning_rate <= 0 or batch_size <= 0 or max_epochs <= 0 or patience <= 0:
        raise ValueError("学習設定は正の値にしてください。")
    device = next(model.policy.parameters()).device
    parameter_report = configure_graph_family_pretrain(model.policy)
    parameters = [parameter for parameter in model.policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    counts = np.bincount(train.labels, minlength=FAMILY_COUNT).astype(np.float32)
    weights = np.zeros(FAMILY_COUNT, dtype=np.float32)
    present = counts > 0
    weights[present] = 1.0 / np.sqrt(counts[present])
    weights[present] /= weights[present].mean()
    class_weights = torch.as_tensor(weights, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    initial = {
        "validation": _metrics(model.policy, validation, device),
        "test": _metrics(model.policy, test, device),
    }
    best_state = {key: value.detach().cpu().clone()
                  for key, value in model.policy.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    train_observations = torch.as_tensor(train.observations)
    train_masks = torch.as_tensor(train.family_masks, dtype=torch.bool)
    train_labels = torch.as_tensor(train.labels, dtype=torch.long)

    for epoch in range(1, max_epochs + 1):
        model.policy.train()
        permutation = torch.randperm(len(train), generator=generator)
        total_loss = 0.0
        for start in range(0, len(train), batch_size):
            indices = permutation[start:start + batch_size]
            observations = train_observations[indices].to(device)
            masks = train_masks[indices].to(device)
            labels = train_labels[indices].to(device)
            logits = _family_logits(model.policy, observations).masked_fill(~masks, -1e8)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)

        model.policy.eval()
        with torch.no_grad():
            observations = torch.as_tensor(validation.observations, device=device)
            masks = torch.as_tensor(validation.family_masks, dtype=torch.bool, device=device)
            labels = torch.as_tensor(validation.labels, dtype=torch.long, device=device)
            validation_loss = float(criterion(
                _family_logits(model.policy, observations).masked_fill(~masks, -1e8), labels
            ))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train),
                        "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.policy.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.policy.load_state_dict(best_state)
    final = {
        "validation": _metrics(model.policy, validation, device),
        "test": _metrics(model.policy, test, device),
    }
    return {
        "parameter_selection": parameter_report,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "before": initial,
        "after": final,
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_setup_n1024_s01_v001")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--gnn-experiment", default="initial_setup_gnn_600_20260923")
    parser.add_argument("--train-games", type=int, default=200)
    parser.add_argument("--validation-games", type=int, default=50)
    parser.add_argument("--test-games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=210_001)
    parser.add_argument("--training-seed", type=int, default=20261601)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()
    if min(args.train_games, args.validation_games, args.test_games) <= 0:
        parser.error("各データセットのゲーム数は正の整数にしてください。")

    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。")
    output.mkdir(parents=True)
    source = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    if type(source.model.policy) is not HierarchicalCandidateMaskablePolicy:
        raise TypeError("移行元はHierarchicalCandidateMaskablePolicyである必要があります。")
    setup = GraphInitialSetupPolicy(source.model.policy)
    setup.load_state_dict(torch.load(
        root / "experiments" / args.gnn_experiment / "gnn_policy.pt",
        map_location="cpu", weights_only=True,
    ))
    setup.eval()
    setup_agent = GraphInitialSetupAgent(setup)

    train_start = args.seed_start
    validation_start = train_start + args.train_games
    test_start = validation_start + args.validation_games
    train = collect_family_examples(setup_agent, range(train_start, validation_start))
    validation = collect_family_examples(
        setup_agent, range(validation_start, test_start)
    )
    test = collect_family_examples(
        setup_agent, range(test_start, test_start + args.test_games)
    )
    environment = CatanEnv(CatanEnvConfig(observation_version="v2"),
                           initial_placement_agent=setup_agent)
    model = MaskablePPO(
        GraphFamilyHierarchicalCandidateMaskablePolicy,
        environment,
        n_steps=source.model.n_steps,
        batch_size=source.model.batch_size,
        n_epochs=source.model.n_epochs,
        gamma=source.model.gamma,
        gae_lambda=source.model.gae_lambda,
        ent_coef=source.model.ent_coef,
        policy_kwargs={"net_arch": [], "ortho_init": False},
        seed=args.training_seed,
        device="cpu",
        verbose=0,
    )
    initialize_graph_policy_from_hierarchical(model.policy, source.model.policy)
    report = train_graph_family_policy(
        model, train, validation, test,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        seed=args.training_seed,
    )
    report["datasets"] = {
        "train_games": args.train_games,
        "validation_games": args.validation_games,
        "test_games": args.test_games,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "test_examples": len(test),
        "seed_start": args.seed_start,
    }
    torch.save(model.policy.state_dict(), output / "graph_family_policy.pt")
    _write(output / "report.json", report)
    environment.close()
    print(json.dumps({
        "artifact": str(output / "graph_family_policy.pt"),
        "datasets": report["datasets"],
        "before_test": report["before"]["test"],
        "after_test": report["after"]["test"],
        "best_epoch": report["best_epoch"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
