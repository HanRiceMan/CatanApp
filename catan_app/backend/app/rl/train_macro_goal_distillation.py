"""Step 3: Championを変えず、明示Teacher GoalからMacro Headだけを学習する。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.agents.ppo import PPOAgent

from .macro_goal_policy import (
    MacroGoalHead,
    TRAINABLE_MACRO_GOALS,
    class_weights,
    classification_metrics,
    extract_macro_features,
    grouped_metrics,
    iter_minibatches,
    load_macro_teacher_examples,
    make_macro_game_split,
    module_state_digest,
    predict_macro_head,
    save_macro_head,
    split_example_indices,
)


def _evaluate(
    head: MacroGoalHead,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    predicted, probabilities = predict_macro_head(head, features[indices])
    return (
        classification_metrics(labels[indices], predicted,
                               probabilities=probabilities),
        predicted,
        probabilities,
    )


def train_macro_head(
    head: MacroGoalHead,
    features: np.ndarray,
    labels: np.ndarray,
    indices: dict[str, np.ndarray],
    *,
    seed: int,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
) -> tuple[list[dict], int]:
    torch.manual_seed(seed)
    np_generator = np.random.default_rng(seed)
    device = next(head.parameters()).device
    weights = class_weights(labels[indices["train"]]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    best_state = deepcopy(head.state_dict())
    best_epoch = 0
    best_balanced_accuracy = -1.0
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict] = []
    for epoch in range(1, max_epochs + 1):
        head.train()
        total_loss = 0.0
        total_examples = 0
        for batch_indices in iter_minibatches(
            indices["train"], batch_size=batch_size, generator=np_generator
        ):
            batch_features = torch.as_tensor(
                features[batch_indices], dtype=torch.float32, device=device
            )
            batch_labels = torch.as_tensor(
                labels[batch_indices], dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(batch_features), batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_indices)
            total_examples += len(batch_indices)

        head.eval()
        validation_logits = head(torch.as_tensor(
            features[indices["validation"]], dtype=torch.float32, device=device
        ))
        validation_labels = torch.as_tensor(
            labels[indices["validation"]], dtype=torch.long, device=device
        )
        validation_loss = float(criterion(
            validation_logits, validation_labels
        ).detach())
        validation_probabilities = validation_logits.softmax(dim=1).detach().cpu().numpy()
        validation_predicted = validation_probabilities.argmax(axis=1)
        validation_metrics = classification_metrics(
            labels[indices["validation"]], validation_predicted,
            probabilities=validation_probabilities,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
        }
        history.append(row)
        balanced = validation_metrics["balanced_accuracy"]
        improved = (
            balanced > best_balanced_accuracy + 1e-8
            or (abs(balanced - best_balanced_accuracy) <= 1e-8
                and validation_loss < best_loss)
        )
        if improved:
            best_balanced_accuracy = balanced
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(head.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    head.load_state_dict(best_state)
    return history, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--seed", type=int, default=20262901)
    parser.add_argument("--split-seed", type=int, default=20262901,
                        help="game単位splitを固定するseed")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."}):
        parser.error("experiment-idが不正です。")
    if min(args.max_epochs, args.patience, args.batch_size) <= 0:
        parser.error("epoch、patience、batch-sizeは正数です。")
    if args.learning_rate <= 0:
        parser.error("learning-rateは正数です。")

    root = Path(__file__).resolve().parents[2]
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = root / dataset_dir
    output = root / "experiments" / args.experiment_id
    examples = load_macro_teacher_examples(dataset_dir)
    split = make_macro_game_split(examples.records, seed=args.split_seed)
    indices = split_example_indices(examples.records, split)

    champion = PPOAgent(args.model_id, device="cpu")
    actor = champion.model.policy.mlp_extractor
    source_digest_before = module_state_digest(actor)
    parity_observations = examples.observations[:64]
    with torch.no_grad():
        parity_logits_before = actor.forward_actor(torch.as_tensor(
            parity_observations, dtype=torch.float32
        )).detach().clone()
    features = extract_macro_features(actor, examples.observations)

    torch.manual_seed(args.seed)
    head = MacroGoalHead().to("cpu")
    before = {}
    for name, split_indices in indices.items():
        before[name], _, _ = _evaluate(
            head, features, examples.labels, split_indices
        )
    history, best_epoch = train_macro_head(
        head, features, examples.labels, indices,
        seed=args.seed,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    after = {}
    test_predictions = None
    test_probabilities = None
    for name, split_indices in indices.items():
        after[name], predicted, probabilities = _evaluate(
            head, features, examples.labels, split_indices
        )
        if name == "test":
            test_predictions = predicted
            test_probabilities = probabilities
    assert test_predictions is not None and test_probabilities is not None

    source_digest_after = module_state_digest(actor)
    with torch.no_grad():
        parity_logits_after = actor.forward_actor(torch.as_tensor(
            parity_observations, dtype=torch.float32
        )).detach()
    source_unchanged = (
        source_digest_before == source_digest_after
        and torch.equal(parity_logits_before, parity_logits_after)
    )
    if not source_unchanged:
        raise AssertionError("Macro Head学習によりChampion Actorが変化しました。")

    test_indices = indices["test"]
    metrics = {
        "before": before,
        "after": after,
        "test_by_opponent_profile": grouped_metrics(
            examples.labels[test_indices], test_predictions, test_probabilities,
            examples.records, test_indices, "opponent_profile",
        ),
        "test_by_seat": grouped_metrics(
            examples.labels[test_indices], test_predictions, test_probabilities,
            examples.records, test_indices, "seat",
        ),
        "source_actor_unchanged": source_unchanged,
        "source_actor_digest_before": source_digest_before,
        "source_actor_digest_after": source_digest_after,
    }
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model_id": args.model_id,
        "source_dataset": dataset_dir.name,
        "dataset_schema_version": examples.definition["dataset_schema_version"],
        "observation_version": examples.definition["observation_version"],
        "observation_size": int(examples.observations.shape[1]),
        "training_seed": args.seed,
        "split_seed": args.split_seed,
        "best_epoch": best_epoch,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "class_weighting": "inverse_frequency",
        "training_goal_labels": list(TRAINABLE_MACRO_GOALS),
        "excluded_counts": dict(examples.excluded_counts),
        "split_counts": {
            name: {
                "games": len(getattr(split, f"{name}_game_ids")),
                "decisions": int(len(split_indices)),
            }
            for name, split_indices in indices.items()
        },
        "runtime_connected": False,
        "action_logits_connected": False,
    }
    save_macro_head(
        output, head, metadata=metadata, split=split, metrics=metrics,
        history=history,
    )
    print(json.dumps({
        "experiment": args.experiment_id,
        "best_epoch": best_epoch,
        "excluded_counts": examples.excluded_counts,
        "split_counts": metadata["split_counts"],
        "test": after["test"],
        "source_actor_unchanged": source_unchanged,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
