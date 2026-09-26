"""実対局のreturn/advantageで通常手番のGraph候補Headを学習する。"""

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
from .action_space import ACTION_SPACE_SIZE, id_to_action
from .candidate_policy import (
    GraphHierarchicalCandidateMaskablePolicy,
    configure_graph_candidate_pretrain,
)
from .env import CatanEnv, CatanEnvConfig
from .evaluate import EvaluationConfig, evaluate_agent
from .hierarchical_distribution import ACTION_FAMILY_IDS


TARGET_FAMILIES = frozenset(("robber_move", "road", "settlement", "city"))
_FAMILY_IDS = np.asarray(ACTION_FAMILY_IDS, dtype=np.int64)


@dataclass(frozen=True)
class OutcomeExamples:
    observations: np.ndarray
    action_masks: np.ndarray
    action_ids: np.ndarray
    families: np.ndarray
    returns: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    collection: dict[str, object]

    def __len__(self) -> int:
        return int(self.action_ids.shape[0])


def collect_outcome_examples(
    policy_agent: PPOAgent,
    seeds: Iterable[int],
    *,
    gamma: float = 0.99,
    sampling_seed: int = 42,
) -> OutcomeExamples:
    """現行方策を確率的に対戦させ、空間ActionへMonte-Carlo advantageを付ける。"""
    if not 0 < gamma <= 1:
        raise ValueError("gammaは0より大きく1以下にしてください。")
    environment = CatanEnv(CatanEnvConfig(
        learning_player_id=1,
        opponent_controller="heuristic",
        policy_compatible_opponents=True,
        observation_version="v2",
    ))
    policy_agent.model.set_random_seed(sampling_seed)
    device = next(policy_agent.model.policy.parameters()).device
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    action_ids: list[int] = []
    families: list[int] = []
    returns: list[float] = []
    values: list[float] = []
    episode_results: list[float] = []
    family_counts: Counter[str] = Counter()
    try:
        for seed in seeds:
            observation, info = environment.reset(seed=int(seed))
            terminated = truncated = False
            episode: list[dict[str, object]] = []
            while not terminated and not truncated:
                game = environment.game
                if game is None:
                    raise RuntimeError("Environmentにゲームがありません。")
                action_mask = np.asarray(info["action_mask"], dtype=np.bool_)
                if game.phase == "setup_ready":
                    action = policy_agent.select_action(game, environment.learning_player_id)
                    selected_id = int(np.flatnonzero(action_mask)[0])
                    from .action_space import action_to_id
                    selected_id = action_to_id(action)
                else:
                    selected, _ = policy_agent.model.predict(
                        observation, deterministic=False, action_masks=action_mask,
                    )
                    selected_id = int(np.asarray(selected).item())
                if not action_mask[selected_id]:
                    raise AssertionError("方策がAction Mask外を選択しました。")
                action = id_to_action(selected_id)
                family = action_family(action)
                family_index = ACTION_FAMILY_INDEX[family]
                family_mask = action_mask & (_FAMILY_IDS == family_index)
                record = game.phase != "setup_ready" and family in TARGET_FAMILIES \
                    and family_mask.sum() >= 2
                value = 0.0
                if record:
                    tensor = torch.as_tensor(
                        np.asarray(observation)[None, :], dtype=torch.float32, device=device
                    )
                    with torch.no_grad():
                        value = float(policy_agent.model.policy.predict_values(tensor).item())
                next_observation, reward, terminated, truncated, next_info = environment.step(
                    selected_id
                )
                episode.append({
                    "observation": np.asarray(observation, dtype=np.float32).copy(),
                    "mask": family_mask,
                    "action_id": selected_id,
                    "family": family_index,
                    "family_name": family,
                    "value": value,
                    "reward": float(reward),
                    "record": record,
                })
                observation, info = next_observation, next_info
            if truncated:
                raise AssertionError(f"Outcome収集が打ち切られました: seed={seed}")
            future_return = 0.0
            recorded_return = 0.0
            for step in reversed(episode):
                future_return = float(step["reward"]) + gamma * future_return
                if bool(step["record"]):
                    observations.append(step["observation"])
                    masks.append(step["mask"])
                    action_ids.append(int(step["action_id"]))
                    families.append(int(step["family"]))
                    values.append(float(step["value"]))
                    returns.append(future_return)
                    family_counts[str(step["family_name"])] += 1
                    recorded_return = future_return
            episode_results.append(recorded_return)
    finally:
        environment.close()
    if not observations:
        raise AssertionError("Outcome候補データを収集できませんでした。")
    returns_array = np.asarray(returns, dtype=np.float32)
    values_array = np.asarray(values, dtype=np.float32)
    advantages_array = returns_array - values_array
    return OutcomeExamples(
        observations=np.stack(observations),
        action_masks=np.stack(masks),
        action_ids=np.asarray(action_ids, dtype=np.int64),
        families=np.asarray(families, dtype=np.int64),
        returns=returns_array,
        values=values_array,
        advantages=advantages_array,
        collection={
            "examples": len(action_ids),
            "family_counts": dict(family_counts),
            "mean_return": float(returns_array.mean()),
            "mean_value": float(values_array.mean()),
            "mean_advantage": float(advantages_array.mean()),
            "advantage_std": float(advantages_array.std()),
            "positive_advantage_rate": float((advantages_array > 0).mean()),
            "episode_tail_return_mean": float(np.mean(episode_results)),
        },
    )


def _candidate_logits(policy, observations: torch.Tensor) -> torch.Tensor:
    return policy.mlp_extractor.forward_actor(observations)[:, :ACTION_SPACE_SIZE]


def _weights(examples: OutcomeExamples, temperature: float) -> np.ndarray:
    advantages = examples.advantages
    normalized = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-6)
    return np.exp(np.clip(normalized / temperature, -2.0, 2.0)).astype(np.float32)


def _metrics(policy, examples: OutcomeExamples, device: torch.device,
             temperature: float) -> dict[str, object]:
    policy.eval()
    with torch.no_grad():
        observations = torch.as_tensor(examples.observations, device=device)
        masks = torch.as_tensor(examples.action_masks, dtype=torch.bool, device=device)
        logits = _candidate_logits(policy, observations).masked_fill(~masks, -1e8)
        predictions = logits.argmax(dim=1).cpu().numpy()
        log_probabilities = torch.log_softmax(logits, dim=1)
        selected_log_probabilities = log_probabilities[
            torch.arange(len(examples), device=device),
            torch.as_tensor(examples.action_ids, dtype=torch.long, device=device),
        ].cpu().numpy()
    weights = _weights(examples, temperature)
    agreement = predictions == examples.action_ids
    threshold = float(np.quantile(examples.advantages, 0.75))
    high = examples.advantages >= threshold
    return {
        "examples": len(examples),
        "logged_action_agreement": float(agreement.mean()),
        "high_advantage_agreement": float(agreement[high].mean()) if high.any() else None,
        "weighted_negative_log_likelihood": float(
            -(selected_log_probabilities * weights).sum() / weights.sum()
        ),
        "advantage_top_quartile": threshold,
        "collection": examples.collection,
    }


def train_graph_outcome_policy(
    policy: GraphHierarchicalCandidateMaskablePolicy,
    train: OutcomeExamples,
    validation: OutcomeExamples,
    test: OutcomeExamples,
    *,
    learning_rate: float = 3e-5,
    batch_size: int = 256,
    max_epochs: int = 15,
    patience: int = 4,
    temperature: float = 1.0,
    anchor_weight: float = 3.0,
    seed: int = 42,
) -> dict[str, object]:
    if min(learning_rate, batch_size, max_epochs, patience, temperature) <= 0 \
            or anchor_weight < 0:
        raise ValueError("学習設定が不正です。")
    device = next(policy.parameters()).device
    anchor_policy = deepcopy(policy).to(device).eval()
    for parameter in anchor_policy.parameters():
        parameter.requires_grad = False
    parameter_report = configure_graph_candidate_pretrain(policy)
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction="none")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_weights = torch.as_tensor(_weights(train, temperature))
    observations = torch.as_tensor(train.observations)
    masks = torch.as_tensor(train.action_masks, dtype=torch.bool)
    labels = torch.as_tensor(train.action_ids, dtype=torch.long)
    initial = {
        "validation": _metrics(policy, validation, device, temperature),
        "test": _metrics(policy, test, device, temperature),
    }
    best_state = {key: value.detach().cpu().clone()
                  for key, value in policy.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        policy.train()
        permutation = torch.randperm(len(train), generator=generator)
        total_loss = 0.0
        total_examples = 0
        for start in range(0, len(train), batch_size):
            indices = permutation[start:start + batch_size]
            batch_observations = observations[indices].to(device)
            batch_masks = masks[indices].to(device)
            batch_labels = labels[indices].to(device)
            batch_weights = train_weights[indices].to(device)
            logits = _candidate_logits(policy, batch_observations).masked_fill(
                ~batch_masks, -1e8
            )
            imitation = (criterion(logits, batch_labels) * batch_weights).sum() \
                / batch_weights.sum()
            with torch.no_grad():
                anchor_logits = _candidate_logits(
                    anchor_policy, batch_observations
                ).masked_fill(~batch_masks, -1e8)
            anchor = F.kl_div(
                torch.log_softmax(logits, dim=1), torch.softmax(anchor_logits, dim=1),
                reduction="batchmean",
            )
            loss = imitation + anchor_weight * anchor
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            total_examples += len(indices)

        policy.eval()
        with torch.no_grad():
            val_observations = torch.as_tensor(validation.observations, device=device)
            val_masks = torch.as_tensor(
                validation.action_masks, dtype=torch.bool, device=device
            )
            val_labels = torch.as_tensor(
                validation.action_ids, dtype=torch.long, device=device
            )
            val_weights = torch.as_tensor(
                _weights(validation, temperature), device=device
            )
            val_logits = _candidate_logits(policy, val_observations).masked_fill(
                ~val_masks, -1e8
            )
            validation_loss = float(
                (criterion(val_logits, val_labels) * val_weights).sum()
                / val_weights.sum()
            )
        history.append({"epoch": epoch, "train_loss": total_loss / total_examples,
                        "validation_weighted_nll": validation_loss})
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
        "temperature": temperature,
        "anchor_weight": anchor_weight,
        "best_epoch": best_epoch,
        "best_validation_weighted_nll": best_loss,
        "history": history,
        "before": initial,
        "after": {
            "validation": _metrics(policy, validation, device, temperature),
            "test": _metrics(policy, test, device, temperature),
        },
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--train-games", type=int, default=200)
    parser.add_argument("--validation-games", type=int, default=50)
    parser.add_argument("--test-games", type=int, default=50)
    parser.add_argument("--evaluation-games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=320_001)
    parser.add_argument("--evaluation-seed-start", type=int, default=330_001)
    parser.add_argument("--sampling-seed", type=int, default=20261801)
    parser.add_argument("--training-seed", type=int, default=20261802)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=3.0)
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
        raise TypeError("GraphHierarchicalCandidateMaskablePolicyが必要です。")
    validation_start = args.seed_start + args.train_games
    test_start = validation_start + args.validation_games
    train = collect_outcome_examples(
        agent, range(args.seed_start, validation_start), gamma=args.gamma,
        sampling_seed=args.sampling_seed,
    )
    validation = collect_outcome_examples(
        agent, range(validation_start, test_start), gamma=args.gamma,
        sampling_seed=args.sampling_seed + 1,
    )
    test = collect_outcome_examples(
        agent, range(test_start, test_start + args.test_games), gamma=args.gamma,
        sampling_seed=args.sampling_seed + 2,
    )
    evaluation_seeds = tuple(range(
        args.evaluation_seed_start, args.evaluation_seed_start + args.evaluation_games
    ))
    config = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                              policy_compatible_opponents=True, observation_version="v2")
    before = evaluate_agent(agent, evaluation_seeds, config=config,
                            agent_name=f"{args.model_id}:before_outcome").to_dict()
    report = train_graph_outcome_policy(
        agent.model.policy, train, validation, test,
        learning_rate=args.learning_rate, batch_size=args.batch_size,
        max_epochs=args.max_epochs, temperature=args.temperature,
        anchor_weight=args.anchor_weight, seed=args.training_seed,
    )
    after = evaluate_agent(agent, evaluation_seeds, config=config,
                           agent_name=f"{args.model_id}:after_outcome").to_dict()
    report["datasets"] = {
        "train_games": args.train_games,
        "validation_games": args.validation_games,
        "test_games": args.test_games,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "test_examples": len(test),
        "seed_start": args.seed_start,
        "gamma": args.gamma,
    }
    report["gameplay_evaluation"] = {
        "seeds": evaluation_seeds,
        "before": before["summary"],
        "after": after["summary"],
    }
    torch.save(agent.model.policy.state_dict(), output / "graph_outcome_policy.pt")
    _write(output / "report.json", report)
    _write(output / "evaluation_before.json", before)
    _write(output / "evaluation_after.json", after)
    print(json.dumps({
        "artifact": str(output / "graph_outcome_policy.pt"),
        "datasets": report["datasets"],
        "train_collection": train.collection,
        "before_test": report["before"]["test"],
        "after_test": report["after"]["test"],
        "gameplay_before": before["summary"],
        "gameplay_after": after["summary"],
        "best_epoch": report["best_epoch"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
