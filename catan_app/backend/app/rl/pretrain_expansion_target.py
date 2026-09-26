"""現行PPOを教師に、拡張カリキュラムで除いた行動だけを弱めるKL事前調整。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO

from app.agents.ppo import PPOAgent

from .candidate_policy import (
    ExpansionGraphHierarchicalCandidateMaskablePolicy,
    GraphHierarchicalCandidateMaskablePolicy,
    configure_expansion_graph_pretrain,
    initialize_expansion_policy_from_graph,
)
from .action_space import action_to_id
from .env import CatanEnv, CatanEnvConfig, expansion_target_weights
from .gnn_setup_policy import GraphInitialSetupAgent


@dataclass(frozen=True)
class TargetExamples:
    """通常Maskと、そこから不要行動だけを除いた教師分布の組。"""

    observations: np.ndarray
    action_masks: np.ndarray
    target_probabilities: np.ndarray
    changed: np.ndarray

    def __len__(self) -> int:
        return int(self.observations.shape[0])


def _probabilities(policy, observations: torch.Tensor,
                   masks: torch.Tensor) -> torch.Tensor:
    """階層分布を含め、377 Actionの合法確率を返す。"""
    logits = policy.mlp_extractor.forward_actor(observations)
    distribution = policy.action_dist.proba_distribution(logits)
    distribution.apply_masking(masks)
    return distribution.distribution.probs


def collect_target_examples(source: PPOAgent, seeds: Iterable[int]) -> TargetExamples:
    """現行PPOの到達状態を集め、制限行動を除いた条件付き分布を教師にする。"""
    if source.initial_setup_policy is None:
        raise ValueError("初期配置GNNを持つ現行PPOモデルが必要です。")
    environment = CatanEnv(
        CatanEnvConfig(
            learning_player_id=1,
            opponent_controller="heuristic",
            policy_compatible_opponents=True,
            observation_version=source.manifest.observation_version,
        ),
        initial_placement_agent=GraphInitialSetupAgent(source.initial_setup_policy),
    )
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    changed: list[bool] = []
    device = next(source.model.policy.parameters()).device
    source.model.policy.eval()
    try:
        for seed in seeds:
            observation, info = environment.reset(seed=int(seed))
            terminated = truncated = False
            while not terminated and not truncated:
                game = environment.game
                if game is None:
                    raise RuntimeError("Environmentにゲームがありません。")
                action_mask = np.asarray(info["action_mask"], dtype=np.bool_)
                if game.phase == "action" and action_mask.sum() >= 2:
                    target_weights = expansion_target_weights(
                        game, environment.learning_player_id, action_mask
                    )
                    with torch.no_grad():
                        teacher = _probabilities(
                            source.model.policy,
                            torch.as_tensor(observation[None, :], device=device),
                            torch.as_tensor(action_mask[None, :], device=device),
                        )[0].cpu().numpy()
                    # 現行PPOの分布を基準に、計画外の支出だけを軟らかく減衰して
                    # 再正規化する。0にはしないため、例外を学ぶ余地を残す。
                    target = teacher * target_weights
                    target /= target.sum()
                    observations.append(np.asarray(observation, dtype=np.float32).copy())
                    masks.append(action_mask.copy())
                    targets.append(target.astype(np.float32))
                    changed.append(bool(np.any(
                        action_mask & (target_weights < 1.0)
                    )))

                action = source.select_action(game, environment.learning_player_id)
                # Action objectから固定IDへ変換は環境の既存検証経路と同じにする。
                action_id = action_to_id(action)
                if not action_mask[action_id]:
                    raise AssertionError("現行PPOが通常Mask外のActionを選びました。")
                observation, _, terminated, truncated, info = environment.step(action_id)
            if truncated:
                raise AssertionError(f"教師データ収集が打ち切られました: seed={seed}")
    finally:
        environment.close()
    if not observations or not any(changed):
        raise AssertionError("カリキュラムで弱める局面を収集できませんでした。")
    return TargetExamples(
        observations=np.stack(observations),
        action_masks=np.stack(masks),
        target_probabilities=np.stack(targets),
        changed=np.asarray(changed, dtype=np.bool_),
    )


def _metrics(policy, examples: TargetExamples, device: torch.device) -> dict[str, float | int]:
    policy.eval()
    with torch.no_grad():
        probabilities = _probabilities(
            policy,
            torch.as_tensor(examples.observations, device=device),
            torch.as_tensor(examples.action_masks, device=device),
        ).clamp_min(1e-12)
        targets = torch.as_tensor(examples.target_probabilities, device=device)
        kl = F.kl_div(probabilities.log(), targets, reduction="batchmean")
        changed = torch.as_tensor(examples.changed, dtype=torch.bool, device=device)
        target_actions = targets.argmax(dim=1)
        predicted_actions = probabilities.argmax(dim=1)
    return {
        "examples": len(examples),
        "restricted_examples": int(changed.sum().item()),
        "kl_to_target": float(kl),
        "target_mode_agreement": float((predicted_actions == target_actions).float().mean()),
        "restricted_target_mode_agreement": float(
            (predicted_actions[changed] == target_actions[changed]).float().mean()
        ),
    }


def _target_kl(probabilities: torch.Tensor, targets: torch.Tensor,
               changed: torch.Tensor, restricted_weight: float) -> torch.Tensor:
    """通常局面は保存しつつ、今回だけ除外する局面を十分に学ばせる。"""
    per_example = F.kl_div(probabilities.clamp_min(1e-12).log(), targets,
                           reduction="none").sum(dim=1)
    weights = torch.where(changed, torch.as_tensor(restricted_weight,
                                                    dtype=per_example.dtype,
                                                    device=per_example.device), 1.0)
    return (per_example * weights).sum() / weights.sum()


def train_expansion_target(
    model: MaskablePPO,
    train: TargetExamples,
    validation: TargetExamples,
    test: TargetExamples,
    *,
    learning_rate: float = 3e-5,
    batch_size: int = 256,
    max_epochs: int = 30,
    patience: int = 5,
    restricted_weight: float = 8.0,
    seed: int = 42,
) -> dict[str, object]:
    """新規の拡張残差だけを、再正規化済み教師分布へKL投影する。"""
    if not isinstance(model.policy, ExpansionGraphHierarchicalCandidateMaskablePolicy):
        raise TypeError("ExpansionGraphHierarchicalCandidateMaskablePolicyが必要です。")
    if min(learning_rate, batch_size, max_epochs, patience, restricted_weight) <= 0:
        raise ValueError("学習設定は正の値にしてください。")
    device = next(model.policy.parameters()).device
    selection = configure_expansion_graph_pretrain(model.policy)
    parameters = [parameter for parameter in model.policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    before = {"validation": _metrics(model.policy, validation, device),
              "test": _metrics(model.policy, test, device)}
    best_state = {key: value.detach().cpu().clone()
                  for key, value in model.policy.state_dict().items()}
    # 新規残差はゼロ初期化で既存PPOと完全一致する。検証KLがそれを上回れない
    # 場合は、必ず残差ゼロの安全な初期状態を採用する。
    with torch.no_grad():
        initial_validation_loss = _target_kl(
            _probabilities(
                model.policy,
                torch.as_tensor(validation.observations, device=device),
                torch.as_tensor(validation.action_masks, dtype=torch.bool, device=device),
            ),
            torch.as_tensor(validation.target_probabilities, device=device),
            torch.as_tensor(validation.changed, dtype=torch.bool, device=device),
            restricted_weight,
        )
    best_loss = float(initial_validation_loss)
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    train_observations = torch.as_tensor(train.observations)
    train_masks = torch.as_tensor(train.action_masks, dtype=torch.bool)
    train_targets = torch.as_tensor(train.target_probabilities)
    train_changed = torch.as_tensor(train.changed, dtype=torch.bool)

    for epoch in range(1, max_epochs + 1):
        model.policy.train()
        permutation = torch.randperm(len(train), generator=generator)
        total_loss = 0.0
        for start in range(0, len(train), batch_size):
            indices = permutation[start:start + batch_size]
            probabilities = _probabilities(
                model.policy, train_observations[indices].to(device),
                train_masks[indices].to(device),
            ).clamp_min(1e-12)
            targets = train_targets[indices].to(device)
            loss = _target_kl(probabilities, targets,
                              train_changed[indices].to(device), restricted_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)

        model.policy.eval()
        with torch.no_grad():
            validation_probabilities = _probabilities(
                model.policy, torch.as_tensor(validation.observations, device=device),
                torch.as_tensor(validation.action_masks, dtype=torch.bool, device=device),
            )
            validation_loss = _target_kl(
                validation_probabilities,
                torch.as_tensor(validation.target_probabilities, device=device),
                torch.as_tensor(validation.changed, dtype=torch.bool, device=device),
                restricted_weight,
            )
        history.append({"epoch": epoch, "train_kl": total_loss / len(train),
                        "validation_kl": float(validation_loss)})
        if validation_loss < best_loss - 1e-6:
            best_loss = float(validation_loss)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.policy.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.policy.load_state_dict(best_state)
    return {
        "parameter_selection": selection,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "restricted_weight": restricted_weight,
        "best_epoch": best_epoch,
        "best_validation_kl": best_loss,
        "initial_validation_weighted_kl": float(initial_validation_loss),
        "history": history,
        "before": before,
        "after": {"validation": _metrics(model.policy, validation, device),
                  "test": _metrics(model.policy, test, device)},
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--source-checkpoint", type=Path, default=None,
                        help="現行モデルではなく、保存済み拡張PPOを教師兼初期値にする")
    parser.add_argument("--train-games", type=int, default=160)
    parser.add_argument("--validation-games", type=int, default=40)
    parser.add_argument("--test-games", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=250_001)
    parser.add_argument("--training-seed", type=int, default=20261611)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--restricted-weight", type=float, default=8.0)
    args = parser.parse_args()
    if min(args.train_games, args.validation_games, args.test_games) <= 0:
        parser.error("各データセットのゲーム数は正の整数にしてください。")

    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。")
    output.mkdir(parents=True)
    source = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    source_checkpoint = args.source_checkpoint.resolve() if args.source_checkpoint else None
    if source_checkpoint is not None:
        if not source_checkpoint.is_file():
            parser.error(f"source-checkpointが見つかりません: {source_checkpoint}")
        source.model = MaskablePPO.load(str(source_checkpoint), device="cpu")
        if type(source.model.policy) is not ExpansionGraphHierarchicalCandidateMaskablePolicy:
            raise TypeError("source-checkpointは拡張GNN方策である必要があります。")
    elif type(source.model.policy) is not GraphHierarchicalCandidateMaskablePolicy:
        raise TypeError("移行元は現行GraphHierarchicalCandidate方策である必要があります。")
    boundaries = (args.seed_start, args.seed_start + args.train_games,
                  args.seed_start + args.train_games + args.validation_games)
    train = collect_target_examples(source, range(boundaries[0], boundaries[1]))
    validation = collect_target_examples(source, range(boundaries[1], boundaries[2]))
    test = collect_target_examples(source, range(boundaries[2], boundaries[2] + args.test_games))
    environment = None
    if source_checkpoint is None:
        environment = CatanEnv(CatanEnvConfig(
            observation_version=source.manifest.observation_version
        ))
        model = MaskablePPO(
            ExpansionGraphHierarchicalCandidateMaskablePolicy, environment,
            n_steps=source.model.n_steps, batch_size=source.model.batch_size,
            n_epochs=source.model.n_epochs, gamma=source.model.gamma,
            gae_lambda=source.model.gae_lambda, ent_coef=source.model.ent_coef,
            policy_kwargs={"net_arch": [], "ortho_init": False},
            seed=args.training_seed, device="cpu", verbose=0,
        )
        added = initialize_expansion_policy_from_graph(model.policy, source.model.policy)
    else:
        model = source.model
        added = []
    report = train_expansion_target(
        model, train, validation, test, learning_rate=args.learning_rate,
        batch_size=args.batch_size, max_epochs=args.max_epochs,
        patience=args.patience, restricted_weight=args.restricted_weight,
        seed=args.training_seed,
    )
    report["datasets"] = {
        "train_games": args.train_games, "validation_games": args.validation_games,
        "test_games": args.test_games, "train_examples": len(train),
        "validation_examples": len(validation), "test_examples": len(test),
        "train_restricted_examples": int(train.changed.sum()),
        "validation_restricted_examples": int(validation.changed.sum()),
        "test_restricted_examples": int(test.changed.sum()),
        "seed_start": args.seed_start,
    }
    report["source_model"] = args.model_id
    report["source_checkpoint"] = str(source_checkpoint) if source_checkpoint else None
    report["target"] = "現行PPOを基準に、計画外道路と建設が近い時の発展購入だけを軟らかく減衰して再正規化"
    report["added_parameter_keys"] = added
    torch.save(model.policy.state_dict(), output / "expansion_target_policy.pt")
    _write(output / "report.json", report)
    if environment is not None:
        environment.close()
    print(json.dumps({"artifact": str(output / "expansion_target_policy.pt"),
                      "datasets": report["datasets"], "before_test": report["before"]["test"],
                      "after_test": report["after"]["test"], "best_epoch": report["best_epoch"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
