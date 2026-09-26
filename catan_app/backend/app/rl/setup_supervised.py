"""初期開拓地の教師あり診断。通常ゲームのAIやモデルRegistryには接続しない。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.domain.actions import PlaceInitialSettlementAction

from .action_space import action_to_id
from .audit import runtime_versions, source_fingerprint
from .compare import PolicyCompatibleHeuristicAgent
from .env import CatanEnv, CatanEnvConfig
from .observation import OBSERVATION_VECTOR_SIZES, get_observation
from .setup_diagnostics import vertex_pips


VERTICES = 54


def candidate_features(game, player_id: int, placement: int) -> np.ndarray:
    """教師あり収集と自律配置の双方で同じ公開候補特徴を作る。"""
    observed = get_observation(game, player_id, version="v2")
    own_pips = np.zeros(5, dtype=np.float32)
    for vertex in observed.vertices:
        if vertex.owner_id == player_id:
            own_pips += np.asarray(vertex.production_pips, dtype=np.float32)
    rows = []
    for vertex in observed.vertices:
        resource_pips = np.asarray(vertex.production_pips, dtype=np.float32)
        rows.append(np.concatenate((resource_pips / 15,
                                    [resource_pips.sum() / 15],
                                    (own_pips > 0).astype(np.float32),
                                    [float(placement == 2)])))
    return np.stack(rows).astype(np.float32)


def collect_examples(seeds: range) -> dict[str, np.ndarray]:
    """seed単位で分割できるよう、各盤面から学習者の開拓地2軒を収集する。"""
    observations, candidates, masks, labels, pips, placements, board_seeds = [], [], [], [], [], [], []
    teacher = PolicyCompatibleHeuristicAgent()
    for index, seed in enumerate(seeds):
        player_id = index % 4 + 1
        env = CatanEnv(CatanEnvConfig(learning_player_id=player_id,
                                     policy_compatible_opponents=True,
                                     observation_version="v2"))
        try:
            observation, info = env.reset(seed=seed)
            placement = 0
            for _ in range(24):
                game = env.game
                if game is None:
                    raise AssertionError("盤面がありません。")
                action = teacher.select_action(game, player_id)
                action_id = action_to_id(action)
                if not info["action_mask"][action_id]:
                    raise AssertionError(f"教師の非法Action: seed={seed}")
                if isinstance(action, PlaceInitialSettlementAction):
                    placement += 1
                    first_id = action_to_id(PlaceInitialSettlementAction(0))
                    vertex_mask = info["action_mask"][first_id:first_id + VERTICES].copy()
                    if not vertex_mask[action.vertex_id]:
                        raise AssertionError("教師の初期開拓地が合法候補にありません。")
                    observations.append(observation.copy())
                    candidates.append(candidate_features(game, player_id, placement))
                    masks.append(vertex_mask)
                    labels.append(action.vertex_id)
                    pips.append([vertex_pips(game.board, vertex) for vertex in range(VERTICES)])
                    placements.append(placement)
                    board_seeds.append(seed)
                observation, _, terminated, truncated, info = env.step(action_id)
                if placement == 2 and game.pending_settlement_id is None:
                    break
                if terminated or truncated:
                    raise AssertionError("初期配置の途中で対局が終了しました。")
            if placement != 2:
                raise AssertionError(f"seed={seed}: 開拓地2軒を収集できませんでした。")
        finally:
            env.close()
    return {
        "observations": np.stack(observations).astype(np.float32),
        "candidates": np.stack(candidates).astype(np.float32),
        "masks": np.stack(masks).astype(np.bool_),
        "labels": np.asarray(labels, dtype=np.int64),
        "pips": np.asarray(pips, dtype=np.int16),
        "placements": np.asarray(placements, dtype=np.int8),
        "seeds": np.asarray(board_seeds, dtype=np.int64),
    }


class SharedCandidateScorer(nn.Module):
    """交差点IDごとの重みを持たず、すべての候補に同じネットを使う。"""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 64),
                                     nn.ReLU(), nn.Linear(64, 1))

    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        return self.network(candidates).squeeze(-1)


def make_model(architecture: str = "mlp") -> nn.Module:
    if architecture == "mlp":
        return nn.Sequential(nn.Linear(OBSERVATION_VECTOR_SIZES["v2"], 256), nn.ReLU(),
                             nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, VERTICES))
    if architecture == "shared":
        return SharedCandidateScorer()
    raise ValueError(f"未対応のarchitecture: {architecture}")


def masked_logits(model: nn.Module, examples: dict[str, np.ndarray]) -> torch.Tensor:
    if isinstance(model, SharedCandidateScorer):
        features = torch.from_numpy(examples["candidates"])
    else:
        features = torch.from_numpy(examples["observations"])
    mask = torch.from_numpy(examples["masks"])
    return model(features).masked_fill(~mask, -1e9)


def assess(model: nn.Module, examples: dict[str, np.ndarray]) -> dict:
    model.eval()
    with torch.no_grad():
        predictions = masked_logits(model, examples).argmax(dim=1).numpy()
    labels = examples["labels"]
    masks = examples["masks"]
    pips = examples["pips"]
    best = np.where(masks, pips, -1).max(axis=1)
    selected = pips[np.arange(len(predictions)), predictions]
    teacher = pips[np.arange(len(labels)), labels]
    counts = Counter(predictions.tolist())
    return {
        "examples": len(labels),
        "teacher_agreement": float(np.mean(predictions == labels)),
        "mean_selected_pips": float(np.mean(selected)),
        "mean_teacher_pips": float(np.mean(teacher)),
        "mean_best_legal_pips": float(np.mean(best)),
        "mean_pips_regret": float(np.mean(best - selected)),
        "most_common_vertex_share": counts.most_common(1)[0][1] / len(predictions),
        "distinct_selected_vertices": len(counts),
        "by_placement": {
            str(number): {
                "teacher_agreement": float(np.mean(predictions[examples["placements"] == number]
                                                   == labels[examples["placements"] == number])),
                "mean_selected_pips": float(np.mean(selected[examples["placements"] == number])),
            } for number in (1, 2)
        },
    }


def train(train_data: dict[str, np.ndarray], validation: dict[str, np.ndarray], *, seed: int,
          epochs: int, batch_size: int = 128, architecture: str = "mlp") -> tuple[nn.Module, list[dict]]:
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = make_model(architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    history = []
    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in np.array_split(rng.permutation(len(train_data["labels"])),
                                      max(1, int(np.ceil(len(train_data["labels"]) / batch_size)))):
            batch = {key: value[indices] for key, value in train_data.items()}
            loss = nn.functional.cross_entropy(masked_logits(model, batch),
                                               torch.from_numpy(batch["labels"]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(nn.functional.cross_entropy(
                masked_logits(model, validation), torch.from_numpy(validation["labels"])))
        history.append({"epoch": epoch, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 8:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--train-boards", type=int, default=600)
    parser.add_argument("--validation-boards", type=int, default=150)
    parser.add_argument("--test-boards", type=int, default=150)
    parser.add_argument("--seed-start", type=int, default=31001)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--architecture", choices=("mlp", "shared"), default="mlp")
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    if min(args.train_boards, args.validation_boards, args.test_boards, args.epochs) <= 0:
        parser.error("盤面数とepochsは正の整数で指定してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。別IDを指定してください。")
    start = args.seed_start
    ranges = {
        "train": range(start, start + args.train_boards),
        "validation": range(start + args.train_boards, start + args.train_boards + args.validation_boards),
        "test": range(start + args.train_boards + args.validation_boards,
                      start + args.train_boards + args.validation_boards + args.test_boards),
    }
    datasets = {name: collect_examples(seeds) for name, seeds in ranges.items()}
    model, history = train(datasets["train"], datasets["validation"],
                           seed=args.training_seed, epochs=args.epochs,
                           architecture=args.architecture)
    summary = {name: assess(model, data) for name, data in datasets.items()}
    output.mkdir(parents=True)
    definition = {"experiment_id": args.experiment_id, "seed_ranges": {
        name: [seeds.start, seeds.stop - 1] for name, seeds in ranges.items()},
        "training_seed": args.training_seed, "epochs_max": args.epochs,
        "source_sha256": source_fingerprint(), "runtime_versions": runtime_versions(),
        "observation_version": "v2", "teacher": "PolicyCompatibleHeuristicAgent",
        "task": "initial_settlement_imitation_only", "architecture": args.architecture}
    (output / "definition.json").write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), output / "diagnostic_model.pt")
    print(json.dumps({"summary": summary, "best_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
