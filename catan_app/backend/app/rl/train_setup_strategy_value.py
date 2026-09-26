"""公開された初期盤面から、3種類の初期配置方式の対局価値を予測する。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.agents.base import Agent
from app.agents.ppo import PPOAgent
from app.domain.actions import Action
from app.domain.game import GameState

from .compare import PolicyCompatibleHeuristicAgent
from .evaluate import EvaluationConfig, evaluate_agent
from .evaluate_construction_priority import VALIDATED_PLANNING
from .observation import encode_observation, get_observation
from .settlement_planning_agent import SettlementPlanningAgent
from .setup_diagnostics import GreedyPipsSetupAgent
from .train_development_budget import write


@dataclass(frozen=True)
class SetupOnlyAgent:
    policy: Agent
    setup: Agent

    def select_action(self, game: GameState, player_id: int) -> Action:
        selected = self.setup if game.phase == "setup_ready" else self.policy
        return selected.select_action(game, player_id)


class RecordingAgent:
    def __init__(self, policy: Agent):
        self.policy = policy
        self.contexts: dict[tuple[int, int], np.ndarray] = {}

    def select_action(self, game: GameState, player_id: int) -> Action:
        key = (game.seed, player_id)
        if game.phase == "setup_ready" and key not in self.contexts:
            self.contexts[key] = encode_observation(get_observation(
                game, player_id, version="v2",
            ))
        return self.policy.select_action(game, player_id)


class SetupStrategyValueNetwork(nn.Module):
    def __init__(self, input_size: int, strategies: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 96), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(96, strategies),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _selection_metrics(predictions: np.ndarray, utilities: np.ndarray,
                       wins: np.ndarray, scores: np.ndarray, names: list[str]) -> dict:
    indices = predictions.argmax(axis=1)
    rows = np.arange(len(indices))
    return {
        "games": len(indices),
        "win_rate": float(wins[rows, indices].mean()),
        "average_score": float(scores[rows, indices].mean()),
        "average_utility": float(utilities[rows, indices].mean()),
        "oracle_win_rate": float(wins.max(axis=1).mean()),
        "current_win_rate": float(wins[:, 0].mean()),
        "choices": dict(Counter(names[index] for index in indices)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--train-games", type=int, default=100)
    parser.add_argument("--validation-games", type=int, default=30)
    parser.add_argument("--test-games", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=800001)
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    total = args.train_games + args.validation_games + args.test_games
    if (Path(args.experiment_id).name != args.experiment_id or min(
            args.train_games, args.validation_games, args.test_games, args.epochs) <= 0):
        parser.error("実験名・局数・epochが不正です。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)

    torch.manual_seed(20260925)
    np.random.seed(20260925)
    base = PPOAgent("ppo_gnn_board_65k_s03_exp_v003", device="cpu")
    current = SettlementPlanningAgent(base, **VALIDATED_PLANNING)
    policies = {
        "current_gnn_filtered": current,
        "heuristic_setup": SetupOnlyAgent(current, PolicyCompatibleHeuristicAgent()),
        "greedy_pips_setup": SetupOnlyAgent(current, GreedyPipsSetupAgent()),
    }
    config = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                              policy_compatible_opponents=True, observation_version="v2")
    seeds = range(args.seed_start, args.seed_start + total)
    reports, recorders = {}, {}
    for name, policy in policies.items():
        recorder = RecordingAgent(policy)
        report = evaluate_agent(recorder, seeds, config=config, agent_name=name).to_dict()
        if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
            raise AssertionError(f"{name}: 評価が正常終了していません。")
        if len(recorder.contexts) != total:
            raise AssertionError(f"{name}: 初期盤面を全局記録できませんでした。")
        reports[name], recorders[name] = report, recorder
        write(output / f"evaluation_{name}.json", report)
        print(json.dumps({"stage": name, "win_rate": report["summary"]["win_rate"],
                          "score": report["summary"]["average_score"]},
                         ensure_ascii=False), flush=True)

    names = list(policies)
    episodes = reports[names[0]]["episodes"]
    keys = [(episode["seed"], episode["learner_id"]) for episode in episodes]
    for name in names[1:]:
        candidate_keys = [
            (episode["seed"], episode["learner_id"])
            for episode in reports[name]["episodes"]
        ]
        if candidate_keys != keys:
            raise AssertionError(f"{name}: 対局結果のseed・席順が基準と一致しません。")
        for key in keys:
            if not np.array_equal(
                    recorders[names[0]].contexts[key], recorders[name].contexts[key]):
                raise AssertionError(f"{name}: 初期配置前の公開観測が基準と一致しません。")
    contexts = np.stack([recorders[names[0]].contexts[key] for key in keys]).astype(np.float32)
    wins = np.stack([
        np.asarray([episode["learner_won"] for episode in reports[name]["episodes"]],
                   dtype=np.float32)
        for name in names
    ], axis=1)
    scores = np.stack([
        np.asarray([episode["learner_score"] for episode in reports[name]["episodes"]],
                   dtype=np.float32)
        for name in names
    ], axis=1)
    # 勝利を最優先し、同じ勝敗の配置方式は最終得点で区別する。
    utilities = 2.0 * wins + scores / 10.0
    train_end = args.train_games
    validation_end = train_end + args.validation_games
    splits = {
        "train": np.arange(0, train_end),
        "validation": np.arange(train_end, validation_end),
        "test": np.arange(validation_end, total),
    }

    model = SetupStrategyValueNetwork(contexts.shape[1], len(names))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    criterion = nn.SmoothL1Loss()
    train_x = torch.from_numpy(contexts[splits["train"]])
    train_y = torch.from_numpy(utilities[splits["train"]])
    validation_x = torch.from_numpy(contexts[splits["validation"]])
    validation_y = torch.from_numpy(utilities[splits["validation"]])
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    generator = torch.Generator().manual_seed(20260925)
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(permutation), 32):
            indices = permutation[start:start + 32]
            loss = criterion(model(train_x[indices]), train_y[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_x), validation_y))
        history.append({"epoch": epoch, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 15:
                break
    if best_state is None:
        raise AssertionError("価値モデルを学習できませんでした。")
    model.load_state_dict(best_state)
    model.eval()

    metrics = {}
    with torch.no_grad():
        predictions = model(torch.from_numpy(contexts)).numpy()
    for name, indices in splits.items():
        metrics[name] = _selection_metrics(
            predictions[indices], utilities[indices], wins[indices], scores[indices], names,
        )
        metrics[name]["individual_win_rates"] = {
            strategy: float(wins[indices, position].mean())
            for position, strategy in enumerate(names)
        }
    torch.save({"state_dict": model.state_dict(), "input_size": contexts.shape[1],
                "strategies": names}, output / "setup_strategy_value.pt")
    np.savez_compressed(output / "dataset.npz", contexts=contexts, wins=wins,
                        scores=scores, utilities=utilities)
    write(output / "training_summary.json", {
        "definition": vars(args), "strategies": names, "best_validation_loss": best_loss,
        "epochs_ran": len(history), "metrics": metrics, "history": history,
        "information_boundary": "各プレイヤーの最初の初期配置手番で公開済みのv2観測だけ",
    })
    print(json.dumps({"stage": "trained", "metrics": metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
