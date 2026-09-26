"""初期配置候補を盤面上の説明可能な特徴へ圧縮し、方式を選択する。"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.agents.ppo import PPOAgent

from .compare import PolicyCompatibleHeuristicAgent
from .evaluate_construction_priority import VALIDATED_PLANNING
from .settlement_planning_agent import SettlementPlanningAgent
from .setup_diagnostics import GreedyPipsSetupAgent, probe_setup
from .train_development_budget import write


PORTS = ("generic", "wood", "brick", "sheep", "wheat", "ore")
RESOURCES = ("wood", "brick", "sheep", "wheat", "ore")


class Selector(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        if hidden_size:
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size), nn.ReLU(),
                nn.Dropout(0.12), nn.Linear(hidden_size, 3),
            )
        else:
            self.network = nn.Linear(input_size, 3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _strategy_features(row: dict) -> list[float]:
    values = [
        row["first_pips"], row["second_pips"], row["total_pips"],
        row["resource_kinds"], row["second_new_resource_kinds"],
        row["missing_road_resources"],
    ]
    values.extend(row["resource_pips"][resource] for resource in RESOURCES)
    values.extend(row["road_extension_pips_after_setup"])
    values.extend(
        choice["best_legal_pips"] - choice["selected_pips"]
        for choice in row["settlement_choices"]
    )
    values.extend(
        choice["best_legal_extension_pips"] - choice["selected_extension_pips"]
        for choice in row["road_choices"]
    )
    values.extend(float(port in row["ports"]) for port in PORTS)
    return [float(value) for value in values]


def _selection_metrics(logits: np.ndarray, wins: np.ndarray, scores: np.ndarray,
                       names: list[str], threshold: float) -> dict:
    current = logits[:, 0]
    alternatives = logits[:, 1:]
    alternative = alternatives.argmax(axis=1) + 1
    best_alternative = alternatives.max(axis=1)
    indices = np.where(best_alternative > current + threshold, alternative, 0)
    rows = np.arange(len(indices))
    utilities = 2.0 * wins + scores / 10.0
    return {
        "games": len(indices),
        "win_rate": float(wins[rows, indices].mean()),
        "average_score": float(scores[rows, indices].mean()),
        "average_utility": float(utilities[rows, indices].mean()),
        "current_win_rate": float(wins[:, 0].mean()),
        "oracle_win_rate": float(wins.max(axis=1).mean()),
        "switch_rate": float(np.mean(indices != 0)),
        "choices": dict(Counter(names[index] for index in indices)),
    }


def _fit(train_x: torch.Tensor, train_y: torch.Tensor,
         validation_x: torch.Tensor, validation_y: torch.Tensor,
         hidden_size: int) -> tuple[Selector, list[dict]]:
    torch.manual_seed(20260925 + hidden_size)
    model = Selector(train_x.shape[1], hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    generator = torch.Generator().manual_seed(20260925)
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, 301):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(permutation), 32):
            indices = permutation[start:start + 32]
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(train_x[indices]), train_y[indices],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(nn.functional.binary_cross_entropy_with_logits(
                model(validation_x), validation_y,
            ))
        history.append({"epoch": epoch, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 25:
                break
    if best_state is None:
        raise AssertionError("特徴量セレクタを学習できませんでした。")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    if any(Path(name).name != name for name in (args.source_experiment, args.experiment_id)):
        parser.error("実験名はフォルダ名だけで指定してください。")
    experiments = Path(__file__).resolve().parents[2] / "experiments"
    source = experiments / args.source_experiment
    output = experiments / args.experiment_id
    output.mkdir(exist_ok=False)
    with (source / "training_summary.json").open(encoding="utf-8") as handle:
        source_summary = json.load(handle)
    arrays = np.load(source / "dataset.npz")
    wins = arrays["wins"].astype(np.float32)
    scores = arrays["scores"].astype(np.float32)
    names = source_summary["strategies"]
    if names != ["current_gnn_filtered", "heuristic_setup", "greedy_pips_setup"]:
        raise AssertionError("想定外の初期配置方式です。")
    definition = source_summary["definition"]
    total = definition["train_games"] + definition["validation_games"] + definition["test_games"]
    if wins.shape != (total, len(names)) or scores.shape != wins.shape:
        raise AssertionError("対局結果と実験定義の件数が一致しません。")

    base = PPOAgent("ppo_gnn_board_65k_s03_exp_v003", device="cpu")
    strategies = {
        "current_gnn_filtered": SettlementPlanningAgent(base, **VALIDATED_PLANNING),
        "heuristic_setup": PolicyCompatibleHeuristicAgent(),
        "greedy_pips_setup": GreedyPipsSetupAgent(),
    }
    rows_by_strategy: dict[str, list[dict]] = {}
    for name, agent in strategies.items():
        rows = [
            probe_setup(agent, definition["seed_start"] + index, index,
                        observation_version="v2")
            for index in range(total)
        ]
        rows_by_strategy[name] = rows
        write(output / f"setup_features_{name}.json", {"episodes": rows})
        print(json.dumps({"stage": name, "setups": len(rows)}, ensure_ascii=False), flush=True)

    feature_rows = []
    for index in range(total):
        seat = rows_by_strategy[names[0]][index]["seat"]
        values = [float(seat == candidate) for candidate in range(1, 5)]
        for name in names:
            row = rows_by_strategy[name][index]
            if row["seed"] != definition["seed_start"] + index or row["seat"] != seat:
                raise AssertionError("初期配置特徴のseedまたは席順が一致しません。")
            values.extend(_strategy_features(row))
        feature_rows.append(values)
    features = np.asarray(feature_rows, dtype=np.float32)
    train_end = definition["train_games"]
    validation_end = train_end + definition["validation_games"]
    splits = {
        "train": np.arange(train_end),
        "validation": np.arange(train_end, validation_end),
        "test": np.arange(validation_end, total),
    }
    mean = features[splits["train"]].mean(axis=0)
    scale = features[splits["train"]].std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (features - mean) / scale
    train_x = torch.from_numpy(normalized[splits["train"]])
    train_y = torch.from_numpy(wins[splits["train"]])
    validation_x = torch.from_numpy(normalized[splits["validation"]])
    validation_y = torch.from_numpy(wins[splits["validation"]])

    candidates = []
    for hidden_size in (0, 24):
        model, history = _fit(train_x, train_y, validation_x, validation_y, hidden_size)
        with torch.no_grad():
            logits = model(torch.from_numpy(normalized)).numpy()
        for threshold in (0.0, 0.15, 0.3, 0.5, 0.75, 1.0):
            validation = _selection_metrics(
                logits[splits["validation"]], wins[splits["validation"]],
                scores[splits["validation"]], names, threshold,
            )
            candidates.append({
                "hidden_size": hidden_size, "threshold": threshold,
                "validation": validation, "model": model,
                "history": history, "logits": logits,
            })
    selected = max(candidates, key=lambda row: (
        row["validation"]["win_rate"], row["validation"]["average_utility"],
        -row["validation"]["switch_rate"], -row["hidden_size"], row["threshold"],
    ))
    metrics = {
        split: _selection_metrics(selected["logits"][indices], wins[indices],
                                  scores[indices], names, selected["threshold"])
        for split, indices in splits.items()
    }
    torch.save({
        "state_dict": selected["model"].state_dict(),
        "input_size": features.shape[1], "hidden_size": selected["hidden_size"],
        "threshold": selected["threshold"], "mean": mean, "scale": scale,
        "strategies": names,
    }, output / "setup_feature_selector.pt")
    np.savez_compressed(output / "features.npz", features=features, wins=wins, scores=scores)
    report = {
        "source_experiment": args.source_experiment,
        "information_boundary": (
            "初期盤面と既知の固定Opponent方式から対局開始前に再生できる初期配置、"
            "席順、pips、資源種類、港、道路先。通常ターンのdice・手札・勝敗は入力しない。"
        ),
        "feature_count": int(features.shape[1]),
        "selected": {"hidden_size": selected["hidden_size"],
                     "threshold": selected["threshold"]},
        "metrics": metrics,
        "candidate_validation": [
            {"hidden_size": row["hidden_size"], "threshold": row["threshold"],
             "validation": row["validation"]}
            for row in candidates
        ],
        "history": selected["history"],
    }
    write(output / "training_summary.json", report)
    print(json.dumps({"stage": "trained", **report["selected"],
                      "metrics": metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
