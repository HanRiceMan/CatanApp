"""初期配置ランキング学習のtemperatureまたは教師盤面数を単独比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.agents.ppo import PPOAgent

from .audit import runtime_versions, source_fingerprint
from .initial_setup_policy import train_ranked_initial_setup_policy


def _numbers(value: str, cast):
    try:
        result = [cast(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not result or any(number <= 0 for number in result):
        raise argparse.ArgumentTypeError("正の値をカンマ区切りで指定してください。")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_hierarchical_50k_v001")
    parser.add_argument("--models-root", type=Path, default=None)
    parser.add_argument("--temperatures", default="1,2,5,10")
    parser.add_argument("--train-sizes", default="600")
    parser.add_argument("--train-seed-start", type=int, default=100_001)
    parser.add_argument("--validation-seed-start", type=int, default=106_001)
    parser.add_argument("--test-seed-start", type=int, default=106_501)
    parser.add_argument("--validation-boards", type=int, default=300)
    parser.add_argument("--test-boards", type=int, default=500)
    parser.add_argument("--training-seed", type=int, default=20260928)
    args = parser.parse_args()
    temperatures = _numbers(args.temperatures, float)
    train_sizes = _numbers(args.train_sizes, int)
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    if args.train_seed_start + max(train_sizes) > args.validation_seed_start:
        parser.error("学習seedと検証seedが重なっています。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。別IDを指定してください。")
    output.mkdir(parents=True)
    definition = {
        "experiment_id": args.experiment_id,
        "source_model": args.model_id,
        "temperatures": temperatures,
        "train_sizes": train_sizes,
        "train_seed_start": args.train_seed_start,
        "validation_seed_range": [args.validation_seed_start,
                                  args.validation_seed_start + args.validation_boards - 1],
        "test_seed_range": [args.test_seed_start, args.test_seed_start + args.test_boards - 1],
        "training_seed": args.training_seed,
        "source_sha256": source_fingerprint(),
        "runtime_versions": runtime_versions(),
    }
    (output / "definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = []
    for train_size in train_sizes:
        for temperature in temperatures:
            agent = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
            policy, report = train_ranked_initial_setup_policy(
                agent.model.policy, train_boards=train_size,
                validation_boards=args.validation_boards, test_boards=args.test_boards,
                seed_start=args.train_seed_start, validation_seed_start=args.validation_seed_start,
                test_seed_start=args.test_seed_start, training_seed=args.training_seed,
                temperature=temperature)
            condition = f"n{train_size}_t{temperature:g}"
            torch.save(policy.state_dict(), output / f"{condition}.pt")
            (output / f"{condition}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            row = {"condition": condition, "train_boards": train_size,
                   "temperature": temperature, "best_epoch": report["best_epoch"],
                   "test": report["after_test"]}
            summaries.append(row)
            (output / "summary.json").write_text(
                json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
