"""50k階層PPOの初期配置を、Heuristic委譲・未学習・head模倣で比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .analyze_supervised_setup import paired_effect
from .audit import runtime_versions, source_fingerprint
from .evaluate import EvaluationConfig, evaluate_agent
from .pretrain_candidate_setup import pretrain_initial_setup_heads


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_hierarchical_50k_v001")
    parser.add_argument("--models-root", type=Path, default=None)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=69_001)
    parser.add_argument("--train-boards", type=int, default=600)
    parser.add_argument("--validation-boards", type=int, default=150)
    parser.add_argument("--test-boards", type=int, default=150)
    parser.add_argument("--teacher-seed-start", type=int, default=67_001)
    parser.add_argument("--training-seed", type=int, default=20260928)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    if min(args.eval_games, args.train_boards, args.validation_boards, args.test_boards) <= 0:
        parser.error("対局数と教師盤面数は正の整数です。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。別IDを指定してください。")
    output.mkdir(parents=True)
    evaluation_seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    definition = {
        "experiment_id": args.experiment_id,
        "source_model": args.model_id,
        "models_root": str(args.models_root) if args.models_root else None,
        "conditions": ["heuristic_setup", "untrained_policy_setup", "imitation_setup_heads"],
        "evaluation_seeds": evaluation_seeds,
        "evaluation_opponents": "heuristic_x3",
        "rotate_player_ids": True,
        "teacher": "PolicyCompatibleHeuristicAgent",
        "teacher_seed_ranges": {
            "train": [args.teacher_seed_start, args.teacher_seed_start + args.train_boards - 1],
            "validation": [args.teacher_seed_start + args.train_boards,
                           args.teacher_seed_start + args.train_boards + args.validation_boards - 1],
            "test": [args.teacher_seed_start + args.train_boards + args.validation_boards,
                     args.teacher_seed_start + args.train_boards + args.validation_boards
                     + args.test_boards - 1],
        },
        "training_seed": args.training_seed,
        "frozen_during_imitation": "all_except_initial_settlement_head_and_initial_road_head",
        "source_sha256": source_fingerprint(),
        "runtime_versions": runtime_versions(),
    }
    _write(output / "definition.json", definition)

    agents = {}
    heuristic_setup = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    if not heuristic_setup.heuristic_initial_placement:
        raise ValueError("比較元モデルはHeuristic初期配置モデルではありません。")
    agents["heuristic_setup"] = heuristic_setup

    untrained = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    untrained.heuristic_initial_placement = False
    agents["untrained_policy_setup"] = untrained

    imitation = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    pretraining = pretrain_initial_setup_heads(
        imitation.model.policy,
        train_boards=args.train_boards,
        validation_boards=args.validation_boards,
        test_boards=args.test_boards,
        seed_start=args.teacher_seed_start,
        training_seed=args.training_seed,
    )
    imitation.heuristic_initial_placement = False
    imitation.model.save(str(output / "imitation_setup_heads_model.zip"))
    _write(output / "pretraining.json", pretraining)
    agents["imitation_setup_heads"] = imitation

    reports = {}
    summaries = {}
    for name, agent in agents.items():
        report = evaluate_agent(
            agent,
            evaluation_seeds,
            config=EvaluationConfig(
                heuristic_opponents=3,
                rotate_player_ids=True,
                policy_compatible_opponents=True,
                observation_version="v2",
            ),
            agent_name=f"{args.model_id}:{name}",
        ).to_dict()
        summary = report["summary"]
        if summary["illegal_action_count"] or summary["truncated"]:
            raise AssertionError(f"非法Actionまたは打ち切り: {name}")
        reports[name] = report
        summaries[name] = summary
        _write(output / f"{name}.json", report)
        _write(output / "summary.json", summaries)
        print(json.dumps({
            "condition": name,
            "win_rate": summary["win_rate"],
            "average_score": summary["average_score"],
            "average_initial_pips": summary["average_initial_pips"],
        }, ensure_ascii=False), flush=True)

    effects = {}
    baseline = reports["heuristic_setup"]
    for name in ("untrained_policy_setup", "imitation_setup_heads"):
        effects[name] = {
            "win_rate": paired_effect(reports[name], baseline, "learner_won"),
            "average_score": paired_effect(reports[name], baseline, "learner_score"),
        }
    _write(output / "paired_effects.json", effects)


if __name__ == "__main__":
    main()
