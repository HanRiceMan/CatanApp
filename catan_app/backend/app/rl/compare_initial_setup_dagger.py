"""初期配置の教師軌跡学習とDAggerを、同一対局で比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.agents.ppo import PPOAgent

from .analyze_supervised_setup import paired_effect
from .audit import runtime_versions, source_fingerprint
from .compare_initial_setup_adapter import IndependentSetupAgent
from .evaluate import EvaluationConfig, evaluate_agent
from .initial_setup_policy import (train_initial_setup_policy,
                                   train_initial_setup_policy_dagger)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_hierarchical_50k_v001")
    parser.add_argument("--models-root", type=Path, default=None)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=85_001)
    parser.add_argument("--train-boards", type=int, default=600)
    parser.add_argument("--validation-boards", type=int, default=150)
    parser.add_argument("--test-boards", type=int, default=150)
    parser.add_argument("--teacher-seed-start", type=int, default=80_001)
    parser.add_argument("--dagger-seed-start", type=int, default=81_001)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-boards", type=int, default=300)
    parser.add_argument("--dagger-validation-boards", type=int, default=100)
    parser.add_argument("--training-seed", type=int, default=20260928)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。別IDを指定してください。")
    output.mkdir(parents=True)
    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    _write(output / "definition.json", {
        "experiment_id": args.experiment_id,
        "source_model": args.model_id,
        "conditions": ["heuristic_setup", "teacher_trajectory", "dagger"],
        "evaluation_seeds": seeds,
        "teacher_seed_start": args.teacher_seed_start,
        "dagger_seed_start": args.dagger_seed_start,
        "dagger_rounds": args.dagger_rounds,
        "dagger_boards_per_round": args.dagger_boards,
        "source_sha256": source_fingerprint(),
        "runtime_versions": runtime_versions(),
    })

    baseline = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    supervised_base = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    supervised_policy, supervised_report = train_initial_setup_policy(
        supervised_base.model.policy, train_boards=args.train_boards,
        validation_boards=args.validation_boards, test_boards=args.test_boards,
        seed_start=args.teacher_seed_start, training_seed=args.training_seed)
    supervised_agent = IndependentSetupAgent(supervised_base, supervised_policy)

    dagger_base = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    dagger_policy, dagger_report = train_initial_setup_policy_dagger(
        dagger_base.model.policy, train_boards=args.train_boards,
        validation_boards=args.validation_boards, test_boards=args.test_boards,
        seed_start=args.teacher_seed_start, training_seed=args.training_seed,
        dagger_rounds=args.dagger_rounds, dagger_boards=args.dagger_boards,
        dagger_validation_boards=args.dagger_validation_boards,
        dagger_seed_start=args.dagger_seed_start)
    dagger_agent = IndependentSetupAgent(dagger_base, dagger_policy)

    torch.save(supervised_policy.state_dict(), output / "teacher_trajectory_policy.pt")
    torch.save(dagger_policy.state_dict(), output / "dagger_policy.pt")
    _write(output / "training.json", {
        "teacher_trajectory": supervised_report,
        "dagger": dagger_report,
    })

    agents = {"heuristic_setup": baseline,
              "teacher_trajectory": supervised_agent,
              "dagger": dagger_agent}
    config = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                              policy_compatible_opponents=True, observation_version="v2")
    reports = {}
    summaries = {}
    for name, agent in agents.items():
        report = evaluate_agent(agent, seeds, config=config,
                                agent_name=f"{args.model_id}:{name}").to_dict()
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
            "average_rank": summary["average_rank"],
            "average_initial_pips": summary["average_initial_pips"],
        }, ensure_ascii=False), flush=True)

    effects = {}
    for name in ("teacher_trajectory", "dagger"):
        effects[name] = {
            "win_rate": paired_effect(reports[name], reports["heuristic_setup"], "learner_won"),
            "average_score": paired_effect(
                reports[name], reports["heuristic_setup"], "learner_score"),
        }
    _write(output / "paired_effects.json", effects)


if __name__ == "__main__":
    main()
