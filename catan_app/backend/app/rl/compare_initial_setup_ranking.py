"""初期配置の1点教師と全候補ランキング教師を同一条件で比較する。"""

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
                                   train_ranked_initial_setup_policy)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_hierarchical_50k_v001")
    parser.add_argument("--models-root", type=Path, default=None)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=92_001)
    parser.add_argument("--train-boards", type=int, default=600)
    parser.add_argument("--validation-boards", type=int, default=150)
    parser.add_argument("--test-boards", type=int, default=150)
    parser.add_argument("--teacher-seed-start", type=int, default=90_001)
    parser.add_argument("--training-seed", type=int, default=20260928)
    parser.add_argument("--temperature", type=float, default=5.0)
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
        "conditions": ["heuristic_setup", "single_action_teacher", "all_candidate_ranking"],
        "evaluation_seeds": seeds,
        "teacher_seed_start": args.teacher_seed_start,
        "teacher_boards": {"train": args.train_boards, "validation": args.validation_boards,
                           "test": args.test_boards},
        "ranking_temperature": args.temperature,
        "source_sha256": source_fingerprint(),
        "runtime_versions": runtime_versions(),
    })

    baseline = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    single_base = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    single_policy, single_report = train_initial_setup_policy(
        single_base.model.policy, train_boards=args.train_boards,
        validation_boards=args.validation_boards, test_boards=args.test_boards,
        seed_start=args.teacher_seed_start, training_seed=args.training_seed)
    single_agent = IndependentSetupAgent(single_base, single_policy)

    ranked_base = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    ranked_policy, ranked_report = train_ranked_initial_setup_policy(
        ranked_base.model.policy, train_boards=args.train_boards,
        validation_boards=args.validation_boards, test_boards=args.test_boards,
        seed_start=args.teacher_seed_start, training_seed=args.training_seed,
        temperature=args.temperature)
    ranked_agent = IndependentSetupAgent(ranked_base, ranked_policy)

    torch.save(single_policy.state_dict(), output / "single_action_policy.pt")
    torch.save(ranked_policy.state_dict(), output / "ranking_policy.pt")
    _write(output / "training.json", {
        "single_action_teacher": single_report,
        "all_candidate_ranking": ranked_report,
    })

    agents = {"heuristic_setup": baseline,
              "single_action_teacher": single_agent,
              "all_candidate_ranking": ranked_agent}
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
        resource_kinds = sum(row["learner_initial_resource_kinds"]
                             for row in report["episodes"]) / len(report["episodes"])
        print(json.dumps({
            "condition": name,
            "win_rate": summary["win_rate"],
            "average_score": summary["average_score"],
            "average_rank": summary["average_rank"],
            "average_initial_pips": summary["average_initial_pips"],
            "average_initial_resource_kinds": resource_kinds,
        }, ensure_ascii=False), flush=True)

    effects = {}
    for name in ("single_action_teacher", "all_candidate_ranking"):
        effects[name] = {
            "win_rate": paired_effect(reports[name], reports["heuristic_setup"], "learner_won"),
            "average_score": paired_effect(
                reports[name], reports["heuristic_setup"], "learner_score"),
        }
    _write(output / "paired_effects.json", effects)


if __name__ == "__main__":
    main()
