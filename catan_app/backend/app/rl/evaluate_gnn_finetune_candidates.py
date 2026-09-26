"""GNN初期配置付きPPO継続学習checkpointを未使用盤面で比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from app.agents.ppo import PPOAgent

from .compare_initial_setup_adapter import IndependentSetupAgent
from .evaluate import EvaluationConfig, evaluate_agent
from .gnn_setup_policy import GraphInitialSetupPolicy


def _candidate(value: str) -> tuple[str, Path]:
    name, separator, path_text = value.partition("=")
    if not separator or not name or not path_text:
        raise argparse.ArgumentTypeError("candidateは NAME=CHECKPOINT_PATH 形式です。")
    path = Path(path_text).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpointがありません: {path}")
    return name, path


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _mean(summaries: list[dict], key: str) -> float:
    return sum(float(summary[key]) for summary in summaries) / len(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--gnn-experiment", required=True)
    parser.add_argument("--model-id", default="ppo_hierarchical_50k_v001")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--candidate", action="append", type=_candidate, required=True)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=140_001)
    args = parser.parse_args()

    if args.eval_games <= 0:
        parser.error("eval-gamesは0より大きくしてください。")
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        parser.error("candidate名が重複しています。")

    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。")
    output.mkdir(parents=True)

    source = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    gnn = GraphInitialSetupPolicy(source.model.policy)
    gnn_path = root / "experiments" / args.gnn_experiment / "gnn_policy.pt"
    gnn.load_state_dict(torch.load(gnn_path, map_location="cpu", weights_only=True))
    gnn.eval()

    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    config = EvaluationConfig(
        heuristic_opponents=3,
        rotate_player_ids=True,
        policy_compatible_opponents=True,
        observation_version="v2",
    )
    definition = {
        "source_model": args.model_id,
        "gnn_experiment": args.gnn_experiment,
        "candidates": {name: str(path) for name, path in candidates.items()},
        "evaluation_seeds": seeds,
        "selection_metrics": ["win_rate", "average_score", "average_rank"],
    }
    _write(output / "definition.json", definition)

    agents: dict[str, IndependentSetupAgent] = {
        "before_finetune": IndependentSetupAgent(source, gnn)
    }
    for name, checkpoint in candidates.items():
        agent = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
        agent.model = MaskablePPO.load(str(checkpoint), device="cpu")
        agents[name] = IndependentSetupAgent(agent, gnn)

    summaries: dict[str, dict] = {}
    for name, agent in agents.items():
        report = evaluate_agent(
            agent,
            seeds,
            config=config,
            agent_name=f"{args.model_id}:gnn_setup:{name}",
        ).to_dict()
        summary = report["summary"]
        if summary["illegal_action_count"] or summary["truncated"]:
            raise AssertionError(f"非法Actionまたは打ち切り: {name}")
        summaries[name] = summary
        _write(output / f"{name}.json", report)
        _write(output / "summary.json", summaries)
        print(json.dumps({
            "candidate": name,
            "win_rate": summary["win_rate"],
            "average_score": summary["average_score"],
            "average_rank": summary["average_rank"],
        }, ensure_ascii=False), flush=True)

    candidate_summaries = [summaries[name] for name in candidates]
    aggregate = {
        "candidate_seed_average": {
            "win_rate": _mean(candidate_summaries, "win_rate"),
            "average_score": _mean(candidate_summaries, "average_score"),
            "average_rank": _mean(candidate_summaries, "average_rank"),
        },
        "baseline": {
            key: summaries["before_finetune"][key]
            for key in ("win_rate", "average_score", "average_rank")
        },
    }
    _write(output / "aggregate.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
