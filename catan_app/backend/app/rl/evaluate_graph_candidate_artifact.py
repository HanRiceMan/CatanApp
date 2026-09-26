"""保存済みGraph候補方策を元モデルと完全未使用seedで比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.agents.ppo import PPOAgent

from .candidate_policy import GraphHierarchicalCandidateMaskablePolicy
from .evaluate import EvaluationConfig, evaluate_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--policy-artifact", type=Path, required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=300_001)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("gamesは正の整数にしてください。")
    artifact = args.policy_artifact.resolve()
    if not artifact.is_file():
        parser.error(f"方策artifactがありません: {artifact}")

    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。")
    output.mkdir(parents=True)
    baseline = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    candidate = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    if type(candidate.model.policy) is not GraphHierarchicalCandidateMaskablePolicy:
        raise TypeError("GraphHierarchicalCandidateMaskablePolicyが必要です。")
    candidate.model.policy.load_state_dict(
        torch.load(artifact, map_location="cpu", weights_only=True)
    )
    candidate.model.policy.eval()
    seeds = tuple(range(args.seed_start, args.seed_start + args.games))
    config = EvaluationConfig(
        heuristic_opponents=3,
        rotate_player_ids=True,
        policy_compatible_opponents=True,
        observation_version="v2",
    )
    baseline_report = evaluate_agent(
        baseline, seeds, config=config, agent_name=f"{args.model_id}:baseline"
    ).to_dict()
    candidate_report = evaluate_agent(
        candidate, seeds, config=config, agent_name=f"{args.model_id}:graph_candidate"
    ).to_dict()
    report = {
        "model_id": args.model_id,
        "policy_artifact": str(artifact),
        "seeds": seeds,
        "baseline": baseline_report,
        "candidate": candidate_report,
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "baseline": baseline_report["summary"],
        "candidate": candidate_report["summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
