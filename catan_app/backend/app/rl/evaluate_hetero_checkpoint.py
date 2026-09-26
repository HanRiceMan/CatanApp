"""異種GNN v2のPPO checkpointを現行モデルと未使用seedで比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from app.agents.ppo import PPOAgent

from .candidate_policy import HeteroGraphHierarchicalCandidateMaskablePolicy
from .evaluate import EvaluationConfig, evaluate_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=390_001)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("gamesは正の整数にしてください。")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpointがありません: {checkpoint}")
    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。")
    output.mkdir(parents=True)

    baseline = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    candidate = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    candidate.model = MaskablePPO.load(str(checkpoint), device="cpu")
    if type(candidate.model.policy) is not HeteroGraphHierarchicalCandidateMaskablePolicy:
        raise TypeError("HeteroGraphHierarchicalCandidateMaskablePolicyが必要です。")
    seeds = tuple(range(args.seed_start, args.seed_start + args.games))
    config = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                              policy_compatible_opponents=True, observation_version="v2")
    baseline_report = evaluate_agent(
        baseline, seeds, config=config, agent_name=f"{args.model_id}:baseline"
    ).to_dict()
    candidate_report = evaluate_agent(
        candidate, seeds, config=config, agent_name=f"{args.model_id}:hetero_gnn_v2"
    ).to_dict()
    report = {
        "model_id": args.model_id,
        "checkpoint": str(checkpoint),
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
