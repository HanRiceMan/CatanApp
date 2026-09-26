"""保存済み候補を、現行最強AIと同一seed・席順で再評価する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.heuristic import HeuristicAgent

from .settlement_planning_agent import SettlementPlanningAgent
from .train_strongest_league import _match, _strongest
from app.agents.ppo import PPOAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-models-root", type=Path, required=True)
    parser.add_argument("--source-model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--source-models-root", type=Path, default=Path("models"))
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=2_600_001)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponents", choices=("all", "strongest", "heuristic"),
                        default="all")
    parser.add_argument("--gate-threshold", type=float, default=None,
                        help="ゲート付き候補の推論閾値を評価時だけ上書きする。")
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("gamesは正の値です。")

    _, source = _strongest(args.source_model_id, args.source_models_root, args.device)
    candidate_base = PPOAgent(
        args.candidate_id, models_root=args.candidate_models_root, device=args.device,
    )
    if args.gate_threshold is not None:
        if not 0 <= args.gate_threshold <= 1:
            parser.error("gate-thresholdは0〜1です。")
        extractor = candidate_base.model.policy.mlp_extractor
        if not hasattr(extractor, "goal_gate_threshold"):
            parser.error("候補はゲート付き方策ではありません。")
        extractor.goal_gate_threshold.fill_(args.gate_threshold)
    if candidate_base.settlement_planning_config is None:
        raise ValueError("候補に戦略計画設定がありません。")
    candidate = SettlementPlanningAgent(
        candidate_base, **candidate_base.settlement_planning_config,
    )
    seeds = range(args.seed_start, args.seed_start + args.games)
    result = {}
    if args.opponents in {"all", "strongest"}:
        result.update({
            "source_vs_strongest_x3": _match(source, source, seeds),
            "candidate_vs_strongest_x3": _match(candidate, source, seeds),
        })
    if args.opponents in {"all", "heuristic"}:
        result.update({
            "source_vs_heuristic_x3": _match(source, HeuristicAgent(), seeds),
            "candidate_vs_heuristic_x3": _match(candidate, HeuristicAgent(), seeds),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({name: {
        key: value[key] for key in ("games", "win_rate", "score", "rank")
    } for name, value in result.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
