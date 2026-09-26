"""Heuristicと最高pipsだけを見る初期配置の、対局結果を同条件で比較する。"""

import argparse
import json
from pathlib import Path

from .audit import runtime_versions, source_fingerprint
from .compare import PolicyCompatibleHeuristicAgent
from .evaluate import EvaluationConfig, evaluate_agent
from .setup_diagnostics import GreedyPipsSettlementAgent, GreedyPipsSetupAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=26001)
    parser.add_argument("--heuristic-counts", nargs="+", type=int, choices=range(4), default=[1, 3])
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    if args.games <= 0:
        parser.error("gamesは正の整数です。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。別IDを指定してください。")
    output.mkdir(parents=True)
    definition = {"experiment_id": args.experiment_id,
                  "seeds": list(range(args.seed_start, args.seed_start + args.games)),
                  "heuristic_counts": args.heuristic_counts, "rotate_player_ids": True,
                  "policy_compatible_opponents": True, "source_sha256": source_fingerprint(),
                  "runtime_versions": runtime_versions(),
                  "agents": ["heuristic", "greedy_settlement", "greedy_pips"]}
    (output / "definition.json").write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")
    agents = {"heuristic": PolicyCompatibleHeuristicAgent(),
              "greedy_settlement": GreedyPipsSettlementAgent(),
              "greedy_pips": GreedyPipsSetupAgent()}
    summaries = {}
    for count in args.heuristic_counts:
        config = EvaluationConfig(heuristic_opponents=count, rotate_player_ids=True,
                                  policy_compatible_opponents=True)
        for name, agent in agents.items():
            cell = f"{name}_H{count}_R{3-count}"
            report = evaluate_agent(agent, definition["seeds"], config=config, agent_name=name).to_dict()
            (output / f"{cell}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            summaries[cell] = report["summary"]
            (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
            summary = report["summary"]
            print(json.dumps({"cell": cell, **{key: summary[key] for key in
                              ("win_rate", "average_score", "average_rank", "illegal_action_count", "truncated")}},
                             ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
