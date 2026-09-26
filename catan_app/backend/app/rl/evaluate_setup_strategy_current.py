"""現行の通常手番を固定し、初期配置方式だけを比較する。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from app.agents.base import Agent
from app.agents.ppo import PPOAgent
from app.domain.actions import Action
from app.domain.game import GameState

from .compare import PolicyCompatibleHeuristicAgent
from .evaluate import EvaluationConfig, evaluate_agent
from .settlement_planning_agent import SettlementPlanningAgent
from .setup_diagnostics import GreedyPipsSettlementAgent, GreedyPipsSetupAgent
from .train_development_budget import paired_comparison, summary, write


@dataclass(frozen=True)
class SetupOnlyAgent:
    policy: Agent
    setup: Agent

    def select_action(self, game: GameState, player_id: int) -> Action:
        if game.phase == "setup_ready":
            return self.setup.select_action(game, player_id)
        return self.policy.select_action(game, player_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=720001)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.games <= 0:
        parser.error("実験名・ゲーム数を確認してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent(args.model_id, device="cpu")
    common = {
        "target_sites": 4, "max_wait_rounds": 9,
        "first_city_max_deficit": 2, "second_city_max_deficit": 2,
        "second_city_min_score": 6, "max_city_wait_rounds": 9,
        "expansion_aware_setup": True, "expansion_setup_min_pips": 3,
        "setup_min_wheat_pips": 1, "endgame_point_reserve_score": 8,
        "prioritize_affordable_city": True, "prioritize_immediate_win": True,
        "prioritize_immediate_titles": True, "title_horizon": 2,
    }
    current = SettlementPlanningAgent(base, **common)
    agents = {
        "current_gnn_filtered": current,
        "heuristic_setup": SetupOnlyAgent(current, PolicyCompatibleHeuristicAgent()),
        "greedy_settlement_setup": SetupOnlyAgent(current, GreedyPipsSettlementAgent()),
        "greedy_pips_setup": SetupOnlyAgent(current, GreedyPipsSetupAgent()),
    }
    config = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                              policy_compatible_opponents=True, observation_version="v2")
    reports, summaries = {}, {}
    seeds = range(args.seed_start, args.seed_start + args.games)
    for name, candidate in agents.items():
        report = evaluate_agent(candidate, seeds, config=config, agent_name=name).to_dict()
        if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
            raise AssertionError(f"{name}: 評価が正常に完了していません。")
        write(output / f"evaluation_{name}.json", report)
        reports[name], summaries[name] = report, summary(report)
        print(json.dumps({"stage": name, **summaries[name]}, ensure_ascii=False), flush=True)
    baseline = reports["current_gnn_filtered"]
    comparisons = {name: paired_comparison(baseline, report)
                   for name, report in reports.items() if report is not baseline}
    write(output / "evaluation_summary.json", {
        "definition": vars(args), "summaries": summaries,
        "paired_vs_current": comparisons,
    })


if __name__ == "__main__":
    main()
