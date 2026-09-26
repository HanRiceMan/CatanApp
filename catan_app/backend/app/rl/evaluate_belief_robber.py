"""現行・公開脅威・ベイズ手札推定つき盗賊を同一盤面で比較する。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.agents.ppo import PPOAgent
from app.domain.actions import MoveRobberAction, StealResourceAction
from app.domain.game import GameState

from .evaluate import EvaluationConfig, evaluate_agent
from .evaluate_construction_priority import VALIDATED_PLANNING
from .settlement_planning_agent import SettlementPlanningAgent
from .train_development_budget import paired_comparison, summary, write


class RobberAuditAgent:
    def __init__(self, delegate: SettlementPlanningAgent):
        self.delegate = delegate
        self.terrains: Counter[str] = Counter()
        self.numbers: Counter[str] = Counter()
        self.victims: Counter[int] = Counter()

    def select_action(self, game: GameState, player_id: int):
        action = self.delegate.select_action(game, player_id)
        if isinstance(action, MoveRobberAction):
            tile = game.board.tiles[action.tile_id]
            self.terrains[tile.terrain] += 1
            self.numbers[str(tile.number)] += 1
        elif isinstance(action, StealResourceAction):
            self.victims[action.victim_id] += 1
        return action

    def audit(self) -> dict:
        total = sum(self.terrains.values())
        return {
            "moves": total,
            "terrain_counts": dict(self.terrains),
            "number_counts": dict(self.numbers),
            "victim_counts": {str(key): value for key, value in self.victims.items()},
            "ore_rate": self.terrains["ore"] / total if total else 0.0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=990001)
    parser.add_argument("--skip-public", action="store_true")
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."} or args.games <= 0):
        parser.error("実験名・ゲーム数が不正です。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent("ppo_gnn_board_65k_s03_exp_v003", device="cpu")
    configs = {
        "ppo": {**VALIDATED_PLANNING, "belief_aware_robber": False},
        "bayesian_threat": {**VALIDATED_PLANNING, "belief_aware_robber": True},
    }
    if not args.skip_public:
        configs = {
            "ppo": configs["ppo"],
            "public_threat": {**VALIDATED_PLANNING, "belief_aware_robber": False,
                              "opponent_aware_robber": True},
            "bayesian_threat": configs["bayesian_threat"],
        }
    write(output / "definition.json", {"arguments": vars(args), "planning": configs})
    evaluation = EvaluationConfig(
        heuristic_opponents=3, rotate_player_ids=True,
        policy_compatible_opponents=True, observation_version="v2",
    )
    reports, summaries = {}, {}
    for name, planning in configs.items():
        candidate = RobberAuditAgent(SettlementPlanningAgent(base, **planning))
        report = evaluate_agent(
            candidate, range(args.seed_start, args.seed_start + args.games),
            config=evaluation, agent_name=name,
        ).to_dict()
        if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
            raise AssertionError("評価が正常終了していません。")
        reports[name] = report
        summaries[name] = {**summary(report), "robber_audit": candidate.audit()}
        write(output / f"evaluation_{name}.json", report)
        print(json.dumps({"stage": name, **summaries[name]}, ensure_ascii=False), flush=True)
    write(output / "evaluation_summary.json", {
        "summaries": summaries,
        "paired_vs_ppo": {
            name: paired_comparison(reports["ppo"], reports[name])
            for name in reports if name != "ppo"
        },
    })


if __name__ == "__main__":
    main()
