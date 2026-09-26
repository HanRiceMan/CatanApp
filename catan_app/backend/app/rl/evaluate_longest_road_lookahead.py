"""最長交易路の1手評価と、行き止まりを避ける2手先評価を比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.ppo import PPOAgent
from app.domain.actions import BuildRoadAction
from app.domain.game import GameState, _building_owner

from .construction_metrics import construction_summary
from .evaluate import EvaluationConfig, evaluate_agent
from .evaluate_construction_priority import VALIDATED_PLANNING
from .settlement_planning_agent import SettlementPlanningAgent
from .train_development_budget import paired_comparison, summary, write


class RoadAuditAgent:
    def __init__(self, delegate: SettlementPlanningAgent):
        self.delegate = delegate
        self.paid_roads = 0
        self.blocked_endpoint_roads = 0
        self.contested_endpoint_roads = 0

    def select_action(self, game: GameState, player_id: int):
        action = self.delegate.select_action(game, player_id)
        if isinstance(action, BuildRoadAction) and not game.free_road_remaining:
            self.paid_roads += 1
            if any(
                _building_owner(game, vertex_id) not in {None, player_id}
                for vertex_id in game.board.edges[action.edge_id].vertex_ids
            ):
                self.blocked_endpoint_roads += 1
            if any(
                _building_owner(game, vertex_id) is None
                and any(game.roads.get(edge_id) not in {None, player_id}
                        for edge_id in game.board.vertices[vertex_id].edge_ids)
                for vertex_id in game.board.edges[action.edge_id].vertex_ids
            ):
                self.contested_endpoint_roads += 1
        return action

    def audit(self) -> dict:
        return {
            "paid_road_decisions": self.paid_roads,
            "blocked_endpoint_road_decisions": self.blocked_endpoint_roads,
            "contested_endpoint_road_decisions": self.contested_endpoint_roads,
            "blocked_endpoint_rate": (
                self.blocked_endpoint_roads / self.paid_roads if self.paid_roads else 0.0
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=950001)
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."} or args.games <= 0):
        parser.error("実験名・ゲーム数が不正です。")

    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    base = PPOAgent("ppo_gnn_board_65k_s03_exp_v003", device="cpu")
    configs = {
        "one_step": dict(VALIDATED_PLANNING),
        "two_step": {**VALIDATED_PLANNING, "longest_road_lookahead": True},
    }
    write(output / "definition.json", {"arguments": vars(args), "planning": configs})
    evaluation = EvaluationConfig(
        heuristic_opponents=3, rotate_player_ids=True,
        policy_compatible_opponents=True, observation_version="v2",
    )
    reports, summaries, audits = {}, {}, {}
    for name, planning in configs.items():
        candidate = RoadAuditAgent(SettlementPlanningAgent(base, **planning))
        report = evaluate_agent(
            candidate, range(args.seed_start, args.seed_start + args.games),
            config=evaluation, agent_name=name,
        ).to_dict()
        if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
            raise AssertionError("評価が正常終了していません。")
        reports[name] = report
        audits[name] = candidate.audit()
        summaries[name] = {
            **summary(report), "construction": construction_summary(report),
            "road_audit": audits[name],
        }
        write(output / f"evaluation_{name}.json", report)
        print(json.dumps({"stage": name, **summaries[name]}, ensure_ascii=False), flush=True)
    write(output / "evaluation_summary.json", {
        "summaries": summaries,
        "paired": paired_comparison(reports["one_step"], reports["two_step"]),
    })


if __name__ == "__main__":
    main()
