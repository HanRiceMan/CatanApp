"""評価JSONから4拠点未到達局の初期生産と行動を比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from app.domain.game import RESOURCES, create_game

from .expansion_planner import vertex_production


def mean(values):
    return fmean(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = []
    for episode in report["episodes"]:
        game = create_game(episode["seed"])
        production = dict.fromkeys(RESOURCES, 0)
        for vertex_id in episode["learner_initial_vertices"]:
            for resource, pips in vertex_production(game, vertex_id).items():
                production[resource] += pips
        pieces = episode["learner_final_pieces"]
        rows.append({
            "seed": episode["seed"],
            "reached_four_sites": pieces["settlements"] + pieces["cities"] >= 4,
            "won": episode["learner_won"],
            "score": episode["learner_score"],
            "initial_production": production,
            "initial_resource_kinds": sum(value > 0 for value in production.values()),
            "paid_roads": episode["learner_actions"].get("build_road_paid", 0),
            "development": episode["learner_actions"].get("buy_development", 0),
            "bank_trades": episode["learner_actions"].get("bank_trade", 0),
            "turn_number": episode["turn_number"],
        })

    def group(reached):
        selected = [row for row in rows if row["reached_four_sites"] is reached]
        return {
            "games": len(selected),
            "win_rate": mean([row["won"] for row in selected]),
            "score": mean([row["score"] for row in selected]),
            "initial_production": {
                resource: mean([row["initial_production"][resource] for row in selected])
                for resource in RESOURCES
            },
            "initial_resource_kinds": mean([row["initial_resource_kinds"] for row in selected]),
            "paid_roads": mean([row["paid_roads"] for row in selected]),
            "development": mean([row["development"] for row in selected]),
            "bank_trades": mean([row["bank_trades"] for row in selected]),
            "turn_number": mean([row["turn_number"] for row in selected]),
        }

    result = {
        "report": str(args.report),
        "reached_four_sites": group(True),
        "missed_four_sites": group(False),
        "missed_rows": [row for row in rows if not row["reached_four_sites"]],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
