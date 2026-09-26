"""評価JSONを勝敗・最終都市数で分け、初期配置の資源生産を診断する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from app.domain.game import RESOURCES, create_game

from .expansion_planner import vertex_production


def _mean(values):
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
        rows.append({
            "won": episode["learner_won"],
            "cities": episode["learner_final_pieces"]["cities"],
            "score": episode["learner_score"],
            "production": production,
        })

    def summarize(selected):
        return {
            "games": len(selected),
            "score": _mean([row["score"] for row in selected]),
            "initial_production": {
                resource: _mean([row["production"][resource] for row in selected])
                for resource in RESOURCES
            },
            "zero_resource_games": {
                resource: sum(row["production"][resource] == 0 for row in selected)
                for resource in RESOURCES
            },
        }

    result = {
        "wins": summarize([row for row in rows if row["won"]]),
        "losses": summarize([row for row in rows if not row["won"]]),
        "losses_zero_cities": summarize([
            row for row in rows if not row["won"] and row["cities"] == 0
        ]),
        "losses_with_cities": summarize([
            row for row in rows if not row["won"] and row["cities"] > 0
        ]),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
