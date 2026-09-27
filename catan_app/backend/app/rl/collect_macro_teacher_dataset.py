"""Step 2A: ChampionのMacro Teacher Datasetを収集してtaxonomyを診断する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .macro_teacher_dataset import (collect_teacher_dataset,
                                    save_teacher_dataset,
                                    verify_action_parity)
from .settlement_planning_agent import SettlementPlanningAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=2330001)
    parser.add_argument("--parity-games", type=int, default=0)
    parser.add_argument("--parity-seed-start", type=int, default=2331001)
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."}
            or args.games <= 0 or args.parity_games < 0):
        parser.error("実験名・ゲーム数が不正です。")
    collection_seeds = set(range(args.seed_start, args.seed_start + args.games))
    parity_seeds = set(range(
        args.parity_seed_start, args.parity_seed_start + args.parity_games,
    ))
    if collection_seeds & parity_seeds:
        parser.error("Teacher収集seedとparity評価seedは分離してください。")

    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    base = PPOAgent(args.model_id, device="cpu")
    planning = base.settlement_planning_config
    if planning is None:
        raise ValueError("指定モデルにはChampion計画設定がありません。")
    champion = SettlementPlanningAgent(base, **planning)
    dataset = collect_teacher_dataset(
        champion, sorted(collection_seeds),
    )
    result = save_teacher_dataset(output, dataset, definition={
        "model_id": args.model_id,
        "observation_version": base.manifest.observation_version,
        "seed_start": args.seed_start,
        "planning": planning,
    })
    print(json.dumps({"stage": "dataset", **result["manifest"]},
                     ensure_ascii=False), flush=True)
    print(json.dumps({"stage": "taxonomy", **result["summary"]},
                     ensure_ascii=False), flush=True)

    if args.parity_games:
        # 共有modelは推論専用。Plannerインスタンスだけを分け、両経路を比較する。
        plain = SettlementPlanningAgent(base, **planning)
        diagnostic = SettlementPlanningAgent(base, **planning)
        parity = verify_action_parity(
            plain, diagnostic, sorted(parity_seeds),
        )
        parity_path = output / "parity_summary.json"
        parity_path.write_text(
            json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(json.dumps({"stage": "parity", **parity},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
