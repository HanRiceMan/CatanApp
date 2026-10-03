"""Step 2A: profile別Macro Teacher Datasetを収集してtaxonomyを診断する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.heuristic import HeuristicAgent
from app.agents.ppo import PPOAgent

from .macro_teacher_dataset import (OPPONENT_PROFILES,
                                    collect_teacher_dataset,
                                    merge_teacher_datasets,
                                    save_teacher_dataset,
                                    verify_action_parity)
from .settlement_planning_agent import SettlementPlanningAgent


def _seed_ranges(
    profiles: list[str], start: int, count: int, stride: int,
) -> dict[str, range]:
    if stride < count:
        raise ValueError("profile-seed-strideはゲーム数以上にしてください。")
    return {
        profile: range(
            start + OPPONENT_PROFILES.index(profile) * stride,
            start + OPPONENT_PROFILES.index(profile) * stride + count,
        )
        for profile in profiles
    }


def _validate_disjoint(*groups: dict[str, range]) -> None:
    seen: set[int] = set()
    for group in groups:
        for profile, seeds in group.items():
            current = set(seeds)
            if seen & current:
                raise ValueError(f"{profile}のseed範囲が他用途と重複しています。")
            seen.update(current)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--robber-model-id",
                        default="ppo_robber_value_73k_s01_exp_v001")
    parser.add_argument("--opponent-profiles", nargs="+",
                        choices=OPPONENT_PROFILES, default=["rule"])
    parser.add_argument("--games", type=int, default=8,
                        help="profileごとの収集ゲーム数")
    parser.add_argument("--seed-start", type=int, default=2330001)
    parser.add_argument("--profile-seed-stride", type=int, default=1000)
    parser.add_argument("--parity-games", type=int, default=0,
                        help="profileごとのparityゲーム数")
    parser.add_argument("--parity-seed-start", type=int, default=2335001)
    args = parser.parse_args()
    profiles = list(dict.fromkeys(args.opponent_profiles))
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."}
            or args.games <= 0 or args.parity_games < 0):
        parser.error("実験名・ゲーム数が不正です。")
    try:
        collection_seeds = _seed_ranges(
            profiles, args.seed_start, args.games, args.profile_seed_stride,
        )
        parity_seeds = _seed_ranges(
            profiles, args.parity_seed_start, args.parity_games,
            args.profile_seed_stride,
        )
        _validate_disjoint(collection_seeds, parity_seeds)
    except ValueError as error:
        parser.error(str(error))

    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    base = PPOAgent(args.model_id, device="cpu")
    planning = base.settlement_planning_config
    if planning is None:
        raise ValueError("指定Championモデルには計画設定がありません。")
    champion = SettlementPlanningAgent(base, **planning)

    robber_base = None
    robber_planning = None
    if "mixed" in profiles:
        robber_base = PPOAgent(args.robber_model_id, device="cpu")
        robber_planning = robber_base.settlement_planning_config
        if robber_planning is None:
            raise ValueError("指定盗賊PPOモデルには計画設定がありません。")

    def champion_factory():
        return SettlementPlanningAgent(base, **planning)

    def robber_factory():
        if robber_base is None or robber_planning is None:
            raise RuntimeError("盗賊PPOが読み込まれていません。")
        return SettlementPlanningAgent(robber_base, **robber_planning)

    factories = {
        "rule": HeuristicAgent,
        "champion": champion_factory,
        "robber_ppo": robber_factory,
    }
    datasets = []
    for profile in profiles:
        dataset = collect_teacher_dataset(
            champion, collection_seeds[profile],
            opponent_profile=profile,
            opponent_factories=factories,
            champion_model_id=args.model_id,
            robber_model_id=args.robber_model_id,
        )
        datasets.append(dataset)
        print(json.dumps({
            "stage": "profile_collected",
            "opponent_profile": profile,
            "games": len(dataset.games),
            "decisions": len(dataset.records),
        }, ensure_ascii=False), flush=True)
    combined = merge_teacher_datasets(
        datasets, observation_size=base.manifest.observation_size,
    )
    result = save_teacher_dataset(output, combined, definition={
        "model_id": args.model_id,
        "robber_model_id": args.robber_model_id,
        "opponent_profiles": profiles,
        "games_per_profile": args.games,
        "observation_version": base.manifest.observation_version,
        "seed_ranges": {
            profile: [seeds.start, seeds.stop - 1]
            for profile, seeds in collection_seeds.items()
        },
        "planning": planning,
    })
    print(json.dumps({"stage": "dataset", **result["manifest"]},
                     ensure_ascii=False), flush=True)
    print(json.dumps({"stage": "profile_stability",
                      **result["profile_stability"]},
                     ensure_ascii=False), flush=True)
    print(json.dumps({"stage": "strategic_intents",
                      **result["strategic_intents"]},
                     ensure_ascii=False), flush=True)
    context = result["structured_context"]
    print(json.dumps({
        "stage": "structured_context",
        "decision_count": context["decision_count"],
        "phase_distribution": context["phase_distribution"],
        "execution_distribution": context["execution_distribution"],
        "observability_gap_decisions": context["observability_gap_decisions"],
        "sentinel_or_invalid_value_count": context["sentinel_or_invalid_value_count"],
    }, ensure_ascii=False), flush=True)

    if args.parity_games:
        parity_profiles = {}
        for profile in profiles:
            plain = SettlementPlanningAgent(base, **planning)
            diagnostic = SettlementPlanningAgent(base, **planning)
            parity_profiles[profile] = verify_action_parity(
                plain, diagnostic, parity_seeds[profile],
                opponent_profile=profile,
                opponent_factories=factories,
                champion_model_id=args.model_id,
                robber_model_id=args.robber_model_id,
            )
            print(json.dumps({"stage": "parity",
                              **parity_profiles[profile]},
                             ensure_ascii=False), flush=True)
        parity = {
            "profiles": parity_profiles,
            "games": sum(item["games"] for item in parity_profiles.values()),
            "all_actions": sum(
                item["all_actions"] for item in parity_profiles.values()
            ),
            "all_equal": all(
                item["action_trace_equal"]
                and item["winner_equal"]
                and item["final_score_equal"]
                and item["final_rank_equal"]
                and item["final_game_state_equal"]
                and item["rng_counter_equal"]
                and item["ppo_weight_equal"] is not False
                and item["actor_logits_equal"] is not False
                for item in parity_profiles.values()
            ),
            "seed_ranges": {
                profile: [seeds.start, seeds.stop - 1]
                for profile, seeds in parity_seeds.items()
            },
        }
        (output / "parity_summary.json").write_text(
            json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8",
        )


if __name__ == "__main__":
    main()
