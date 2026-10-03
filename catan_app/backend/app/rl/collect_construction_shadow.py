"""36局程度のCONSTRUCTION_READY_CHOICE Shadow収集と完全parity検証。"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from app.agents.heuristic import HeuristicAgent
from app.agents.ppo import PPOAgent
from app.domain.game import create_game

from .action_space import get_action_mask
from .construction_shadow import ConstructionShadowAgent, summarize_shadow
from .context_aware_policy import (create_context_aware_experiment_agent,
                                   load_context_aware_artifact,
                                   save_context_aware_artifact)
from .macro_teacher_dataset import OPPONENT_PROFILES, play_teacher_episode
from .observation import encode_observation, get_observation
from .settlement_planning_agent import SettlementPlanningAgent


def verify_artifact_reload(original, loaded) -> dict:
    """保存前後の全weight・Actor/Critic・masked distribution・Actionを比較。"""
    if original.manifest.to_dict() != loaded.manifest.to_dict():
        raise AssertionError("Context manifestが再読込後に変わりました。")
    old_weights = original.model.policy.state_dict()
    new_weights = loaded.model.policy.state_dict()
    if old_weights.keys() != new_weights.keys() or any(
        not torch.equal(old_weights[key], new_weights[key]) for key in old_weights
    ):
        raise AssertionError("保存→再読込でPolicy weightが変わりました。")
    game = create_game(20261035)
    game.phase = "action"
    game.current_player_id = 1
    observation = encode_observation(get_observation(game, 1, version="v7"))
    reloaded_observation = encode_observation(get_observation(game, 1, version="v7"))
    mask = get_action_mask(game, 1)
    tensors = [original.model.policy.obs_to_tensor(observation)[0],
               loaded.model.policy.obs_to_tensor(reloaded_observation)[0]]
    with torch.no_grad():
        logits = [policy.mlp_extractor.forward_actor(tensor)
                  for policy, tensor in zip((original.model.policy, loaded.model.policy),
                                            tensors, strict=True)]
        values = [policy.value_net(policy.mlp_extractor.forward_critic(tensor))
                  for policy, tensor in zip((original.model.policy, loaded.model.policy),
                                            tensors, strict=True)]
        distributions = [policy.get_distribution(tensor, action_masks=mask).distribution.probs
                         for policy, tensor in zip((original.model.policy, loaded.model.policy),
                                                   tensors, strict=True)]
    checks = {
        "manifest_equal": True,
        "all_policy_tensors_bitwise_equal": True,
        "context_observation_equal": np.array_equal(observation, reloaded_observation),
        "actor_logits_bitwise_equal": torch.equal(*logits),
        "critic_value_bitwise_equal": torch.equal(*values),
        "masked_distribution_bitwise_equal": torch.equal(*distributions),
        "selected_action_equal": original.select_action(game, 1)
                                 == loaded.select_action(game, 1),
    }
    if not all(checks.values()):
        raise AssertionError(f"save/reload parity失敗: {checks}")
    return checks


def collect_shadow_games(champion: PPOAgent, shadow, robber: PPOAgent, *,
                         games_per_profile: int, seed_start: int) -> tuple[list[dict], list[dict], dict]:
    planning = champion.settlement_planning_config
    robber_planning = robber.settlement_planning_config
    if planning is None or robber_planning is None:
        raise ValueError("ChampionとrobberモデルにPlanner configが必要です。")

    def champion_factory():
        return SettlementPlanningAgent(champion, **planning)

    def robber_factory():
        return SettlementPlanningAgent(robber, **robber_planning)

    factories = {"rule": HeuristicAgent, "champion": champion_factory,
                 "robber_ppo": robber_factory}
    records: list[dict] = []
    games: list[dict] = []
    timings: dict[str, list[float]] = {
        "surface_detection": [], "observation_generation": [],
        "runtime_context_generation": [], "policy_forward": [],
        "environment_step": [],
    }
    for profile_index, profile in enumerate(OPPONENT_PROFILES):
        for index in range(games_per_profile):
            seed = seed_start + profile_index * 1000 + index
            seat = index % 4 + 1
            plain = play_teacher_episode(
                SettlementPlanningAgent(champion, **planning), seed,
                champion_seat=seat, opponent_profile=profile,
                opponent_factories=factories, collect_reports=False,
            )
            agent = ConstructionShadowAgent(champion, shadow, **planning)
            agent.game_seed = seed
            agent.opponent_profile = profile
            sampled = play_teacher_episode(
                agent, seed, champion_seat=seat, opponent_profile=profile,
                opponent_factories=factories, collect_reports=False,
                step_duration_sink=timings["environment_step"].append,
            )
            if plain.action_trace != sampled.action_trace:
                raise AssertionError(f"{profile}/{seed}: 全Action列不一致")
            if (plain.final_state != sampled.final_state
                    or plain.final_views != sampled.final_views):
                raise AssertionError(f"{profile}/{seed}: 全GameState不一致")
            for key in ("winner", "final_vp", "final_rank", "final_scores", "rng_counter"):
                if plain.game[key] != sampled.game[key]:
                    raise AssertionError(f"{profile}/{seed}: {key}不一致")
            for row in agent.records:
                if row["game_seed"] != seed or row["seat"] != seat:
                    raise AssertionError("Shadow Decisionのgame/seat参照が不正です。")
            records.extend(agent.records)
            for name, values in agent.timings.items():
                timings[name].extend(values)
            games.append({
                "game_seed": seed, "opponent_profile": profile, "seat": seat,
                "action_count": len(plain.action_trace),
                "surface_count": len(agent.records),
                "surface_candidates": dict(agent.surface_counts),
                "hard_exclusions": dict(agent.exclusion_counts),
                "winner": plain.game["winner"],
                "final_scores": plain.game["final_scores"],
                "final_rank": plain.game["final_rank"],
                "rng_counter": plain.game["rng_counter"],
                "parity": True,
            })
            print(f"{profile} seed={seed} seat={seat} "
                  f"actions={len(plain.action_trace)} surface={len(agent.records)}",
                  flush=True)
    return records, games, summarize_shadow(records, games, timings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games-per-profile", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=2360001)
    parser.add_argument("--champion-model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--robber-model-id", default="ppo_robber_value_73k_s01_exp_v001")
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."}
            or args.games_per_profile < 1):
        parser.error("実験名または局数が不正です。")
    root = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if root.exists():
        parser.error("実験フォルダが既にあります。")
    champion = PPOAgent(args.champion_model_id, device="cpu")
    robber = PPOAgent(args.robber_model_id, device="cpu")
    original = create_context_aware_experiment_agent(champion)
    root.mkdir(parents=True)
    save_context_aware_artifact(original, root / "context_policy")
    shadow = load_context_aware_artifact(root / "context_policy", champion)
    reload_checks = verify_artifact_reload(original, shadow)
    records, games, summary = collect_shadow_games(
        champion, shadow, robber,
        games_per_profile=args.games_per_profile,
        seed_start=args.seed_start,
    )
    summary["save_reload"] = reload_checks
    summary["games"] = games
    summary["seed_start"] = args.seed_start
    summary["games_per_profile"] = args.games_per_profile
    summary["source_model_id"] = args.champion_model_id
    summary["shadow_artifact"] = shadow.manifest.to_dict()
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    with gzip.open(root / "surface_decisions.jsonl.gz", "wt", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"games": len(games), "surface_decisions": len(records),
                      "action_parity": summary["all_games_parity"],
                      "artifact_reload": all(reload_checks.values())},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
