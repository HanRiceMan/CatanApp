"""学習なしのv7 Context Policyと現行Championの固定seed parity。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.agents.heuristic import HeuristicAgent
from app.agents.ppo import PPOAgent
from app.domain.game import GameState

from .action_space import get_action_mask
from .context_aware_policy import create_context_aware_experiment_agent
from .macro_teacher_dataset import OPPONENT_PROFILES, play_teacher_episode
from .observation import encode_observation, get_observation
from .runtime_context import (RUNTIME_CONTEXT_SIZE, encode_runtime_context,
                              runtime_context_definition)
from .settlement_planning_agent import SettlementPlanningAgent


def compare_policy_probe(champion: PPOAgent, candidate, game: GameState,
                         player_id: int) -> dict[str, Any]:
    """同一状態・maskで旧Actor/新ActorとCriticをbitwise比較。"""
    old_observation = encode_observation(get_observation(game, player_id, version="v2"))
    new_observation = encode_observation(get_observation(game, player_id, version="v7"))
    if not np.array_equal(old_observation, new_observation[:len(old_observation)]):
        raise AssertionError("v7のv2 prefixが一致しません。")
    old_policy = champion.model.policy
    new_policy = candidate.model.policy
    old_tensor, _ = old_policy.obs_to_tensor(old_observation)
    new_tensor, _ = new_policy.obs_to_tensor(new_observation)
    mask = get_action_mask(game, player_id)
    with torch.no_grad():
        old_logits = old_policy.mlp_extractor.forward_actor(old_tensor)
        new_logits = new_policy.mlp_extractor.forward_actor(new_tensor)
        old_value = old_policy.value_net(old_policy.mlp_extractor.forward_critic(old_tensor))
        new_value = new_policy.value_net(new_policy.mlp_extractor.forward_critic(new_tensor))
        if np.any(mask):
            old_probs = old_policy.get_distribution(old_tensor, action_masks=mask).distribution.probs
            new_probs = new_policy.get_distribution(new_tensor, action_masks=mask).distribution.probs
        else:
            old_probs = new_probs = None
    results = {
        "phase": game.phase,
        "actor_logits_bitwise_equal": torch.equal(old_logits, new_logits),
        "critic_value_bitwise_equal": torch.equal(old_value, new_value),
        "masked_distribution_bitwise_equal": (
            torch.equal(old_probs, new_probs) if old_probs is not None else None
        ),
        "selected_action_equal": champion.select_action(game, player_id)
        == candidate.select_action(game, player_id),
    }
    if not all(value for key, value in results.items()
               if key != "phase" and value is not None):
        raise AssertionError(f"固定probeの出力に差があります: {results}")
    return results


class ContextSamplingPlanningAgent(SettlementPlanningAgent):
    """対局に介入せずv7 Contextの値域とphase別probeを集める。"""

    def __init__(self, policy, champion: PPOAgent, **planning):
        super().__init__(policy, **planning)
        self.champion = champion
        self.samples: list[np.ndarray] = []
        self.probes: dict[str, dict[str, Any]] = {}

    def select_action(self, game: GameState, player_id: int):
        vector = encode_runtime_context(game, player_id,
                                        max_plan_roads=self.max_plan_roads)
        self.samples.append(vector)
        if game.phase not in self.probes:
            self.probes[game.phase] = compare_policy_probe(
                self.champion, self.policy, game, player_id,
            )
        return super().select_action(game, player_id)


def verify_context_policy_games(
    champion: PPOAgent, candidate, robber: PPOAgent,
    *, seed_start: int = 2350001, games_per_profile: int = 1,
) -> dict[str, Any]:
    planning = champion.settlement_planning_config
    robber_planning = robber.settlement_planning_config
    if planning is None or robber_planning is None:
        raise ValueError("対象モデルのPlanner configがありません。")

    def champion_factory():
        return SettlementPlanningAgent(champion, **planning)

    def robber_factory():
        return SettlementPlanningAgent(robber, **robber_planning)

    factories = {"rule": HeuristicAgent, "champion": champion_factory,
                 "robber_ppo": robber_factory}
    results = {}
    for profile_index, profile in enumerate(OPPONENT_PROFILES):
        profile_games = []
        for index in range(games_per_profile):
            seed = seed_start + profile_index * 1000 + index
            seat = index % 4 + 1
            plain = play_teacher_episode(
                SettlementPlanningAgent(champion, **planning), seed,
                champion_seat=seat, opponent_profile=profile,
                opponent_factories=factories, collect_reports=False,
            )
            sampled = ContextSamplingPlanningAgent(candidate, champion, **planning)
            new = play_teacher_episode(
                sampled, seed, champion_seat=seat, opponent_profile=profile,
                opponent_factories=factories, collect_reports=False,
            )
            if plain.action_trace != new.action_trace:
                raise AssertionError(f"{profile}/{seed}: 全Action列不一致")
            if plain.final_state != new.final_state or plain.final_views != new.final_views:
                raise AssertionError(f"{profile}/{seed}: 最終GameState不一致")
            keys = ("winner", "final_vp", "final_rank", "final_scores", "rng_counter")
            if any(plain.game[key] != new.game[key] for key in keys):
                raise AssertionError(f"{profile}/{seed}: 対局結果またはRNG不一致")
            if not sampled.samples:
                raise AssertionError("Context sampleがありません。")
            matrix = np.stack(sampled.samples)
            if (matrix.shape[1] != RUNTIME_CONTEXT_SIZE
                    or not np.isfinite(matrix).all()
                    or np.any(matrix < 0) or np.any(matrix > 1)):
                raise AssertionError("Context値域または次元が不正です。")
            profile_games.append({
                "seed": seed, "seat": plain.game["seat"],
                "all_actions": len(plain.action_trace),
                "winner": plain.game["winner"],
                "final_scores": plain.game["final_scores"],
                "sampled_decisions": len(sampled.samples),
                "context_dimension": matrix.shape[1],
                "context_min": float(matrix.min()),
                "context_max": float(matrix.max()),
                "all_context_finite_in_range": True,
                "phase_probes": sampled.probes,
                "action_trace_equal": True,
                "final_state_equal": True,
                "outcome_equal": True,
                "rng_counter_equal": True,
            })
        results[profile] = profile_games
    return {
        "source_model_id": champion.manifest.model_id,
        "candidate_observation_version": "v7",
        "context_dimension": RUNTIME_CONTEXT_SIZE,
        "profiles": results,
        "game_count": sum(map(len, results.values())),
        "all_actions": sum(game["all_actions"]
                           for games in results.values() for game in games),
        "all_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--robber-model-id", default="ppo_robber_value_73k_s01_exp_v001")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed-start", type=int, default=2350001)
    parser.add_argument("--games-per-profile", type=int, default=1)
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."}
            or args.games_per_profile < 1):
        parser.error("実験名またはゲーム数が不正です。")
    root = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if root.exists():
        parser.error("実験フォルダが既にあります。")
    champion = PPOAgent(args.model_id, device="cpu")
    robber = PPOAgent(args.robber_model_id, device="cpu")
    candidate = create_context_aware_experiment_agent(champion)
    result = verify_context_policy_games(
        champion, candidate, robber,
        seed_start=args.seed_start, games_per_profile=args.games_per_profile,
    )
    root.mkdir(parents=True)
    (root / "runtime_context_definition.json").write_text(
        json.dumps(runtime_context_definition(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "parity_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"experiment": args.experiment_id,
                      "games": result["game_count"],
                      "all_actions": result["all_actions"],
                      "context_dimension": result["context_dimension"],
                      "all_equal": result["all_equal"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
