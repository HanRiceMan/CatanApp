"""PPOの合法手上の選択確率をオフライン記録する。UIや学習には影響しない。"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

from app.agents.ppo import PPOAgent
from app.domain.game import player_for

from .action_space import ACTION_CATALOG, action_to_id, get_action_mask
from .audit import source_fingerprint
from .evaluate import EvaluationConfig, evaluate_agent
from .observation import encode_observation, get_observation
from .policy_selection import action_family


class TracedPPO:
    def __init__(self, agent: PPOAgent, comparison_model=None):
        self.agent = agent
        self.comparison_model = comparison_model
        self.decisions: list[dict] = []

    @staticmethod
    def _probabilities(model, observation, mask):
        tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = model.policy.get_distribution(tensor, action_masks=mask)
            return distribution.distribution.probs.cpu().numpy().reshape(-1)

    def select_action(self, game, player_id):
        action = self.agent.select_action(game, player_id)
        action_id = action_to_id(action)
        mask = get_action_mask(game, player_id)
        observation = encode_observation(get_observation(game, player_id,
                                                        version=self.agent.manifest.observation_version))
        probabilities = self._probabilities(self.agent.model, observation, mask)
        legal = np.flatnonzero(mask)
        p = probabilities[legal]
        positive = p[p > 0]
        entropy = float(-(positive * np.log(positive)).sum())
        top = sorted(legal, key=lambda i: float(probabilities[i]), reverse=True)[:5]
        family_ids = defaultdict(list)
        for index in legal:
            family_ids[action_family(ACTION_CATALOG[int(index)].action)].append(int(index))
        road_ids = family_ids["road"]
        family_probability_mass = {family: float(sum(probabilities[index] for index in ids))
                                   for family, ids in family_ids.items() if ids}
        family_max_probability = {family: float(max(probabilities[index] for index in ids))
                                  for family, ids in family_ids.items() if ids}
        comparison = {}
        if self.comparison_model is not None:
            other = self._probabilities(self.comparison_model, observation, mask)
            other_choice = max(legal, key=lambda index: float(other[index]))
            comparison = {
                "comparison_selected_family": action_family(ACTION_CATALOG[int(other_choice)].action),
                "comparison_road_probability_mass": float(sum(other[index] for index in road_ids)),
                "comparison_family_probability_mass": {
                    family: float(sum(other[index] for index in ids))
                    for family, ids in family_ids.items() if ids
                },
            }
        self.decisions.append({
            "seed": game.seed, "player_id": player_id, "turn": game.turn_number,
            "phase": game.phase, "legal_count": len(legal),
            "entropy": entropy,
            "normalized_entropy": entropy / float(np.log(len(legal))) if len(legal) > 1 else None,
            "selected": ACTION_CATALOG[action_id].name,
            "selected_family": action_family(action),
            "selected_probability": float(probabilities[action_id]),
            "paid_road_legal_count": len(road_ids) if game.phase == "action" and not game.free_road_remaining else 0,
            "road_probability_mass": float(sum(probabilities[index] for index in road_ids)),
            "family_legal_count": {family: len(ids) for family, ids in family_ids.items() if ids},
            "family_probability_mass": family_probability_mass,
            "family_max_probability": family_max_probability,
            **comparison,
            "own_resources": dict(player_for(game, player_id).resources),
            "top_legal_actions": [{"action": ACTION_CATALOG[int(i)].name, "probability": float(probabilities[i])} for i in top],
        })
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=20001)
    parser.add_argument("--models-root", type=Path, default=None,
                        help="実験用モデルが標準models/以外にある場合の保存先")
    parser.add_argument("--checkpoint-steps", type=int, default=None,
                        help="完成モデルの代わりに指定stepの保存済みチェックポイントを診断する")
    parser.add_argument("--compare-checkpoint-steps", type=int, default=None,
                        help="同一局面を別stepのチェックポイントにも入力し、反実仮想の選択を記録する")
    parser.add_argument("--stochastic", action="store_true",
                        help="決定論的argmaxの代わりに保存方策の分布からActionを標本抽出する")
    parser.add_argument("--policy-seed", type=int, default=20260923,
                        help="確率的Action標本抽出の再現用seed")
    args = parser.parse_args()
    if args.games <= 0 or Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("gamesとexperiment-idを確認してください。")
    if any(steps is not None and steps <= 0
           for steps in (args.checkpoint_steps, args.compare_checkpoint_steps)):
        parser.error("チェックポイントstep数は正の整数です。")
    if args.compare_checkpoint_steps is not None and (args.checkpoint_steps is None or args.stochastic):
        parser.error("同一局面比較は--checkpoint-stepsを指定した決定論的実行のみ対応です。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダは既に存在します。別IDを指定してください。")
    agent = TracedPPO(PPOAgent(args.model_id, models_root=args.models_root,
                              deterministic=not args.stochastic, device="cpu"))
    def load_checkpoint(steps):
        from sb3_contrib import MaskablePPO
        checkpoint = (agent.agent.registry.models_root / args.model_id
                      / f"checkpoint_{steps}_steps.zip")
        if not checkpoint.is_file():
            parser.error(f"チェックポイントが見つかりません: {checkpoint}")
        model = MaskablePPO.load(str(checkpoint), device="cpu")
        if model.num_timesteps != steps:
            parser.error("チェックポイント内のstep数が指定値と一致しません。")
        return model
    if args.checkpoint_steps is not None:
        agent.agent.model = load_checkpoint(args.checkpoint_steps)
    if args.compare_checkpoint_steps is not None:
        agent.comparison_model = load_checkpoint(args.compare_checkpoint_steps)
    if args.stochastic:
        agent.agent.model.set_random_seed(args.policy_seed)
    report = evaluate_agent(agent, range(args.seed_start, args.seed_start + args.games),
                            config=EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                                                    policy_compatible_opponents=True,
                                                    observation_version=agent.agent.manifest.observation_version),
                            agent_name=args.model_id)
    groups = defaultdict(list)
    for item in agent.decisions:
        if item["legal_count"] > 1:
            groups[item["phase"]].append(item)
    summary = {phase: {"decisions": len(items),
                       "mean_normalized_entropy": fmean(x["normalized_entropy"] for x in items),
                       "mean_max_probability": fmean(x["selected_probability"] for x in items),
                       "probability_over_99pct_rate": fmean(x["selected_probability"] >= .99 for x in items)}
               for phase, items in groups.items()}
    paid_road_opportunities = [item for item in agent.decisions if item["paid_road_legal_count"] > 0]
    summary["paid_road_opportunities"] = {
        "decisions": len(paid_road_opportunities),
        "selected_rate": (fmean(item["selected"].startswith("build_road")
                                for item in paid_road_opportunities) if paid_road_opportunities else None),
        "mean_road_probability_mass": (fmean(item["road_probability_mass"]
                                            for item in paid_road_opportunities) if paid_road_opportunities else None),
        "mean_max_road_probability": (fmean(item["family_max_probability"]["road"]
                                            for item in paid_road_opportunities) if paid_road_opportunities else None),
        "selected_families": dict(Counter(item["selected_family"] for item in paid_road_opportunities)),
        "mean_family_probability_mass": {
            family: fmean(item["family_probability_mass"].get(family, 0.0)
                          for item in paid_road_opportunities)
            for family in ("road", "settlement", "city", "development_purchase",
                           "bank_trade", "development_use", "end_turn")
        } if paid_road_opportunities else {},
    }
    if agent.comparison_model is not None:
        summary["same_state_comparison"] = {
            "paid_road_decisions": len(paid_road_opportunities),
            "reference_argmax_road_rate": summary["paid_road_opportunities"]["selected_rate"],
            "comparison_argmax_road_rate": (
                fmean(item["comparison_selected_family"] == "road" for item in paid_road_opportunities)
                if paid_road_opportunities else None),
            "reference_mean_road_probability_mass": summary["paid_road_opportunities"]["mean_road_probability_mass"],
            "comparison_mean_road_probability_mass": (
                fmean(item["comparison_road_probability_mass"] for item in paid_road_opportunities)
                if paid_road_opportunities else None),
            "comparison_selected_families": dict(Counter(
                item["comparison_selected_family"] for item in paid_road_opportunities)),
        }
    result = {"experiment_id": args.experiment_id, "model_id": args.model_id,
              "action_selection": "stochastic" if args.stochastic else "deterministic",
              "policy_seed": args.policy_seed if args.stochastic else None,
              "checkpoint_steps": args.checkpoint_steps,
              "compare_checkpoint_steps": args.compare_checkpoint_steps,
              "source_sha256": source_fingerprint(), "training_steps": agent.agent.model.num_timesteps,
              "summary_multi_choice_only": summary, "report": report.to_dict(), "decisions": agent.decisions}
    output.mkdir(parents=True)
    (output / "trace.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"DONE {output}")


if __name__ == "__main__":
    main()
