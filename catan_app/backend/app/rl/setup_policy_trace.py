"""PPOが初期開拓地候補に与える確率を、同じ盤面群で記録する。"""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

from app.agents.ppo import PPOAgent
from app.domain.actions import PlaceInitialSettlementAction

from .action_space import action_to_id, get_action_mask, id_to_action
from .audit import model_audit, runtime_versions, source_fingerprint
from .model_registry import ModelRegistry
from .observation import encode_observation, get_observation
from .setup_diagnostics import probe_setup, vertex_pips


class TracedSetupPPO:
    def __init__(self, agent: PPOAgent):
        self.agent = agent
        self.decisions: list[dict] = []
        self._settlement_number: dict[int, int] = {}

    def select_action(self, game, player_id):
        action = self.agent.select_action(game, player_id)
        if not isinstance(action, PlaceInitialSettlementAction):
            return action
        mask = get_action_mask(game, player_id)
        observation = encode_observation(get_observation(game, player_id,
                                                        version=self.agent.manifest.observation_version))
        tensor, _ = self.agent.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.agent.model.policy.get_distribution(tensor, action_masks=mask)
            probabilities = distribution.distribution.probs.cpu().numpy().reshape(-1)
        candidates = [(index, id_to_action(int(index)).vertex_id) for index in np.flatnonzero(mask)]
        if not candidates or not all(isinstance(id_to_action(int(index)), PlaceInitialSettlementAction)
                                     for index, _ in candidates):
            raise AssertionError("初期開拓地選択時のAction Maskが想定外です。")
        candidate_pips = [(index, vertex, vertex_pips(game.board, vertex)) for index, vertex in candidates]
        best_pips = max(pips for _, _, pips in candidate_pips)
        positive = probabilities[mask]
        positive = positive[positive > 0]
        entropy = float(-(positive * np.log(positive)).sum())
        top = sorted(candidate_pips, key=lambda item: float(probabilities[item[0]]), reverse=True)[:5]
        number = self._settlement_number.get(game.seed, 0) + 1
        self._settlement_number[game.seed] = number
        self.decisions.append({
            "seed": game.seed, "player_id": player_id, "placement_number": number,
            "selected_vertex_id": action.vertex_id,
            "selected_pips": vertex_pips(game.board, action.vertex_id),
            "best_legal_pips": best_pips,
            "legal_count": len(candidates),
            "selected_probability": float(probabilities[action_to_id(action)]),
            "max_probability": float(max(probabilities[index] for index, _ in candidates)),
            "normalized_entropy": entropy / float(np.log(len(candidates))) if len(candidates) > 1 else 0.0,
            "probability_mass_on_best_pips": float(sum(probabilities[index] for index, _, pips
                                                        in candidate_pips if pips == best_pips)),
            "policy_expected_pips": float(sum(probabilities[index] * pips for index, _, pips
                                               in candidate_pips)),
            "top_candidates": [{"vertex_id": vertex, "pips": pips,
                                "probability": float(probabilities[index])} for index, vertex, pips in top],
        })
        return action


def summarize(decisions: list[dict]) -> dict:
    result = {}
    for number in (1, 2):
        rows = [row for row in decisions if row["placement_number"] == number]
        if not rows:
            raise ValueError(f"{number}軒目の記録がありません。")
        counts = Counter(row["selected_vertex_id"] for row in rows)
        result[str(number)] = {
            "decisions": len(rows),
            "selected_vertex_unique": len(counts),
            "selected_vertex_most_common": {"vertex_id": counts.most_common(1)[0][0],
                                            "share": counts.most_common(1)[0][1] / len(rows)},
            "mean_selected_probability": fmean(row["selected_probability"] for row in rows),
            "mean_max_probability": fmean(row["max_probability"] for row in rows),
            "mean_normalized_entropy": fmean(row["normalized_entropy"] for row in rows),
            "mean_probability_mass_on_best_pips": fmean(row["probability_mass_on_best_pips"] for row in rows),
            "mean_policy_expected_pips": fmean(row["policy_expected_pips"] for row in rows),
            "mean_best_legal_pips": fmean(row["best_legal_pips"] for row in rows),
            "mean_selected_pips": fmean(row["selected_pips"] for row in rows),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=26001)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    if args.games <= 0:
        parser.error("gamesは正の整数です。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。別IDを指定してください。")
    registry = ModelRegistry()
    traces = {}
    audits = {}
    for model_id in args.models:
        agent = PPOAgent(model_id, models_root=registry.models_root, device="cpu")
        if agent.heuristic_initial_placement:
            parser.error("PPOが自分で初期配置を選ぶモデルだけを指定してください。")
        traced = TracedSetupPPO(agent)
        for index, seed in enumerate(range(args.seed_start, args.seed_start + args.games)):
            probe_setup(traced, seed, index, observation_version=agent.manifest.observation_version)
        traces[model_id] = {"summary": summarize(traced.decisions), "decisions": traced.decisions}
        audits[model_id] = model_audit(model_id, agent.model, registry)
        print(json.dumps({"model": model_id, **traces[model_id]["summary"]}, ensure_ascii=False), flush=True)
    output.mkdir(parents=True)
    definition = {"experiment_id": args.experiment_id, "models": args.models,
                  "seeds": list(range(args.seed_start, args.seed_start + args.games)),
                  "source_sha256": source_fingerprint(), "runtime_versions": runtime_versions(),
                  "model_sha256": {name: data["model_sha256"] for name, data in audits.items()},
                  "opponents": "heuristic_x3", "rotate_player_ids": True, "setup_only": True}
    (output / "definition.json").write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")
    for model_id, trace in traces.items():
        (output / f"{model_id}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({model_id: trace["summary"] for model_id, trace
                                                     in traces.items()}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
