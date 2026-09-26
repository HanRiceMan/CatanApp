"""教師あり開拓地方策を自律配置・完走対局で評価する（通常UIには接続しない）。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from app.domain.actions import Action, PlaceInitialSettlementAction
from app.domain.game import GameState, legal_settlement_ids

from .audit import runtime_versions, source_fingerprint
from .compare import PolicyCompatibleHeuristicAgent
from .evaluate import EvaluationConfig, evaluate_agent
from .observation import encode_observation, get_observation
from .setup_diagnostics import GreedyPipsSettlementAgent, probe_setup, summarize
from .setup_supervised import candidate_features, make_model


class SupervisedSetupAgent:
    """初期開拓地だけ診断モデルを使い、道路と通常ターンは既存Heuristicへ戻す。"""

    def __init__(self, model_file: Path, architecture: str):
        self.architecture = architecture
        self.model = make_model(architecture)
        self.model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
        self.model.eval()
        self.fallback = PolicyCompatibleHeuristicAgent()

    def select_action(self, game: GameState, player_id: int) -> Action:
        if game.phase != "setup_ready" or game.pending_settlement_id is not None:
            return self.fallback.select_action(game, player_id)
        if self.architecture == "shared":
            placement = 1 + sum(owner == player_id for owner in game.settlements.values())
            features = torch.from_numpy(candidate_features(game, player_id, placement)[None, ...])
        else:
            observation = get_observation(game, player_id, version="v2")
            features = torch.from_numpy(encode_observation(observation)[None, ...])
        legal = legal_settlement_ids(game)
        if not legal:
            raise AssertionError("初期開拓地の合法候補がありません。")
        with torch.no_grad():
            logits = self.model(features).squeeze(0).numpy()
        mask = np.full(len(logits), -np.inf, dtype=np.float32)
        mask[legal] = 0
        return PlaceInitialSettlementAction(int(np.argmax(logits + mask)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=32001)
    parser.add_argument("--heuristic-counts", nargs="+", type=int, choices=range(4), default=[3])
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    if args.games <= 0:
        parser.error("gamesは正の整数です。")
    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。別IDを指定してください。")
    model_specs = {
        "supervised_mlp": (root / "experiments" / "setup_supervised_mlp_recheck_20260922" /
                           "diagnostic_model.pt", "mlp"),
        "supervised_shared": (root / "experiments" / "setup_supervised_shared_20260922" /
                              "diagnostic_model.pt", "shared"),
    }
    agents = {name: SupervisedSetupAgent(path, architecture)
              for name, (path, architecture) in model_specs.items()}
    agents["heuristic"] = PolicyCompatibleHeuristicAgent()
    agents["greedy_pips"] = GreedyPipsSettlementAgent()
    seeds = list(range(args.seed_start, args.seed_start + args.games))
    output.mkdir(parents=True)
    definition = {"experiment_id": args.experiment_id, "seeds": seeds,
                  "training_seed_ranges": [[31001, 31600], [31601, 31750], [31751, 31900]],
                  "heuristic_counts": args.heuristic_counts, "rotate_player_ids": True,
                  "policy_compatible_opponents": True, "source_sha256": source_fingerprint(),
                  "runtime_versions": runtime_versions(),
                  "model_sha256": {name: _sha256(path) for name, (path, _) in model_specs.items()},
                  "agents": list(agents), "setup_road_policy": "heuristic",
                  "normal_turn_policy": "heuristic"}
    (output / "definition.json").write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")
    setup_summary = {}
    for name, agent in agents.items():
        rows = [probe_setup(agent, seed, index, observation_version="v2")
                for index, seed in enumerate(seeds)]
        summary = summarize(rows)
        (output / f"{name}_setup.json").write_text(
            json.dumps({"summary": summary, "episodes": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        setup_summary[name] = summary
        (output / "setup_summary.json").write_text(json.dumps(setup_summary, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
        print(json.dumps({"setup": name, "total_pips": summary["total_pips"],
                          "resource_kinds": summary["resource_kinds"]}, ensure_ascii=False), flush=True)
    results = {}
    for count in args.heuristic_counts:
        config = EvaluationConfig(heuristic_opponents=count, rotate_player_ids=True,
                                  policy_compatible_opponents=True, observation_version="v2")
        for name, agent in agents.items():
            cell = f"{name}_H{count}_R{3-count}"
            report = evaluate_agent(agent, seeds, config=config, agent_name=name).to_dict()
            (output / f"{cell}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            results[cell] = report["summary"]
            (output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            summary = report["summary"]
            print(json.dumps({"cell": cell, **{key: summary[key] for key in
                              ("win_rate", "average_score", "average_rank", "illegal_action_count", "truncated")}},
                             ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
