"""同じ盤面seed群でPPOと既存AIを比較する。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from app.agents.heuristic import HeuristicAgent
from app.agents.base import Agent
from app.agents.random_agent import RandomAgent
from app.domain.actions import Action
from app.domain.game import GameState

from .action_space import legal_policy_actions
from .evaluate import EvaluationConfig, evaluate_agent
from .model_registry import ModelRegistry
from .policy_adapter import policy_compatible_action


@dataclass(frozen=True)
class PolicyCompatibleHeuristicAgent:
    """通常のHeuristicをPPO v1環境のダイス・破棄・交渉仕様へ合わせる。"""

    def select_action(self, game: GameState, player_id: int) -> Action:
        proposed = policy_compatible_action(HeuristicAgent().select_action(game, player_id))
        if proposed not in legal_policy_actions(game, player_id):
            raise ValueError(f"Heuristicの選択がPPO v1環境で不正です: {proposed}")
        return proposed


@dataclass(frozen=True)
class HeuristicSetupAgent:
    """学習済みPPOは維持し、初期配置だけ既存Heuristicに任せる診断用Agent。"""

    policy: Agent

    def select_action(self, game: GameState, player_id: int) -> Action:
        if game.phase == "setup_ready":
            return PolicyCompatibleHeuristicAgent().select_action(game, player_id)
        return self.policy.select_action(game, player_id)


def compare_agents(model_id: str, seeds: tuple[int, ...], *, models_root: Path | None = None,
                   learning_player_id: int = 1, opponent_controller: str = "heuristic",
                   setup_ablation: bool = False) -> dict:
    """比較条件・各試合・集計をJSON化可能なdictで返す。"""
    if not seeds:
        raise ValueError("seedを1つ以上指定してください。")
    from app.agents.ppo import PPOAgent

    registry = ModelRegistry(models_root)
    manifest = registry.load(model_id)
    config = EvaluationConfig(learning_player_id=learning_player_id,
                              opponent_controller=opponent_controller,
                              observation_version=manifest.observation_version)
    candidates = {
        "ppo": PPOAgent(model_id, models_root=registry.models_root),
        "heuristic": PolicyCompatibleHeuristicAgent(),
        "random": RandomAgent(),
    }
    if setup_ablation:
        candidates["ppo_heuristic_setup"] = HeuristicSetupAgent(candidates["ppo"])
    reports = {name: evaluate_agent(agent, seeds, config=config, agent_name=name).to_dict()
               for name, agent in candidates.items()}
    return {
        "model_id": model_id,
        "training_steps": manifest.training_steps,
        "seeds": list(seeds),
        "learning_player_id": learning_player_id,
        "opponent_controller": opponent_controller,
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="保存済みPPOを同一seedのHeuristic/Randomと比較する")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=8001)
    parser.add_argument("--learning-player-id", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--opponent", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--output-name", help="モデルフォルダ内に保存するJSONファイル名（省略時はcomparison.json）")
    parser.add_argument("--setup-ablation", action="store_true", help="PPOの初期配置だけHeuristicに代える診断を追加")
    values = parser.parse_args()
    if values.games <= 0:
        parser.error("--gamesは正の整数にしてください。")
    if values.output_name and (Path(values.output_name).name != values.output_name
                               or Path(values.output_name).suffix != ".json"):
        parser.error("--output-nameはパスを含まない.jsonファイル名にしてください。")
    report = compare_agents(values.model_id, tuple(range(values.seed_start, values.seed_start + values.games)),
                            learning_player_id=values.learning_player_id,
                            opponent_controller=values.opponent,
                            setup_ablation=values.setup_ablation)
    filename = values.output_name or ("comparison.json" if values.learning_player_id == 1
                                      else f"comparison_p{values.learning_player_id}.json")
    output = ModelRegistry().models_root / values.model_id / filename
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({name: result["summary"] for name, result in report["reports"].items()},
                     ensure_ascii=False, indent=2))
    print(f"結果: {output}")


if __name__ == "__main__":
    main()
