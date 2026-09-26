"""初期配置head学習と、独立した初期配置専用ネットワークを比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.agents.ppo import PPOAgent
from app.domain.actions import Action
from app.domain.game import GameState

from .action_space import get_action_mask, id_to_action
from .analyze_supervised_setup import paired_effect
from .audit import runtime_versions, source_fingerprint
from .evaluate import EvaluationConfig, evaluate_agent
from .initial_setup_policy import InitialSetupPolicy, train_initial_setup_policy
from .observation import encode_observation, get_observation
from .pretrain_candidate_setup import pretrain_initial_setup_heads


class IndependentSetupAgent:
    """初期配置だけ専用方策、それ以外は既存PPOを使う実験用Agent。"""

    def __init__(self, ppo_agent: PPOAgent, setup_policy: InitialSetupPolicy):
        self.ppo_agent = ppo_agent
        self.setup_policy = setup_policy

    def select_action(self, game: GameState, player_id: int) -> Action:
        if game.phase != "setup_ready":
            return self.ppo_agent.select_action(game, player_id)
        version = self.ppo_agent.manifest.observation_version
        observation = encode_observation(get_observation(game, player_id, version=version))
        action_mask = get_action_mask(game, player_id)
        return id_to_action(self.setup_policy.select_action_id(observation, action_mask))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_hierarchical_50k_v001")
    parser.add_argument("--models-root", type=Path, default=None)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=73_001)
    parser.add_argument("--train-boards", type=int, default=600)
    parser.add_argument("--validation-boards", type=int, default=150)
    parser.add_argument("--test-boards", type=int, default=150)
    parser.add_argument("--teacher-seed-start", type=int, default=70_001)
    parser.add_argument("--training-seed", type=int, default=20260928)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。別IDを指定してください。")
    output.mkdir(parents=True)

    seeds = tuple(range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    definition = {
        "experiment_id": args.experiment_id,
        "source_model": args.model_id,
        "conditions": ["heuristic_setup", "setup_heads_only", "independent_setup_policy"],
        "evaluation_seeds": seeds,
        "evaluation_opponents": "heuristic_x3",
        "rotate_player_ids": True,
        "teacher_seed_start": args.teacher_seed_start,
        "teacher_boards": {
            "train": args.train_boards,
            "validation": args.validation_boards,
            "test": args.test_boards,
        },
        "source_sha256": source_fingerprint(),
        "runtime_versions": runtime_versions(),
    }
    _write(output / "definition.json", definition)

    baseline = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    if not baseline.heuristic_initial_placement:
        raise ValueError("比較元モデルはHeuristic初期配置モデルではありません。")

    heads_agent = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    heads_report = pretrain_initial_setup_heads(
        heads_agent.model.policy,
        train_boards=args.train_boards,
        validation_boards=args.validation_boards,
        test_boards=args.test_boards,
        seed_start=args.teacher_seed_start,
        training_seed=args.training_seed,
    )
    heads_agent.heuristic_initial_placement = False
    heads_agent.model.save(str(output / "setup_heads_only_model.zip"))

    adapter_base = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    setup_policy, adapter_report = train_initial_setup_policy(
        adapter_base.model.policy,
        train_boards=args.train_boards,
        validation_boards=args.validation_boards,
        test_boards=args.test_boards,
        seed_start=args.teacher_seed_start,
        training_seed=args.training_seed,
    )
    torch.save(setup_policy.state_dict(), output / "independent_setup_policy.pt")
    adapter_agent = IndependentSetupAgent(adapter_base, setup_policy)
    _write(output / "pretraining.json", {
        "setup_heads_only": heads_report,
        "independent_setup_policy": adapter_report,
    })

    agents = {
        "heuristic_setup": baseline,
        "setup_heads_only": heads_agent,
        "independent_setup_policy": adapter_agent,
    }
    reports = {}
    summaries = {}
    config = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                              policy_compatible_opponents=True, observation_version="v2")
    for name, agent in agents.items():
        report = evaluate_agent(agent, seeds, config=config,
                                agent_name=f"{args.model_id}:{name}").to_dict()
        summary = report["summary"]
        if summary["illegal_action_count"] or summary["truncated"]:
            raise AssertionError(f"非法Actionまたは打ち切り: {name}")
        reports[name] = report
        summaries[name] = summary
        _write(output / f"{name}.json", report)
        _write(output / "summary.json", summaries)
        print(json.dumps({
            "condition": name,
            "win_rate": summary["win_rate"],
            "average_score": summary["average_score"],
            "average_rank": summary["average_rank"],
            "average_initial_pips": summary["average_initial_pips"],
        }, ensure_ascii=False), flush=True)

    effects = {}
    for name in ("setup_heads_only", "independent_setup_policy"):
        effects[name] = {
            "win_rate": paired_effect(reports[name], reports["heuristic_setup"], "learner_won"),
            "average_score": paired_effect(
                reports[name], reports["heuristic_setup"], "learner_score"),
        }
    _write(output / "paired_effects.json", effects)


if __name__ == "__main__":
    main()
