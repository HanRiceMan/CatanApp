"""同じ10k条件に教師あり初期配置ウォームスタートだけを追加する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.ppo import PPOAgent

from .evaluate import EvaluationConfig, evaluate_agent
from .setup_diagnostics import probe_setup, summarize
from .train import TrainConfig, train_maskable_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="candidate_ab_10k_s20260923")
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    root = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    definition = json.loads((root / "definition.json").read_text(encoding="utf-8"))
    steps = definition["training_steps"]
    seed = definition["training_seed"]
    model_id = f"candidate_pretrained_{steps}_s{seed}"
    if (root / "models" / model_id).exists():
        parser.error("同名モデルが既にあります。再学習して上書きしません。")
    config = TrainConfig(model_id=model_id, models_root=root / "models", total_timesteps=steps,
                         seed=seed, observation_version="v2", policy_architecture="candidate",
                         pretrain_initial_settlement=True,
                         n_steps=definition["n_steps"], batch_size=definition["batch_size"],
                         n_epochs=definition["n_epochs"], learning_rate=definition["learning_rate"],
                         gamma=definition["gamma"], gae_lambda=definition["gae_lambda"],
                         ent_coef=definition["ent_coef"], checkpoint_interval_steps=max(256, steps // 2),
                         evaluation_seeds=tuple(range(definition["evaluation_seeds"][0] + 1000,
                                                      definition["evaluation_seeds"][0] + 1004)))
    train_maskable_ppo(config)
    agent = PPOAgent(model_id, models_root=root / "models", device="cpu")
    seeds = definition["evaluation_seeds"]
    setup_rows = [probe_setup(agent, board_seed, index, observation_version="v2")
                  for index, board_seed in enumerate(seeds)]
    setup_summary = summarize(setup_rows)
    (root / "candidate_pretrained_setup.json").write_text(
        json.dumps({"summary": setup_summary, "episodes": setup_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    summaries = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    summaries["candidate_pretrained_setup"] = setup_summary
    print(json.dumps({"architecture": "candidate_pretrained", "setup_pips": setup_summary["total_pips"],
                      "first_vertex_share": setup_summary["settlement_most_common"][0]["share"]},
                     ensure_ascii=False), flush=True)
    for heuristic_count in (3, 1):
        report = evaluate_agent(agent, seeds,
                                config=EvaluationConfig(heuristic_opponents=heuristic_count,
                                                        rotate_player_ids=True,
                                                        policy_compatible_opponents=True,
                                                        observation_version="v2"),
                                agent_name=model_id).to_dict()
        key = f"candidate_pretrained_H{heuristic_count}_R{3-heuristic_count}"
        (root / f"{key}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries[key] = report["summary"]
        (root / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"cell": key, "win_rate": report["summary"]["win_rate"],
                          "average_score": report["summary"]["average_score"],
                          "illegal_action_count": report["summary"]["illegal_action_count"],
                          "truncated": report["summary"]["truncated"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
