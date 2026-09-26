"""同じランキング教師で初期配置MLPと小規模GNNを比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.agents.ppo import PPOAgent

from .audit import runtime_versions, source_fingerprint
from .gnn_setup_policy import train_graph_initial_setup_policy
from .initial_setup_policy import train_ranked_initial_setup_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-id", default="ppo_hierarchical_50k_v001")
    parser.add_argument("--models-root", type=Path, default=None)
    parser.add_argument("--train-boards", type=int, default=600)
    parser.add_argument("--validation-boards", type=int, default=300)
    parser.add_argument("--test-boards", type=int, default=500)
    parser.add_argument("--train-seed-start", type=int, default=100_001)
    parser.add_argument("--validation-seed-start", type=int, default=106_001)
    parser.add_argument("--test-seed-start", type=int, default=106_501)
    parser.add_argument("--training-seed", type=int, default=20260928)
    parser.add_argument("--temperature", type=float, default=5.0)
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。別IDを指定してください。")
    output.mkdir(parents=True)
    definition = {**vars(args), "models_root": str(args.models_root) if args.models_root else None,
                  "source_sha256": source_fingerprint(), "runtime_versions": runtime_versions()}
    (output / "definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")

    mlp_agent = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    mlp, mlp_report = train_ranked_initial_setup_policy(
        mlp_agent.model.policy, train_boards=args.train_boards,
        validation_boards=args.validation_boards, test_boards=args.test_boards,
        seed_start=args.train_seed_start, validation_seed_start=args.validation_seed_start,
        test_seed_start=args.test_seed_start, training_seed=args.training_seed,
        temperature=args.temperature)
    gnn_agent = PPOAgent(args.model_id, models_root=args.models_root, device="cpu")
    gnn, gnn_report = train_graph_initial_setup_policy(
        gnn_agent.model.policy, train_boards=args.train_boards,
        validation_boards=args.validation_boards, test_boards=args.test_boards,
        train_seed_start=args.train_seed_start, validation_seed_start=args.validation_seed_start,
        test_seed_start=args.test_seed_start, training_seed=args.training_seed,
        temperature=args.temperature)
    torch.save(mlp.state_dict(), output / "mlp_policy.pt")
    torch.save(gnn.state_dict(), output / "gnn_policy.pt")
    report = {"mlp": mlp_report, "gnn": gnn_report}
    (output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: values["after_test"] for name, values in report.items()},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
