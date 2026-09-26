"""長時間学習をせず、保存済みPPOとBaselineを混合相手・座席回転で診断する。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import monotonic

from app.agents.ppo import PPOAgent
from app.agents.random_agent import RandomAgent

from .audit import model_audit, runtime_versions, source_fingerprint
from .compare import HeuristicSetupAgent, PolicyCompatibleHeuristicAgent
from .evaluate import EvaluationConfig, evaluate_agent
from .model_registry import ModelRegistry
from .observation import OBSERVATION_VERSION


def _save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--models", nargs="+", default=["ppo_heuristic_300k_v001", "ppo_heuristic_1m_v001"])
    parser.add_argument("--models-root", type=Path, default=None,
                        help="実験用モデルが標準models/以外にある場合の保存先")
    parser.add_argument("--setup-ablation-models", nargs="*", default=[],
                        help="指定モデルに対し、評価時だけ初期配置をHeuristicに替えた候補も追加")
    parser.add_argument("--setup-ablation-only", action="store_true",
                        help="指定モデル自体は再評価せず、初期配置を替えた候補だけ評価")
    parser.add_argument("--games", type=int, default=48)
    parser.add_argument("--seed-start", type=int, default=20001)
    parser.add_argument("--heuristic-counts", nargs="+", type=int, choices=range(4), default=[0, 1, 2, 3])
    parser.add_argument("--max-policy-steps", type=int, default=5000)
    parser.add_argument("--no-baselines", action="store_true",
                        help="Random/Heuristicの再評価を省き、指定したモデルだけを比較する")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    if args.games <= 0 or args.max_policy_steps <= 0:
        parser.error("games/max-policy-stepsは正の整数です。")
    if any(model_id not in args.models for model_id in args.setup_ablation_models):
        parser.error("--setup-ablation-modelsには--models内のモデルを指定してください。")
    if args.setup_ablation_only and set(args.setup_ablation_models) != set(args.models):
        parser.error("--setup-ablation-onlyでは全モデルを--setup-ablation-modelsへ指定してください。")
    registry = ModelRegistry(args.models_root)
    candidates = ({} if args.no_baselines else
                  {"heuristic": PolicyCompatibleHeuristicAgent(), "random": RandomAgent()})
    audits = {}
    candidate_versions = {name: OBSERVATION_VERSION for name in candidates}
    for model_id in args.models:
        agent = PPOAgent(model_id, models_root=registry.models_root, device="cpu")
        audit = model_audit(model_id, agent.model, registry)
        if not args.setup_ablation_only:
            candidates[model_id] = agent
            audits[model_id] = audit
            candidate_versions[model_id] = agent.manifest.observation_version
        if model_id in args.setup_ablation_models:
            ablation_id = f"{model_id}__inference_setup"
            candidates[ablation_id] = HeuristicSetupAgent(agent)
            audits[ablation_id] = audit
            candidate_versions[ablation_id] = agent.manifest.observation_version

    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    definition = {"experiment_id": args.experiment_id, "source_sha256": source_fingerprint(),
                  "seeds": list(range(args.seed_start, args.seed_start + args.games)),
                  "heuristic_counts": args.heuristic_counts, "models": args.models,
                  "setup_ablation_models": args.setup_ablation_models,
                  "setup_ablation_only": args.setup_ablation_only,
                  "candidate_observation_versions": candidate_versions,
                  "include_baselines": not args.no_baselines,
                  "model_sha256": {name: data["model_sha256"] for name, data in audits.items()},
                  "max_policy_steps": args.max_policy_steps,
                  "policy_compatible_opponents": True, "rotate_player_ids": True,
                  "deterministic_ppo": True, "reward_changed": False}
    if output.exists():
        if not args.resume:
            parser.error("実験フォルダが既にあります。別IDか--resumeを指定してください。")
        existing = json.loads((output / "definition.json").read_text(encoding="utf-8"))
        if existing != definition:
            parser.error("コード・モデル・条件が異なるため同じ実験には追記できません。")
    else:
        output.mkdir(parents=True)
        _save(output / "definition.json", definition)
        _save(output / "audit.json", {"created_at": datetime.now(timezone.utc).isoformat(),
                                      "versions": runtime_versions(), "models": audits})

    summaries = {}
    for count in args.heuristic_counts:
        for name, agent in candidates.items():
            config = EvaluationConfig(heuristic_opponents=count, rotate_player_ids=True,
                                      policy_compatible_opponents=True, max_policy_steps=args.max_policy_steps,
                                      observation_version=candidate_versions[name])
            cell_id = f"{name}_H{count}_R{3-count}"
            path = output / f"{cell_id}.json"
            if path.exists():
                report = json.loads(path.read_text(encoding="utf-8"))
            else:
                print(f"START {cell_id}: {args.games} games", flush=True)
                started = monotonic()
                report = evaluate_agent(agent, definition["seeds"], config=config, agent_name=name).to_dict()
                report.update({"experiment_id": args.experiment_id, "cell_id": cell_id,
                               "elapsed_seconds": monotonic() - started,
                               "model_audit_id": name if name in audits else None})
                _save(path, report)
            summaries[cell_id] = report["summary"]
            _save(output / "summary.json", summaries)
            summary = report["summary"]
            print(json.dumps({"cell": cell_id, **{key: summary[key] for key in
                              ("win_rate", "average_score", "average_rank", "truncated", "illegal_action_count")}},
                             ensure_ascii=False), flush=True)
    print(f"DONE {output}", flush=True)


if __name__ == "__main__":
    main()
