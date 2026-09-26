"""道路v4を固定した発展購入専用残差の事前調整と、同一盤面での比較。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from app.agents.ppo import PPOAgent
from app.domain.actions import BuyDevelopmentAction
from .action_space import action_to_id
from .audit import source_fingerprint
from .development_budget import DevelopmentBudgetFeatures
from .development_policy import (DevelopmentBudgetMaskablePolicy, DEVELOPMENT_LOGIT,
                                  configure_development_only, initialize_development_policy)
from .audit_development_policy import audit as audit_policy
from .env import CatanEnv, CatanEnvConfig
from .evaluate import EvaluationConfig, evaluate_agent
from .gnn_setup_policy import GraphInitialSetupAgent


DEV_ID = action_to_id(BuyDevelopmentAction())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def collect(source, seeds):
    observations, masks, bought = [], [], []
    for index, seed in enumerate(seeds):
        pid = index % 4 + 1
        env = CatanEnv(CatanEnvConfig(
            learning_player_id=pid, policy_compatible_opponents=True, observation_version="v2",
        ), initial_placement_agent=GraphInitialSetupAgent(source.initial_setup_policy))
        try:
            observation, info = env.reset(seed=seed)
            while True:
                action = source.select_action(env.game, pid)
                action_id = action_to_id(action)
                if info["action_mask"][DEV_ID]:
                    observations.append(observation.copy())
                    masks.append(info["action_mask"].copy())
                    bought.append(action_id == DEV_ID)
                observation, _, terminated, truncated, info = env.step(action_id)
                if truncated:
                    raise AssertionError(f"収集が打ち切られました: {seed}")
                if terminated:
                    break
        finally:
            env.close()
    if not observations:
        raise ValueError("発展購入可能局面がありません。")
    data = {"observations": torch.as_tensor(np.stack(observations)),
            "masks": torch.as_tensor(np.stack(masks)), "bought": torch.tensor(bought)}
    with torch.no_grad():
        data.update(DevelopmentBudgetFeatures()(data["observations"]))
    return data


def audit_data(data):
    selected = data["bought"]
    names = ("reserve", "connected", "army_exception", "robber_exception", "endgame_exception", "stalled")
    return {
        "opportunities": len(selected), "actual_purchases": int(selected.sum()),
        "opportunity_counts": {name: int(data[name].sum()) for name in names},
        "purchase_counts": {name: int((data[name] & selected).sum()) for name in names},
        "mean_predicted_delay_when_buying": float(data["delay"][selected].mean()) if selected.any() else None,
    }


def train(model, datasets, *, epochs, seed):
    torch.manual_seed(seed)
    trainable = configure_development_only(model.policy)
    actor = model.policy.mlp_extractor
    optimizer = torch.optim.Adam(actor.development_head.parameters(), lr=1e-3)
    training, validation, test = datasets
    # delayの大小を専用headへ教える。建設機会を奪う時だけoddsを1/2〜1/10にする。
    def target(data):
        return -(data["delay"] * 0.75).clamp(np.log(2), np.log(10))

    def loss(data):
        # clamp境界0で勾配が止まるruntimeでも学習できるよう、負の教師へrawを回帰。
        # 対局時の安全な補正範囲はdevelopment_correction()で適用する。
        predicted = actor.development_head(data["features"]).squeeze(1)
        keep = data["reserve"]
        if not keep.any():
            raise ValueError("資源温存対象局面を収集できませんでした。")
        return (predicted[keep] - target(data)[keep]).square().mean()

    history = []
    best = {key: value.clone() for key, value in actor.development_head.state_dict().items()}
    with torch.no_grad():
        initial = float(loss(validation))
    best_loss, best_epoch = initial, 0
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        training_loss = loss(training)
        training_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.development_head.parameters(), 1)
        optimizer.step()
        with torch.no_grad():
            validation_loss = float(loss(validation))
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch = validation_loss, epoch
            best = {key: value.clone() for key, value in actor.development_head.state_dict().items()}
        history.append({"epoch": epoch, "train_mse": float(training_loss.detach()),
                        "validation_mse": validation_loss})
    actor.development_head.load_state_dict(best)
    with torch.no_grad():
        test_loss = float(loss(test))
    return {"trainable_parameters": trainable, "initial_validation_mse": initial,
            "best_validation_mse": best_loss, "best_epoch": best_epoch,
            "test_mse": test_loss, "history": history}


def verify_frozen(source, candidate, data):
    old = source.policy.state_dict()
    new = candidate.policy.state_dict()
    changed = [key for key, value in old.items() if not torch.equal(value, new[key])]
    if changed:
        raise AssertionError(f"既存の重みが変わりました: {changed}")
    observation = data["observations"][:128]
    with torch.no_grad():
        a = source.policy.mlp_extractor.forward_actor(observation)
        b = candidate.policy.mlp_extractor.forward_actor(observation)
    keep = torch.arange(a.shape[1]) != DEVELOPMENT_LOGIT
    difference = float((a[:, keep] - b[:, keep]).abs().max())
    if difference != 0:
        raise AssertionError(f"発展購入以外のlogitが変わりました: {difference}")
    return {"all_source_tensors_identical": True, "non_development_logit_max_diff": difference}


def summary(report):
    s = report["summary"]
    a, o, t = s["average_actions"], s["average_opportunities"], s["average_strategy"]
    episodes = report["episodes"]
    reached = [e for e in episodes if e["strategy"].get("third_site_reached", 0)]
    return {
        "games": len(episodes), "wins": sum(e["learner_won"] for e in episodes),
        "win_rate": s["win_rate"], "score": s["average_score"], "rank": s["average_rank"],
        "settlements": a.get("build_settlement", 0), "cities": a.get("build_city", 0),
        "paid_roads": a.get("build_road_paid", 0), "development": a.get("buy_development", 0),
        "third_site_rate": len(reached) / len(episodes),
        "third_site_turn_if_reached": (float(np.mean([e["strategy"]["third_site_own_turn"]
                                                     for e in reached])) if reached else None),
        "third_site_by_turn20_rate": sum(e["strategy"]["third_site_own_turn"] <= 20
                                         for e in reached) / len(episodes),
        "largest_army_rate": s["largest_army_acquisition_rate"],
        "longest_road_rate": s["longest_road_acquisition_rate"],
        "dev_consumes_settlement_reserve": o.get("development_consumes_settlement_reserve", 0),
        "dev_consumes_city_reserve": o.get("development_consumes_city_reserve", 0),
        "dev_settlement_deficit_increase": t.get("development_settlement_deficit_increase", 0),
        "dev_city_deficit_increase": t.get("development_city_deficit_increase", 0),
        "road_settlement_deficit_increase": t.get("paid_road_settlement_deficit_increase", 0),
        "road_while_planner_waits": o.get("build_road_while_planner_waits", 0),
        "connected_turns": o.get("connected_site_turns", 0),
        "connected_turns_missing_wood_brick": o.get("connected_site_turns_missing_wood_brick", 0),
        "connected_turns_missing_only_sheep_wheat": o.get("connected_site_turns_missing_only_sheep_wheat", 0),
        "own_turns": t.get("own_turns_observed", 0),
    }


def paired_comparison(reference, candidate, *, seed=9182):
    # 盤面seed・学習者座席を対応させて再標本化。独立二標本の誤差扱いをしない。
    first, second = reference["episodes"], candidate["episodes"]
    if [(e["seed"], e["learner_id"]) for e in first] != [(e["seed"], e["learner_id"]) for e in second]:
        raise AssertionError("比較盤面または座席が一致していません。")
    extract = {
        "win_rate": lambda e: float(e["learner_won"]),
        "score": lambda e: e["learner_score"],
        "settlements": lambda e: e["learner_actions"].get("build_settlement", 0),
        "cities": lambda e: e["learner_actions"].get("build_city", 0),
        "development": lambda e: e["learner_actions"].get("buy_development", 0),
        "third_site_rate": lambda e: e["strategy"].get("third_site_reached", 0),
    }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(first), size=(4000, len(first)))
    result = {}
    for name, extract_one in extract.items():
        delta = np.array([extract_one(b) - extract_one(a) for a, b in zip(first, second)])
        interval = np.quantile(delta[indices].mean(1), [.025, .975])
        result[name] = {"difference": float(delta.mean()), "paired_bootstrap_95_ci": interval.tolist()}
    return result


def evaluate(output, agents, seeds):
    reports, summaries = {}, {}
    config = EvaluationConfig(heuristic_opponents=3, rotate_player_ids=True,
                              policy_compatible_opponents=True, observation_version="v2")
    for name, agent in agents.items():
        report = evaluate_agent(agent, seeds, config=config, agent_name=name).to_dict()
        if report["summary"]["illegal_action_count"] or report["summary"]["truncated"]:
            raise AssertionError(f"{name}: 評価が正常に完了していません。")
        write(output / f"evaluation_{name}.json", report)
        reports[name], summaries[name] = report, summary(report)
        print(json.dumps({"stage": name, **summaries[name]}, ensure_ascii=False), flush=True)
    comparisons = {name: paired_comparison(reports["road_v4"], report)
                   for name, report in reports.items() if name != "road_v4"}
    write(output / "evaluation_summary.json", {"summaries": summaries,
                                               "paired_vs_road_v4": comparisons})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source-checkpoint", type=Path, default=Path(
        "experiments/expansion_strategy_soft_plan_v4_s01_20260924/checkpoint_normal.zip"))
    parser.add_argument("--model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--train-games", type=int, default=80)
    parser.add_argument("--validation-games", type=int, default=20)
    parser.add_argument("--test-games", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=360001)
    parser.add_argument("--training-seed", type=int, default=20261751)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-seed-start", type=int, default=370001)
    parser.add_argument("--data-from", type=Path, help="同じsourceとseedの収集済み局面を再利用")
    parser.add_argument("--holdout-checkpoint", type=Path,
                        help="追加学習せず保存候補をoriginal・road_v4と比較する")
    parser.add_argument("--candidate-strengths", type=float, nargs="+", default=[0.5, 1.0],
                        help="温存対象局面で発展購入logitへ掛ける補正倍率")
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}
            or min(args.train_games, args.validation_games, args.test_games,
                   args.epochs, args.eval_games, *args.candidate_strengths) <= 0):
        parser.error("実験名・ゲーム数・epoch数を確認してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.training_seed)
    original = PPOAgent(args.model_id, device="cpu")
    source = PPOAgent(args.model_id, device="cpu")
    source.model = MaskablePPO.load(str(args.source_checkpoint), device="cpu")
    definition = {**vars(args), "source_fingerprint": source_fingerprint(),
                  "source_checkpoint_sha256": sha256(args.source_checkpoint.read_bytes()).hexdigest(),
                  "assumptions": "expected production; surplus-only bank trades; 3-road horizon; no discard/future bank prediction",
                  "policy_update": "development head only; no road or critic update"}
    definition = {key: str(value) if isinstance(value, Path) else value for key, value in definition.items()}
    write(output / "definition.json", definition)
    if args.holdout_checkpoint:
        candidate = PPOAgent(args.model_id, device="cpu")
        candidate.model = MaskablePPO.load(str(args.holdout_checkpoint), device="cpu")
        evaluate(output, {"original": original, "road_v4": source, "development": candidate},
                 range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
        return

    datasets = []
    start = args.seed_start
    if args.data_from:
        previous = json.loads((args.data_from / "definition.json").read_text(encoding="utf-8"))
        for key in ("source_checkpoint_sha256", "seed_start", "train_games", "validation_games", "test_games"):
            if previous[key] != definition[key]:
                raise ValueError(f"再利用データの条件が違います: {key}")
    for name, games in (("train", args.train_games), ("validation", args.validation_games),
                        ("test", args.test_games)):
        if args.data_from:
            saved = torch.load(args.data_from / f"data_{name}.pt", weights_only=True)
            data = {key: saved[key] for key in ("observations", "masks", "bought")}
            with torch.no_grad():
                data.update(DevelopmentBudgetFeatures()(data["observations"]))
        else:
            data = collect(source, range(start, start + games))
        torch.save(data, output / f"data_{name}.pt")
        audit = audit_data(data)
        write(output / f"audit_{name}.json", audit)
        print(json.dumps({"stage": name, **audit}, ensure_ascii=False), flush=True)
        datasets.append(data)
        start += games
    env = CatanEnv(CatanEnvConfig(observation_version="v2"))
    base = source.model
    model = MaskablePPO(DevelopmentBudgetMaskablePolicy, env, device="cpu",
                        n_steps=base.n_steps, batch_size=base.batch_size, n_epochs=base.n_epochs,
                        gamma=base.gamma, gae_lambda=base.gae_lambda,
                        policy_kwargs={"net_arch": [], "ortho_init": False}, seed=args.training_seed)
    initialize_development_policy(model.policy, base.policy)
    model.num_timesteps = base.num_timesteps  # 事前調整は環境PPO stepとして加算しない。
    report = train(model, datasets, epochs=args.epochs, seed=args.training_seed)
    if report["best_epoch"] == 0:
        write(output / "training.json", report)
        raise RuntimeError("検証誤差が初期値から改善しませんでした。候補の対戦評価を中止します。")
    report["preservation"] = verify_frozen(base, model, datasets[2])
    write(output / "training.json", report)
    agents = {"original": original, "road_v4": source}
    for strength in args.candidate_strengths:
        label = str(strength).rstrip("0").rstrip(".").replace(".", "p")
        name = f"development_x{label}"
        model.policy.mlp_extractor.development_strength.fill_(strength)
        path = output / f"{name}.zip"
        model.save(str(path))
        agent = PPOAgent(args.model_id, device="cpu")
        agent.model = MaskablePPO.load(str(path), device="cpu")
        agents[name] = agent
        write(output / f"audit_{name}.json", audit_policy(
            source.model, agent.model, datasets[2], [strength]
        ))
    evaluate(output, agents, range(args.eval_seed_start, args.eval_seed_start + args.eval_games))
    env.close()


if __name__ == "__main__":
    main()
