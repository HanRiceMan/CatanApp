"""Gated Step 3A-9 smoke, one-update rollout, and held-out evaluation."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path

import numpy as np
import torch

from app.agents.ppo import PPOAgent

from .construction_timing import BUILD_NOW, DEFER, ConstructionTimingGate, SurfaceCritic
from .construction_timing_ppo import one_timing_ppo_update
from .construction_timing_rollout import run_timing_episode, summarize_episodes
from .runtime_context import RUNTIME_CONTEXT_VERSION


CHAMPION_ID = "ppo_gnn_board_65k_s03_exp_v003"
ROBBER_ID = "ppo_robber_value_73k_s01_exp_v001"
PROFILES = ("rule", "champion", "mixed")
FORCED_CASES = (
    (2370001, 1, "rule", "SETTLEMENT_ONLY", BUILD_NOW, "BuildSettlementAction"),
    (2370001, 1, "rule", "CITY_ONLY", BUILD_NOW, "BuildCityAction"),
    (2370001, 1, "rule", "SETTLEMENT_ONLY", DEFER, "EndTurnAction"),
    (2370001, 1, "rule", "CITY_ONLY", DEFER, "BuyDevelopmentAction"),
    (2370007, 3, "rule", "CITY_ONLY", DEFER, "BankTradeAction"),
    (2370001, 1, "rule", "CITY_ONLY", DEFER, "BuildRoadAction"),
    (2371002, 2, "champion", "SETTLEMENT_ONLY", DEFER, "ProposeTradeAction"),
)


def _write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _models():
    champion = PPOAgent(CHAMPION_ID, device="cpu")
    robber = PPOAgent(ROBBER_ID, device="cpu")
    latent_dim = champion.model.policy.mlp_extractor.latent_dim_pi
    gate = ConstructionTimingGate(latent_dim)
    critic = SurfaceCritic(latent_dim)
    return champion, robber, gate, critic


def _run(champion, robber, gate, critic, seed, seat, profile, probability,
         *, delegation_seed=None, baseline=False, stochastic=True, force=None):
    return run_timing_episode(
        champion, robber, gate, critic, seed=seed, seat=seat, profile=profile,
        delegation_probability=probability,
        delegation_rng_seed=(delegation_seed if delegation_seed is not None
                             else seed + 500000),
        use_delegator=not baseline, stochastic_gate=stochastic,
        integration_force=force)


def smoke(root, champion, robber, gate, critic):
    if (root / "smoke.json").exists():
        raise FileExistsError("Smoke report already exists")
    with torch.no_grad():
        latent = torch.zeros(2, champion.model.policy.mlp_extractor.latent_dim_pi)
        context = torch.zeros(2, 79)
        subtype = torch.eye(2)
        initial_prob = torch.softmax(gate(latent, context, subtype), dim=-1).tolist()
        base = torch.tensor([[0.2], [0.3]])
        initial_critic = critic(base, latent, context, subtype).tolist()
    if initial_prob != [[0.5, 0.5], [0.5, 0.5]] or initial_critic != base.tolist():
        raise AssertionError("Gate/Critic are not zero-impact initialized")
    forced = []
    for seed, seat, profile, subtype, gate_action, action_name in FORCED_CASES:
        episode = _run(champion, robber, gate, critic, seed, seat, profile, 0,
                       force=(subtype, gate_action, action_name))
        if not (episode.integration_forced and episode.terminal
                and len(episode.transitions) == 1 and episode.decisions[0]["action_type"]
                == action_name):
            raise AssertionError(f"Forced integration failed: {subtype}/{action_name}")
        if action_name == "ProposeTradeAction":
            if (not episode.decisions[0]["trade_proposed"]
                    or episode.decisions[0]["trade_phase_after_step"] not in
                    {"action", "turn_pre_roll", "game_over"}):
                raise AssertionError("PlayerTrade did not return to learner boundary")
        forced.append({"seed": seed, "seat": seat, "profile": profile,
                       "surface": subtype, "action": action_name,
                       "policy_steps": episode.policy_steps,
                       "macro_duration": episode.transitions[0].macro_duration,
                       "trade_crossed": episode.transitions[0].player_trades_crossed,
                       "terminal": episode.terminal})
        print(f"forced {subtype}/{action_name}: PASS", flush=True)

    parity = []
    for profile_index, profile in enumerate(PROFILES):
        for seat in range(1, 5):
            seed = 2410000 + profile_index * 100 + seat
            baseline = _run(champion, robber, gate, critic, seed, seat, profile, 0,
                            baseline=True)
            delegated_zero = _run(champion, robber, gate, critic, seed, seat,
                                  profile, 0)
            if not (baseline.action_trace == delegated_zero.action_trace
                    and baseline.final_state == delegated_zero.final_state
                    and baseline.rng_counter == delegated_zero.rng_counter
                    and baseline.policy_steps == delegated_zero.policy_steps
                    and baseline.winner == delegated_zero.winner
                    and baseline.final_vp == delegated_zero.final_vp
                    and baseline.final_rank == delegated_zero.final_rank
                    and not delegated_zero.transitions):
                raise AssertionError(f"Zero-delegation parity failed: {profile}/{seat}")
            parity.append({"profile": profile, "seat": seat, "seed": seed,
                           "actions": len(baseline.action_trace),
                           "policy_steps": baseline.policy_steps, "parity": True})
            print(f"zero parity {profile}/seat{seat}: PASS", flush=True)

    smoke_episodes = []
    for profile_index, profile in enumerate(PROFILES):
        for seat in range(1, 5):
            seed = 2420000 + profile_index * 100 + seat
            episode = _run(champion, robber, gate, critic, seed, seat, profile, 0.10)
            if not episode.terminal or episode.truncated:
                raise AssertionError("10% delegation episode did not finish")
            if any(item.macro_duration < 1 or not isfinite(item.reward_accumulated)
                   for item in episode.transitions):
                raise AssertionError("Invalid macro transition")
            smoke_episodes.append(episode)
            print(f"10% smoke {profile}/seat{seat}: "
                  f"delegated={len(episode.transitions)}", flush=True)
    summary = summarize_episodes(smoke_episodes)
    if summary["transitions"] < 2 or summary["truncated_transitions"]:
        raise AssertionError("10% smoke produced insufficient valid transitions")
    if summary["defer_immediate_same_surface"] > 20:
        raise AssertionError("Potential repeated DEFER loop")
    _write(root / "smoke.json", {
        "forced_integration": forced,
        "zero_gate_probability": initial_prob,
        "zero_critic_residual": True,
        "zero_delegation_parity": parity,
        "ten_percent": summary,
        "ten_percent_episodes": [dict(seed=item.seed, seat=item.seat,
                                      profile=item.profile,
                                      delegation_rng_seed=item.delegation_rng_seed,
                                      transitions=len(item.transitions),
                                      terminal=item.terminal,
                                      policy_steps=item.policy_steps)
                                 for item in smoke_episodes],
        "passed": True,
    })
    print("INTEGRATION SMOKE PASSED", flush=True)


def train(root, champion, robber, gate, critic):
    smoke_report = json.loads((root / "smoke.json").read_text(encoding="utf-8"))
    if not smoke_report.get("passed"):
        raise RuntimeError("Integration smoke has not passed")
    if (root / "update.json").exists():
        raise FileExistsError("Update already exists; exactly one update permitted")
    episodes = []
    delegated_count = 0
    for block in range(20):
        for profile_index, profile in enumerate(PROFILES):
            for seat in range(1, 5):
                index = block * 12 + profile_index * 4 + seat - 1
                seed = 2430001 + index
                episode = _run(champion, robber, gate, critic, seed, seat, profile, 0.25,
                               delegation_seed=3430001 + index)
                if not episode.terminal or episode.truncated:
                    raise AssertionError("Training rollout truncated before terminal")
                episodes.append(episode)
                delegated_count += len(episode.transitions)
                print(f"rollout {len(episodes)} {profile}/seat{seat}: "
                      f"delegated={len(episode.transitions)} total={delegated_count}",
                      flush=True)
        if delegated_count >= 128:
            break
    if delegated_count < 128:
        raise RuntimeError("Failed to collect 128 delegated on-policy transitions")
    rollout = summarize_episodes(episodes)
    if not (rollout["all_terminal"] and rollout["truncated_transitions"] == 0):
        raise AssertionError("Rollout completeness gate failed")
    if rollout["gate_action"]["BUILD_NOW"] == 0 or rollout["gate_action"]["DEFER"] == 0:
        raise AssertionError("Only one Gate Action was observed")
    if (rollout["next_delegated_surface"] == 0
            or rollout["nondelegated_surface_crossings"] == 0):
        raise AssertionError("Conservative semi-MDP bootstrap boundary was not exercised")
    _write(root / "train_rollout.json", {
        "summary": rollout,
        "episodes": [dict(seed=item.seed, seat=item.seat, profile=item.profile,
                          delegation_rng_seed=item.delegation_rng_seed,
                          transitions=len(item.transitions), winner=item.winner,
                          final_vp=item.final_vp, final_rank=item.final_rank,
                          policy_steps=item.policy_steps)
                     for item in episodes],
    })
    result = one_timing_ppo_update(champion, gate, critic, episodes,
                                   training_seed=3439001)
    report = {"accepted": result.accepted, "reason": result.reason,
              "metrics": result.metrics,
              "rollout": rollout,
              "configuration": {
                  "delegation_probability": 0.25,
                  "actor_lr": 1e-4, "critic_lr": 3e-4,
                  "batch_size": 32, "epochs": 2, "clip_range": 0.1,
                  "target_kl": 0.01, "entropy_coef": 0.005,
                  "gamma": champion.model.gamma,
                  "gae_lambda": champion.model.gae_lambda,
                  "training_seed": 3439001,
              }}
    _write(root / "update.json", report)
    if result.accepted:
        artifact = root / "candidate_gate.pt"
        if artifact.exists():
            raise FileExistsError("Candidate artifact already exists")
        torch.save({"gate": gate.state_dict(), "critic": critic.state_dict(),
                    "parent_model_id": CHAMPION_ID,
                    "runtime_context_version": RUNTIME_CONTEXT_VERSION,
                    "update_count": 1}, artifact)
    print(json.dumps({"accepted": result.accepted, "reason": result.reason,
                      "transitions": delegated_count,
                      "mean_kl": result.metrics["mean_kl"]}), flush=True)


def evaluate(root, champion, robber, gate, critic, *, games_per_profile=20):
    update = json.loads((root / "update.json").read_text(encoding="utf-8"))
    if not update["accepted"]:
        raise RuntimeError("Rejected update must not be evaluated as candidate")
    artifact = torch.load(root / "candidate_gate.pt", map_location="cpu",
                          weights_only=True)
    if artifact["parent_model_id"] != CHAMPION_ID or artifact["update_count"] != 1:
        raise ValueError("Candidate artifact parent/update count mismatch")
    gate.load_state_dict(artifact["gate"])
    critic.load_state_dict(artifact["critic"])
    results = {"baseline": [], "conservative_25": [], "full_surface_100": []}
    for profile_index, profile in enumerate(PROFILES):
        for index in range(games_per_profile):
            seat = index % 4 + 1
            seed = 2450001 + profile_index * 1000 + index
            baseline = _run(champion, robber, gate, critic, seed, seat, profile,
                            0, baseline=True)
            conservative = _run(champion, robber, gate, critic, seed, seat,
                                profile, 0.25, delegation_seed=3450001 + seed)
            full = _run(champion, robber, gate, critic, seed, seat,
                        profile, 1.0, delegation_seed=3460001 + seed)
            for name, item in (("baseline", baseline),
                               ("conservative_25", conservative),
                               ("full_surface_100", full)):
                if not item.terminal or item.truncated:
                    raise AssertionError(f"Evaluation did not terminate: {name}/{seed}")
                results[name].append(item)
            print(f"evaluation {profile} seed={seed} seat={seat} "
                  f"VP baseline/25/100="
                  f"{baseline.final_vp}/{conservative.final_vp}/{full.final_vp}",
                  flush=True)
    _write(root / "evaluation.json", {
        "games_per_profile": games_per_profile,
        "seed_start": 2450001,
        "summary": {name: summarize_episodes(items)
                    for name, items in results.items()},
        "by_profile": {
            name: {profile: summarize_episodes([
                item for item in items if item.profile == profile])
                   for profile in PROFILES}
            for name, items in results.items()},
        "games": [dict(seed=item.seed, profile=item.profile, seat=item.seat,
                       variant=name, winner=item.winner, vp=item.final_vp,
                       rank=item.final_rank, steps=item.policy_steps,
                       transitions=len(item.transitions))
                  for name, items in results.items() for item in items],
    })
    print("EVALUATION COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "train", "evaluate"))
    parser.add_argument("--experiment-id", default="construction_timing_step3a9_20261003")
    parser.add_argument("--games-per-profile", type=int, default=20)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id:
        parser.error("Invalid experiment id")
    root = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    root.mkdir(parents=True, exist_ok=True)
    champion, robber, gate, critic = _models()
    if args.stage == "smoke":
        smoke(root, champion, robber, gate, critic)
    elif args.stage == "train":
        train(root, champion, robber, gate, critic)
    else:
        evaluate(root, champion, robber, gate, critic,
                 games_per_profile=args.games_per_profile)


if __name__ == "__main__":
    main()
