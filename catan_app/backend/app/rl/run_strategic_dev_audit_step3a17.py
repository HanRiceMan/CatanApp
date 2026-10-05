"""Replay the 36 archived T=2 full-surface games and audit Dev choices."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from typing import Any

import torch

from app.agents.ppo import PPOAgent

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .run_strategic_gate_step3a14 import CHAMPION_ID, PROFILES, ROBBER_ID
from .strategic_dev_audit_step3a17 import (
    capture_dev_state, paired_dev_continuation, select_representative_states,
    summarize_dev_audit,
)
from .strategic_gate_ppo_step3a16 import load_strategic_artifact
from .strategic_gate_rollout_step3a14 import run_strategic_episode


SCHEMA = "strategic_development_causal_audit_step3a17_v1"
BASE = Path(__file__).resolve().parents[2] / "experiments"
PRIOR_ROOT = BASE / "strategic_gate_step3a16_20261005"
AUDIT_ROOT = BASE / "strategic_dev_audit_step3a17_20261005_balanced"
FUTURE_SEED_START = 3170000


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_audit(*, representative_count: int = 24, future_count: int = 8,
              profiles: tuple[str, ...] = PROFILES,
              output: Path = AUDIT_ROOT) -> dict[str, Any]:
    if future_count < 8 or representative_count < 1:
        raise ValueError("Step 3A-17 requires at least eight paired futures")
    output.mkdir(parents=True, exist_ok=True)
    champion = PPOAgent(CHAMPION_ID, device="cpu")
    robber = PPOAgent(ROBBER_ID, device="cpu")
    dim = champion.model.policy.mlp_extractor.latent_dim_pi
    gate, critic, manifest = load_strategic_artifact(
        PRIOR_ROOT / "Strategic-T2-preupdate", dim)
    if manifest["temperature"] != 2.0:
        raise AssertionError("Not the frozen Step 3A-16 T=2 pre-update artifact")
    champion_digest_before = _weight_digest(champion)
    gate_before = {name: value.detach().clone()
                   for name, value in gate.state_dict().items()}
    captured = []
    parity = []
    for profile_index, profile in enumerate(PROFILES):
        if profile not in profiles:
            continue
        for index in range(12):
            seed = 3260000 + profile_index * 1000 + index
            seat = index % 4 + 1
            archived = _read(PRIOR_ROOT / f"eval-pre100-{profile}-{seed}.json")
            local = []

            def observe(env, planner, snapshot, mask, record):
                if record["selected_family"] != "BUY_DEVELOPMENT":
                    return
                state = capture_dev_state(env, planner, snapshot, mask, record)
                state.row.update({"profile": profile, "seat": seat, "seed": seed,
                                  "learner_id": env.learning_player_id})
                local.append(state)

            episode = run_strategic_episode(
                champion, robber, seed=seed, seat=seat, profile=profile,
                mode="strategic", delegation_probability=1.0,
                delegation_seed=seed + 12345, gate_sampling_seed=seed + 54321,
                stochastic_gate=True, prior_temperature=2.0,
                strategic_gate=gate, strategic_critic=critic,
                decision_audit_observer=observe)
            actual = [(item["pre_policy_step"], item["selected_family"],
                       item["concrete_action"]) for item in episode.decisions]
            expected = [(item["pre_policy_step"], item["selected_family"],
                         item["concrete_action"]) for item in archived["decisions"]]
            if (actual != expected or episode.winner != archived["winner"]
                    or episode.final_vp != archived["final_vp"]
                    or episode.final_rank != archived["final_rank"]
                    or episode.policy_steps != archived["policy_steps"]
                    or episode.rng_counter != archived["rng_counter"]):
                raise AssertionError(f"Archived T=2 full-surface replay mismatch: {profile}/{seed}")
            captured.extend(local)
            parity.append({"profile": profile, "seed": seed, "seat": seat,
                           "decisions": len(actual), "dev_decisions": len(local),
                           "action_outcome_rng_parity": True})
            print(f"replay {profile} {seed} dev={len(local)}", flush=True)
    rows = [state.row for state in captured]
    _write(output / "dev_decisions.json", {"schema": SCHEMA,
                                            "source": "Strategic-T2-preupdate full100",
                                            "rows": rows})
    _write(output / "replay_parity.json", parity)
    if profiles != PROFILES:
        return {"captured": len(rows), "profiles": dict(Counter(row["profile"] for row in rows))}
    if len(rows) != 153:
        raise AssertionError(f"Expected all 153 archived Development choices, got {len(rows)}")
    chosen = select_representative_states(captured, representative_count)
    selected = [{"state_id": f"{state.row['profile']}-{state.row['seed']}-{state.row['pre_policy_step']}",
                 "profile": state.row["profile"], "seat": state.row["seat"],
                 "probability": state.row["selected_dev_probability"],
                 "candidate_set": state.row["candidate_set"],
                 "alternative_family": state.row["strongest_non_dev_family"],
                 "champion_family": state.row["champion_counterfactual_family"]}
                for state in chosen]
    _write(output / "selected_states.json", selected)
    paired = []
    for state_index, state in enumerate(chosen):
        state_id = selected[state_index]["state_id"]
        for future_index in range(future_count):
            path = output / f"pair-{state_index:02d}-{future_index:02d}.json"
            if path.exists():
                pair = _read(path)
                if pair["state_id"] != state_id:
                    raise AssertionError("Paired checkpoint state selection changed")
            else:
                future_seed = FUTURE_SEED_START + state_index * 100 + future_index
                pair = paired_dev_continuation(state, future_seed)
                pair["state_id"] = state_id
                pair["profile"] = state.row["profile"]
                pair["candidate_set"] = state.row["candidate_set"]
                pair["dev_probability"] = state.row["selected_dev_probability"]
                pair["alternative_family"] = state.row["strongest_non_dev_family"]
                pair["context"] = state.row["context"]
                _write(path, pair)
            paired.append(pair)
            print(f"pair {state_index+1}/{len(chosen)} future={future_index+1}/{future_count} "
                  f"delta={pair['delta_return']:.3f}", flush=True)
    summary = summarize_dev_audit(rows, paired)
    if _weight_digest(champion) != champion_digest_before:
        raise AssertionError("Audit changed Champion weights")
    if any(not torch.equal(value, gate.state_dict()[name])
           for name, value in gate_before.items()):
        raise AssertionError("Audit changed Strategic Gate weights")
    with gzip.open(output / "paired_continuations.json.gz", "wt", encoding="utf-8") as stream:
        json.dump({"schema": SCHEMA, "pairs": paired}, stream,
                  ensure_ascii=False, allow_nan=False)
    summary.update({"schema": SCHEMA, "replay_games": len(parity),
                    "all_replay_action_outcome_rng_parity": True,
                    "future_seeds_per_state": future_count,
                    "paired_continuation_policy": "frozen Champion Planner after one forced family",
                    "future_rng_method": "common future seed per branch, counter reset at fork",
                    "reward_gamma": .99,
                    "diagnostic_horizon_policy_steps": 30,
                    "no_optimizer_update": True})
    summary["frozen_champion_weight_digest"] = champion_digest_before
    summary["gate_weights_unchanged"] = True
    _write(output / "summary.json", summary)
    return summary


def resummarize_saved(output: Path = AUDIT_ROOT) -> dict[str, Any]:
    """Rebuild aggregate diagnostics from the committed compact evidence."""
    rows = _read(output / "dev_decisions.json")["rows"]
    with gzip.open(output / "paired_continuations.json.gz", "rt", encoding="utf-8") as stream:
        paired = json.load(stream)["pairs"]
    if len(rows) != 153 or len(paired) != 192 or len({item["state_id"] for item in paired}) != 24:
        raise AssertionError("Step 3A-17 compact evidence is incomplete")
    result = summarize_dev_audit(rows, paired)
    previous = _read(output / "summary.json")
    result.update({key: value for key, value in previous.items()
                   if key not in result})
    _write(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representative-count", type=int, default=24)
    parser.add_argument("--future-count", type=int, default=8)
    parser.add_argument("--profile", choices=PROFILES, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.profile and args.output is None:
        parser.error("Partial profile replay requires a separate --output directory")
    root = args.output or AUDIT_ROOT
    result = (resummarize_saved(root) if args.summarize_only else
              run_audit(representative_count=args.representative_count,
                        future_count=args.future_count,
                        profiles=tuple(args.profile) if args.profile else PROFILES,
                        output=root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
