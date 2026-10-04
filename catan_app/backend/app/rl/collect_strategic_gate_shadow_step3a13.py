"""Step 3A-13 Shadow audit of six-family Gate, executors, prior, and wins."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from math import log
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from app.agents.heuristic import HeuristicAgent
from app.agents.ppo import PPOAgent
from app.domain.actions import Action, ProposeTradeAction
from app.domain.game import GameState

from .action_space import action_to_id, get_action_mask
from .macro_teacher_dataset import OPPONENT_PROFILES, play_teacher_episode
from .settlement_planning_agent import SettlementPlanningAgent
from .strategic_action_surface import (FAMILIES, SURFACE_VERSION,
                                       detect_strategic_action_surface, family_of)
from .strategic_gate_step3a13 import (FrozenStrategicExecutors,
                                     StrategicActionGate,
                                     StrategicSurfaceCritic, family_priors,
                                     frozen_snapshot, resolve_immediate_win)
from .trade_strategy import choose_proactive_trade_with_diagnostic


CHAMPION_ID = "ppo_gnn_board_65k_s03_exp_v003"
ROBBER_ID = "ppo_robber_value_73k_s01_exp_v001"
SHADOW_SCHEMA = "strategic_gate_shadow_step3a13_v1"


def _action_record(action: Action | None) -> dict[str, Any] | None:
    if action is None:
        return None
    try:
        catalog_id = action_to_id(action)
    except ValueError:
        catalog_id = None
    return {"type": type(action).__name__, "family": family_of(action),
            "catalog_id": catalog_id, "fields": asdict(action)}


def _entropy(probabilities: list[float]) -> float:
    return -sum(value * log(value) for value in probabilities if value > 0)


def _corr(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _weight_digest(policy) -> str:
    digest = sha256()
    for name, tensor in policy.model.policy.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class StrategicGateShadowAgent(SettlementPlanningAgent):
    """All proposals remain shadow; actual action is selected once by Champion."""

    def __init__(self, policy, **planning):
        super().__init__(policy, **planning)
        latent_dim = policy.model.policy.mlp_extractor.latent_dim_pi
        self.gate = StrategicActionGate(latent_dim)
        self.critic = StrategicSurfaceCritic(latent_dim)
        self.executors = FrozenStrategicExecutors(self)
        self.records: list[dict] = []
        self.win_events: list[dict] = []
        self.profile: str = ""
        self.seed: int = -1
        self.actor_logits_checks = 0
        self.zero_prior_checks = 0

    def _protected_trade_candidate(self, game: GameState, player_id: int) -> bool:
        if not self.proactive_player_trade:
            return False
        candidate = choose_proactive_trade_with_diagnostic(
            deepcopy(game), player_id,
            target_sites=self.target_sites,
            max_plan_roads=self.max_plan_roads,
            max_wait_rounds=self.max_wait_rounds,
            max_city_wait_rounds=self.max_city_wait_rounds,
            max_scarcity_deficit=self.proactive_trade_max_scarcity_deficit,
            strategic_surplus_trade=self.strategic_surplus_trade,
            opponent_score_limit=self.proactive_trade_opponent_score_limit,
        )
        return candidate is not None

    def select_action(self, game: GameState, player_id: int) -> Action:
        if game.phase not in {"action", "turn_pre_roll"}:
            return super().select_action(game, player_id)
        if game.phase == "turn_pre_roll":
            win = resolve_immediate_win(game, player_id)
            action = super().select_action(game, player_id)
            if win.action is not None:
                self._record_win(game, player_id, win, action)
            return action

        source_before = deepcopy(game)
        mask = get_action_mask(game, player_id)
        surface = detect_strategic_action_surface(game, player_id, mask)
        win = resolve_immediate_win(game, player_id, mask)
        if not surface.eligible:
            action, _, _ = self._select_action_with_diagnostics(game, player_id)
            if win.action is not None:
                self._record_win(game, player_id, win, action)
            if game != source_before:
                raise AssertionError("Non-Gate win audit modified source state")
            return action

        snapshot = frozen_snapshot(self.policy, game, player_id,
                                   max_plan_roads=self.max_plan_roads)
        prior = family_priors(snapshot, mask, surface.family_mask)
        latent = snapshot.actor_latent
        context = torch.as_tensor(snapshot.runtime_context,
                                  dtype=latent.dtype, device=latent.device).unsqueeze(0)
        family_mask = torch.tensor([surface.family_mask], dtype=torch.bool,
                                   device=latent.device)
        native_scores = torch.tensor(
            [prior["raw_scores"]["native_family_logits"]],
            dtype=latent.dtype, device=latent.device)
        with torch.no_grad():
            gate_logits = self.gate(latent, context, family_mask, native_scores)
            gate_probs = torch.softmax(gate_logits, dim=-1)[0].cpu().numpy()
            surface_value = self.critic(snapshot.critic_value, latent,
                                        context, family_mask)
        if not np.array_equal(gate_logits[0, family_mask[0]].cpu().numpy(),
                              native_scores[0, family_mask[0]].cpu().numpy()):
            raise AssertionError("Zero residual changed frozen family prior logits")
        if not np.allclose(gate_probs, prior["probabilities"]["native_family_logits"],
                           rtol=0, atol=1e-6):
            raise AssertionError(
                "Zero residual changed frozen family prior distribution: "
                f"{np.max(np.abs(gate_probs - prior['probabilities']['native_family_logits']))}")
        if not torch.equal(surface_value, snapshot.critic_value):
            raise AssertionError("Zero residual changed frozen base Critic value")
        self.zero_prior_checks += 1

        executor_records: dict[str, dict] = {}
        for family, available in zip(FAMILIES, surface.family_mask, strict=True):
            if not available:
                continue
            methods = ("A_PLANNER_EXTRACTION", "B_FROZEN_PPO") if family in {
                "BUILD_ROAD", "BANK_TRADE"} else ("B_FROZEN_PPO",)
            executor_records[family] = {}
            for method in methods:
                result = self.executors.execute(
                    game, player_id, family, method=method, mask=mask,
                    snapshot=snapshot)
                executor_records[family][method] = {
                    "success": result.success, "failure": result.failure,
                    "action": _action_record(result.action),
                }
        protected_available = self._protected_trade_candidate(game, player_id)
        action, _, trace = self._select_action_with_diagnostics(game, player_id)
        post_snapshot = frozen_snapshot(self.policy, game, player_id,
                                        max_plan_roads=self.max_plan_roads)
        if (not np.array_equal(snapshot.candidate_logits, post_snapshot.candidate_logits)
                or not np.array_equal(snapshot.old_family_logits,
                                      post_snapshot.old_family_logits)
                or not torch.equal(snapshot.critic_value, post_snapshot.critic_value)):
            raise AssertionError("Shadow changed Champion Actor/Critic logits")
        self.actor_logits_checks += 1
        if game != source_before:
            raise AssertionError("Strategic Shadow modified source GameState or RNG")
        if win.action is not None:
            self._record_win(game, player_id, win, action)

        actual_family = family_of(action)
        proposed_family = FAMILIES[int(np.argmax(gate_probs))]
        record = surface.to_dict()
        record.update({
            "record_schema": SHADOW_SCHEMA,
            "game_seed": self.seed, "game_id": game.id,
            "profile": self.profile,
            "seat": game.seat_order.index(player_id) + 1,
            "player_id": player_id, "turn": game.turn_number,
            "revision": game.revision,
            "rng_counter": game.random_source.counter,
            "prior": prior,
            "gate_logits": gate_logits[0].cpu().tolist(),
            "gate_probabilities": gate_probs.tolist(),
            "gate_proposed_family": proposed_family,
            "surface_critic_value": float(surface_value.item()),
            "executors": executor_records,
            "all_candidates_executor_success": all(
                executor_records[family]["B_FROZEN_PPO"]["success"]
                for family, available in zip(FAMILIES, surface.family_mask, strict=True)
                if available),
            "protected_player_trade_available": protected_available,
            "protected_player_trade_selected_by_champion": isinstance(
                action, ProposeTradeAction),
            "champion_action": _action_record(action),
            "champion_family": actual_family,
            "champion_execution_source": trace.execution_source.value,
            "champion_in_candidate_set": (
                actual_family in FAMILIES
                and surface.family_mask[FAMILIES.index(actual_family)]),
            "winning_action_detected": win.action is not None,
            "winning_action": _action_record(win.action),
            "champion_chose_any_winning_action": (
                win.action is not None and _action_record(action)["catalog_id"]
                in win.winning_action_ids),
        })
        self.records.append(record)
        return action

    def _record_win(self, game: GameState, player_id: int, win,
                    action: Action) -> None:
        self.win_events.append({
            "game_seed": self.seed, "profile": self.profile,
            "seat": game.seat_order.index(player_id) + 1,
            "phase": game.phase, "turn": game.turn_number,
            "revision": game.revision,
            "winning_action_ids": list(win.winning_action_ids),
            "resolved_action": _action_record(win.action),
            "champion_action": _action_record(action),
            "champion_chose_any_winning_action": (
                _action_record(action)["catalog_id"] in win.winning_action_ids),
        })


def _summarize_group(records: list[dict]) -> dict[str, Any]:
    candidate_counts = Counter(row["candidate_count"] for row in records)
    priors = {}
    for method in ("uniform", "max_legal_logit", "log_mean_exp",
                   "native_family_logits"):
        tops = Counter()
        entropies = []
        agreement = 0
        compared = 0
        for row in records:
            probs = row["prior"]["probabilities"][method]
            top_index = int(np.argmax(probs))
            tops[FAMILIES[top_index]] += 1
            entropies.append(_entropy(probs))
            if row["champion_in_candidate_set"]:
                compared += 1
                agreement += FAMILIES[top_index] == row["champion_family"]
        priors[method] = {
            "mean_entropy": mean(entropies) if entropies else None,
            "top_family": dict(tops),
            "champion_family_agreement": agreement / compared if compared else None,
            "agreement_denominator": compared,
        }
    available = Counter()
    successes = Counter()
    failures = Counter()
    method_stats = defaultdict(lambda: {"available": 0, "success": 0,
                                        "champion_same_family": 0,
                                        "champion_exact_match": 0,
                                        "failures": Counter(),
                                        "road_quality": Counter()})
    for row in records:
        for family, methods in row["executors"].items():
            available[family] += 1
            main = methods["B_FROZEN_PPO"]
            successes[family] += bool(main["success"])
            if not main["success"]:
                failures[f"{family}:{main['failure']}"] += 1
            for method, result in methods.items():
                entry = method_stats[f"{family}:{method}"]
                entry["available"] += 1
                entry["success"] += bool(result["success"])
                if not result["success"]:
                    entry["failures"][result["failure"]] += 1
                if family == "BUILD_ROAD" and result["success"]:
                    edge = result["action"]["fields"]["edge_id"]
                    for purpose in ("expansion_edges", "longest_progress_edges",
                                    "both_edges", "other_edges"):
                        entry["road_quality"][purpose] += (
                            edge in row["road_purposes"][purpose])
                if row["champion_family"] == family:
                    entry["champion_same_family"] += 1
                    entry["champion_exact_match"] += (
                        result["success"]
                        and result["action"] == row["champion_action"])
    for entry in method_stats.values():
        entry["failures"] = dict(entry["failures"])
        entry["road_quality"] = dict(entry["road_quality"])
    correlations = {}
    for family in ("BUILD_ROAD", "BANK_TRADE"):
        relevant = [row for row in records if family in row["executors"]]
        index = FAMILIES.index(family)
        counts = [row["prior"]["legal_action_counts"][index] for row in relevant]
        correlations[family] = {
            "n": len(relevant), "min_legal_actions": min(counts) if counts else None,
            "max_legal_actions": max(counts) if counts else None,
            "count_vs_probability_pearson": {
                method: _corr(counts, [row["prior"]["probabilities"][method][index]
                                       for row in relevant])
                for method in ("uniform", "max_legal_logit", "log_mean_exp",
                               "native_family_logits")},
        }
    return {
        "decisions": len(records),
        "candidate_count": dict(candidate_counts),
        "protected_player_trade_available": sum(
            row["protected_player_trade_available"] for row in records),
        "protected_player_trade_selected": sum(
            row["protected_player_trade_selected_by_champion"] for row in records),
        "all_candidates_executor_success": sum(
            row["all_candidates_executor_success"] for row in records),
        "executor_available": dict(available),
        "executor_success": dict(successes),
        "executor_failure": dict(failures),
        "executor_method": dict(method_stats),
        "priors": priors,
        "cardinality": correlations,
    }


def summarize(records: list[dict], wins: list[dict], games: list[dict],
              *, weights_unchanged: bool, actor_logits_checks: int,
              zero_prior_checks: int) -> dict:
    groups = {"all": records,
              "no_protected_overlap": [row for row in records
                                       if not row["protected_player_trade_selected_by_champion"]],
              "protected_player_trade_overlap": [row for row in records
                                                 if row["protected_player_trade_selected_by_champion"]]}
    return {
        "schema_version": SHADOW_SCHEMA,
        "game_count": len(games),
        "profile": dict(Counter(row["profile"] for row in records)),
        "seat": dict(Counter(row["seat"] for row in records)),
        "profile_seat_games": dict(Counter(
            f"{row['profile']}:{row['seat']}" for row in games)),
        "shadow_decisions": len(records),
        "groups": {name: _summarize_group(rows) for name, rows in groups.items()},
        "by_profile": {profile: _summarize_group([
            row for row in records if row["profile"] == profile])
            for profile in OPPONENT_PROFILES},
        "by_seat": {str(seat): _summarize_group([
            row for row in records if row["seat"] == seat])
            for seat in range(1, 5)},
        "winning_action_detected": len(wins),
        "winning_action_missed_by_champion": sum(
            not row["champion_chose_any_winning_action"] for row in wins),
        "winning_action_family": dict(Counter(
            row["resolved_action"]["family"] for row in wins)),
        "winning_miss_champion_family": dict(Counter(
            row["champion_action"]["family"] for row in wins
            if not row["champion_chose_any_winning_action"])),
        "parity_games": sum(row["parity"] for row in games),
        "champion_weights_unchanged": weights_unchanged,
        "actor_critic_logits_parity_checks": actor_logits_checks,
        "zero_residual_prior_checks": zero_prior_checks,
    }


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def collect(root: Path, *, games_per_profile: int, seed_start: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    champion = PPOAgent(CHAMPION_ID, device="cpu")
    robber = PPOAgent(ROBBER_ID, device="cpu")
    if champion.settlement_planning_config is None or robber.settlement_planning_config is None:
        raise ValueError("Both frozen models require their existing Planner config")
    weight_before = _weight_digest(champion)
    factories = {
        "rule": HeuristicAgent,
        "champion": lambda: SettlementPlanningAgent(
            champion, **champion.settlement_planning_config),
        "robber_ppo": lambda: SettlementPlanningAgent(
            robber, **robber.settlement_planning_config),
    }
    records: list[dict] = []
    wins: list[dict] = []
    games: list[dict] = []
    actor_checks = zero_checks = 0
    for profile_index, profile in enumerate(OPPONENT_PROFILES):
        for index in range(games_per_profile):
            seed = seed_start + profile_index * 1000 + index
            seat = index % 4 + 1
            checkpoint = root / f"game-{profile}-{seed}.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                records.extend(saved["records"])
                wins.extend(saved["wins"])
                games.append(saved["game"])
                actor_checks += saved["actor_checks"]
                zero_checks += saved["zero_checks"]
                continue
            agent = StrategicGateShadowAgent(
                champion, **champion.settlement_planning_config)
            agent.profile, agent.seed = profile, seed
            baseline = play_teacher_episode(
                SettlementPlanningAgent(champion, **champion.settlement_planning_config),
                seed, champion_seat=seat, opponent_profile=profile,
                opponent_factories=factories, collect_reports=False)
            shadow = play_teacher_episode(
                agent, seed, champion_seat=seat, opponent_profile=profile,
                opponent_factories=factories, collect_reports=False)
            parity_keys = ("winner", "final_vp", "final_rank", "final_scores",
                           "rng_counter", "all_action_count")
            parity = (baseline.action_trace == shadow.action_trace
                      and baseline.final_state == shadow.final_state
                      and baseline.final_views == shadow.final_views
                      and all(baseline.game[key] == shadow.game[key]
                              for key in parity_keys))
            if not parity:
                raise AssertionError(f"Champion/Shadow parity failed: {profile}/{seed}")
            game = {
                "profile": profile, "game_seed": seed, "seat": seat,
                "action_count": len(baseline.action_trace),
                "shadow_decisions": len(agent.records),
                "winning_action_detected": len(agent.win_events),
                "winner": baseline.game["winner"],
                "final_vp": baseline.game["final_vp"],
                "final_rank": baseline.game["final_rank"],
                "rng_counter": baseline.game["rng_counter"],
                "parity": parity,
            }
            _write_json(checkpoint, {
                "game": game, "records": agent.records,
                "wins": agent.win_events,
                "actor_checks": agent.actor_logits_checks,
                "zero_checks": agent.zero_prior_checks,
            })
            records.extend(agent.records)
            wins.extend(agent.win_events)
            games.append(game)
            actor_checks += agent.actor_logits_checks
            zero_checks += agent.zero_prior_checks
            print(f"{profile} seed={seed} seat={seat} shadow={len(agent.records)} "
                  f"wins={len(agent.win_events)}", flush=True)
    weight_after = _weight_digest(champion)
    if weight_before != weight_after:
        raise AssertionError("Champion policy weights changed in Shadow audit")
    summary = summarize(records, wins, games,
                        weights_unchanged=True,
                        actor_logits_checks=actor_checks,
                        zero_prior_checks=zero_checks)
    summary.update({"seed_start": seed_start,
                    "games_per_profile": games_per_profile,
                    "champion_model_id": CHAMPION_ID,
                    "robber_model_id": ROBBER_ID,
                    "champion_weight_digest": weight_after,
                    "candidate_surface_version": SURFACE_VERSION})
    _write_json(root / "summary.json", summary)
    _write_json(root / "games.json", games)
    with (root / "decisions.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (root / "winning_actions.jsonl").open("w", encoding="utf-8") as stream:
        for row in wins:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="strategic_gate_shadow_step3a13_20261004")
    parser.add_argument("--games-per-profile", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=2940000)
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."} or args.games_per_profile < 1):
        parser.error("Invalid experiment ID or games-per-profile")
    root = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if (root / "summary.json").exists():
        parser.error("Experiment already complete; use a fresh ID")
    summary = collect(root, games_per_profile=args.games_per_profile,
                      seed_start=args.seed_start)
    print(json.dumps({key: summary[key] for key in
                      ("game_count", "shadow_decisions", "winning_action_detected",
                       "winning_action_missed_by_champion", "parity_games")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
