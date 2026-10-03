"""Step 3A-7: 建設可能局面のstate-based Surfaceと非介入Shadow診断。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import fmean, median
from time import perf_counter
from typing import Any

import numpy as np
import torch

from app.domain.actions import (Action, BuildCityAction, BuildSettlementAction,
                                ProposeTradeAction)
from app.domain.game import GameState, player_for

from .action_families import action_family
from .action_space import (ACTION_CATALOG, ACTION_SPACE_SIZE, action_to_id,
                           get_action_mask, id_to_action)
from .hierarchical_distribution import (FAMILY_COUNT,
                                        HierarchicalMaskableCategoricalDistribution)
from .observation import encode_observation, get_observation
from .reward import actual_score
from .runtime_context import (RUNTIME_CONTEXT_SIZE, encode_runtime_context,
                              runtime_context_definition)
from .settlement_planning_agent import SettlementPlanningAgent
from .structured_context import build_structured_context_report


SURFACE_VERSION = "construction_ready_choice_v1"
SHADOW_RECORD_VERSION = "construction_shadow_v1"
TARGET_BUILD_FAMILIES = frozenset({"settlement", "city"})
CONTEXT_ANALYSIS_FIELDS = (
    "settlement.candidate_count", "settlement.min_additional_roads",
    "settlement.eta", "city.best_production_gain", "city.eta",
    "risk.hand_size", "risk.discard_exposure", "robber.blocked_pips",
    "road.acquisition_gap", "road.defense_gap", "knight.plays_needed",
    "knight.draws_needed",
)
_FIELD_DEFINITIONS = {field["name"]: field
                      for field in runtime_context_definition()["field_order"]}


@dataclass(frozen=True)
class ConstructionSurface:
    subtype: str | None
    exclusion: str | None
    legal_settlement_count: int
    legal_city_count: int

    @property
    def eligible(self) -> bool:
        return self.subtype is not None and self.exclusion is None


def detect_construction_surface(game: GameState, player_id: int,
                                mask: np.ndarray | None = None) -> ConstructionSurface:
    """最終Actionに一切依存せず、Action前stateと合法maskだけで判定。"""
    if game.phase != "action" or game.current_player_id != player_id:
        return ConstructionSurface(None, "OUTSIDE_ACTION_PHASE", 0, 0)
    mask = get_action_mask(game, player_id) if mask is None else mask
    if mask.shape != (ACTION_SPACE_SIZE,):
        raise ValueError("377 Action maskの次元が不正です。")
    settlements = sum(bool(mask[item.id]) and isinstance(item.action, BuildSettlementAction)
                      for item in ACTION_CATALOG)
    cities = sum(bool(mask[item.id]) and isinstance(item.action, BuildCityAction)
                 for item in ACTION_CATALOG)
    if not settlements and not cities:
        return ConstructionSurface(None, "NO_IMMEDIATE_CONSTRUCTION", 0, 0)
    subtype = ("SETTLEMENT_AND_CITY" if settlements and cities
               else "SETTLEMENT_ONLY" if settlements else "CITY_ONLY")
    if game.winner_id is not None:
        return ConstructionSurface(subtype, "GAME_ALREADY_OVER", settlements, cities)
    if game.free_road_remaining:
        return ConstructionSurface(subtype, "FREE_ROAD_SEQUENCE", settlements, cities)
    if game.pending_trade is not None:
        return ConstructionSurface(subtype, "PENDING_TRADE", settlements, cities)
    if mask.sum() <= 1:
        return ConstructionSurface(subtype, "FORCED_SINGLE_ACTION", settlements, cities)
    score = actual_score(game, player_id)
    if score >= 9:
        return ConstructionSurface(subtype, "IMMEDIATE_BUILD_WIN", settlements, cities)
    if score >= 8:
        context = build_structured_context_report(game, player_id)
        if (context.sections["road_title"]["immediate_win"].value
                or context.sections["knight_title"]["immediate_win"].value):
            return ConstructionSurface(subtype, "IMMEDIATE_TITLE_WIN", settlements, cities)
    return ConstructionSurface(subtype, None, settlements, cities)


def _action_family(action: Action) -> str:
    if isinstance(action, ProposeTradeAction):
        return "player_trade"
    try:
        return action_family(action)
    except ValueError:
        return type(action).__name__


def _action_record(action: Action) -> dict[str, Any]:
    try:
        catalog_id = action_to_id(action)
    except ValueError:
        catalog_id = None
    return {"type": type(action).__name__, "family": _action_family(action),
            "catalog_id": catalog_id, "fields": asdict(action)}


def _decoded_context(vector: np.ndarray) -> dict[str, float | None]:
    values = {}
    for name in CONTEXT_ANALYSIS_FIELDS:
        field = _FIELD_DEFINITIONS[name]
        index = field["vector_index"]
        if field["valid_mask"] and not vector[index + 1]:
            values[name] = None
        elif field["normalization"] == "four_minus_ratio_over_two":
            values[name] = 4.0 - 2.0 * float(vector[index])
        else:
            values[name] = float(vector[index]) * field["scale"]
    return values


def _shadow_proposal(model, observation: np.ndarray,
                     mask: np.ndarray) -> dict[str, Any]:
    """実適用せず階層分布のdeterministic modeを取得。top-kは全377分布順。"""
    if not mask.any():
        raise AssertionError("Surface上のlegal maskは空になりません。")
    policy = model.policy
    with torch.no_grad():
        tensor, _ = policy.obs_to_tensor(observation)
        logits = policy.action_net(policy.mlp_extractor.forward_actor(tensor))
        distribution = HierarchicalMaskableCategoricalDistribution()
        distribution.proba_distribution(logits)
        distribution.apply_masking(mask)
        selected_id = int(distribution.mode().item())
        probabilities = distribution.distribution.probs[0].cpu().numpy()
        raw_logits = logits[0].cpu().numpy()
    if raw_logits.shape != (ACTION_SPACE_SIZE + FAMILY_COUNT,):
        raise AssertionError("Shadow logitsの377 Action + family次元が不正です。")
    if not mask[selected_id]:
        raise AssertionError("Shadow提案が非法Actionです。")
    top_indices = np.argsort(-probabilities, kind="stable")[:5]
    return {
        "selected_action": _action_record(id_to_action(selected_id)),
        "selected_probability": float(probabilities[selected_id]),
        "top_k": [{"action_id": int(index), "action_name": ACTION_CATALOG[int(index)].name,
                   "family": _action_family(id_to_action(int(index))),
                   "probability": float(probabilities[index])}
                  for index in top_indices if mask[index]],
        "action_logits": raw_logits[:ACTION_SPACE_SIZE].astype(float).tolist(),
        "family_logits": raw_logits[ACTION_SPACE_SIZE:].astype(float).tolist(),
        "legal_mask": mask.astype(bool).tolist(),
    }


class ConstructionShadowAgent(SettlementPlanningAgent):
    """Champion選択を一度だけ行い、Surface局面でv7提案のみ観測。"""

    def __init__(self, champion, shadow, **planning):
        super().__init__(champion, **planning)
        self.shadow = shadow
        self.records: list[dict[str, Any]] = []
        self.surface_counts: Counter[str] = Counter()
        self.exclusion_counts: Counter[str] = Counter()
        self.timings: dict[str, list[float]] = {
            "surface_detection": [], "observation_generation": [],
            "runtime_context_generation": [], "policy_forward": [],
        }
        self.game_seed: int | None = None
        self.opponent_profile: str | None = None

    def select_action(self, game: GameState, player_id: int) -> Action:
        if game.phase != "action":
            return super().select_action(game, player_id)
        counter_before = game.random_source.counter
        start = perf_counter()
        mask = get_action_mask(game, player_id)
        surface = detect_construction_surface(game, player_id, mask)
        self.timings["surface_detection"].append(perf_counter() - start)
        if surface.exclusion != "NO_IMMEDIATE_CONSTRUCTION":
            self.surface_counts[surface.subtype or "NONE"] += 1
            if surface.exclusion is not None:
                self.exclusion_counts[surface.exclusion] += 1
        if not surface.eligible:
            return super().select_action(game, player_id)

        start = perf_counter()
        v2 = encode_observation(get_observation(game, player_id, version="v2"))
        self.timings["observation_generation"].append(perf_counter() - start)
        start = perf_counter()
        context = encode_runtime_context(game, player_id,
                                         max_plan_roads=self.max_plan_roads)
        self.timings["runtime_context_generation"].append(perf_counter() - start)
        if context.shape != (RUNTIME_CONTEXT_SIZE,) or not np.isfinite(context).all():
            raise AssertionError("Contextのshapeまたは有限性が不正です。")
        observation = np.concatenate((v2, context)).astype(np.float32, copy=False)
        action, _, trace = self._select_action_with_diagnostics(game, player_id)
        start = perf_counter()
        proposal = _shadow_proposal(self.shadow.model, observation, mask)
        self.timings["policy_forward"].append(perf_counter() - start)
        if game.random_source.counter != counter_before:
            raise AssertionError("Shadow診断がゲームRNGを消費しました。")
        source = trace.execution_source.value
        if source == "BASE_PPO" and (trace.contributed_reasons or trace.contributed_intents):
            raise AssertionError("BASE_PPOへPlanner由来の理由またはIntentを誤帰属しました。")
        reasons = ([item.value for item in trace.contributed_reasons]
                   if source != "BASE_PPO" else [])
        if source == "BASE_PPO" and reasons:
            raise AssertionError("BASE_PPOへPlanner Reasonを帰属させました。")
        shadow_action = id_to_action(proposal["selected_action"]["catalog_id"])
        if source == "BASE_PPO" and action != shadow_action:
            raise AssertionError("BASE_PPOとzero-impact Shadow Actionが不一致です。")
        player = player_for(game, player_id)
        self.records.append({
            "record_schema": SHADOW_RECORD_VERSION,
            "surface_version": SURFACE_VERSION,
            "game_seed": self.game_seed,
            "game_id": game.id,
            "opponent_profile": self.opponent_profile,
            "turn": game.turn_number,
            "round": game.round_number,
            "phase": game.phase,
            "player_id": player_id,
            "seat": game.seat_order.index(player_id) + 1,
            "score": actual_score(game, player_id),
            "site_count": player.settlements + player.cities,
            "city_count": player.cities,
            "surface_subtype": surface.subtype,
            "legal_settlement_count": surface.legal_settlement_count,
            "legal_city_count": surface.legal_city_count,
            "context_features": _decoded_context(context),
            "runtime_context_vector": context.astype(float).tolist(),
            "champion_action": _action_record(action),
            "champion_execution_source": source,
            "champion_source_reasons": reasons,
            "champion_contributed_intents": ([item.value for item in trace.contributed_intents]
                                            if source != "BASE_PPO" else []),
            "shadow": proposal,
            "exact_action_equal": action == shadow_action,
            "family_equal": _action_family(action) == _action_family(shadow_action),
            "target_build_equal": ((_action_family(action) in TARGET_BUILD_FAMILIES)
                                   == (_action_family(shadow_action) in TARGET_BUILD_FAMILIES)),
        })
        return action


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None,
                "max": None, "p95": None}
    ordered = sorted(values)
    return {"count": len(values), "mean": fmean(values), "median": median(values),
            "min": ordered[0], "max": ordered[-1],
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],}


def summarize_shadow(records: list[dict[str, Any]],
                     games: list[dict[str, Any]],
                     timings: dict[str, list[float]]) -> dict[str, Any]:
    """勝率ではなくSurface coverage・Planner介入・不一致の構造を集計。"""
    count = len(records)
    champion_families = Counter(row["champion_action"]["family"] for row in records)
    shadow_families = Counter(row["shadow"]["selected_action"]["family"] for row in records)
    subtypes = Counter(row["surface_subtype"] for row in records)
    sources = Counter(row["champion_execution_source"] for row in records)
    exact = sum(row["exact_action_equal"] for row in records)
    family = sum(row["family_equal"] for row in records)
    build = sum(row["target_build_equal"] for row in records)
    both_construction = [row for row in records
                         if row["champion_action"]["family"] in TARGET_BUILD_FAMILIES
                         and row["shadow"]["selected_action"]["family"] in TARGET_BUILD_FAMILIES]
    mismatch = [row for row in records if not row["exact_action_equal"]]
    end_turn_mismatch = [row for row in mismatch
                         if "end_turn" in (row["champion_action"]["family"],
                                           row["shadow"]["selected_action"]["family"])]
    groups = {"all": records, "mismatch": mismatch,
              "end_turn_mismatch": end_turn_mismatch,
              "city_planner_mismatch": [row for row in mismatch
                                        if row["champion_execution_source"] == "CITY_PLANNER"],
              "settlement_planner_mismatch": [row for row in mismatch
                                              if row["champion_execution_source"] == "SETTLEMENT_PLANNER"]}
    context_groups = {}
    for group_name, rows in groups.items():
        context_groups[group_name] = {
            field: _numeric_summary([float(row["context_features"][field])
                                     for row in rows if row["context_features"][field] is not None])
            for field in CONTEXT_ANALYSIS_FIELDS
        }
        for field in ("score", "site_count", "city_count"):
            context_groups[group_name][field] = _numeric_summary(
                [float(row[field]) for row in rows])

    per_profile = {}
    for profile in ("rule", "champion", "mixed"):
        subset = [row for row in records if row["opponent_profile"] == profile]
        per_profile[profile] = {
            "games": sum(game["opponent_profile"] == profile for game in games),
            "surface_decisions": len(subset),
            "seat_counts": dict(sorted(Counter(row["seat"] for row in subset).items())),
            "subtypes": dict(Counter(row["surface_subtype"] for row in subset)),
            "champion_families": dict(Counter(row["champion_action"]["family"]
                                              for row in subset)),
            "shadow_families": dict(Counter(row["shadow"]["selected_action"]["family"]
                                            for row in subset)),
        }
    transitions = Counter(
        f'{row["champion_action"]["family"]}->{row["shadow"]["selected_action"]["family"]}'
        for row in records)
    return {
        "record_schema": SHADOW_RECORD_VERSION,
        "surface_version": SURFACE_VERSION,
        "game_count": len(games),
        "surface_decisions": count,
        "subtypes": dict(subtypes),
        "per_profile": per_profile,
        "champion_action_families": dict(champion_families),
        "shadow_action_families": dict(shadow_families),
        "family_transitions": dict(transitions),
        "exact_action_agreement": {"count": exact, "rate": exact / count if count else None},
        "family_agreement": {"count": family, "rate": family / count if count else None},
        "target_build_vs_other_agreement": {"count": build,
                                            "rate": build / count if count else None},
        "settlement_vs_city_agreement_when_both_build": {
            "denominator": len(both_construction),
            "count": sum(row["family_equal"] for row in both_construction),
            "rate": (sum(row["family_equal"] for row in both_construction)
                     / len(both_construction) if both_construction else None),
        },
        "champion_execution_sources": dict(sources),
        "planner_route_rate": (sum(source != "BASE_PPO" for source in sources.elements())
                               / count if count else None),
        "effective_override_rate": len(mismatch) / count if count else None,
        "end_turn_mismatch_count": len(end_turn_mismatch),
        "context_by_group": context_groups,
        "timings_ms": {name: _numeric_summary([value * 1000 for value in values])
                       for name, values in timings.items()},
        "all_games_parity": all(game["parity"] for game in games),
        "base_ppo_reason_misattribution": sum(
            bool(row["champion_source_reasons"])
            for row in records if row["champion_execution_source"] == "BASE_PPO"
        ),
    }
