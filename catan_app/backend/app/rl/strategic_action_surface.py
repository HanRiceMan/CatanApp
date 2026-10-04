"""Step 3A-12: non-invasive audit of an algorithmic strategic-action mask.

This module does not choose a family or a concrete action. In particular,
Champion-selected actions and Planner scores never influence availability.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from app.domain.actions import (Action, BankTradeAction, BuildCityAction,
                                BuildRoadAction, BuildSettlementAction,
                                BuyDevelopmentAction, EndTurnAction,
                                ProposeTradeAction, UseDevelopmentAction)
from app.domain.game import GameState, _longest_road_length, player_for

from .action_space import ACTION_CATALOG, ACTION_SPACE_SIZE, action_to_id, get_action_mask
from .expansion_planner import _routes_from_network, _settlement_candidates
from .reward import actual_score
from .settlement_planning_agent import SettlementPlanningAgent


SURFACE_VERSION = "strategic_action_surface_step3a12_v1"
FAMILIES = ("BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD",
            "BUY_DEVELOPMENT", "BANK_TRADE", "END_TURN")
_FAMILY_TYPES = (BuildSettlementAction, BuildCityAction, BuildRoadAction,
                 BuyDevelopmentAction, BankTradeAction, EndTurnAction)
_CATALOG_BY_FAMILY = {
    name: tuple(item for item in ACTION_CATALOG if isinstance(item.action, action_type))
    for name, action_type in zip(FAMILIES, _FAMILY_TYPES, strict=True)
}


def family_of(action: Action) -> str:
    for family, action_type in zip(FAMILIES, _FAMILY_TYPES, strict=True):
        if isinstance(action, action_type):
            return family
    if isinstance(action, ProposeTradeAction):
        return "PROTECTED_PLAYER_TRADE"
    if isinstance(action, UseDevelopmentAction):
        return "PROTECTED_PLAY_DEVELOPMENT"
    return f"PROTECTED_{type(action).__name__}"


def _road_purposes(game: GameState, player_id: int,
                   roads: list[BuildRoadAction]) -> dict[str, Any]:
    """Structural expansion and longest-length facts, not a road-value score."""
    if not roads:
        return {"expansion_edges": [], "longest_progress_edges": [],
                "both_edges": [], "other_edges": []}
    routes = _routes_from_network(game, player_id)
    expansion = {
        edge_id
        for vertex_id in _settlement_candidates(game)
        if (route := routes.get(vertex_id)) is not None and 1 <= route[0] <= 3
        for edge_id in route[1]
    }
    before = _longest_road_length(game, player_id)
    progress: set[int] = set()
    for road in roads:
        if road.edge_id in game.roads:
            raise ValueError("Road diagnostic requires unoccupied legal edges")
        game.roads[road.edge_id] = player_id
        try:
            if _longest_road_length(game, player_id) > before:
                progress.add(road.edge_id)
        finally:
            del game.roads[road.edge_id]
    legal_ids = {road.edge_id for road in roads}
    expansion &= legal_ids
    return {
        "expansion_edges": sorted(expansion),
        "longest_progress_edges": sorted(progress),
        "both_edges": sorted(expansion & progress),
        "other_edges": sorted(legal_ids - expansion - progress),
    }


def _hard_exclusion(game: GameState, player_id: int,
                    legal: dict[str, list[Action]], mask: np.ndarray) -> str | None:
    if game.winner_id is not None:
        return "GAME_OVER"
    if game.free_road_remaining:
        return "FREE_ROAD"
    if game.pending_trade is not None:
        return "PENDING_TRADE"
    if int(mask.sum()) <= 1:
        return "FORCED_SINGLE_ACTION"
    score = actual_score(game, player_id)
    if score >= 9 and (legal["BUILD_SETTLEMENT"] or legal["BUILD_CITY"]):
        return "IMMEDIATE_CONSTRUCTION_WIN"
    if score >= 8:
        for road in legal["BUILD_ROAD"]:
            if SettlementPlanningAgent._road_wins_now(game, player_id, road):
                return "IMMEDIATE_ROAD_TITLE_WIN"
        knight_legal = bool(mask[action_to_id(UseDevelopmentAction("knight"))])
        if knight_legal and game.largest_army_holder_id != player_id:
            player = player_for(game, player_id)
            rival = max(other.played_knights for other in game.players
                        if other.id != player_id)
            if player.played_knights + 1 >= max(3, rival + 1):
                return "IMMEDIATE_KNIGHT_TITLE_WIN"
    return None


@dataclass(frozen=True)
class StrategicActionSurface:
    phase: str
    family_mask: tuple[bool, ...]
    legal_action_counts: tuple[int, ...]
    hard_exclusion: str | None
    road_purposes: dict[str, Any]

    @property
    def candidate_count(self) -> int:
        return sum(self.family_mask)

    @property
    def eligible(self) -> bool:
        return (self.hard_exclusion is None and self.candidate_count >= 2
                and any(self.family_mask[:-1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_version": SURFACE_VERSION,
            "phase": self.phase,
            "families": list(FAMILIES),
            "candidate_mask": list(self.family_mask),
            "legal_action_counts": dict(zip(FAMILIES, self.legal_action_counts,
                                            strict=True)),
            "candidate_count": self.candidate_count,
            "hard_exclusion": self.hard_exclusion,
            "gate_eligible": self.eligible,
            "road_purposes": self.road_purposes,
        }


def detect_strategic_action_surface(
    game: GameState, player_id: int, mask: np.ndarray | None = None,
) -> StrategicActionSurface:
    """State-based predicate: immediate legal families plus a few hard exceptions."""
    if game.phase != "action" or game.current_player_id != player_id:
        return StrategicActionSurface(game.phase, (False,) * len(FAMILIES),
                                      (0,) * len(FAMILIES), "OUTSIDE_ACTION_PHASE",
                                      _road_purposes(game, player_id, []))
    mask = get_action_mask(game, player_id) if mask is None else mask
    if mask.shape != (ACTION_SPACE_SIZE,):
        raise ValueError("Expected 377-element legal Action mask")
    legal = {family: [item.action for item in _CATALOG_BY_FAMILY[family]
                      if mask[item.id]] for family in FAMILIES}
    counts = tuple(len(legal[family]) for family in FAMILIES)
    # Development-card actions are protected, but count for forced-action test.
    hard = _hard_exclusion(game, player_id, legal, mask)
    roads = _road_purposes(game, player_id, legal["BUILD_ROAD"])
    return StrategicActionSurface(game.phase, tuple(count > 0 for count in counts),
                                  counts, hard, roads)


def summarize_strategic_surface(records: list[dict[str, Any]],
                                games: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage is descriptive; the Champion's family is never a Teacher label."""
    all_count = len(records)
    eligible = [row for row in records if row["gate_eligible"]]
    profile = Counter(row["profile"] for row in eligible)
    seat = Counter(row["seat"] for row in eligible)
    candidate_count = Counter(row["candidate_count"] for row in records)
    eligible_candidate_count = Counter(row["candidate_count"] for row in eligible)
    available = Counter(family for row in eligible for family, present in
                        zip(FAMILIES, row["candidate_mask"], strict=True) if present)
    pairs = Counter(f"{a}+{b}" for row in eligible for a, b in combinations(
        [family for family, present in zip(FAMILIES, row["candidate_mask"], strict=True)
         if present], 2))
    actual = Counter(row["champion_family"] for row in eligible)
    source = Counter(row["champion_execution_source"] for row in eligible)
    misses = Counter(row["champion_family"] for row in eligible
                     if not row["champion_in_candidate_set"])
    road = Counter()
    for row in eligible:
        if row["candidate_mask"][FAMILIES.index("BUILD_ROAD")]:
            purposes = row["road_purposes"]
            road["road_candidate_states"] += 1
            for name in ("expansion_edges", "longest_progress_edges",
                         "both_edges", "other_edges"):
                road[f"states_with_{name}"] += bool(purposes[name])
                road[f"legal_{name}"] += len(purposes[name])
            if row["champion_family"] == "BUILD_ROAD":
                edge = row["champion_action_fields"]["edge_id"]
                road["champion_road_actions"] += 1
                for name in ("expansion_edges", "longest_progress_edges",
                             "both_edges", "other_edges"):
                    road[f"champion_in_{name}"] += edge in purposes[name]
    return {
        "schema_version": SURFACE_VERSION,
        "game_count": len(games), "action_phase_decisions": all_count,
        "gate_eligible_decisions": len(eligible),
        "profile_eligible": dict(profile), "seat_eligible": dict(seat),
        "candidate_count_all": dict(candidate_count),
        "candidate_count_eligible": dict(eligible_candidate_count),
        "single_candidate_count": candidate_count.get(1, 0),
        "two_candidate_count": candidate_count.get(2, 0),
        "three_or_more_candidate_count": sum(value for count, value in
                                              candidate_count.items() if count >= 3),
        "family_availability_eligible": dict(available),
        "family_cooccurrence_eligible": dict(pairs),
        "champion_family_eligible": dict(actual),
        "champion_execution_source_eligible": dict(source),
        "champion_family_represented_eligible": sum(
            row["champion_in_candidate_set"] for row in eligible),
        "champion_family_unrepresented_eligible": dict(misses),
        "hard_exclusions": dict(Counter(row["hard_exclusion"] for row in records
                                       if row["hard_exclusion"] is not None)),
        "road_diagnostic": dict(road),
        "parity_games": sum(game["parity"] for game in games),
    }
