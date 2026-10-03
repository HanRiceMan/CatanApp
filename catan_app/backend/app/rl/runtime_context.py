"""Observation v7の決定的Context。Plannerの選択・理由・対局結果は含めない。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import numpy as np

from app.domain.game import RESOURCES, GameState, player_for

from .structured_context import build_structured_context_report_with_plan


RUNTIME_CONTEXT_VERSION = "runtime_context_v1"
RESOURCE_ORDER = RESOURCES
NEW_CARD_ORDER = ("knight", "road_building", "year_of_plenty", "monopoly", "victory_point")
NEW_CARD_MAX = (14, 2, 2, 2, 5)


@dataclass(frozen=True)
class ContextField:
    name: str
    unit: str
    range: str
    normalization: str
    scale: float
    provenance: str
    semantics: str
    valid_mask: bool = False

    def definition(self) -> dict[str, Any]:
        result = asdict(self)
        result["valid_mask_field"] = f"{self.name}.valid" if self.valid_mask else None
        return result


def _field(name: str, unit: str, upper: float, provenance: str,
           semantics: str, *, valid_mask: bool = False,
           normalization: str = "divide_by_fixed_scale") -> ContextField:
    raw_range = ("2..4" if normalization == "four_minus_ratio_over_two"
                 else f"0..{upper:g}")
    return ContextField(name, unit, raw_range, normalization, upper,
                        provenance, semantics, valid_mask)


RAW = "RAW_V2_DUPLICATE"
DERIVED = "DETERMINISTIC_FROM_V2"
BOARD = "GNN_BOARD_SUMMARY"
MISSING = "MISSING_FROM_V2"


CONTEXT_FIELDS: tuple[ContextField, ...] = (
    _field("settlement.candidate_exists", "bool", 1, BOARD, "道路3本以内に、現在使える開拓コマで届く候補がある"),
    _field("settlement.candidate_count", "vertices", 54, BOARD, "到達可能な開拓候補の数"),
    _field("settlement.min_additional_roads", "roads", 15, BOARD, "候補へ必要な最少道路数。Planner順位は使わない", valid_mask=True),
    *(_field(f"settlement.shortage.{resource}", "cards", maximum, DERIVED,
             "最少道路候補への道路+開拓地予算の不足。最短道路、ETA、vertex ID順に決定",
             valid_mask=True)
      for resource, maximum in (("wood", 4), ("brick", 4), ("sheep", 1), ("wheat", 1))),
    _field("settlement.immediate_build", "bool", 1, DERIVED, "今、開拓地Actionが合法"),
    _field("settlement.eta", "rounds", 12, DERIVED, "最少道路候補の生産・港による概算待ち", valid_mask=True),

    _field("city.upgradeable_count", "vertices", 5, BOARD, "将来都市化できる自分の開拓地数"),
    _field("city.best_production_gain", "pips", 15, BOARD, "既存開拓地の都市化による最大pip増分", valid_mask=True),
    _field("city.shortage.wheat", "cards", 2, DERIVED, "都市費用の小麦不足"),
    _field("city.shortage.ore", "cards", 3, DERIVED, "都市費用の鉱石不足"),
    _field("city.immediate_build", "bool", 1, DERIVED, "今、都市Actionが合法"),
    _field("city.eta", "rounds", 12, DERIVED, "都市資源の概算待ち", valid_mask=True),
    _field("city.pieces_remaining", "pieces", 4, DERIVED, "残り都市コマ"),

    _field("road.current_length", "edges", 15, BOARD, "自分の最長連結道路長"),
    _field("road.acquisition_gap", "edges", 15, BOARD, "非保持時に称号取得へ必要な長さ差", valid_mask=True),
    _field("road.defense_gap", "edges", 15, BOARD, "保持時の他者最長長との差", valid_mask=True),
    _field("road.structurally_extendable_count", "edges", 72, BOARD, "手札を無視した合法延伸edge数"),
    _field("road.affordable_title_directed_count", "edges", 72, MISSING, "今合法かつ最長長を増すedge数"),
    _field("road.immediate_acquisition", "bool", 1, DERIVED, "道1本で即称号獲得可能"),
    _field("road.immediate_win", "bool", 1, DERIVED, "道1本と称号点で即勝利可能"),
    _field("road.shortage.wood", "cards", 1, DERIVED, "有料道路1本の木材不足"),
    _field("road.shortage.brick", "cards", 1, DERIVED, "有料道路1本のレンガ不足"),

    _field("knight.used", "cards", 14, RAW, "使用済み騎士数"),
    _field("knight.usable", "cards", 14, MISSING, "新規購入カードを除いた使用可能騎士数"),
    _field("knight.plays_needed", "plays", 14, DERIVED, "最大騎士力へ追加で必要な使用数"),
    _field("knight.draws_needed", "cards", 14, DERIVED, "保持騎士を全使用しても必要な追加獲得枚数"),
    _field("knight.deck_remaining", "cards", 25, RAW, "発展山札の残数"),
    _field("knight.development_affordable", "bool", 1, DERIVED, "山札と資源上、発展を購入可能"),
    _field("knight.playable_now", "bool", 1, MISSING, "今騎士使用Actionが合法"),
    _field("knight.immediate_acquisition", "bool", 1, MISSING, "騎士1枚で即称号獲得"),
    _field("knight.immediate_win", "bool", 1, MISSING, "騎士1枚と称号点で即勝利"),

    _field("robber.blocked_pips", "pips", 30, BOARD, "盗賊が塞ぐ自分の生産pip"),
    _field("robber.blocked_ratio", "ratio", 1, BOARD, "全生産pip中の盗賊阻害割合", valid_mask=True),
    *(_field(f"robber.blocked.{resource}", "pips", 30, BOARD,
             "盗賊が塞ぐ資源別生産pip") for resource in RESOURCE_ORDER),
    _field("robber.playable_knight", "bool", 1, MISSING, "今騎士で盗賊を動かせる"),
    _field("robber.removable_now", "bool", 1, MISSING, "今騎士か盗賊移動phaseで解除可能"),

    _field("risk.hand_size", "cards", 95, RAW, "本人資源手札合計"),
    _field("risk.discard_exposure", "bool", 1, DERIVED, "7なら半数破棄の対象"),
    _field("risk.discard_count_if_seven", "cards", 47, DERIVED, "今7が出た場合の破棄枚数"),
    _field("risk.resource_diversity", "types", 5, DERIVED, "保有資源種類数"),
    *(_field(f"risk.development_shortage.{resource}", "cards", 1, DERIVED,
             "発展購入資源の不足") for resource in ("sheep", "wheat", "ore")),
    _field("risk.bank_trade_options", "pairs", 20, DERIVED, "銀行在庫も考慮した交換可能な資源ペア数"),
    *(_field(f"risk.port_advantage.{resource}", "advantage", 1, DERIVED,
             "(4-交換比)/2。4:1=0、3:1=0.5、2:1=1",
             normalization="four_minus_ratio_over_two") for resource in RESOURCE_ORDER),
    *(_field(f"risk.legal_family.{family}", "actions", maximum, MISSING,
             "377 mask内の現在合法なAction family件数")
      for family, maximum in (("road", 72), ("settlement", 54), ("city", 54),
                              ("development", 1), ("bank_trade", 20),
                              ("play_development", 22))),
    _field("risk.immediate_construction", "bool", 1, DERIVED, "道・開拓地・都市を今建設可能"),

    *(_field(f"new_development.{card}", "cards", maximum, MISSING,
             "本人の今手番に購入した発展カード。相手カードは含まない")
      for card, maximum in zip(NEW_CARD_ORDER, NEW_CARD_MAX, strict=True)),
)

RUNTIME_CONTEXT_SIZE = sum(1 + int(field.valid_mask) for field in CONTEXT_FIELDS)
CONTEXT_FIELD_ORDER = tuple(field.name for field in CONTEXT_FIELDS)


def runtime_context_definition() -> dict[str, Any]:
    order = []
    index = 0
    for field in CONTEXT_FIELDS:
        definition = field.definition()
        definition["vector_index"] = index
        definition["encoded_range"] = "0..1"
        order.append(definition)
        index += 1 + int(field.valid_mask)
    return {
        "version": RUNTIME_CONTEXT_VERSION,
        "base_observation_version": "v2",
        "dimension": RUNTIME_CONTEXT_SIZE,
        "field_order": order,
        "invalid_value_encoding": "zero_with_separate_valid_mask",
        "excluded_student_inputs": [
            "objective_goal", "strategic_intent", "execution_source",
            "reason_contributed_to_selection", "source_confirmed_reasons",
            "planner_selected_action", "planner_selected_vertex", "site_value",
            "plan_value", "cross_goal_planner_score", "winner", "final_vp",
            "final_rank", "future_outcome", "teacher_attribution",
            "bayesian_hand_belief", "resource_events", "trade_history",
        ],
    }


def _source_values(game: GameState, player_id: int, max_plan_roads: int) -> tuple[dict[str, Any], dict[str, bool]]:
    report, plan = build_structured_context_report_with_plan(
        game, player_id, max_plan_roads=max_plan_roads)
    sections = report.sections
    values: dict[str, Any] = {}
    valid: dict[str, bool] = {}

    def put(name: str, section: str, field: str) -> None:
        signal = sections[section][field]
        valid[name] = signal.valid
        values[name] = signal.value if signal.valid else None

    for name, section, field in (
        ("settlement.candidate_exists", "settlement", "candidate_exists"),
        ("settlement.candidate_count", "settlement", "candidate_count"),
        ("settlement.min_additional_roads", "settlement", "min_additional_roads"),
        ("settlement.immediate_build", "settlement", "immediate_build_possible"),
        ("city.upgradeable_count", "city", "upgradeable_count"),
        ("city.best_production_gain", "city", "best_future_production_gain"),
        ("city.immediate_build", "city", "immediate_build_possible"),
        ("city.eta", "city", "estimated_wait_rounds"),
        ("city.pieces_remaining", "city", "city_pieces_remaining"),
        ("road.current_length", "road_title", "current_length"),
        ("road.acquisition_gap", "road_title", "acquisition_gap"),
        ("road.defense_gap", "road_title", "defense_gap"),
        ("road.structurally_extendable_count", "road_title", "structurally_extendable_edge_count"),
        ("road.affordable_title_directed_count", "road_title", "affordable_title_directed_edge_count"),
        ("road.immediate_acquisition", "road_title", "immediate_acquisition"),
        ("road.immediate_win", "road_title", "immediate_win"),
        ("knight.used", "knight_title", "used_knights"),
        ("knight.usable", "knight_title", "usable_knights"),
        ("knight.plays_needed", "knight_title", "plays_needed"),
        ("knight.draws_needed", "knight_title", "draws_needed"),
        ("knight.deck_remaining", "knight_title", "development_deck_remaining"),
        ("knight.development_affordable", "knight_title", "development_affordable"),
        ("knight.playable_now", "knight_title", "knight_playable_now"),
        ("knight.immediate_acquisition", "knight_title", "immediate_acquisition"),
        ("knight.immediate_win", "knight_title", "immediate_win"),
        ("robber.blocked_pips", "robber", "blocked_production_pips"),
        ("robber.blocked_ratio", "robber", "blocked_production_ratio"),
        ("robber.playable_knight", "robber", "playable_knight_available"),
        ("robber.removable_now", "robber", "immediate_removal_possible"),
        ("risk.hand_size", "resource_risk", "hand_size"),
        ("risk.discard_exposure", "resource_risk", "discard_exposure"),
        ("risk.discard_count_if_seven", "resource_risk", "discard_count_if_seven"),
        ("risk.resource_diversity", "resource_risk", "resource_diversity"),
        ("risk.bank_trade_options", "resource_risk", "available_bank_trade_options"),
        ("risk.immediate_construction", "resource_risk", "immediate_construction_available"),
    ):
        put(name, section, field)

    # 最短距離・同距離では最短ETAで決める。site/plan_valueを読まない。
    player = player_for(game, player_id)
    canonical = (min(plan.targets, key=lambda target: (
        target.additional_roads, target.wait_rounds, target.vertex_id,
    )) if plan.targets and player.settlements < 5 else None)
    resources = (dict(zip(RESOURCE_ORDER, canonical.required_resources, strict=True))
                 if canonical else {})
    for resource in ("wood", "brick", "sheep", "wheat"):
        name = f"settlement.shortage.{resource}"
        values[name] = max(0, resources[resource] - player.resources[resource]) if canonical else None
        valid[name] = canonical is not None
    eta = canonical.wait_rounds if canonical else None
    values["settlement.eta"] = eta if eta is not None and isfinite(eta) and eta <= 12 else None
    valid["settlement.eta"] = values["settlement.eta"] is not None

    for resource in ("wheat", "ore"):
        name = f"city.shortage.{resource}"
        values[name] = sections["city"]["resource_shortage"].value[resource]
        valid[name] = True
    for resource in ("wood", "brick"):
        name = f"road.shortage.{resource}"
        values[name] = sections["road_title"]["resource_shortage"].value[resource]
        valid[name] = True
    for resource in RESOURCE_ORDER:
        name = f"robber.blocked.{resource}"
        values[name] = sections["robber"]["blocked_resource_vector"].value[resource]
        valid[name] = True
        name = f"risk.port_advantage.{resource}"
        values[name] = sections["resource_risk"]["port_trade_ratios"].value[resource]
        valid[name] = True
    for resource in ("sheep", "wheat", "ore"):
        name = f"risk.development_shortage.{resource}"
        values[name] = sections["resource_risk"]["build_shortage"].value["development"][resource]
        valid[name] = True

    legal = sections["resource_risk"]["legal_action_count_by_family"].value
    families = {
        "road": ("BuildRoadAction",),
        "settlement": ("BuildSettlementAction",),
        "city": ("BuildCityAction",),
        "development": ("BuyDevelopmentAction",),
        "bank_trade": ("BankTradeAction",),
        "play_development": ("UseDevelopmentAction",),
    }
    for family, action_names in families.items():
        name = f"risk.legal_family.{family}"
        values[name] = sum(legal.get(action, 0) for action in action_names)
        valid[name] = True
    new_cards = player.new_development_cards
    for card in NEW_CARD_ORDER:
        name = f"new_development.{card}"
        values[name] = new_cards.count(card)
        valid[name] = True
    return values, valid


def encode_runtime_context(game: GameState, player_id: int, *,
                           max_plan_roads: int = 3) -> np.ndarray:
    """Action前の状態だけを読む。valid=falseは値0+独立maskで符号化。"""
    values, valid = _source_values(game, player_id, max_plan_roads)
    expected = set(CONTEXT_FIELD_ORDER)
    if set(values) != expected or set(valid) != expected:
        raise AssertionError(f"Runtime Context field不一致: {sorted(expected ^ set(values))}")
    result: list[float] = []
    for field in CONTEXT_FIELDS:
        usable = valid[field.name]
        raw = values[field.name]
        if usable:
            if raw is None or not isfinite(float(raw)):
                raise ValueError(f"有効Context値が未定義です: {field.name}")
            normalized = ((4.0 - float(raw)) / 2.0
                          if field.normalization == "four_minus_ratio_over_two"
                          else float(raw) / field.scale)
            if not -1e-7 <= normalized <= 1 + 1e-7:
                raise ValueError(f"Context値域外: {field.name}={raw}")
            result.append(normalized)
        else:
            if raw is not None:
                raise ValueError(f"invalid Contextに値が入っています: {field.name}")
            result.append(0.0)
        if field.valid_mask:
            result.append(float(usable))
    encoded = np.asarray(result, dtype=np.float32)
    if encoded.shape != (RUNTIME_CONTEXT_SIZE,) or not np.isfinite(encoded).all():
        raise AssertionError("Runtime Context vectorの形状または有限性が不正です。")
    return encoded
