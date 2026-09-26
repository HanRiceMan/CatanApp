"""PPO v1向けの固定長Action Spaceと合法Action Mask。"""

from dataclasses import dataclass
from itertools import combinations_with_replacement
from numbers import Integral
from typing import Final

import numpy as np

from app.domain.actions import (Action, BankTradeAction, BuildCityAction, BuildRoadAction,
                                BuildSettlementAction, BuyDevelopmentAction, DiscardHalfAction,
                                EndTurnAction, MoveRobberAction, PlaceInitialRoadAction,
                                PlaceInitialSettlementAction, RollOrderAction, RollTurnDiceAction,
                                StartGameAction, StealResourceAction, UseDevelopmentAction)
from app.domain.board import create_topology
from app.domain.game import RESOURCES, GameState, legal_actions

ACTION_SPACE_VERSION: Final[str] = "v1"


@dataclass(frozen=True)
class ActionDefinition:
    id: int
    name: str
    action: Action


def _build_catalog() -> tuple[ActionDefinition, ...]:
    actions: list[tuple[str, Action]] = [
        ("roll_order", RollOrderAction()),
        ("roll_turn", RollTurnDiceAction()),
    ]
    topology = create_topology()
    actions.extend((f"initial_settlement_v{vertex.id}", PlaceInitialSettlementAction(vertex.id))
                   for vertex in topology.vertices)
    actions.extend((f"initial_road_e{edge.id}", PlaceInitialRoadAction(edge.id)) for edge in topology.edges)
    actions.append(("start_game", StartGameAction()))
    actions.extend((f"move_robber_h{tile_id}", MoveRobberAction(tile_id)) for tile_id in range(19))
    actions.extend((f"steal_p{player_id}", StealResourceAction(player_id)) for player_id in range(1, 5))
    actions.extend((f"build_road_e{edge.id}", BuildRoadAction(edge.id)) for edge in topology.edges)
    actions.extend((f"build_settlement_v{vertex.id}", BuildSettlementAction(vertex.id))
                   for vertex in topology.vertices)
    actions.extend((f"build_city_v{vertex.id}", BuildCityAction(vertex.id)) for vertex in topology.vertices)
    actions.append(("buy_development", BuyDevelopmentAction()))
    actions.extend((f"bank_{give}_for_{receive}", BankTradeAction(give, receive))
                   for give in RESOURCES for receive in RESOURCES if give != receive)
    actions.extend((
        ("use_knight", UseDevelopmentAction("knight")),
        ("use_road_building", UseDevelopmentAction("road_building")),
    ))
    actions.extend((f"use_year_of_plenty_{first}_{second}", UseDevelopmentAction("year_of_plenty", (first, second)))
                   for first, second in combinations_with_replacement(RESOURCES, 2))
    actions.extend((f"use_monopoly_{resource}", UseDevelopmentAction("monopoly", (resource,)))
                   for resource in RESOURCES)
    actions.extend((
        ("discard_half_deterministically", DiscardHalfAction()),
        ("end_turn", EndTurnAction()),
    ))
    return tuple(ActionDefinition(index, name, action) for index, (name, action) in enumerate(actions))


ACTION_CATALOG: Final[tuple[ActionDefinition, ...]] = _build_catalog()
ACTION_SPACE_SIZE: Final[int] = len(ACTION_CATALOG)
_ACTION_TO_ID: Final[dict[Action, int]] = {definition.action: definition.id for definition in ACTION_CATALOG}


def action_to_id(action: Action) -> int:
    """PPO v1カタログ内のActionを固定IDへ変換する。

    サイコロの目やプレイヤー間交渉のような、方策が直接選ばない要素は対象外。
    豊作の2資源は ``RESOURCES`` 順の非減少順で渡す必要がある。
    """
    if isinstance(action, (RollOrderAction, RollTurnDiceAction)) and action.dice is not None:
        raise ValueError("固定Action Spaceではサイコロの目を方策が指定できません。")
    if isinstance(action, UseDevelopmentAction) and action.card == "year_of_plenty":
        if action.resources is None or tuple(sorted(action.resources, key=RESOURCES.index)) != action.resources:
            raise ValueError("豊作の資源はRESOURCES順で指定してください。")
    try:
        return _ACTION_TO_ID[action]
    except (KeyError, TypeError) as error:
        raise ValueError(f"PPO v1 Action Spaceでは扱わないActionです: {type(action).__name__}") from error


def id_to_action(action_id: int) -> Action:
    """固定IDを対応するActionへ戻す。``action_to_id`` と可逆。"""
    if not isinstance(action_id, Integral) or action_id < 0 or action_id >= ACTION_SPACE_SIZE:
        raise ValueError(f"Action IDが範囲外です: {action_id}")
    return ACTION_CATALOG[int(action_id)].action


def legal_policy_actions(game: GameState, player_id: int) -> list[Action]:
    """PPO v1が選択できる、固定カタログ内の合法Actionだけを返す。

    既存の人間用の任意破棄・プレイヤー間交渉は維持しつつ、PPO v1では
    方策の出力次元を実用的に保つため除外する。
    """
    legal: list[Action] = []
    for action in legal_actions(game, player_id):
        try:
            action_to_id(action)
        except ValueError:
            continue
        legal.append(action)
    return legal


def get_action_mask(game: GameState, player_id: int) -> np.ndarray:
    """``ACTION_SPACE_SIZE`` 個のbool値でPPO v1の合法手を示す。"""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    for action in legal_policy_actions(game, player_id):
        mask[action_to_id(action)] = True
    return mask
