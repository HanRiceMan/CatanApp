"""固定Action Spaceを、階層方策で使う排他的な行動種別へ分ける。"""

from __future__ import annotations

from typing import Final

from app.domain.actions import (Action, BankTradeAction, BuildCityAction,
                                BuildRoadAction, BuildSettlementAction,
                                BuyDevelopmentAction, DiscardHalfAction,
                                EndTurnAction, MoveRobberAction,
                                PlaceInitialRoadAction,
                                PlaceInitialSettlementAction, RollOrderAction,
                                RollTurnDiceAction, StartGameAction,
                                StealResourceAction, UseDevelopmentAction)


ACTION_FAMILIES: Final[tuple[str, ...]] = (
    "roll_order", "roll_turn", "initial_settlement", "initial_road",
    "start_game", "robber_move", "robber_steal", "road", "settlement",
    "city", "development_purchase", "bank_trade", "development_use",
    "discard", "end_turn",
)
ACTION_FAMILY_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(ACTION_FAMILIES)
}


def action_family(action: Action) -> str:
    for action_type, family in (
        (RollOrderAction, "roll_order"),
        (RollTurnDiceAction, "roll_turn"),
        (PlaceInitialSettlementAction, "initial_settlement"),
        (PlaceInitialRoadAction, "initial_road"),
        (StartGameAction, "start_game"),
        (MoveRobberAction, "robber_move"),
        (StealResourceAction, "robber_steal"),
        (BuildRoadAction, "road"),
        (BuildSettlementAction, "settlement"),
        (BuildCityAction, "city"),
        (BuyDevelopmentAction, "development_purchase"),
        (BankTradeAction, "bank_trade"),
        (UseDevelopmentAction, "development_use"),
        (DiscardHalfAction, "discard"),
        (EndTurnAction, "end_turn"),
    ):
        if isinstance(action, action_type):
            return family
    raise ValueError(f"Action Familyが未定義です: {type(action).__name__}")
