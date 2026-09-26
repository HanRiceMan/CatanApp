"""通常Agentの確率イベント・破棄・交渉をPPO v1の選択範囲へ合わせる。"""

from app.domain.actions import (Action, DiscardHalfAction, DiscardResourcesAction,
                                EndTurnAction, ProposeTradeAction, RollOrderAction,
                                RollTurnDiceAction)


def policy_compatible_action(action: Action) -> Action:
    if isinstance(action, RollOrderAction):
        return RollOrderAction()
    if isinstance(action, RollTurnDiceAction):
        return RollTurnDiceAction()
    if isinstance(action, DiscardResourcesAction):
        return DiscardHalfAction()
    if isinstance(action, ProposeTradeAction):
        return EndTurnAction()
    return action
