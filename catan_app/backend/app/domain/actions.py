"""ゲームエンジンへ渡す、型付きの操作モデル。

UI・HTTP・AIはいずれもこのモジュールの Action を組み立て、
``game.apply_action`` に渡す。ここにはゲーム状態やルール処理を置かない。
"""

from dataclasses import dataclass
from typing import Literal, Mapping, TypeAlias

ResourceName: TypeAlias = Literal["wood", "brick", "sheep", "wheat", "ore"]
DevelopmentCardName: TypeAlias = Literal["knight", "road_building", "year_of_plenty", "monopoly"]
TradeDecision: TypeAlias = Literal["accept", "decline", "counter"]


@dataclass(frozen=True)
class TradeOffer:
    give: Mapping[str, int]
    want: Mapping[str, int]


@dataclass(frozen=True)
class RollOrderAction:
    dice: tuple[int, int] | None = None


@dataclass(frozen=True)
class PlaceInitialSettlementAction:
    vertex_id: int


@dataclass(frozen=True)
class PlaceInitialRoadAction:
    edge_id: int


@dataclass(frozen=True)
class StartGameAction:
    pass


@dataclass(frozen=True)
class RollTurnDiceAction:
    dice: tuple[int, int] | None = None


@dataclass(frozen=True)
class DiscardResourcesAction:
    resources: Mapping[str, int]


@dataclass(frozen=True)
class DiscardHalfAction:
    """RL v1用の決定的な7枚超過時の破棄。

    通常UIは ``DiscardResourcesAction`` で任意の内訳を渡し続ける。
    固定長Action Spaceでは組合せ爆発を避けるため、このActionだけを
    使い、ゲームエンジンが既存の決定的な破棄候補を適用する。
    """
    pass


@dataclass(frozen=True)
class MoveRobberAction:
    tile_id: int


@dataclass(frozen=True)
class StealResourceAction:
    victim_id: int


@dataclass(frozen=True)
class BuildRoadAction:
    edge_id: int


@dataclass(frozen=True)
class BuildSettlementAction:
    vertex_id: int


@dataclass(frozen=True)
class BuildCityAction:
    vertex_id: int


@dataclass(frozen=True)
class BuyDevelopmentAction:
    pass


@dataclass(frozen=True)
class BankTradeAction:
    give_resource: str
    receive_resource: str


@dataclass(frozen=True)
class ProposeTradeAction:
    target_id: int
    offers: tuple[TradeOffer, ...]


@dataclass(frozen=True)
class RespondToTradeAction:
    decision: TradeDecision
    offer_index: int | None = None


@dataclass(frozen=True)
class ProposeCounterTradeAction:
    offers: tuple[TradeOffer, ...]


@dataclass(frozen=True)
class UseDevelopmentAction:
    card: DevelopmentCardName
    resources: tuple[str, ...] | None = None


@dataclass(frozen=True)
class EndTurnAction:
    pass


@dataclass(frozen=True)
class SetPlayerControllerAction:
    is_ai: bool
    # Noneなら、最後に選んだAI種別を維持する（既存API互換）。
    agent_name: str | None = None


Action: TypeAlias = (
    RollOrderAction | PlaceInitialSettlementAction | PlaceInitialRoadAction | StartGameAction
    | RollTurnDiceAction | DiscardResourcesAction | DiscardHalfAction | MoveRobberAction | StealResourceAction
    | BuildRoadAction | BuildSettlementAction | BuildCityAction | BuyDevelopmentAction
    | BankTradeAction | ProposeTradeAction | RespondToTradeAction | ProposeCounterTradeAction
    | UseDevelopmentAction | EndTurnAction | SetPlayerControllerAction
)
