"""ゲーム状態を読むだけのエージェント共通インターフェース。"""

from typing import TYPE_CHECKING, Protocol

from app.domain.actions import Action

if TYPE_CHECKING:
    from app.domain.game import GameState


class Agent(Protocol):
    def select_action(self, game: "GameState", player_id: int) -> Action:
        """状態を変更せず、次に実行したい Action を一つ返す。"""
