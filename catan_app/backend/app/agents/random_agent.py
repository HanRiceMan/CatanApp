"""主に Action 空間の動作確認用の単純なランダムエージェント。"""

from random import Random

from app.domain.actions import Action
from app.domain.game import GameActionError, GameState, legal_actions


class RandomAgent:
    def select_action(self, game: GameState, player_id: int) -> Action:
        actions = legal_actions(game, player_id)
        if not actions:
            raise GameActionError("現在の状態で選べる Action がありません。")
        rng = Random((game.seed << 24) ^ (game.revision << 4) ^ player_id)
        return rng.choice(actions)
