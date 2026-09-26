"""RL環境でのみ使う、勝利に寄せた小さな報酬関数。"""

from dataclasses import asdict, dataclass

from app.domain.game import GameState, player_for


@dataclass(frozen=True)
class RewardConfig:
    """値は学習実験ごとに保存する。ゲームルールには依存させない。"""

    win: float = 1.0
    loss: float = -1.0
    victory_point_delta: float = 0.05


@dataclass(frozen=True)
class RewardBreakdown:
    score_before: int
    score_after: int
    victory_point_reward: float
    terminal_reward: float

    @property
    def total(self) -> float:
        return self.victory_point_reward + self.terminal_reward

    def to_dict(self) -> dict[str, float | int]:
        return {**asdict(self), "total": self.total}


def actual_score(game: GameState, player_id: int) -> int:
    """非公開勝利点を含む、本人にだけ使う実得点。"""
    player = player_for(game, player_id)
    return (player.settlements + 2 * player.cities
            + 2 * (game.longest_road_holder_id == player_id)
            + 2 * (game.largest_army_holder_id == player_id)
            + player.development_cards.count("victory_point"))


def reward_after_transition(game: GameState, learning_player_id: int, score_before: int,
                            config: RewardConfig) -> RewardBreakdown:
    score_after = actual_score(game, learning_player_id)
    victory_point_reward = config.victory_point_delta * (score_after - score_before)
    terminal_reward = 0.0
    if game.phase == "game_over":
        terminal_reward = config.win if game.winner_id == learning_player_id else config.loss
    return RewardBreakdown(score_before, score_after, victory_point_reward, terminal_reward)
