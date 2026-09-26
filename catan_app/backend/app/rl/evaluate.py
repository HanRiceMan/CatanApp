"""学習前後のAgentを、同じCatanEnv上で数値比較する評価ランナー。"""

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from statistics import fmean
from typing import Iterable, Literal, Mapping

from app.agents.base import Agent
from app.domain.actions import (Action, BankTradeAction, BuildCityAction, BuildRoadAction,
                                BuildSettlementAction, BuyDevelopmentAction, EndTurnAction,
                                PlaceInitialRoadAction, PlaceInitialSettlementAction,
                                UseDevelopmentAction)
from app.domain.board import BoardRules
from app.domain.game import GameState

from .action_space import action_to_id
from .env import CatanEnv, CatanEnvConfig, IllegalPolicyAction
from .observation import OBSERVATION_VERSION
from .reward import RewardConfig, actual_score
from .diagnostics_metrics import EvaluationTelemetry, endgame_rank


@dataclass(frozen=True)
class EvaluationConfig:
    """評価条件。学習側・評価側で同じ内容を保存して比較する。"""

    learning_player_id: int = 1
    opponent_controller: Literal["heuristic", "random"] = "heuristic"
    board_rules: BoardRules = field(default_factory=BoardRules)
    reward: RewardConfig = field(default_factory=RewardConfig)
    max_policy_steps: int = 5_000
    opponent_by_player: Mapping[int, Literal["heuristic", "random"]] = field(default_factory=dict)
    policy_compatible_opponents: bool = False
    rotate_player_ids: bool = False
    heuristic_opponents: int | None = None
    observation_version: str = OBSERVATION_VERSION


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    winner_id: int | None
    learner_won: bool
    learner_score: int
    turn_number: int
    policy_steps: int
    cumulative_reward: float
    illegal_action_count: int
    truncated: bool
    learner_activity: Mapping[str, int]
    learner_actions: Mapping[str, int]
    learner_final_pieces: Mapping[str, int]
    learner_initial_vertices: tuple[int, ...]
    learner_initial_pips: int
    learner_initial_resource_kinds: int
    learner_id: int
    learner_seat: int
    seat_order: tuple[int, ...]
    opponent_controllers: Mapping[int, str]
    player_scores: Mapping[int, int]
    learner_rank: float | None
    second_place_credit: float
    acquired_longest_road: bool
    acquired_largest_army: bool
    trade_ratios: Mapping[str, int]
    opportunities: Mapping[str, int]
    strategy: Mapping[str, int | float]
    score_changes: tuple[dict, ...]
    observation_range: tuple[float, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    agent_name: str
    config: EvaluationConfig
    episodes: tuple[EpisodeResult, ...]

    @property
    def win_rate(self) -> float:
        return fmean(result.learner_won for result in self.episodes) if self.episodes else 0.0

    @property
    def average_score(self) -> float:
        return fmean(result.learner_score for result in self.episodes) if self.episodes else 0.0

    @property
    def average_turns(self) -> float:
        return fmean(result.turn_number for result in self.episodes) if self.episodes else 0.0

    @property
    def illegal_action_count(self) -> int:
        return sum(result.illegal_action_count for result in self.episodes)

    def to_dict(self) -> dict:
        completed = [episode for episode in self.episodes if not episode.truncated]
        action_names = sorted({key for episode in self.episodes for key in episode.learner_actions})
        opportunity_names = sorted({key for episode in self.episodes for key in episode.opportunities})
        strategy_names = sorted({key for episode in self.episodes for key in episode.strategy})
        def mean(values):
            return fmean(values) if self.episodes else 0.0
        return {
            "evaluation_version": "v3",
            "agent_name": self.agent_name,
            "config": asdict(self.config),
            "episodes": [episode.to_dict() for episode in self.episodes],
            "summary": {
                "episodes": len(self.episodes),
                "win_rate": self.win_rate,
                "average_score": self.average_score,
                "average_turns": self.average_turns,
                "illegal_action_count": self.illegal_action_count,
                "completed": len(completed),
                "truncated": len(self.episodes) - len(completed),
                "average_rank": fmean(e.learner_rank for e in completed) if completed else None,
                "second_place_rate": fmean(e.second_place_credit for e in completed) if completed else None,
                "average_actions": {key: mean([e.learner_actions.get(key, 0) for e in self.episodes]) for key in action_names},
                "average_opportunities": {key: mean([e.opportunities.get(key, 0) for e in self.episodes]) for key in opportunity_names},
                "average_strategy": {key: mean([e.strategy.get(key, 0) for e in self.episodes]) for key in strategy_names},
                "average_trade_ratios": {key: mean([e.trade_ratios.get(key, 0) for e in self.episodes]) for key in ("4:1", "3:1", "2:1")},
                "longest_road_acquisition_rate": mean([e.acquired_longest_road for e in self.episodes]),
                "largest_army_acquisition_rate": mean([e.acquired_largest_army for e in self.episodes]),
                "average_initial_pips": mean([e.learner_initial_pips for e in self.episodes]),
                "learner_by_id": {str(pid): {"games": sum(e.learner_id == pid for e in self.episodes),
                    "wins": sum(e.learner_won and e.learner_id == pid for e in self.episodes)} for pid in range(1, 5)},
                "learner_by_starting_seat": {str(seat): {"games": sum(e.learner_seat == seat for e in self.episodes),
                    "wins": sum(e.learner_won and e.learner_seat == seat for e in self.episodes)} for seat in range(1, 5)},
                "winner_by_id": dict(Counter(str(e.winner_id) for e in completed)),
                "winner_by_starting_seat": dict(Counter(str(e.seat_order.index(e.winner_id) + 1) for e in completed)),
            },
        }


def _environment(config: EvaluationConfig, telemetry: EvaluationTelemetry) -> CatanEnv:
    return CatanEnv(CatanEnvConfig(
        learning_player_id=config.learning_player_id,
        opponent_controller=config.opponent_controller,
        board_rules=config.board_rules,
        reward=config.reward,
        max_policy_steps=config.max_policy_steps,
        controller_by_player=config.opponent_by_player,
        policy_compatible_opponents=config.policy_compatible_opponents,
        observation_version=config.observation_version,
    ), action_observer=telemetry.after_action)


def episode_config(config: EvaluationConfig, index: int) -> EvaluationConfig:
    """4つのPlayer IDと混合相手の割当を、Agent間で同じ順に回す。"""
    player_id = (config.learning_player_id - 1 + index) % 4 + 1 if config.rotate_player_ids else config.learning_player_id
    opponents = dict(config.opponent_by_player)
    if config.heuristic_opponents is not None:
        if config.heuristic_opponents not in range(4):
            raise ValueError("heuristic_opponentsは0〜3です。")
        if opponents:
            raise ValueError("heuristic_opponentsとopponent_by_playerは同時に指定できません。")
        others = [pid for pid in range(1, 5) if pid != player_id]
        offset = (index // 4 if config.rotate_player_ids else index) % 3
        others = others[offset:] + others[:offset]
        opponents = {pid: "heuristic" if n < config.heuristic_opponents else "random" for n, pid in enumerate(others)}
    if any(pid not in range(1, 5) or pid == player_id or kind not in {"heuristic", "random"} for pid, kind in opponents.items()):
        raise ValueError("opponent_by_playerは相手Player IDとheuristic/randomで指定してください。")
    return replace(config, learning_player_id=player_id, opponent_by_player=opponents)


def _action_name(action: Action, *, free_road: bool = False) -> str:
    """非公開カードを記録せず、学習者が選んだ操作の種類だけを集計する。"""
    if isinstance(action, BuildRoadAction):
        return "build_road_free" if free_road else "build_road_paid"
    if isinstance(action, BuildSettlementAction):
        return "build_settlement"
    if isinstance(action, BuildCityAction):
        return "build_city"
    if isinstance(action, BuyDevelopmentAction):
        return "buy_development"
    if isinstance(action, UseDevelopmentAction):
        return f"use_development_{action.card}"
    if isinstance(action, BankTradeAction):
        return "bank_trade"
    if isinstance(action, PlaceInitialSettlementAction):
        return "initial_settlement"
    if isinstance(action, PlaceInitialRoadAction):
        return "initial_road"
    if isinstance(action, EndTurnAction):
        return "end_turn"
    return type(action).__name__.removesuffix("Action").lower()


_NUMBER_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def _initial_production(game: GameState, vertex_ids: tuple[int, ...]) -> tuple[int, int]:
    """初期2開拓地の出目頻度合計と資源種類数（港などは考慮しない）。"""
    tiles = [game.board.tiles[tile_id] for vertex_id in vertex_ids
             for tile_id in game.board.vertices[vertex_id].hex_ids]
    return (sum(_NUMBER_PIPS.get(tile.number, 0) for tile in tiles),
            len({tile.terrain for tile in tiles if tile.terrain != "desert"}))


def evaluate_agent(agent: Agent, seeds: Iterable[int], *, config: EvaluationConfig | None = None,
                   agent_name: str | None = None) -> EvaluationReport:
    """固定seed群でAgentを評価する。非法Actionは握りつぶさず失敗にする。"""
    requested_config = config or EvaluationConfig()
    results: list[EpisodeResult] = []
    for index, seed in enumerate(seeds):
        evaluation_config = episode_config(requested_config, index)
        telemetry = EvaluationTelemetry(evaluation_config.learning_player_id)
        env = _environment(evaluation_config, telemetry)
        observation, info = env.reset(seed=seed)
        telemetry.check_observation(observation)
        total_reward = 0.0
        illegal_actions = 0
        learner_actions: Counter[str] = Counter()
        initial_vertices: list[int] = []
        while True:
            if info["external_action_required"] is not None:
                raise RuntimeError("評価Environmentにexternal操作席は使用できません。")
            game = env.game
            assert game is not None
            action = agent.select_action(game, evaluation_config.learning_player_id)
            try:
                action_id = action_to_id(action)
                if not info["action_mask"][action_id]:
                    raise IllegalPolicyAction(f"AgentがMask外のActionを選択しました: {action_id}")
            except (ValueError, IllegalPolicyAction) as error:
                illegal_actions += 1
                raise IllegalPolicyAction(f"seed {seed}: {error}") from error
            action_name = _action_name(action, free_road=bool(game.free_road_remaining))
            telemetry.before_action(game, action, info["action_mask"])
            observation, reward, terminated, truncated, info = env.step(action_id)
            telemetry.check_observation(observation)
            learner_actions[action_name] += 1
            if isinstance(action, PlaceInitialSettlementAction):
                initial_vertices.append(action.vertex_id)
            total_reward += reward
            if terminated or truncated:
                break
        activity = Counter(entry["kind"] for entry in game.activity_log
                           if entry["player_id"] == evaluation_config.learning_player_id)
        player = game.players[evaluation_config.learning_player_id - 1]
        initial_pips, initial_resource_kinds = _initial_production(game, tuple(initial_vertices))
        player_scores = {pid: actual_score(game, pid) for pid in range(1, 5)}
        rank, second_place = endgame_rank(player_scores, player.id, game.winner_id)
        results.append(EpisodeResult(
            seed=seed,
            winner_id=game.winner_id,
            learner_won=game.winner_id == evaluation_config.learning_player_id,
            learner_score=actual_score(game, evaluation_config.learning_player_id),
            turn_number=game.turn_number,
            policy_steps=env.policy_steps,
            cumulative_reward=total_reward,
            illegal_action_count=illegal_actions,
            truncated=truncated,
            learner_activity=dict(sorted(activity.items())),
            learner_actions=dict(sorted(learner_actions.items())),
            learner_final_pieces={
                "roads": sum(owner == player.id for owner in game.roads.values()),
                "settlements": player.settlements,
                "cities": player.cities,
            },
            learner_initial_vertices=tuple(initial_vertices),
            learner_initial_pips=initial_pips,
            learner_initial_resource_kinds=initial_resource_kinds,
            learner_id=player.id,
            learner_seat=game.seat_order.index(player.id) + 1,
            seat_order=tuple(game.seat_order),
            opponent_controllers={pid: kind for pid, kind in env.controllers.items() if pid != player.id},
            player_scores=player_scores,
            learner_rank=rank,
            second_place_credit=second_place,
            acquired_longest_road=player.id in telemetry.titles["longest_road"],
            acquired_largest_army=player.id in telemetry.titles["largest_army"],
            trade_ratios=dict(telemetry.trades),
            opportunities=dict(telemetry.opportunities),
            strategy=dict(telemetry.strategy),
            score_changes=tuple(telemetry.score_changes),
            observation_range=(telemetry.observation_min, telemetry.observation_max),
        ))
        env.close()
    return EvaluationReport(agent_name or type(agent).__name__, requested_config, tuple(results))
