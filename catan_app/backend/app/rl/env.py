"""PPO v1のための、FastAPIを通さない単一学習者Environment。"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Literal, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.agents.runner import get_agent_for_player, set_agent_for_player
from app.agents.base import Agent
from app.agents.heuristic import HeuristicAgent
from app.domain.actions import (Action, BuildRoadAction, BuyDevelopmentAction,
                                EndTurnAction, MoveRobberAction,
                                ProposeCounterTradeAction, ProposeTradeAction,
                                RespondToTradeAction, StealResourceAction)
from app.domain.board import BoardRules
from app.domain.game import (GameState, apply_action, create_game, player_for,
                             setup_player_id)

from .action_space import (ACTION_CATALOG, ACTION_SPACE_SIZE, action_to_id,
                           get_action_mask, id_to_action)
from .expansion_planner import analyze_expansion_plan
from .observation import (OBSERVATION_VERSION, OBSERVATION_VECTOR_SIZES,
                          encode_observation, get_observation)
from .reward import RewardConfig, actual_score, reward_after_transition
from .policy_adapter import policy_compatible_action

SeatController = Literal["learner", "heuristic", "random", "external"]
WAIT_ROAD_TARGET_WEIGHT = 0.25
OFF_PLAN_ROAD_TARGET_WEIGHT = 0.20
NEAR_BUILD_DEVELOPMENT_TARGET_WEIGHT = 0.35


class ExternalActionRequired(RuntimeError):
    """外部操作席が手番を持つため、学習者の ``step`` を続けられない。"""


class IllegalPolicyAction(ValueError):
    """Action MaskがFalseのAction IDが与えられた。"""


@dataclass(frozen=True)
class CatanEnvConfig:
    """Environment単位の設定。通常のUI対局設定とは独立している。"""

    learning_player_id: int = 1
    opponent_controller: Literal["heuristic", "random"] = "heuristic"
    # Noneは従来の固定相手。指定時は各episodeのseedで相手3人の割当を回す。
    heuristic_opponents: int | None = None
    controller_by_player: Mapping[int, SeatController] = field(default_factory=dict)
    board_rules: BoardRules = field(default_factory=BoardRules)
    reward: RewardConfig = field(default_factory=RewardConfig)
    max_policy_steps: int = 5_000
    allow_player_trades: bool = False
    # 新しい公平比較専用。Falseは既存モデルの学習条件を維持する。
    policy_compatible_opponents: bool = False
    # 初期配置のみ既存Heuristicへ委譲し、通常ターンをPPOが学ぶ実験用。
    heuristic_initial_placement: bool = False
    observation_version: str = OBSERVATION_VERSION
    # 学習専用。通常UIの合法手は変えず、盤面拡張を学ぶ初期段階だけ
    # 「今すぐ建設できる」場面の有料道路・早すぎる発展購入を隠す。
    expansion_curriculum: bool = False
    # 学習専用。通常の合法手は隠さず、上のカリキュラムが弱めたいActionを
    # 実際に選んだ時だけ小さな負の報酬を与える。通常UI・評価は既定値0。
    early_development_penalty: float = 0.0
    unproductive_road_penalty: float = 0.0
    # 盗賊計画をPPOへ移管するカリキュラム専用。実戦・通常学習は0。
    robber_teacher_reward: float = 0.0
    robber_tile_value_reward: float = 0.0
    robber_stolen_resource_reward: float = 0.0


def robber_teacher_action(game: GameState, player_id: int) -> Action | None:
    """公開情報の共同ベイズ教師が選ぶ盗賊Actionを返す。"""
    if game.phase not in {"robber_move", "robber_steal"}:
        return None
    from .hand_belief import BayesianHandEstimator
    from .victory_race import best_belief_robber_tile, best_belief_robber_victim
    beliefs = BayesianHandEstimator().estimate(game, player_id)
    if game.phase == "robber_move":
        legal = [definition.action.tile_id for definition in ACTION_CATALOG
                 if get_action_mask(game, player_id)[definition.id]
                 and isinstance(definition.action, MoveRobberAction)]
        return MoveRobberAction(best_belief_robber_tile(
            game, player_id, legal, beliefs,
        ))
    return StealResourceAction(best_belief_robber_victim(
        game, player_id, game.robber_victim_ids, beliefs,
    ))


def robber_tile_quality_reward(game: GameState, player_id: int,
                               action_mask: np.ndarray,
                               action: Action, scale: float) -> float:
    """選択タイルの妨害・略奪評価を合法候補内[-scale, scale]へ正規化する。"""
    if scale <= 0 or not isinstance(action, MoveRobberAction):
        return 0.0
    from .hand_belief import BayesianHandEstimator
    from .victory_race import belief_robber_tile_value
    beliefs = BayesianHandEstimator().estimate(game, player_id)
    values = {
        definition.action.tile_id: belief_robber_tile_value(
            game, player_id, definition.action.tile_id, beliefs
        )
        for definition in ACTION_CATALOG
        if action_mask[definition.id]
        and isinstance(definition.action, MoveRobberAction)
    }
    if action.tile_id not in values or len(values) < 2:
        return 0.0
    low, high = min(values.values()), max(values.values())
    if high - low <= 1e-9:
        return 0.0
    quality = (values[action.tile_id] - low) / (high - low)
    return float(scale * (2.0 * quality - 1.0))


def stolen_resource_utility_reward(weights: Mapping[str, float],
                                   resource: str | None,
                                   scale: float) -> float:
    """実際に奪った資源が現在の建設不足を埋める度合いを[0, scale]で返す。"""
    if scale <= 0 or resource is None:
        return 0.0
    if resource not in weights:
        return 0.0
    maximum = max(weights.values())
    return float(scale * weights[resource] / maximum) if maximum > 0 else 0.0


def _stolen_resource_since(game: GameState, event_index: int,
                           player_id: int) -> str | None:
    for event in game.resource_events[event_index:]:
        transfer = event.get("hidden_transfer")
        if (event.get("kind") == "robber_steal" and transfer
                and transfer.get("to_player_id") == player_id
                and player_id in transfer.get("known_to", ())):
            return transfer.get("private_resource")
    return None


def expansion_curriculum_mask(game: GameState, player_id: int,
                              action_mask: np.ndarray) -> np.ndarray:
    """拡張方針を学ぶ時だけ使う、通常ルールからの限定的な行動制限。

    接続済みの有望な開拓地候補があるなら、購入資源が揃うまで待つのを基本にする。
    より遠い候補の価値が道路費用を含めても十分高い場合だけ、そこへ向かう道路を残す。
    発展カードは開拓地・都市の建設が遠い停滞局面か、明確な例外理由がある時に残す。
    これは通常UI・評価には一切適用しない。
    """
    result = np.asarray(action_mask, dtype=np.bool_).copy()
    if game.phase != "action":
        return result
    player = player_for(game, player_id)
    plan = analyze_expansion_plan(game, player_id)
    if not game.free_road_remaining:
        for definition in ACTION_CATALOG:
            if not result[definition.id] or not isinstance(definition.action, BuildRoadAction):
                continue
            if plan.should_wait_for_settlement:
                result[definition.id] = False
            elif plan.recommended_road_ids and definition.action.edge_id not in plan.recommended_road_ids:
                result[definition.id] = False

    # 建設までの不足資源を現在の生産・港交換で埋める概算が6ラウンド未満なら、
    # 発展カードより建設資源を温存する。盗賊対応・最大騎士力・終盤は例外。
    robber_pressure = any(
        game.robber_tile_id in game.board.vertices[vertex_id].hex_ids
        for vertices in (game.settlements, game.cities)
        for vertex_id, owner_id in vertices.items()
        if owner_id == player_id
    )
    army_pursuit = player.played_knights >= 2
    endgame_development = actual_score(game, player_id) >= 8
    construction_stalled = plan.build_wait_rounds >= 6.0
    if not (construction_stalled or robber_pressure or army_pursuit or endgame_development):
        result[action_to_id(BuyDevelopmentAction())] = False
    if not result.any():
        raise RuntimeError("拡張カリキュラムがすべての合法Actionを隠しました。")
    return result


def expansion_target_weights(game: GameState, player_id: int,
                             action_mask: np.ndarray) -> np.ndarray:
    """現行PPOを壊さず、計画外の支出だけを弱める軟らかい教師重み。"""
    weights = np.ones_like(action_mask, dtype=np.float32)
    if game.phase != "action":
        return weights
    plan = analyze_expansion_plan(game, player_id)
    if not game.free_road_remaining:
        for definition in ACTION_CATALOG:
            if not action_mask[definition.id] or not isinstance(definition.action, BuildRoadAction):
                continue
            if plan.should_wait_for_settlement:
                weights[definition.id] = WAIT_ROAD_TARGET_WEIGHT
            elif (plan.recommended_road_ids
                  and definition.action.edge_id not in plan.recommended_road_ids):
                weights[definition.id] = OFF_PLAN_ROAD_TARGET_WEIGHT

    player = player_for(game, player_id)
    robber_pressure = any(
        game.robber_tile_id in game.board.vertices[vertex_id].hex_ids
        for buildings in (game.settlements, game.cities)
        for vertex_id, owner_id in buildings.items()
        if owner_id == player_id
    )
    development_exception = (
        plan.build_wait_rounds >= 6.0
        or robber_pressure
        or player.played_knights >= 2
        or actual_score(game, player_id) >= 8
    )
    development_id = action_to_id(BuyDevelopmentAction())
    if action_mask[development_id] and not development_exception:
        weights[development_id] = NEAR_BUILD_DEVELOPMENT_TARGET_WEIGHT
    return weights


def expansion_strategy_penalty(game: GameState, player_id: int,
                               action_mask: np.ndarray, action: Action, *,
                               early_development: float,
                               unproductive_road: float) -> float:
    """例外条件を含むカリキュラム方針に反したActionだけを軽く減点する。"""
    if game.phase != "action" or not isinstance(action, (BuildRoadAction, BuyDevelopmentAction)):
        return 0.0
    preferred = expansion_curriculum_mask(game, player_id, action_mask)
    if preferred[action_to_id(action)]:
        return 0.0
    if isinstance(action, BuyDevelopmentAction):
        return -early_development
    return -unproductive_road


class CatanEnv(gym.Env[np.ndarray, int]):
    """P1など1人だけを学習させるGymnasium互換Environment。

    通常は学習者以外を既存のHeuristicAgentまたはRandomAgentで自動進行する。
    ``external`` 席は教材・デバッグ用の人間操作席で、学習ループには向かない。
    """

    metadata = {"render_modes": []}

    def __init__(self, config: CatanEnvConfig | None = None, *,
                 action_observer: Callable[[GameState, int, Action], None] | None = None,
                 initial_placement_agent: Agent | None = None,
                 opponent_agents: Mapping[int, Agent] | None = None,
                 learner_auxiliary_agent: Agent | None = None):
        super().__init__()
        self.config = config or CatanEnvConfig()
        if self.config.observation_version not in OBSERVATION_VECTOR_SIZES:
            raise ValueError(f"未対応のObservation版です: {self.config.observation_version}")
        if min(self.config.early_development_penalty,
               self.config.unproductive_road_penalty,
               self.config.robber_teacher_reward,
               self.config.robber_tile_value_reward,
               self.config.robber_stolen_resource_reward) < 0:
            raise ValueError("学習用penalty/rewardは0以上にしてください。")
        self._action_observer = action_observer
        self._initial_placement_agent = initial_placement_agent
        self._opponent_agents = dict(opponent_agents or {})
        invalid_opponents = (
            set(self._opponent_agents) - set(range(1, 5))
            or ({self.config.learning_player_id}
                if self.config.learning_player_id in self._opponent_agents else set())
        )
        if invalid_opponents:
            raise ValueError(f"固定Opponentの席指定が不正です: {sorted(invalid_opponents)}")
        self._learner_auxiliary_agent = learner_auxiliary_agent
        if self._initial_placement_agent is not None and self.config.heuristic_initial_placement:
            raise ValueError("初期配置AgentとHeuristic初期配置は同時に指定できません。")
        self._controllers = self._resolve_controllers(self.config)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(OBSERVATION_VECTOR_SIZES[self.config.observation_version],),
                                            dtype=np.float32)
        self.game: GameState | None = None
        self.episode_seed: int | None = None
        self.policy_steps = 0
        self._request_number = 0
        self._learning_score = 0
        self._pending_external_player_id: int | None = None
        self._is_truncated = False
        self._learner_auxiliary_actions = 0

    @staticmethod
    def _resolve_controllers(config: CatanEnvConfig) -> dict[int, SeatController]:
        if config.learning_player_id not in range(1, 5):
            raise ValueError("learning_player_idは1〜4で指定してください。")
        if config.opponent_controller not in {"heuristic", "random"}:
            raise ValueError("opponent_controllerはheuristicまたはrandomです。")
        if config.heuristic_opponents is not None:
            if config.heuristic_opponents not in range(4):
                raise ValueError("heuristic_opponentsは0〜3で指定してください。")
            if config.controller_by_player:
                raise ValueError("heuristic_opponentsとcontroller_by_playerは同時に指定できません。")
        if config.max_policy_steps <= 0:
            raise ValueError("max_policy_stepsは正の整数にしてください。")
        controllers: dict[int, SeatController] = {player_id: config.opponent_controller for player_id in range(1, 5)}
        for player_id, controller in config.controller_by_player.items():
            if player_id not in range(1, 5) or controller not in {"learner", "heuristic", "random", "external"}:
                raise ValueError("controller_by_playerの指定が不正です。")
            controllers[player_id] = controller
        if controllers[config.learning_player_id] not in {config.opponent_controller, "learner"}:
            raise ValueError("learning_player_idのcontrollerはlearnerにしてください。")
        controllers[config.learning_player_id] = "learner"
        if sum(controller == "learner" for controller in controllers.values()) != 1:
            raise ValueError("学習者は1人だけにしてください。")
        return controllers

    def _episode_controllers(self, seed: int) -> dict[int, SeatController]:
        if self.config.heuristic_opponents is None:
            return self._resolve_controllers(self.config)
        opponents = [pid for pid in range(1, 5) if pid != self.learning_player_id]
        offset = seed % len(opponents)
        opponents = opponents[offset:] + opponents[:offset]
        controllers: dict[int, SeatController] = {self.learning_player_id: "learner"}
        for index, player_id in enumerate(opponents):
            controllers[player_id] = "heuristic" if index < self.config.heuristic_opponents else "random"
        return controllers

    @property
    def learning_player_id(self) -> int:
        return self.config.learning_player_id

    @property
    def pending_external_player_id(self) -> int | None:
        return self._pending_external_player_id

    @property
    def controllers(self) -> dict[int, SeatController]:
        return dict(self._controllers)

    def _require_game(self) -> GameState:
        if self.game is None:
            raise RuntimeError("先にreset()を呼んでください。")
        return self.game

    def _request_id(self, prefix: str) -> str:
        self._request_number += 1
        return f"env-{prefix}-{self._request_number:08d}"

    def _required_player_id(self, game: GameState) -> int | None:
        if game.phase == "rolling_order":
            return game.pending_rollers[0] if game.pending_rollers else None
        if game.phase == "setup_ready":
            return setup_player_id(game)
        if game.phase == "ready_to_play":
            return game.seat_order[0] if game.seat_order else None
        if game.phase == "discarding":
            if self.learning_player_id in game.pending_discards:
                return self.learning_player_id
            return game.pending_discards[0] if game.pending_discards else None
        if game.phase in {"trade_response", "trade_counter_offer"}:
            return game.pending_trade["recipient_id"] if game.pending_trade else None
        if game.phase in {"turn_pre_roll", "robber_move", "robber_steal", "action"}:
            return game.current_player_id
        return None

    def _apply(self, player_id: int, action: Action, prefix: str) -> None:
        game = self._require_game()
        apply_action(game, player_id, action, game.revision, self._request_id(prefix))
        if self._action_observer is not None:
            self._action_observer(game, player_id, action)

    def _agent_action(self, player_id: int) -> Action:
        game = self._require_game()
        override = self._opponent_agents.get(player_id)
        if override is not None:
            action = override.select_action(game, player_id)
        else:
            controller = self._controllers[player_id]
            set_agent_for_player(game, player_id, controller)
            action = get_agent_for_player(game, player_id).select_action(game, player_id)
        if (self.config.policy_compatible_opponents
                and not (self.config.allow_player_trades
                         and isinstance(action, (ProposeTradeAction,
                                                 RespondToTradeAction,
                                                 ProposeCounterTradeAction)))):
            return policy_compatible_action(action)
        # PPO v1では対人交渉を扱わない。Opponentが交渉を始める前に、その手番を終える。
        if not self.config.allow_player_trades and isinstance(action, ProposeTradeAction):
            return EndTurnAction()
        return action

    def _learner_auxiliary_action(self) -> Action | None:
        """固定Action Space外の交渉だけを補助層へ委譲する。"""
        if self._learner_auxiliary_agent is None or not self.config.allow_player_trades:
            return None
        game = self._require_game()
        player_id = self.learning_player_id
        action = self._learner_auxiliary_agent.select_action(game, player_id)
        expected = {
            "action": ProposeTradeAction,
            "trade_response": RespondToTradeAction,
            "trade_counter_offer": ProposeCounterTradeAction,
        }.get(game.phase)
        if expected is None:
            return None
        if isinstance(action, expected):
            return action
        if game.phase in {"trade_response", "trade_counter_offer"}:
            raise RuntimeError(
                f"学習者の交渉補助層が{game.phase}で{type(action).__name__}を返しました。"
            )
        return None

    def _advance_until_boundary(self) -> None:
        """固定Opponentを進め、学習者または外部操作席で止まる。"""
        game = self._require_game()
        self._pending_external_player_id = None
        while game.phase != "game_over":
            player_id = self._required_player_id(game)
            if player_id is None:
                raise RuntimeError(f"次の操作担当を決定できません: {game.phase}")
            controller = self._controllers[player_id]
            if controller == "learner":
                if self._initial_placement_agent is not None and game.phase == "setup_ready":
                    self._apply(player_id, self._initial_placement_agent.select_action(
                        game, player_id), "learner-setup-agent")
                    continue
                if self.config.heuristic_initial_placement and game.phase == "setup_ready":
                    self._apply(player_id, HeuristicAgent().select_action(game, player_id), "learner-setup")
                    continue
                auxiliary = self._learner_auxiliary_action()
                if auxiliary is not None:
                    self._apply(player_id, auxiliary, "learner-trade")
                    self._learner_auxiliary_actions += 1
                    continue
                return
            if controller == "external":
                self._pending_external_player_id = player_id
                return
            action = self._agent_action(player_id)
            self._apply(player_id, action, "opponent")

    def _observation(self) -> np.ndarray:
        return encode_observation(get_observation(self._require_game(), self.learning_player_id,
                                                  version=self.config.observation_version))

    def _info(self, *, reward_components: dict[str, float | int] | None = None) -> dict[str, Any]:
        game = self._require_game()
        info: dict[str, Any] = {
            "action_mask": self.action_masks(),
            "episode_seed": self.episode_seed,
            "learning_player_id": self.learning_player_id,
            "controllers": dict(self._controllers),
            "expansion_curriculum": self.config.expansion_curriculum,
            "early_development_penalty": self.config.early_development_penalty,
            "unproductive_road_penalty": self.config.unproductive_road_penalty,
            "robber_teacher_reward": self.config.robber_teacher_reward,
            "robber_tile_value_reward": self.config.robber_tile_value_reward,
            "robber_stolen_resource_reward": self.config.robber_stolen_resource_reward,
            "player_trades_enabled": self.config.allow_player_trades,
            "learner_auxiliary_actions": self._learner_auxiliary_actions,
            "fixed_opponent_ids": sorted(self._opponent_agents),
            "phase": game.phase,
            "current_player_id": game.current_player_id,
            "winner_id": game.winner_id,
            "policy_steps": self.policy_steps,
            "external_action_required": self._pending_external_player_id,
        }
        if reward_components is not None:
            info["reward_components"] = reward_components
        return info

    def action_masks(self) -> np.ndarray:
        """MaskablePPOが利用する予定のAction Mask取得口。"""
        game = self._require_game()
        mask = get_action_mask(game, self.learning_player_id)
        if not self.config.expansion_curriculum or game.phase != "action":
            return mask
        return expansion_curriculum_mask(game, self.learning_player_id, mask)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            unknown = set(options) - {"seed"}
            if unknown:
                raise ValueError(f"未対応のreset optionsです: {sorted(unknown)}")
        chosen_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        if options and "seed" in options:
            chosen_seed = int(options["seed"])
        self.episode_seed = chosen_seed
        self._controllers = self._episode_controllers(chosen_seed)
        self.game = create_game(chosen_seed, rules=self.config.board_rules)
        self.policy_steps = 0
        self._request_number = 0
        self._is_truncated = False
        self._learner_auxiliary_actions = 0
        self._learning_score = actual_score(self.game, self.learning_player_id)
        self._advance_until_boundary()
        return self._observation(), self._info()

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        game = self._require_game()
        if game.phase == "game_over":
            raise RuntimeError("このEpisodeは終了しています。reset()を呼んでください。")
        if self._is_truncated:
            raise RuntimeError("このEpisodeは上限stepに達しています。reset()を呼んでください。")
        if self._pending_external_player_id is not None:
            raise ExternalActionRequired(f"プレイヤー {self._pending_external_player_id} の外部操作を先に実行してください。")
        if self._required_player_id(game) != self.learning_player_id:
            raise RuntimeError("現在は学習者の操作タイミングではありません。")
        mask = self.action_masks()
        if not isinstance(action_id, Integral) or action_id < 0 or action_id >= ACTION_SPACE_SIZE or not mask[action_id]:
            raise IllegalPolicyAction(f"Action ID {action_id} は現在合法ではありません。")
        action = id_to_action(int(action_id))
        teacher_action = (
            robber_teacher_action(game, self.learning_player_id)
            if self.config.robber_teacher_reward > 0 else None
        )
        robber_teacher_reward = (
            self.config.robber_teacher_reward
            * (1.0 if action == teacher_action else -1.0)
            if teacher_action is not None else 0.0
        )
        robber_tile_reward = robber_tile_quality_reward(
            game, self.learning_player_id, mask, action,
            self.config.robber_tile_value_reward,
        )
        if (isinstance(action, StealResourceAction)
                and self.config.robber_stolen_resource_reward > 0):
            from .victory_race import desired_resource_weights
            stolen_resource_weights = desired_resource_weights(
                game, self.learning_player_id
            )
        else:
            stolen_resource_weights = {}
        resource_event_index = len(game.resource_events)
        strategy_penalty = expansion_strategy_penalty(
            game, self.learning_player_id, mask, action,
            early_development=self.config.early_development_penalty,
            unproductive_road=self.config.unproductive_road_penalty,
        )
        score_before = self._learning_score
        self._apply(self.learning_player_id, action, "learner")
        stolen_resource = (
            _stolen_resource_since(
                game, resource_event_index, self.learning_player_id
            ) if isinstance(action, StealResourceAction) else None
        )
        robber_resource_reward = stolen_resource_utility_reward(
            stolen_resource_weights, stolen_resource,
            self.config.robber_stolen_resource_reward,
        )
        self.policy_steps += 1
        self._advance_until_boundary()
        breakdown = reward_after_transition(game, self.learning_player_id, score_before, self.config.reward)
        self._learning_score = breakdown.score_after
        terminated = game.phase == "game_over"
        truncated = not terminated and self.policy_steps >= self.config.max_policy_steps
        self._is_truncated = truncated
        reward_components = breakdown.to_dict()
        reward_components["expansion_strategy_penalty"] = strategy_penalty
        reward_components["robber_teacher_reward"] = robber_teacher_reward
        reward_components["robber_tile_value_reward"] = robber_tile_reward
        reward_components["robber_stolen_resource_reward"] = robber_resource_reward
        reward_components["total"] = (
            breakdown.total + strategy_penalty + robber_teacher_reward
            + robber_tile_reward + robber_resource_reward
        )
        return (self._observation(),
                breakdown.total + strategy_penalty + robber_teacher_reward
                + robber_tile_reward + robber_resource_reward, terminated,
                truncated, self._info(reward_components=reward_components))

    def step_external(self, player_id: int, action: int | Action) -> tuple[np.ndarray, dict[str, Any]]:
        """外部操作席（将来の人間UI接続用）を1操作進める。

        intならPPO v1 Action ID、Actionなら通常UIと同じ完全なActionを渡せる。
        学習を止めて人が操作する教材・デバッグ向けであり、無人PPO学習には使わない。
        """
        game = self._require_game()
        if self._pending_external_player_id != player_id or self._required_player_id(game) != player_id:
            raise ExternalActionRequired(f"現在、プレイヤー {player_id} の外部操作は待機していません。")
        resolved = id_to_action(int(action)) if isinstance(action, Integral) else action
        if not self.config.allow_player_trades and isinstance(resolved, (ProposeTradeAction, RespondToTradeAction, ProposeCounterTradeAction)):
            raise ValueError("PPO v1 Environmentではプレイヤー間交渉は無効です。")
        self._apply(player_id, resolved, "external")
        self._advance_until_boundary()
        return self._observation(), self._info()

    def render(self) -> None:
        """UI描画はNext.jsが担当するため、学習Environmentは描画しない。"""
        return None

    def close(self) -> None:
        self.game = None
