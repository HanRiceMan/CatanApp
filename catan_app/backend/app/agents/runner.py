"""Agent の選択結果を共通入口へ適用し、既存のAI実行APIを支える。"""

from app.domain.actions import (BankTradeAction, BuildCityAction, BuildRoadAction, BuildSettlementAction,
                                BuyDevelopmentAction, DiscardResourcesAction, EndTurnAction,
                                MoveRobberAction, PlaceInitialRoadAction, PlaceInitialSettlementAction,
                                ProposeCounterTradeAction, ProposeTradeAction, RespondToTradeAction,
                                RollOrderAction, RollTurnDiceAction, StartGameAction, StealResourceAction,
                                UseDevelopmentAction)
from app.domain.game import GameActionError, GameState, apply_action, pending_ai_player_id

from .base import Agent
from .heuristic import HeuristicAgent
from .random_agent import RandomAgent

AGENT_FACTORIES: dict[str, type[Agent]] = {
    "heuristic": HeuristicAgent,
    "random": RandomAgent,
}
PPO_AGENT_PREFIX = "ppo:"
PLANNED_PPO_AGENT_PREFIX = "ppo-plan:"
_ppo_agents: dict[str, Agent] = {}


def validate_agent_name(agent_name: str) -> None:
    """API/UIから渡されたAI種別とPPOモデルの互換性を事前に検証する。"""
    if agent_name in AGENT_FACTORIES:
        return
    prefix = next((value for value in (PLANNED_PPO_AGENT_PREFIX, PPO_AGENT_PREFIX)
                   if agent_name.startswith(value)), None)
    if prefix is None or not agent_name.removeprefix(prefix):
        raise ValueError("未登録のAI種別です。")
    model_id = agent_name.removeprefix(prefix)
    try:
        from app.agents.ppo import PPOAgent
        # Registryに加えて、現在のPythonプロセスでPPO推論できることも確認する。
        agent = PPOAgent(model_id)
        if prefix == PLANNED_PPO_AGENT_PREFIX and agent.settlement_planning_config is None:
            raise ValueError("このPPOモデルには開拓計画設定がありません。")
    except (KeyError, ValueError, RuntimeError) as error:
        raise ValueError(f"PPOモデルを利用できません: {error}") from error


def get_agent_for_player(game: GameState, player_id: int) -> Agent:
    """プレイヤー単位の種類を解決する。未指定のAIは既存互換の heuristic。"""
    player = next(player for player in game.players if player.id == player_id)
    agent_name = getattr(player, "agent_name", "heuristic")
    prefix = next((value for value in (PLANNED_PPO_AGENT_PREFIX, PPO_AGENT_PREFIX)
                   if agent_name.startswith(value)), None)
    if prefix is not None:
        model_id = agent_name.removeprefix(prefix)
        try:
            from app.agents.ppo import PPOAgent
            if agent_name not in _ppo_agents:
                base = PPOAgent(model_id)
                if prefix == PLANNED_PPO_AGENT_PREFIX:
                    from app.rl.settlement_planning_agent import SettlementPlanningAgent
                    config = base.settlement_planning_config
                    if config is None:
                        raise ValueError("このPPOモデルには開拓計画設定がありません。")
                    _ppo_agents[agent_name] = SettlementPlanningAgent(base, **config)
                else:
                    _ppo_agents[agent_name] = base
            return _ppo_agents[agent_name]
        except (KeyError, ValueError, RuntimeError) as error:
            raise GameActionError(f"PPOモデルを利用できません: {error}") from error
    try:
        return AGENT_FACTORIES[agent_name]()
    except KeyError as error:
        raise GameActionError(f"未登録のAI種別です: {agent_name}") from error


def set_agent_for_player(game: GameState, player_id: int, agent_name: str) -> None:
    """将来の設定UIや実験コードから、ゲームエンジンを触れずにAIを差し替える入口。"""
    validate_agent_name(agent_name)
    next(player for player in game.players if player.id == player_id).agent_name = agent_name


def ai_actor(game: GameState) -> int | None:
    return pending_ai_player_id(game)


def _request_id(game: GameState) -> str:
    return f"ai-{game.ai_action_number:08d}"


def _record(game: GameState, message: str) -> None:
    game.ai_action_number += 1
    game.ai_log.append(message)
    if len(game.ai_log) > 500:
        del game.ai_log[:-500]


def _message_for(player_id: int, action) -> str:
    if isinstance(action, RollOrderAction):
        return f"P{player_id}: 順番決め {sum(action.dice or ())}"
    if isinstance(action, PlaceInitialSettlementAction):
        return f"P{player_id}: 初期開拓地 V{action.vertex_id}"
    if isinstance(action, PlaceInitialRoadAction):
        return f"P{player_id}: 初期道路 E{action.edge_id}"
    if isinstance(action, StartGameAction):
        return f"P{player_id}: ゲーム開始"
    if isinstance(action, RollTurnDiceAction):
        return f"P{player_id}: 通常ダイス {sum(action.dice or ())}"
    if isinstance(action, DiscardResourcesAction):
        return f"P{player_id}: 7で{sum(action.resources.values())}枚破棄"
    if isinstance(action, MoveRobberAction):
        return f"P{player_id}: 盗賊をH{action.tile_id}へ"
    if isinstance(action, StealResourceAction):
        return f"P{player_id}: P{action.victim_id}から1枚略奪"
    if isinstance(action, BuildRoadAction):
        return f"P{player_id}: 道路 E{action.edge_id}"
    if isinstance(action, BuildSettlementAction):
        return f"P{player_id}: 開拓地 V{action.vertex_id}"
    if isinstance(action, BuildCityAction):
        return f"P{player_id}: 都市 V{action.vertex_id}"
    if isinstance(action, BuyDevelopmentAction):
        return f"P{player_id}: 発展カード購入"
    if isinstance(action, BankTradeAction):
        return f"P{player_id}: 銀行・港交易"
    if isinstance(action, ProposeTradeAction):
        return f"P{player_id}: P{action.target_id}へ{len(action.offers)}候補で交渉"
    if isinstance(action, RespondToTradeAction):
        if action.decision == "accept":
            return f"P{player_id}: 交渉候補{(action.offer_index or 0) + 1}を承諾"
        if action.decision == "counter":
            return f"P{player_id}: 逆交渉を選択"
        return f"P{player_id}: 交渉を拒否"
    if isinstance(action, ProposeCounterTradeAction):
        offer = action.offers[0]
        return f"P{player_id}: {next(iter(offer.give))}と{next(iter(offer.want))}の逆交渉"
    if isinstance(action, UseDevelopmentAction):
        return f"P{player_id}: {action.card}を使用"
    if isinstance(action, EndTurnAction):
        return f"P{player_id}: 手番終了"
    return f"P{player_id}: {type(action).__name__}"


def play_ai_step(game: GameState) -> str:
    """AIを一手だけ進める。Agent は選択し、状態変更は apply_action が行う。"""
    player_id = ai_actor(game)
    if player_id is None:
        raise GameActionError("現在はAIが操作する場面ではありません。")
    agent = get_agent_for_player(game, player_id)
    action = agent.select_action(game, player_id)
    message = _message_for(player_id, action)
    diagnostic = None
    if game.phase in {"setup_ready", "action", "robber_move"}:
        from app.rl.decision_diagnostics import diagnose_action
        diagnostic = diagnose_action(game, player_id, action, agent)
    apply_action(game, player_id, action, game.revision, _request_id(game))
    if diagnostic is not None:
        game.ai_decisions.append(diagnostic)
        if len(game.ai_decisions) > 500:
            del game.ai_decisions[:-500]
    _record(game, message)
    return message


def run_ai_until_pause(game: GameState, max_steps: int = 100_000) -> dict:
    start_actions = game.ai_action_number
    while game.phase != "game_over" and ai_actor(game) is not None:
        if game.ai_action_number - start_actions >= max_steps:
            return {"status": "step_limit", "steps": max_steps, "phase": game.phase, "winner_id": game.winner_id}
        play_ai_step(game)
    return {"status": "game_over" if game.phase == "game_over" else "waiting_for_human",
            "steps": game.ai_action_number - start_actions, "phase": game.phase, "winner_id": game.winner_id}
