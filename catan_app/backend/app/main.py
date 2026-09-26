"""ローカル4人対局のAPI。状態はメモリに保存し、再起動すると消える。"""

from secrets import compare_digest, randbelow
from threading import RLock
from typing import Literal
import json

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .domain.actions import (BankTradeAction, BuildCityAction, BuildRoadAction, BuildSettlementAction,
                             BuyDevelopmentAction, DiscardResourcesAction, EndTurnAction, MoveRobberAction,
                             PlaceInitialRoadAction, PlaceInitialSettlementAction, ProposeCounterTradeAction,
                             ProposeTradeAction, RespondToTradeAction, RollOrderAction, RollTurnDiceAction,
                             SetPlayerControllerAction, StartGameAction, StealResourceAction,
                             TradeOffer as DomainTradeOffer,
                             UseDevelopmentAction)
from .domain.board import BoardRules
from .domain.ai import run_ai_until_pause, validate_agent_name
from .domain.game import GameActionError, GameState, apply_action, create_game, game_view
from .rl.model_registry import ModelRegistry

app = FastAPI(title="CATAN Local Table API", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002", "http://127.0.0.1:3002"],
    # 家庭内LANでよく使われるプライベートIPv4からの画面接続だけを許可する。
    allow_origin_regex=r"^http://(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}):3002$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Player-Token"],
)

games: dict[str, GameState] = {}
game_lock = RLock()


class CreateGameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    rules: BoardRules = Field(default_factory=BoardRules)
    ai_player_ids: list[int] = Field(default_factory=list, max_length=4)
    # 既存のai_player_idsはそのまま使える。指定した席だけAI種別を上書きする。
    ai_agents: dict[int, str] = Field(default_factory=dict)
    watch_ai: bool = False


class RollRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0)
    request_id: str = Field(min_length=8, max_length=100)


class PlacementRequest(RollRequest):
    kind: Literal["settlement", "road"]
    target_id: int = Field(ge=0, le=71, strict=True)


class StartPlayRequest(RollRequest):
    pass


class DiscardRequest(RollRequest):
    resources: dict[str, int]


class RobberRequest(RollRequest):
    tile_id: int = Field(ge=0, le=18, strict=True)


class StealRequest(RollRequest):
    victim_id: int = Field(ge=1, le=4, strict=True)


class BuildRequest(RollRequest):
    kind: Literal["road", "settlement", "city", "development"]
    target_id: int | None = Field(default=None, ge=0, le=71, strict=True)


class BankTradeRequest(RollRequest):
    give_resource: str
    receive_resource: str


class TradeOfferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    give: dict[str, int]
    want: dict[str, int]


class TradeProposalRequest(RollRequest):
    target_id: int = Field(ge=1, le=4, strict=True)
    offers: list[TradeOfferRequest]


class TradeResponseRequest(RollRequest):
    decision: Literal["accept", "decline", "counter"]
    offer_index: int | None = Field(default=None, ge=0, le=2, strict=True)


class CounterTradeRequest(RollRequest):
    offers: list[TradeOfferRequest]


class DevelopmentRequest(RollRequest):
    card: Literal["knight", "road_building", "year_of_plenty", "monopoly"]
    resources: list[str] | None = None


class ControllerRequest(RollRequest):
    is_ai: bool
    agent_name: str | None = Field(default=None, min_length=1, max_length=100)


class AiRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = Field(default=100_000, ge=1, le=100_000)


def find_game(game_id: str) -> GameState:
    if game_id not in games:
        raise HTTPException(404, "対局が見つかりません。Pythonを再起動した場合は、新しい対局を作成してください。")
    return games[game_id]


def identify_player(game: GameState, token: str | None) -> int | None:
    if token is None:
        return None
    for player_id, saved_token in game.player_tokens.items():
        if compare_digest(saved_token, token):
            return player_id
    raise HTTPException(403, "プレイヤー用リンクが無効です。作成画面から正しいリンクを開いてください。")


def auto_advance(game: GameState) -> dict:
    if game.ai_watch_mode:
        return {"status": "watching", "steps": 0, "phase": game.phase, "winner_id": game.winner_id}
    return run_ai_until_pause(game)


def ppo_runtime_available() -> tuple[bool, str | None]:
    """通常サーバーが保存済みPPOを推論できるかをUIへ明示する。"""
    try:
        import sb3_contrib  # noqa: F401
    except ImportError:
        return False, "PPOモデルを使うには、Python 3.12の .venv-rl でバックエンドを起動してください。"
    return True, None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": "catan"}


@app.get("/api/ai-models")
def list_ai_models() -> dict:
    """現在のObservation/Action Spaceと互換性がある保存済みモデルだけを返す。"""
    available, message = ppo_runtime_available()
    registry = ModelRegistry()
    models = []
    for manifest in registry.list_available():
        entry = manifest.to_dict()
        config_path = registry.models_root / manifest.model_id / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            config = {}
        entry.update({
            "policy_architecture": config.get("policy_architecture", "mlp"),
            "heuristic_initial_placement": config.get("heuristic_initial_placement", False) is True,
            "initial_setup_policy": (
                manifest.initial_setup.get("type")
                if manifest.initial_setup is not None
                else ("heuristic" if config.get("heuristic_initial_placement", False) is True else "ppo")
            ),
            "settlement_planning": config.get("settlement_planning"),
        })
        models.append(entry)
    return {
        "models": models,
        "ppo_runtime_available": available,
        "ppo_runtime_message": message,
    }


@app.post("/api/games", status_code=201)
def start_game(request: CreateGameRequest | None = None) -> dict:
    request = request or CreateGameRequest()
    seed = request.seed if request.seed is not None else randbelow(2**32)
    try:
        if any(player_id not in range(1, 5) for player_id in request.ai_agents):
            raise ValueError("AI種別を指定できるプレイヤーIDは1〜4です。")
        for agent_name in request.ai_agents.values():
            validate_agent_name(agent_name)
        ai_player_ids = tuple(sorted(set(request.ai_player_ids) | set(request.ai_agents)))
        game = create_game(seed, request.rules, ai_player_ids, request.watch_ai, ai_agent_names=request.ai_agents)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    with game_lock:
        games[game.id] = game
        ai_result = auto_advance(game)
    # 参加用トークンは作成者に一度だけ返す。通常のGETでは公開しない。
    return {"game": game_view(game), "player_access": [
        {"player_id": player_id, "token": token} for player_id, token in game.player_tokens.items()],
        "ai_result": ai_result}


@app.get("/api/games/{game_id}")
def get_game(game_id: str, x_player_token: str | None = Header(default=None)) -> dict:
    with game_lock:
        game = find_game(game_id)
        return game_view(game, identify_player(game, x_player_token))


@app.post("/api/games/{game_id}/order-rolls")
def roll_dice(game_id: str, request: RollRequest,
              x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(RollOrderAction(), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/initial-placements")
def place_piece(game_id: str, request: PlacementRequest,
                x_player_token: str | None = Header(default=None)) -> dict:
    action = PlaceInitialSettlementAction(request.target_id) if request.kind == "settlement" else PlaceInitialRoadAction(request.target_id)
    return action_response(action, game_id, request, x_player_token)


def player_action(game_id: str, token: str | None) -> tuple[GameState, int]:
    game = find_game(game_id)
    player_id = identify_player(game, token)
    if player_id is None:
        raise HTTPException(403, "プレイヤー用タブから操作してください。")
    return game, player_id


def action_response(action, game_id: str, request: RollRequest, token: str | None) -> dict:
    with game_lock:
        game, player_id = player_action(game_id, token)
        try:
            apply_action(game, player_id, action, request.expected_revision, request.request_id)
            auto_advance(game)
        except (GameActionError, ValueError) as error:
            raise HTTPException(409, str(error)) from error
        return game_view(game, player_id)


@app.post("/api/games/{game_id}/start-play")
def start_play(game_id: str, request: StartPlayRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(StartGameAction(), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/turn-rolls")
def turn_roll(game_id: str, request: RollRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(RollTurnDiceAction(), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/discards")
def discard(game_id: str, request: DiscardRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(DiscardResourcesAction(request.resources), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/robber")
def robber(game_id: str, request: RobberRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(MoveRobberAction(request.tile_id), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/robber-steals")
def robber_steal(game_id: str, request: StealRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(StealResourceAction(request.victim_id), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/builds")
def build(game_id: str, request: BuildRequest, x_player_token: str | None = Header(default=None)) -> dict:
    actions = {
        "road": BuildRoadAction(request.target_id),
        "settlement": BuildSettlementAction(request.target_id),
        "city": BuildCityAction(request.target_id),
        "development": BuyDevelopmentAction(),
    }
    return action_response(actions[request.kind], game_id, request, x_player_token)


@app.post("/api/games/{game_id}/bank-trades")
def trade_with_bank(game_id: str, request: BankTradeRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(BankTradeAction(request.give_resource, request.receive_resource), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/player-trades")
def trade_with_player(game_id: str, request: TradeProposalRequest, x_player_token: str | None = Header(default=None)) -> dict:
    offers = tuple(DomainTradeOffer(offer.give, offer.want) for offer in request.offers)
    return action_response(ProposeTradeAction(request.target_id, offers), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/player-trades/respond")
def respond_trade(game_id: str, request: TradeResponseRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(RespondToTradeAction(request.decision, request.offer_index), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/player-trades/counter")
def counter_trade(game_id: str, request: CounterTradeRequest, x_player_token: str | None = Header(default=None)) -> dict:
    offers = tuple(DomainTradeOffer(offer.give, offer.want) for offer in request.offers)
    return action_response(ProposeCounterTradeAction(offers), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/developments/use")
def development(game_id: str, request: DevelopmentRequest, x_player_token: str | None = Header(default=None)) -> dict:
    resources = tuple(request.resources) if request.resources is not None else None
    return action_response(UseDevelopmentAction(request.card, resources), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/end-turn")
def finish_turn(game_id: str, request: RollRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(EndTurnAction(), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/players/{player_id}/controller")
def change_controller(game_id: str, player_id: int, request: ControllerRequest,
                      x_player_token: str | None = Header(default=None)) -> dict:
    with game_lock:
        game, authenticated_id = player_action(game_id, x_player_token)
        if authenticated_id != player_id:
            raise HTTPException(403, "操作担当を切り替えられるのは、そのプレイヤー用タブだけです。")
        try:
            if request.agent_name is not None:
                validate_agent_name(request.agent_name)
            apply_action(game, authenticated_id, SetPlayerControllerAction(request.is_ai, request.agent_name),
                         request.expected_revision, request.request_id)
            auto_advance(game)
        except (GameActionError, ValueError) as error:
            raise HTTPException(409, str(error)) from error
        return game_view(game, authenticated_id)


@app.post("/api/games/{game_id}/ai/run")
def run_ai_game(game_id: str, request: AiRunRequest | None = None) -> dict:
    with game_lock:
        game = find_game(game_id)
        result = run_ai_until_pause(game, (request or AiRunRequest()).max_steps)
        return {"game": game_view(game), "result": result}
