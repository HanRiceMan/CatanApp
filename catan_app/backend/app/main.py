"""ローカル4人対局のAPI。状態はメモリに保存し、再起動すると消える。"""

from secrets import compare_digest, randbelow
from threading import RLock
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .domain.board import BoardRules
from .domain.ai import run_ai_until_pause
from .domain.game import (GameActionError, GameState, bank_trade, build_piece, create_game,
                          discard_for_seven, end_turn, game_view, move_robber,
                          place_initial_piece, propose_counter_trade, propose_trade,
                          respond_to_trade, roll_for_order, roll_turn_dice,
                          set_player_controller, start_normal_game, steal_with_robber,
                          use_development)

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


class TradeOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    give: dict[str, int]
    want: dict[str, int]


class TradeProposalRequest(RollRequest):
    target_id: int = Field(ge=1, le=4, strict=True)
    offers: list[TradeOffer]


class TradeResponseRequest(RollRequest):
    decision: Literal["accept", "decline", "counter"]
    offer_index: int | None = Field(default=None, ge=0, le=2, strict=True)


class CounterTradeRequest(RollRequest):
    offers: list[TradeOffer]


class DevelopmentRequest(RollRequest):
    card: Literal["knight", "road_building", "year_of_plenty", "monopoly"]
    resources: list[str] | None = None


class ControllerRequest(RollRequest):
    is_ai: bool


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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": "catan"}


@app.post("/api/games", status_code=201)
def start_game(request: CreateGameRequest | None = None) -> dict:
    request = request or CreateGameRequest()
    seed = request.seed if request.seed is not None else randbelow(2**32)
    try:
        game = create_game(seed, request.rules, tuple(request.ai_player_ids), request.watch_ai)
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
    with game_lock:
        game = find_game(game_id)
        player_id = identify_player(game, x_player_token)
        if player_id is None:
            raise HTTPException(403, "プレイヤー用タブから操作してください。")
        try:
            roll_for_order(game, player_id, request.expected_revision, request.request_id)
            auto_advance(game)
        except GameActionError as error:
            raise HTTPException(409, str(error)) from error
        return game_view(game, player_id)


@app.post("/api/games/{game_id}/initial-placements")
def place_piece(game_id: str, request: PlacementRequest,
                x_player_token: str | None = Header(default=None)) -> dict:
    with game_lock:
        game = find_game(game_id)
        player_id = identify_player(game, x_player_token)
        if player_id is None:
            raise HTTPException(403, "プレイヤー用タブから操作してください。")
        try:
            place_initial_piece(game, player_id, request.kind, request.target_id,
                                request.expected_revision, request.request_id)
            auto_advance(game)
        except GameActionError as error:
            raise HTTPException(409, str(error)) from error
        return game_view(game, player_id)


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
            action(game, player_id, request)
            auto_advance(game)
        except GameActionError as error:
            raise HTTPException(409, str(error)) from error
        return game_view(game, player_id)


@app.post("/api/games/{game_id}/start-play")
def start_play(game_id: str, request: StartPlayRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(lambda game, pid, body: start_normal_game(game, pid, body.expected_revision, body.request_id), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/turn-rolls")
def turn_roll(game_id: str, request: RollRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(lambda game, pid, body: roll_turn_dice(game, pid, body.expected_revision, body.request_id), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/discards")
def discard(game_id: str, request: DiscardRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(lambda game, pid, body: discard_for_seven(game, pid, body.resources, body.expected_revision, body.request_id), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/robber")
def robber(game_id: str, request: RobberRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(lambda game, pid, body: move_robber(game, pid, body.tile_id, body.expected_revision, body.request_id), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/robber-steals")
def robber_steal(game_id: str, request: StealRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(lambda game, pid, body: steal_with_robber(game, pid, body.victim_id, body.expected_revision, body.request_id), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/builds")
def build(game_id: str, request: BuildRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(lambda game, pid, body: build_piece(game, pid, body.kind, body.target_id, body.expected_revision, body.request_id), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/bank-trades")
def trade_with_bank(game_id: str, request: BankTradeRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(lambda game, pid, body: bank_trade(game, pid, body.give_resource, body.receive_resource, body.expected_revision, body.request_id), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/player-trades")
def trade_with_player(game_id: str, request: TradeProposalRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(lambda game, pid, body: propose_trade(game, pid, body.target_id, [offer.model_dump() for offer in body.offers], body.expected_revision, body.request_id), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/player-trades/respond")
def respond_trade(game_id: str, request: TradeResponseRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(lambda game, pid, body: respond_to_trade(game, pid, body.decision, body.offer_index, body.expected_revision, body.request_id), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/player-trades/counter")
def counter_trade(game_id: str, request: CounterTradeRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(lambda game, pid, body: propose_counter_trade(game, pid, [offer.model_dump() for offer in body.offers], body.expected_revision, body.request_id), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/developments/use")
def development(game_id: str, request: DevelopmentRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(lambda game, pid, body: use_development(game, pid, body.card, body.expected_revision, body.request_id, resources=body.resources), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/end-turn")
def finish_turn(game_id: str, request: RollRequest, x_player_token: str | None = Header(default=None)) -> dict:
    return action_response(lambda game, pid, body: end_turn(game, pid, body.expected_revision, body.request_id), game_id, request, x_player_token)


@app.post("/api/games/{game_id}/players/{player_id}/controller")
def change_controller(game_id: str, player_id: int, request: ControllerRequest,
                      x_player_token: str | None = Header(default=None)) -> dict:
    def change(game: GameState, authenticated_id: int, body: ControllerRequest) -> None:
        if authenticated_id != player_id:
            raise HTTPException(403, "操作担当を切り替えられるのは、そのプレイヤー用タブだけです。")
        set_player_controller(game, authenticated_id, body.is_ai, body.expected_revision, body.request_id)
    return action_response(change, game_id, request, x_player_token)


@app.post("/api/games/{game_id}/ai/run")
def run_ai_game(game_id: str, request: AiRunRequest | None = None) -> dict:
    with game_lock:
        game = find_game(game_id)
        result = run_ai_until_pause(game, (request or AiRunRequest()).max_steps)
        return {"game": game_view(game), "result": result}
