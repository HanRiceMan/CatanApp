import type { BoardRules, CreatedGame, GameSetup, PieceKind, Resource } from "./types";

// スマホから8003番へ直接接続せず、同一オリジンのNext.js経由でPythonへ中継する。
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "/python-api";

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

async function request<T>(path: string, init: RequestInit = {}, token?: string): Promise<T> {
  const headers = new Headers(init.headers);
  if (token) headers.set("X-Player-Token", token);
  if (init.body) headers.set("Content-Type", "application/json");
  const response = await fetch(API_BASE_URL + path, {
    ...init, headers, cache: "no-store", signal: init.signal ?? AbortSignal.timeout(15000),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new ApiError(typeof error?.detail === "string" ? error.detail : "入力または接続を確認してください。", response.status);
  }
  return response.json();
}

export const createGame = (rules: BoardRules, aiPlayerIds: number[] = [], watchAi = false) =>
  request<CreatedGame>("/api/games", { method: "POST", body: JSON.stringify({ rules, ai_player_ids: aiPlayerIds, watch_ai: watchAi }) });

export const getGame = (id: string, token?: string, signal?: AbortSignal) =>
  request<GameSetup>(`/api/games/${encodeURIComponent(id)}`, { signal }, token);

export const rollForOrder = (id: string, token: string, revision: number, requestId: string) =>
  request<GameSetup>(`/api/games/${encodeURIComponent(id)}/order-rolls`, {
    method: "POST", body: JSON.stringify({ expected_revision: revision, request_id: requestId }),
  }, token);

export const placeInitialPiece = (id: string, token: string, revision: number, requestId: string,
  kind: PieceKind, targetId: number) => request<GameSetup>(`/api/games/${encodeURIComponent(id)}/initial-placements`, {
  method: "POST", body: JSON.stringify({ expected_revision: revision, request_id: requestId, kind, target_id: targetId }),
  }, token);

const action = (id: string, token: string, path: string, revision: number, requestId: string, body: Record<string, unknown> = {}) =>
  request<GameSetup>(`/api/games/${encodeURIComponent(id)}${path}`, { method: "POST", body: JSON.stringify({ ...body, expected_revision: revision, request_id: requestId }) }, token);

export const startPlay = (id: string, token: string, revision: number, requestId: string) => action(id, token, "/start-play", revision, requestId);
export const rollTurnDice = (id: string, token: string, revision: number, requestId: string) => action(id, token, "/turn-rolls", revision, requestId);
export const discardCards = (id: string, token: string, revision: number, requestId: string, resources: Record<Resource, number>) => action(id, token, "/discards", revision, requestId, { resources });
export const moveRobber = (id: string, token: string, revision: number, requestId: string, tileId: number) => action(id, token, "/robber", revision, requestId, { tile_id: tileId });
export const stealCard = (id: string, token: string, revision: number, requestId: string, victimId: number) => action(id, token, "/robber-steals", revision, requestId, { victim_id: victimId });
export const buildPiece = (id: string, token: string, revision: number, requestId: string, kind: "road" | "settlement" | "city" | "development", targetId?: number) => action(id, token, "/builds", revision, requestId, { kind, target_id: targetId ?? null });
export const bankTrade = (id: string, token: string, revision: number, requestId: string, give: Resource, receive: Resource) => action(id, token, "/bank-trades", revision, requestId, { give_resource: give, receive_resource: receive });
export type TradeOffer = { give: Record<Resource, number>; want: Record<Resource, number> };
export const proposeTrade = (id: string, token: string, revision: number, requestId: string, targetId: number, offers: TradeOffer[]) => action(id, token, "/player-trades", revision, requestId, { target_id: targetId, offers });
export const respondTrade = (id: string, token: string, revision: number, requestId: string, decision: "accept" | "decline" | "counter", offerIndex?: number) => action(id, token, "/player-trades/respond", revision, requestId, { decision, offer_index: offerIndex ?? null });
export const counterTrade = (id: string, token: string, revision: number, requestId: string, offers: TradeOffer[]) => action(id, token, "/player-trades/counter", revision, requestId, { offers });
export const useDevelopment = (id: string, token: string, revision: number, requestId: string, card: "knight" | "road_building" | "year_of_plenty" | "monopoly", resources?: Resource[]) => action(id, token, "/developments/use", revision, requestId, { card, resources });
export const endTurn = (id: string, token: string, revision: number, requestId: string) => action(id, token, "/end-turn", revision, requestId);
export const setPlayerController = (id: string, token: string, playerId: number, revision: number, requestId: string, isAi: boolean) =>
  action(id, token, `/players/${playerId}/controller`, revision, requestId, { is_ai: isAi });

export const advanceAi = (id: string, steps = 1) => request<{ game: GameSetup; result: { status: string; steps: number; phase: string; winner_id: number | null } }>(
  `/api/games/${encodeURIComponent(id)}/ai/run`, { method: "POST", body: JSON.stringify({ max_steps: steps }) });
