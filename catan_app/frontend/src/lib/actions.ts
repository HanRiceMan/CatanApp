import type { GameSetup, Resource } from "./types";

export const BUILD_COSTS = {
  road: { wood: 1, brick: 1 },
  settlement: { wood: 1, brick: 1, sheep: 1, wheat: 1 },
  city: { wheat: 2, ore: 3 },
  development: { sheep: 1, wheat: 1, ore: 1 },
} satisfies Record<string, Partial<Record<Resource, number>>>;
export type BuildKind = keyof typeof BUILD_COSTS;

/** 押せない理由も返すことで、必要な資源と次にできることを画面で説明する。 */
export function buildUnavailableReason(game: GameSetup, kind: BuildKind): string | null {
  const me = game.players.find(player => player.id === game.viewer_id);
  if (!me || game.current_player_id !== me.id || game.phase !== "action") return "自分の行動フェーズで使えます";
  const freeRoad = kind === "road" && game.action.free_road_remaining > 0;
  if (game.action.free_road_remaining > 0 && !freeRoad) return "先に無料の道を配置してください";
  if ((kind === "road" && me.road_count >= 15) || (kind === "settlement" && me.settlement_count >= 5) || (kind === "city" && me.city_count >= 4)) return "使える駒が残っていません";
  if (!freeRoad) {
    const missing = Object.entries(BUILD_COSTS[kind]).some(([resource, count]) => (me.resources?.[resource as Resource] ?? 0) < count);
    if (missing) return "資源が不足しています";
  }
  if (kind === "development") return game.action.development_remaining === 0 ? "山札がありません" : null;
  const ids = kind === "road" ? game.action.legal.road_ids : kind === "city" ? game.action.legal.city_ids : game.action.legal.settlement_ids;
  return ids.length === 0 ? "建設できる場所・駒がありません" : null;
}
