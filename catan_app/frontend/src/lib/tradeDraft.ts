import type { ActivityEntry } from "./types";

/** 相手は引き継がず、自分が最後に提示した最大3候補だけを再利用する。 */
export function lastOwnTradeOffers(entries: ActivityEntry[] | undefined, viewerId: number | null) {
  if (viewerId === null) return undefined;
  return [...(entries ?? [])].reverse().find(entry =>
    entry.kind === "trade" && entry.player_id === viewerId && entry.trade?.offers?.length)?.trade?.offers;
}
