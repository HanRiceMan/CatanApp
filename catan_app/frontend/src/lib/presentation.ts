import type { GameSetup } from "./types";

export type PresentationKind = "dice" | "development" | "build" | "commerce" | "incident";

/** 新しい確定イベントがある時だけ、盤面・手札の表示をカットイン完了まで保留する。 */
export function presentationKinds(shown: GameSetup, incoming: GameSetup): PresentationKind[] {
  if (shown.id !== incoming.id || incoming.revision <= shown.revision) return [];
  const kinds = new Set<PresentationKind>();
  const dice = incoming.latest_dice_event;
  if (dice && dice.sequence > (shown.latest_dice_event?.sequence ?? 0)) kinds.add("dice");
  const lastActivity = shown.activity_log?.at(-1)?.sequence ?? 0;
  for (const entry of incoming.activity_log ?? []) {
    if (entry.sequence <= lastActivity) continue;
    if (entry.kind === "development") kinds.add("development");
    else if (entry.kind === "build") kinds.add("build");
    else if (["bank_trade", "trade_accepted", "trade_declined", "trade_counter"].includes(entry.kind)) kinds.add("commerce");
    else if (entry.kind === "discard" || entry.kind === "steal") kinds.add("incident");
  }
  return [...kinds];
}
