"use client";

import { useEffect, useRef, useState } from "react";
import type { ActivityEntry, GameSetup } from "../lib/types";

const CARD_LABELS: Record<string, { icon: string; name: string }> = {
  knight: { icon: "♞", name: "騎士" },
  road_building: { icon: "⌁", name: "街道建設" },
  year_of_plenty: { icon: "✦", name: "豊作" },
  monopoly: { icon: "↔", name: "独占" },
};

export function DevelopmentCutIn({ game, onPlayingChange }: {
  game: GameSetup; onPlayingChange: (playing: boolean) => void;
}) {
  const latest = (game.activity_log ?? []).filter(entry => entry.kind === "development").at(-1) ?? null;
  const seen = useRef({ id: game.id, sequence: latest?.sequence ?? 0 });
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [event, setEvent] = useState<ActivityEntry | null>(null);

  useEffect(() => {
    if (seen.current.id !== game.id) {
      seen.current = { id: game.id, sequence: latest?.sequence ?? 0 };
      setEvent(null);
      onPlayingChange(false);
      return;
    }
    if (!latest || latest.sequence === seen.current.sequence) return;
    seen.current.sequence = latest.sequence;
    if (timer.current) clearTimeout(timer.current);
    setEvent(latest);
    onPlayingChange(true);
    timer.current = setTimeout(() => { setEvent(null); onPlayingChange(false); },
      window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 650 : 1700);
  }, [game.id, latest?.sequence, onPlayingChange]);

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  if (!event) return null;
  const card = CARD_LABELS[event.card ?? ""] ?? { icon: "✦", name: "発展カード" };
  return <div key={event.sequence} className="dice-cut-in development-cut-in" role="status" aria-live="polite">
    <div className="development-stage">
      <p>プレイヤー {event.player_id} が発展カードを使用</p>
      <div className="development-cut-card" aria-hidden="true"><span>{card.icon}</span></div>
      <strong>{card.name}</strong>
    </div>
  </div>;
}
