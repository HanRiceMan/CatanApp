"use client";

import { useEffect, useRef, useState } from "react";
import { TERRAIN_INFO, type ActivityEntry, type GameSetup, type Resource } from "../lib/types";

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
      window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 325 : 850);
  }, [game.id, latest?.sequence, onPlayingChange]);

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  if (!event) return null;
  const card = CARD_LABELS[event.card ?? ""] ?? { icon: "✦", name: "発展カード" };
  const details = event.details;
  const plenty = details?.resources && Object.entries(details.resources).filter(([, count]) => count > 0)
    .map(([resource, count]) => `${TERRAIN_INFO[resource as Resource].resource} ${count}枚`).join("・");
  const taken = details?.taken_from && Object.entries(details.taken_from).filter(([, count]) => count > 0);
  return <div key={event.sequence} className="dice-cut-in development-cut-in" role="status" aria-live="polite">
    <div className="development-stage">
      <p>プレイヤー {event.player_id} が発展カードを使用</p>
      <div className="development-cut-card" aria-hidden="true"><span>{card.icon}</span></div>
      <strong>{card.name}</strong>
      {event.card === "year_of_plenty" && plenty && <span className="development-detail">銀行から {plenty} を獲得</span>}
      {event.card === "monopoly" && details?.resource && <div className="development-detail">
        <span>{TERRAIN_INFO[details.resource].resource}を独占</span>
        {taken?.length ? taken.map(([playerId, count]) => <span key={playerId}>プレイヤー {playerId} から {count}枚</span>) : <span>他のプレイヤーからの獲得は0枚</span>}
      </div>}
    </div>
  </div>;
}
