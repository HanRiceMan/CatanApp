"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ActivityEntry, GameSetup } from "../lib/types";

const DISPLAY_MS = 3000;

function buildDetail(entry: ActivityEntry) {
  if (entry.text.includes("発展カード")) {
    return { title: "DEVELOPMENT CARD", label: "発展カードを購入", icon: "?", purchase: true };
  }
  if (entry.text.includes("都市")) return { title: "BUILD", label: "都市へ発展", icon: "♜", purchase: false };
  if (entry.text.includes("開拓地")) return { title: "BUILD", label: "開拓地を建設", icon: "⌂", purchase: false };
  return { title: "BUILD", label: "道を建設", icon: "━", purchase: false };
}

/** 建築・発展カード購入の確定ログを、短い盤面カットインで伝える。 */
export function BuildCutIn({ game, onPlayingChange }: {
  game: GameSetup; onPlayingChange: (playing: boolean) => void;
}) {
  const entries = (game.activity_log ?? []).filter(entry => entry.kind === "build");
  const latestSequence = entries.at(-1)?.sequence ?? 0;
  const seen = useRef({ id: game.id, sequence: latestSequence });
  const pending = useRef<ActivityEntry[]>([]);
  const active = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [event, setEvent] = useState<ActivityEntry | null>(null);

  const showNext = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    const next = pending.current.shift() ?? null;
    setEvent(next);
    active.current = Boolean(next);
    onPlayingChange(Boolean(next));
    if (next) {
      const duration = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 650 : DISPLAY_MS;
      timer.current = setTimeout(showNext, duration);
    }
  }, [onPlayingChange]);

  useEffect(() => {
    if (seen.current.id !== game.id) {
      seen.current = { id: game.id, sequence: latestSequence };
      pending.current = [];
      active.current = false;
      if (timer.current) clearTimeout(timer.current);
      setEvent(null);
      onPlayingChange(false);
      return;
    }
    const fresh = entries.filter(entry => entry.sequence > seen.current.sequence);
    seen.current.sequence = latestSequence;
    if (!fresh.length) return;
    pending.current.push(...fresh);
    if (!active.current) showNext();
  }, [game.id, latestSequence, showNext, onPlayingChange]);

  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
    onPlayingChange(false);
  }, [onPlayingChange]);

  if (!event) return null;
  const detail = buildDetail(event);
  return <div className="dice-cut-in build-cut-in" role="status" aria-live="polite">
    <div className="commerce-stage build-commerce-stage" key={event.sequence}>
      <p className="commerce-kicker">{detail.title}</p>
      <h3>プレイヤー {event.player_id}</h3>
      <div className="build-commerce-content">
        <div className={detail.purchase ? "build-card-back" : "build-mark"} aria-hidden="true">{detail.icon}</div>
        <div className="build-description">
          <strong>{detail.label}</strong>
        </div>
      </div>
    </div>
  </div>;
}
