"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { TERRAIN_INFO, type ActivityEntry, type GameSetup, type Resource } from "../lib/types";

const INCIDENT_KINDS = new Set(["discard", "steal"]);

function IncidentScene({ entry }: { entry: ActivityEntry }) {
  const victimId = entry.details?.victim_id ?? Number(entry.text.match(/プレイヤー\s*(\d+)/)?.[1] ?? 0);
  const discarded = entry.details?.resources && Object.entries(entry.details.resources)
    .filter(([, count]) => count > 0)
    .map(([resource, count]) => `${TERRAIN_INFO[resource as Resource].resource} ${count}枚`);
  return <div className="incident-stage" key={entry.sequence}>
    <p className="incident-kicker">{entry.kind === "steal" ? "ROBBER" : "DISCARD"}</p>
    <div className="incident-mark" aria-hidden="true">{entry.kind === "steal" ? "♞" : "!"}</div>
    {entry.kind === "steal" ? <>
      <strong>プレイヤー {entry.player_id} が<br />プレイヤー {victimId || "?"} から奪った</strong>
      <small>奪われた資源の種類は非公開</small>
    </> : <>
      <strong>プレイヤー {entry.player_id} がバースト</strong>
      <span>{discarded?.length ? discarded.join("・") : entry.text}</span>
    </>}
  </div>;
}

/** 公開される7の破棄と盗賊の被害者だけを表示する。盗まれた資源は表示しない。 */
export function IncidentCutIn({ game, onPlayingChange }: {
  game: GameSetup; onPlayingChange: (playing: boolean) => void;
}) {
  const entries = (game.activity_log ?? []).filter(entry => INCIDENT_KINDS.has(entry.kind));
  const latestSequence = entries.at(-1)?.sequence ?? 0;
  const seen = useRef({ id: game.id, sequence: latestSequence });
  const pending = useRef<ActivityEntry[]>([]);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const active = useRef(false);
  const [event, setEvent] = useState<ActivityEntry | null>(null);

  const showNext = useCallback(function advance() {
    if (timer.current) clearTimeout(timer.current);
    const next = pending.current.shift() ?? null;
    setEvent(next);
    active.current = Boolean(next);
    onPlayingChange(Boolean(next));
    if (next) timer.current = setTimeout(advance,
      window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 325 : 1500);
  }, [onPlayingChange]);

  useEffect(() => {
    if (seen.current.id !== game.id) {
      seen.current = { id: game.id, sequence: latestSequence };
      pending.current = [];
      if (timer.current) clearTimeout(timer.current);
      active.current = false;
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

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  if (!event) return null;
  return <div className="dice-cut-in incident-cut-in" role="status" aria-live="polite"><IncidentScene entry={event} /></div>;
}
