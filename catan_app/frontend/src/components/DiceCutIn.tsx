import { useEffect, useRef, useState } from "react";
import type { GameSetup } from "../lib/types";
import { Dice } from "./PlayerSeat";

type DiceEvent = NonNullable<GameSetup["latest_dice_event"]>;

/** サーバーが確定した出目だけを表示し、演出ではゲームの乱数を変更しない。 */
export function DiceCutIn({ game, onPlayingChange }: { game: GameSetup; onPlayingChange: (playing: boolean) => void }) {
  const [event, setEvent] = useState<DiceEvent | null>(null);
  const [rolling, setRolling] = useState(false);
  const seen = useRef({ id: game.id, sequence: game.latest_dice_event?.sequence ?? 0 });
  const settleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    const next = game.latest_dice_event;
    if (seen.current.id !== game.id) {
      seen.current = { id: game.id, sequence: next?.sequence ?? 0 };
      setEvent(null); onPlayingChange(false);
      return;
    }
    if (!next || seen.current.sequence === next.sequence) return;
    seen.current.sequence = next.sequence;
    if (settleTimer.current) clearTimeout(settleTimer.current);
    if (hideTimer.current) clearTimeout(hideTimer.current);
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    setEvent(next); setRolling(!reduced); onPlayingChange(true);
    settleTimer.current = setTimeout(() => setRolling(false), reduced ? 0 : 900);
    hideTimer.current = setTimeout(() => { setEvent(null); onPlayingChange(false); }, reduced ? 650 : 1700);
  }, [game.id, game.latest_dice_event?.sequence, onPlayingChange]);
  useEffect(() => () => {
    if (settleTimer.current) clearTimeout(settleTimer.current);
    if (hideTimer.current) clearTimeout(hideTimer.current);
  }, []);
  if (!event) return null;
  return <div key={event.sequence} className={`dice-cut-in ${rolling ? "rolling" : "settled"}`} role="status" aria-live="polite">
    <div className="dice-stage">
      <p>プレイヤー {event.player_id} · {event.kind === "order" ? "順番決め" : "サイコロ"}</p>
      <div className="tumbling-dice" aria-hidden="true">{event.dice.map((value, index) => {
        const remaining = [1, 2, 3, 4, 5, 6].filter(face => face !== value && face !== 7 - value);
        const faces = [value, 7 - value, remaining[0], 7 - remaining[0], remaining[1], 7 - remaining[1]];
        return <div key={index} className="die-tumbler">
          <div className="dice-cube">{faces.map((face, side) => <div key={side} className={`cube-face face-${side}`}><Dice value={face} /></div>)}</div>
        </div>;
      })}</div>
      <strong className="dice-total">{rolling ? "転がっています…" : event.total}</strong>
    </div>
  </div>;
}
