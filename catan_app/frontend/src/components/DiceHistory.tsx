"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { GameSetup } from "../lib/types";
import { Dice } from "./PlayerSeat";

export function DiceHistory({ game }: { game: GameSetup }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [browserHistory, setBrowserHistory] = useState<GameSetup["dice_history"]>([]);
  const storageKey = `catan-dice-history:${game.id}`;

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) ?? "[]") as GameSetup["dice_history"];
      setBrowserHistory(Array.isArray(saved) ? saved : []);
    } catch { setBrowserHistory([]); }
  }, [storageKey]);

  useEffect(() => {
    const event = game.latest_dice_event;
    if (!event || event.kind !== "turn") return;
    const roll = { player_id: event.player_id, dice: event.dice, total: event.total, turn_number: event.turn_number ?? game.turn_number };
    setBrowserHistory(previous => {
      const key = `${roll.turn_number}-${roll.player_id}`;
      if (previous.some(item => `${item.turn_number}-${item.player_id}` === key)) return previous;
      const next = [...previous, roll];
      try { localStorage.setItem(storageKey, JSON.stringify(next)); } catch { /* 保存不可でも画面内では保持する */ }
      return next;
    });
  }, [game.latest_dice_event, game.turn_number, storageKey]);

  const history = useMemo(() => {
    const merged = new Map<string, GameSetup["dice_history"][number]>();
    for (const roll of [...browserHistory, ...(game.dice_history ?? [])]) merged.set(`${roll.turn_number}-${roll.player_id}`, roll);
    return [...merged.values()].sort((a, b) => b.turn_number - a.turn_number);
  }, [browserHistory, game.dice_history]);
  const current = game.last_roll;

  return <>
    <button className="dice-history-trigger" type="button" onClick={() => dialog.current?.showModal()}
      aria-label={current ? `この手番の出目は${current.total}。サイコロ履歴を開く` : "この手番は未投。サイコロ履歴を開く"}>
      <span aria-hidden="true">🎲</span><strong>{current?.total ?? "—"}</strong>
    </button>
    <dialog ref={dialog} className="dice-history-dialog" aria-labelledby="dice-history-title"
      onClick={event => { if (event.target === dialog.current) dialog.current.close(); }}>
      <header><div><small>ROLL HISTORY</small><h2 id="dice-history-title">サイコロの履歴</h2></div>
        <button type="button" autoFocus onClick={() => dialog.current?.close()} aria-label="サイコロ履歴を閉じる">閉じる</button></header>
      {history.length ? <ol>{history.map((roll, index) => {
        const player = game.players.find(candidate => candidate.id === roll.player_id);
        return <li key={`${roll.turn_number}-${roll.player_id}-${index}`}>
          <span className="dice-history-turn">第{roll.turn_number}手番</span>
          <strong>{player?.name ?? `プレイヤー ${roll.player_id}`}</strong>
          <span className="dice-history-values" aria-label={`出目 ${roll.dice[0]}と${roll.dice[1]}、合計${roll.total}`}>
            <Dice value={roll.dice[0]} /><Dice value={roll.dice[1]} /><b>{roll.total}</b>
          </span>
        </li>;
      })}</ol> : <p className="dice-history-empty">まだ通常手番のサイコロは振られていません。</p>}
    </dialog>
  </>;
}
