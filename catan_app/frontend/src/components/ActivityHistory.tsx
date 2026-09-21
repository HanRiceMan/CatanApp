"use client";

import { useRef } from "react";
import type { GameSetup } from "../lib/types";

export function ActivityHistory({ game }: { game: GameSetup }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const entries = game.activity_log ?? [];
  const latest = entries.at(-1);

  return <>
    <button className="activity-history-trigger" type="button" onClick={() => dialog.current?.showModal()}
      title={latest ? `プレイヤー ${latest.player_id}：${latest.text}` : "行動履歴を開く"}
      aria-label={latest ? `最新の行動：プレイヤー ${latest.player_id}、${latest.text}。行動履歴を開く` : "行動履歴を開く"}>
      <span aria-hidden="true">▤</span><span className="activity-latest-text">{latest ? `P${latest.player_id} ${latest.text}` : "行動履歴"}</span>
    </button>
    <dialog ref={dialog} className="activity-history-dialog" aria-labelledby="activity-history-title"
      onClick={event => { if (event.target === dialog.current) dialog.current.close(); }}>
      <header><div><small>TABLE LOG</small><h2 id="activity-history-title">行動履歴</h2></div>
        <button type="button" autoFocus onClick={() => dialog.current?.close()} aria-label="行動履歴を閉じる">閉じる</button></header>
      {entries.length ? <ol>{[...entries].reverse().map(entry => <li key={entry.sequence}>
        <span className="activity-history-turn">{entry.turn_number ? `第${entry.turn_number}手番` : "ゲーム準備"}</span>
        <div><strong>プレイヤー {entry.player_id}</strong><p>{entry.text}</p></div>
      </li>)}</ol> : <p className="activity-history-empty">まだ記録された行動はありません。</p>}
    </dialog>
  </>;
}
