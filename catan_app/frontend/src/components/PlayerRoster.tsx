"use client";

import { useRef, useState, type CSSProperties } from "react";
import type { Player } from "../lib/types";
import { TERRAIN_INFO } from "../lib/types";
import styles from "./PlayerRoster.module.css";

const COLORS = { red: "#ed1726", cyan: "#39bddd", purple: "#984bc5", yellow: "#f5d900" };
const DEVELOPMENT_NAMES: Record<string, string> = {
  knight: "騎士", victory_point: "勝利点", monopoly: "独占", road_building: "街道建設", year_of_plenty: "豊作",
};

export function PlayerRoster({ players, order, rankingPending, viewerId, turnPlayerId }: {
  players: Player[]; order: number[]; rankingPending: boolean; viewerId: number | null; turnPlayerId: number | null;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const selected = players.find(player => player.id === selectedId);
  const sorted = order.map(id => players.find(player => player.id === id)).filter((p): p is Player => !!p);
  return <>
    <nav className={styles.roster} aria-label={rankingPending ? "順番決め中のプレイヤー" : "1番手から4番手のプレイヤー"}>
      <ol>
        {sorted.map((player, index) => <li key={player.id}>
          <button type="button" className={`${styles.player} ${player.pending_action ? styles.waiting : ""} ${player.id === turnPlayerId ? styles.currentTurn : ""}`}
            style={{ "--player-color": COLORS[player.color] } as CSSProperties}
            aria-haspopup="dialog"
            aria-label={`${rankingPending ? "順番未確定" : `${index + 1}番手`}、${player.name}、${player.is_ai ? "AI" : "人"}、${player.id === viewerId ? player.score ?? player.public_score : player.public_score}点${player.id === turnPlayerId ? "、現在の手番" : ""}${player.pending_action ? `、操作待ち：${player.pending_action}` : ""}`}
            title={player.pending_action ?? `${index + 1}番手`}
            onClick={() => { setSelectedId(player.id); dialog.current?.showModal(); }}>
            <span className={styles.identity}>
              <i className={styles.lamp} aria-hidden="true" />
              <strong>{player.name}</strong>
            </span>
            <span className={styles.summary}><span className={styles.controller}>{player.is_ai ? "AI" : "人"}</span>
              {(player.has_longest_road || player.has_largest_army) && <span className={styles.awards} aria-label="獲得している特殊カード">
                {player.has_longest_road && <i title="最長交易路" aria-label="最長交易路">⌁</i>}
                {player.has_largest_army && <i title="最大騎士力" aria-label="最大騎士力">♞</i>}
              </span>}
              <span className={styles.score}><b>{player.id === viewerId ? player.score ?? player.public_score : player.public_score}</b> 点</span></span>
          </button>
          {index < sorted.length - 1 && <span className={styles.arrow} aria-hidden="true">↓</span>}
        </li>)}
      </ol>
    </nav>
    <dialog ref={dialog} className={styles.dialog} aria-labelledby="player-detail-title"
      onClick={event => { if (event.target === dialog.current) dialog.current.close(); }}>
      {selected && <div className={styles.detail}>
        <header><h2 id="player-detail-title">{selected.name}</h2><button type="button" autoFocus onClick={() => dialog.current?.close()} aria-label="プレイヤー情報を閉じる">閉じる</button></header>
        {selected.pending_action && <p className={styles.task} role="status">● 操作待ち：{selected.pending_action}</p>}
        <dl className={styles.counts}><div><dt>資源カード</dt><dd>{selected.resource_count}<small>枚</small></dd></div><div><dt>発展カード</dt><dd>{selected.development_count}<small>枚</small></dd></div></dl>
        {selected.resources && <div className={styles.breakdown} aria-label="資源の内訳">{Object.entries(selected.resources).map(([resource, count]) => <span key={resource}>{TERRAIN_INFO[resource as keyof typeof TERRAIN_INFO].resource} <b>{count}</b></span>)}</div>}
        {selected.development_cards && <section><h3>所持している発展カード</h3><CardList cards={selected.development_cards} /></section>}
        <section><h3>使用済みの発展カード</h3><CardList cards={selected.used_development_cards ?? []} /></section>
        {(selected.has_longest_road || selected.has_largest_army) && <p>{selected.has_longest_road && "⌁ 最長交易路　"}{selected.has_largest_army && "♞ 最大騎士力"}</p>}
      </div>}
    </dialog>
  </>;
}

function CardList({ cards }: { cards: string[] }) {
  if (cards.length === 0) return <p className={styles.empty}>なし</p>;
  return <ul className={styles.cardList}>{Object.entries(DEVELOPMENT_NAMES).filter(([key]) => cards.includes(key)).map(([key, name]) =>
    <li key={key}><span>{name}</span><b>{cards.filter(card => card === key).length}枚</b></li>)}</ul>;
}
