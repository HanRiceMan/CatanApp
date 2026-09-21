import { useState } from "react";
import type { TradeOffer } from "../lib/api";
import { TERRAIN_INFO, type GameSetup, type Resource, type Tile } from "../lib/types";
import { buildUnavailableReason, type BuildKind } from "../lib/actions";
import { TradePanel } from "./TradePanel";

const RESOURCES: Resource[] = ["wood", "brick", "sheep", "wheat", "ore"];
const EMPTY = (): Record<Resource, number> => ({ wood: 0, brick: 0, sheep: 0, wheat: 0, ore: 0 });
const CARD_LABEL: Record<string, string> = { knight: "騎士", road_building: "街道建設", year_of_plenty: "豊作", monopoly: "独占", victory_point: "勝利点" };
const BUILD_LABEL: Record<BuildKind, [string, string]> = {
  road: ["道を建設", "木材1・レンガ1"], settlement: ["開拓地を建設", "木材1・レンガ1・羊毛1・小麦1"],
  city: ["都市にする", "小麦2・鉱石3"], development: ["発展カードを購入", "羊毛1・小麦1・鉱石1"],
};
type Category = "build" | "bank" | "trade" | "development";

function ResourceSelect({ value, onChange }: { value: Resource; onChange: (resource: Resource) => void }) {
  return <select value={value} onChange={event => onChange(event.target.value as Resource)}>{RESOURCES.map(resource => <option key={resource} value={resource}>{TERRAIN_INFO[resource].resource}</option>)}</select>;
}

export function NormalTurnPanel({ game, busy, selectedTile, onAction, onPropose, onRespond, onCounter }: {
  game: GameSetup; busy: boolean; selectedTile: Tile | null;
  onAction: (kind: string, payload?: Record<string, unknown>) => void;
  onPropose: (targetId: number, offers: TradeOffer[]) => void;
  onRespond: (decision: "accept" | "decline" | "counter", offerIndex?: number) => void;
  onCounter: (offers: TradeOffer[]) => void;
}) {
  const me = game.players.find(player => player.id === game.viewer_id);
  const [category, setCategory] = useState<Category | null>(null);
  const [discard, setDiscard] = useState(EMPTY);
  const [give, setGive] = useState<Resource>("wood");
  const [receive, setReceive] = useState<Resource>("brick");
  const [devResource, setDevResource] = useState<Resource>("wood");
  const [devResource2, setDevResource2] = useState<Resource>("brick");
  const cards = me?.development_cards ?? [];
  const playable = [...new Set(cards)].filter(card => card !== "victory_point" &&
    cards.filter(value => value === card).length > (me?.new_development_cards ?? []).filter(value => value === card).length);
  const plentyUnavailable = game.bank[devResource] < (devResource === devResource2 ? 2 : 1) || game.bank[devResource2] < 1;
  const developmentControls = <div className="development-controls">
    {playable.some(card => card === "year_of_plenty" || card === "monopoly") && <div className="development-resource-picks">
      <label>独占／豊作の1枚目<ResourceSelect value={devResource} onChange={setDevResource} /></label>
      {playable.includes("year_of_plenty") && <label>豊作の2枚目<ResourceSelect value={devResource2} onChange={setDevResource2} /></label>}
    </div>}
    {playable.map(card => <button type="button" key={card}
      disabled={busy || game.action.used_development_this_turn || (card === "year_of_plenty" && plentyUnavailable)}
      onClick={() => onAction("development", { card, resources: card === "year_of_plenty" ? [devResource, devResource2] : card === "monopoly" ? [devResource] : undefined })}>
      {CARD_LABEL[card]}を使う{card === "year_of_plenty" && plentyUnavailable ? "（銀行の在庫不足）" : ""}
    </button>)}
    {game.action.used_development_this_turn ? <small>この手番では発展カードを1枚使用済みです。</small> :
      playable.length === 0 && <p className="muted">今使える発展カードはありません。購入したカードは次の自分の手番から使えます。</p>}
  </div>;
  const buildButton = (kind: BuildKind) => {
    const reason = buildUnavailableReason(game, kind);
    return <button key={kind} type="button" disabled={busy || !!reason} title={reason ?? BUILD_LABEL[kind][1]}
      onClick={() => onAction(kind === "development" ? "build" : "build_mode", { kind })}>
      <strong>{BUILD_LABEL[kind][0]}</strong><small>{BUILD_LABEL[kind][1]}</small>{reason && <small className="unavailable-reason">{reason}</small>}
    </button>;
  };
  if (game.phase === "game_over") return <section className="turn-card winner-card"><h2>プレイヤー {game.winner_id} の勝利です！</h2></section>;
  if (!me) return <section className="turn-card"><p>プレイヤー用タブで通常手番を操作します。</p></section>;
  const mine = game.current_player_id === me.id;
  if (game.phase === "turn_pre_roll") return <section className="turn-card">
    <h2>プレイヤー {game.current_player_id} の手番</h2>
    {mine ? <><button className="start-button" disabled={busy} onClick={() => onAction("roll")}>サイコロを振る</button>
      <button type="button" aria-expanded={category === "development"} onClick={() => setCategory(category === "development" ? null : "development")}>発展カードを使う</button>
      {category === "development" && developmentControls}
      </> : <p>待機中</p>}
  </section>;
  if (game.phase === "discarding") {
    const count = game.your_discard_count;
    const total = Object.values(discard).reduce((a, b) => a + b, 0);
    const valid = total === count && RESOURCES.every(resource => discard[resource] <= (me.resources?.[resource] ?? 0));
    return <section className="turn-card"><h2>7が出ました</h2>{count ? <>
      <p>捨てる枚数：{count}（選択中 {total}）</p>
      {RESOURCES.map(resource => <label className="discard-row" key={resource}>{TERRAIN_INFO[resource].resource}
        <button type="button" aria-label={TERRAIN_INFO[resource].resource + "を減らす"} disabled={discard[resource] === 0 || busy} onClick={() => setDiscard(previous => ({ ...previous, [resource]: previous[resource] - 1 }))}>−</button>
        <output>{discard[resource]}</output>
        <button type="button" aria-label={TERRAIN_INFO[resource].resource + "を増やす"} disabled={total >= count || discard[resource] >= (me.resources?.[resource] ?? 0) || busy} onClick={() => setDiscard(previous => ({ ...previous, [resource]: previous[resource] + 1 }))}>＋</button>
      </label>)}
      <button disabled={busy || !valid} onClick={() => onAction("discard", { resources: discard })}>{count}枚を捨てる</button>
    </> : <p>他のプレイヤーが資源を捨てるのを待っています。</p>}</section>;
  }
  if (game.phase === "robber_move") {
    const chosen = selectedTile && game.robber.legal_tile_ids.includes(selectedTile.id) ? selectedTile : null;
    return <section className="turn-card"><h2>盗賊を動かす</h2>{mine ? <>
      <button disabled={busy || !chosen} onClick={() => chosen && onAction("robber", { tile_id: chosen.id })}>盗賊をここに置く</button>
    </> : <p>プレイヤー {game.current_player_id} が盗賊を動かしています。</p>}</section>;
  }
  if (game.phase === "robber_steal") return <section className="turn-card"><h2>資源を1枚奪う</h2>{mine ? <>
    {game.robber.victim_ids.map(id => <button key={id} disabled={busy} onClick={() => onAction("steal", { victim_id: id })}>プレイヤー {id} から奪う</button>)}
  </> : <p>プレイヤー {game.current_player_id} が対象を選んでいます。</p>}</section>;
  if (game.phase === "trade_response" || game.phase === "trade_counter_offer") return <TradePanel game={game} busy={busy} onPropose={onPropose} onRespond={onRespond} onCounter={onCounter} />;
  if (game.phase !== "action" || !mine) return <section className="turn-card"><p>プレイヤー {game.current_player_id} の操作を待っています。</p></section>;
  if (game.action.free_road_remaining > 0) return <section className="turn-card"><h2>無料の道を配置</h2>
    <p>残り {game.action.free_road_remaining} 本です。</p>
    <button disabled={busy || !!buildUnavailableReason(game, "road")} onClick={() => onAction("build_mode", { kind: "road" })}>盤面に無料の道を置く</button>
  </section>;
  const bankReason = give === receive ? "異なる資源を選んでください" :
    (me.resources?.[give] ?? 0) < (game.action.port_ratios[give] ?? 4) ? "渡す資源が不足しています" :
      game.bank[receive] < 1 ? "銀行の在庫がありません" : null;
  return <div className="action-workspace">
    <section className="turn-card action-menu-card">
      <div className="action-categories" aria-label="行動の種類">
        {([["build", "⌂", "建築"], ["bank", "⚓", "貿易"], ["trade", "⇄", "交渉"], ["development", "✦", "発展"]] as const).map(([key, icon, label]) =>
          <button type="button" key={key} aria-pressed={category === key} onClick={() => { setCategory(category === key ? null : key); onAction("build_mode", { kind: null }); }}>
            <span aria-hidden="true">{icon}</span>{label}
          </button>)}
      </div>
    </section>
    {category === "build" && <section className="turn-card"><h3>建築</h3><div className="build-actions">{(["road", "settlement", "city"] as const).map(buildButton)}</div>
      <button type="button" className="subtle-button" onClick={() => onAction("build_mode", { kind: null })}>盤面の建設選択を解除</button>
    </section>}
    {category === "bank" && <section className="turn-card"><h3>銀行・港との貿易</h3>
      <p>{game.action.port_ratios[give] ?? 4}枚を渡して、1枚を受け取ります。</p>
      <label>渡す資源<ResourceSelect value={give} onChange={setGive} /></label>
      <label>受け取る資源<ResourceSelect value={receive} onChange={setReceive} /></label>
      <button disabled={busy || !!bankReason} onClick={() => onAction("bank", { give_resource: give, receive_resource: receive })}>交換する</button>
      {bankReason && <small>{bankReason}</small>}
    </section>}
    {category === "trade" && <TradePanel game={game} busy={busy} onPropose={onPropose} onRespond={onRespond} onCounter={onCounter} />}
    {category === "development" && <section className="turn-card"><h3>発展カード</h3><div className="build-actions">{buildButton("development")}</div>
      {developmentControls}
    </section>}
    <button className="end-turn-button" disabled={busy || !game.last_roll} onClick={() => onAction("end")}>手番を終了する</button>
  </div>;
}
