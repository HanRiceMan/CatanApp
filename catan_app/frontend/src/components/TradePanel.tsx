import { useEffect, useState } from "react";
import type { TradeOffer } from "../lib/api";
import { TERRAIN_INFO, type GameSetup, type Resource } from "../lib/types";

const RESOURCES: Resource[] = ["wood", "brick", "sheep", "wheat", "ore"];
const EMPTY = (): Record<Resource, number> => ({ wood: 0, brick: 0, sheep: 0, wheat: 0, ore: 0 });
const blankOffer = (): TradeOffer => ({ give: EMPTY(), want: EMPTY() });

function ResourceEditor({ title, values, opposite, max, onChange }: {
  title: string; values: Record<Resource, number>; opposite: Record<Resource, number>; max: Record<Resource, number>;
  onChange: (resource: Resource, next: number) => void;
}) {
  return <div className="trade-resource-editor"><strong>{title}</strong>{RESOURCES.map(resource => {
    const info = TERRAIN_INFO[resource]; const amount = values[resource];
    const unavailable = opposite[resource] > 0;
    return <div className="trade-stepper" key={resource}>
      <span className={`mini-resource ${resource}`}>{info.resource}</span>
      <button type="button" aria-label={`${info.resource}を1枚減らす`} disabled={amount === 0} onClick={() => onChange(resource, amount - 1)}>−</button>
      <output>{amount}</output>
      <button type="button" aria-label={`${info.resource}を1枚増やす`} disabled={unavailable || amount >= max[resource]} onClick={() => onChange(resource, amount + 1)}>＋</button>
    </div>;
  })}</div>;
}

function OffersEditor({ resources, onSubmit, submitLabel, busy }: {
  resources: Record<Resource, number>; onSubmit: (offers: TradeOffer[]) => void; submitLabel: string; busy: boolean;
}) {
  const [count, setCount] = useState(1);
  const [offers, setOffers] = useState<TradeOffer[]>([blankOffer()]);
  useEffect(() => setOffers(previous => previous.length === count ? previous : Array.from({ length: count }, (_, index) => previous[index] ?? blankOffer())), [count]);
  function change(index: number, side: "give" | "want", resource: Resource, next: number) {
    setOffers(previous => previous.map((offer, offerIndex) => offerIndex !== index ? offer : { ...offer, [side]: { ...offer[side], [resource]: next } }));
  }
  const valid = offers.every(offer => RESOURCES.some(resource => offer.give[resource] > 0) &&
    RESOURCES.some(resource => offer.want[resource] > 0) &&
    RESOURCES.every(resource => offer.give[resource] <= resources[resource] && !(offer.give[resource] > 0 && offer.want[resource] > 0)));
  return <div className="offer-editor">
    <label className="offer-count">候補数 <select value={count} onChange={event => setCount(Number(event.target.value))}>{[1, 2, 3].map(value => <option value={value} key={value}>{value}個</option>)}</select></label>
    {offers.map((offer, index) => <section className="offer-card" key={index}>
      <h4>第{["一", "二", "三"][index]}候補</h4>
      <ResourceEditor title="あなた → 相手" values={offer.give} opposite={offer.want} max={resources} onChange={(resource, next) => change(index, "give", resource, next)} />
      <ResourceEditor title="相手 → あなた" values={offer.want} opposite={offer.give} max={{ wood: 19, brick: 19, sheep: 19, wheat: 19, ore: 19 }} onChange={(resource, next) => change(index, "want", resource, next)} />
    </section>)}
    <button className="trade-submit" type="button" disabled={!valid || busy} onClick={() => onSubmit(offers)}>{submitLabel}</button>
  </div>;
}

function cardsText(cards: Record<Resource, number>) {
  const text = RESOURCES.filter(resource => cards[resource] > 0).map(resource => `${TERRAIN_INFO[resource].resource}${cards[resource]}`).join("・");
  return text || "なし";
}

export function TradePanel({ game, busy, onPropose, onRespond, onCounter }: {
  game: GameSetup; busy: boolean; onPropose: (targetId: number, offers: TradeOffer[]) => void;
  onRespond: (decision: "accept" | "decline" | "counter", offerIndex?: number) => void; onCounter: (offers: TradeOffer[]) => void;
}) {
  const me = game.players.find(player => player.id === game.viewer_id);
  const trade = game.pending_trade;
  const [targetId, setTargetId] = useState(1);
  useEffect(() => { if (game.viewer_id && targetId === game.viewer_id) setTargetId(game.players.find(player => player.id !== game.viewer_id)?.id ?? 1); }, [game.viewer_id, game.players, targetId]);
  if (!me?.resources || game.viewer_id === null) return null;
  if (game.phase === "action" && game.current_player_id === game.viewer_id) {
    const targets = game.players.filter(player => player.id !== game.viewer_id);
    return <section className="turn-card trade-panel"><h3>プレイヤーと交渉</h3>
      {game.trade_count >= game.trade_limit ? <p className="muted">この手番の交渉は5回に達しました。</p> : <>
        <label>交渉相手 <select value={targetId} onChange={event => setTargetId(Number(event.target.value))}>{targets.map(player => <option value={player.id} key={player.id}>プレイヤー {player.id}</option>)}</select></label>
        <OffersEditor resources={me.resources} busy={busy} submitLabel="この内容で提案する" onSubmit={offers => onPropose(targetId, offers)} />
      </>}
    </section>;
  }
  if (!trade?.offers) return trade ? <section className="turn-card trade-panel"><p>交渉中</p></section> : null;
  if (game.phase === "trade_counter_offer" && trade.recipient_id === game.viewer_id) {
    return <section className="turn-card trade-panel"><h3>逆交渉を提案</h3><OffersEditor resources={me.resources} busy={busy} submitLabel="逆交渉を送る" onSubmit={onCounter} /></section>;
  }
  if (game.phase === "trade_response" && trade.recipient_id === game.viewer_id) {
    return <section className="turn-card trade-panel"><h3>{trade.is_counter ? "逆交渉が届きました" : "交渉が届きました"}</h3>
      {trade.offers.map((offer, index) => {
        const affordable = RESOURCES.every(resource => me.resources![resource] >= offer.want[resource]);
        return <article className="received-offer" key={index}><strong>第{["一", "二", "三"][index]}候補</strong><p>あなたが受け取る：{cardsText(offer.give)}</p><p>あなたが渡す：{cardsText(offer.want)}</p><button type="button" disabled={!affordable || busy} onClick={() => onRespond("accept", index)}>{affordable ? "この候補で成立" : "渡す資源が不足"}</button></article>;
      })}
      <div className="trade-response-actions"><button type="button" disabled={busy} onClick={() => onRespond("decline")}>すべて断る</button>{!trade.is_counter && <button type="button" disabled={busy} onClick={() => onRespond("counter")}>逆交渉をする</button>}</div>
    </section>;
  }
  return <section className="turn-card trade-panel"><p>返答待ち</p></section>;
}
