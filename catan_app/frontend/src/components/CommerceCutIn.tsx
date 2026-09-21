"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { TERRAIN_INFO, type ActivityEntry, type GameSetup, type Resource } from "../lib/types";

const RESOURCES: Resource[] = ["wood", "brick", "sheep", "wheat", "ore"];
const COMMERCE_KINDS = new Set(["trade", "trade_accepted", "trade_declined", "trade_counter", "bank_trade"]);
const MIN_PROPOSAL_MS = 900;
const RESULT_MS = 3000;
type Cards = Partial<Record<Resource, number>>;

function CardsLine({ cards }: { cards: Cards }) {
  const selected = RESOURCES.filter(resource => (cards[resource] ?? 0) > 0);
  return <span className="commerce-cards">{selected.length ? selected.map(resource =>
    <span key={resource}>{TERRAIN_INFO[resource].resource}<b> {cards[resource]}</b></span>) : "なし"}</span>;
}

function fallbackParties(entry: ActivityEntry, game: GameSetup): [number, number] {
  if (entry.kind === "trade_counter") {
    // 旧APIの「逆交渉を選びました。」には相手番号がない。直前の提案を参照する。
    const proposal = [...(game.activity_log ?? [])].reverse().find(candidate => candidate.kind === "trade"
      && candidate.sequence < entry.sequence
      && (candidate.trade?.recipient_id === entry.player_id
        || candidate.text.includes(`プレイヤー ${entry.player_id} `)));
    if (proposal) return [proposal.player_id, entry.player_id];
    const pending = game.pending_trade;
    if (pending?.recipient_id === entry.player_id) return [pending.proposer_id, entry.player_id];
    if (pending?.proposer_id === entry.player_id) return [pending.recipient_id, entry.player_id];
  }
  const other = Number(entry.text.match(/プレイヤー\s*(\d+)/)?.[1] ?? 0);
  return entry.kind === "trade" ? [entry.player_id, other] : [other, entry.player_id];
}

function openProposal(game: GameSetup): ActivityEntry | null {
  if (game.phase !== "trade_response" || !game.pending_trade) return null;
  const { proposer_id: proposerId, recipient_id: recipientId } = game.pending_trade;
  const matching = [...(game.activity_log ?? [])].reverse().find(entry => entry.kind === "trade"
    && entry.player_id === proposerId
    && (entry.trade?.recipient_id === recipientId || entry.text.includes(`プレイヤー ${recipientId} `)));
  const proposal = matching ?? { sequence: game.revision, turn_number: game.turn_number, player_id: proposerId,
    kind: "trade", text: `プレイヤー ${recipientId} に交渉を提案しました。` };
  if (proposal.trade || !game.pending_trade.offers) return proposal;
  return { ...proposal, trade: { proposer_id: proposerId, recipient_id: recipientId,
    is_counter: game.pending_trade.is_counter, outcome: "proposed", offers: game.pending_trade.offers } };
}

function previousProposal(entry: ActivityEntry, game: GameSetup): ActivityEntry | null {
  const [proposerId, recipientId] = entry.trade
    ? [entry.trade.proposer_id, entry.trade.recipient_id] : fallbackParties(entry, game);
  return [...(game.activity_log ?? [])].reverse().find(candidate => candidate.kind === "trade"
    && candidate.sequence < entry.sequence && candidate.player_id === proposerId
    && (candidate.trade?.recipient_id === recipientId || candidate.text.includes(`プレイヤー ${recipientId} `))) ?? null;
}

function resourceSummary(cards: Cards): string {
  return RESOURCES.filter(resource => (cards[resource] ?? 0) > 0)
    .map(resource => `${TERRAIN_INFO[resource].resource}${cards[resource]}枚`).join("・");
}

function selectedOfferIndex(entry: ActivityEntry, offers: NonNullable<ActivityEntry["trade"]>["offers"], proposerId: number, recipientId: number): number | undefined {
  if (entry.kind !== "trade_accepted") return undefined;
  if (entry.trade?.selected_offer_index !== undefined) return entry.trade.selected_offer_index;
  // 稼働中の旧APIは選択番号を送らないため、成立ログの資源内訳から候補を照合する。
  const index = offers.findIndex(offer => entry.text.includes(`プレイヤー ${proposerId} が${resourceSummary(offer.give)}`)
    && entry.text.includes(`プレイヤー ${recipientId} が${resourceSummary(offer.want)}`));
  return index >= 0 ? index : undefined;
}

function CommerceScene({ entry, game, proposal }: { entry: ActivityEntry; game: GameSetup; proposal: ActivityEntry | null }) {
  if (entry.kind === "bank_trade") {
    return <div className="commerce-stage" key={entry.sequence}>
      <p className="commerce-kicker">BANK TRADE</p>
      <h3>プレイヤー {entry.player_id} ↔ 銀行</h3>
      {entry.bank ? <div className="commerce-exchange">
        <div><small>プレイヤー {entry.player_id} が渡す</small><CardsLine cards={entry.bank.give} /></div>
        <span className="commerce-swap" aria-hidden="true">⇄</span>
        <div><small>銀行から受け取る</small><CardsLine cards={entry.bank.want} /></div>
      </div> : <p className="commerce-fallback">{entry.text}</p>}
      <strong className="commerce-result accepted">交換成立</strong>
    </div>;
  }

  const original = entry.kind === "trade" ? entry : proposal ?? previousProposal(entry, game);
  const trade = original?.trade ?? entry.trade;
  const [proposerId, recipientId] = trade ? [trade.proposer_id, trade.recipient_id] : fallbackParties(entry, game);
  const pending = game.pending_trade;
  const offers = trade?.offers ?? (entry.kind === "trade" && pending?.proposer_id === proposerId && pending?.recipient_id === recipientId ? pending.offers : undefined);
  const isProposal = entry.kind === "trade";
  const isAccepted = entry.kind === "trade_accepted";
  const isDeclined = entry.kind === "trade_declined";
  const isCounter = entry.kind === "trade_counter";
  const acceptedIndex = offers ? selectedOfferIndex(entry, offers, proposerId, recipientId) : undefined;
  return <div className="commerce-stage" key={original?.sequence ?? entry.sequence}>
    <p className="commerce-kicker">{trade?.is_counter ? "COUNTER OFFER" : "PLAYER TRADE"}</p>
    <h3>プレイヤー {proposerId || "?"} → プレイヤー {recipientId || "?"}</h3>
    {offers?.length ? <div className={`commerce-offers offers-${Math.min(offers.length, 3)}`}>{offers.map((offer, index) =>
      <div className={`commerce-exchange${acceptedIndex === index ? " selected" : ""}`} key={`${original?.sequence ?? entry.sequence}-${index}`}>
        <small className="commerce-option">第{index + 1}候補</small>
        <div><small>P{proposerId} が渡す</small><CardsLine cards={offer.give} /></div>
        <span className="commerce-swap" aria-hidden="true">⇄</span>
        <div><small>P{recipientId} が渡す</small><CardsLine cards={offer.want} /></div>
      </div>)}</div> : <p className="commerce-fallback">{entry.text}</p>}
    {!isProposal && <strong className={`commerce-result ${isAccepted ? "accepted" : isDeclined ? "declined" : "counter"}`}>
      {isAccepted ? "成立" : isDeclined ? "不成立" : "逆交渉へ"}
    </strong>}
  </div>;
}

/** 提案は返答まで残し、結果と銀行貿易は読める時間だけ表示する。再読込中の交渉も復元する。 */
export function CommerceCutIn({ game, onPlayingChange }: {
  game: GameSetup; onPlayingChange: (playing: boolean) => void;
}) {
  const entries = game.activity_log ?? [];
  const latestSequence = entries.at(-1)?.sequence ?? 0;
  const seen = useRef({ id: game.id, sequence: latestSequence });
  const pending = useRef<ActivityEntry[]>([]);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const initialProposal = openProposal(game);
  const active = useRef(Boolean(initialProposal));
  const current = useRef<ActivityEntry | null>(initialProposal);
  const proposalSnapshot = useRef<ActivityEntry | null>(initialProposal);
  const shownAt = useRef(Date.now());
  const [event, setEvent] = useState<ActivityEntry | null>(initialProposal);

  const showNext = useCallback(function advance() {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
    const next = pending.current.shift();
    current.current = next ?? null;
    if (!next) {
      active.current = false;
      proposalSnapshot.current = null;
      setEvent(null);
      onPlayingChange(false);
      return;
    }
    active.current = true;
    if (next.kind === "trade") proposalSnapshot.current = next;
    setEvent(next);
    shownAt.current = Date.now();
    // 提案中は返答ボタンとAIの応答を妨げない。次の結果イベントまで表示し続ける。
    onPlayingChange(next.kind !== "trade");
    if (next.kind === "trade") {
      if (pending.current.length) timer.current = setTimeout(advance, MIN_PROPOSAL_MS);
      return;
    }
    timer.current = setTimeout(advance, RESULT_MS);
  }, [onPlayingChange]);

  useEffect(() => {
    if (seen.current.id !== game.id) {
      seen.current = { id: game.id, sequence: latestSequence };
      pending.current = [];
      if (timer.current) clearTimeout(timer.current);
      timer.current = null;
      const proposal = openProposal(game);
      active.current = Boolean(proposal);
      current.current = proposal;
      proposalSnapshot.current = proposal;
      shownAt.current = Date.now();
      setEvent(proposal);
      onPlayingChange(false);
      return;
    }
    const fresh = entries.filter(entry => entry.sequence > seen.current.sequence && COMMERCE_KINDS.has(entry.kind))
      .map(entry => {
        // 稼働中の旧APIでは提案の内訳が履歴にない。参加者向け応答に残る候補を、その場で固定する。
        const pendingTrade = game.pending_trade;
        if (entry.trade || entry.kind !== "trade" || !pendingTrade?.offers || pendingTrade.proposer_id !== entry.player_id) return entry;
        return { ...entry, trade: { proposer_id: pendingTrade.proposer_id, recipient_id: pendingTrade.recipient_id,
          is_counter: pendingTrade.is_counter, outcome: "proposed" as const, offers: pendingTrade.offers } };
      });
    seen.current.sequence = latestSequence;
    if (!fresh.length) return;
    pending.current.push(...fresh);
    if (!active.current) showNext();
    else if (current.current?.kind === "trade" && !timer.current) {
      const remaining = Math.max(0, MIN_PROPOSAL_MS - (Date.now() - shownAt.current));
      timer.current = setTimeout(showNext, remaining);
    }
  }, [game.id, latestSequence, showNext, onPlayingChange]);

  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
    onPlayingChange(false);
  }, [onPlayingChange]);
  if (!event) return null;
  return <div className="dice-cut-in commerce-cut-in" role="status" aria-live="polite">
    <CommerceScene entry={event} game={game} proposal={proposalSnapshot.current} />
  </div>;
}

