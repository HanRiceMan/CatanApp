import type { OrderRoll, Player, Resource } from "../lib/types";
import { TERRAIN_INFO } from "../lib/types";
import { PortIcon } from "./PortIcon";

const RESOURCES: Resource[] = ["wood", "brick", "sheep", "wheat", "ore"];
const DEVELOPMENT_NAMES: Record<string, string> = {
  knight: "騎士", victory_point: "勝利点", monopoly: "独占",
  road_building: "街道建設", year_of_plenty: "豊作",
};

function DevelopmentCards({ cards, used = false }: { cards: string[]; used?: boolean }) {
  return <div className={`development-hand ${used ? "used-development-hand" : ""}`}>
    {Object.keys(DEVELOPMENT_NAMES).filter(card => !used || card !== "victory_point").map(card => {
      const count = cards.filter(value => value === card).length;
      return <div className={`development-face ${count ? "" : "no-cards"}`} key={card}
        aria-label={`${used ? "使用済み" : "手札"}の${DEVELOPMENT_NAMES[card]} ${count}枚`}>
        <span aria-hidden="true">{DEVELOPMENT_ICONS[card]}</span><span className="development-name">{DEVELOPMENT_NAMES[card]}</span><strong>{count}</strong>
      </div>;
    })}
  </div>;
}
const DEVELOPMENT_ICONS: Record<string, string> = {
  knight: "♞", victory_point: "★", monopoly: "↔", road_building: "⌁", year_of_plenty: "✦",
};

export function Dice({ value }: { value: number }) {
  const spots: Record<number, number[]> = { 1: [4], 2: [0, 8], 3: [0, 4, 8], 4: [0, 2, 6, 8], 5: [0, 2, 4, 6, 8], 6: [0, 2, 3, 5, 6, 8] };
  return <span className="die" role="img" aria-label={`サイコロ ${value}`}>
    {Array.from({ length: 9 }, (_, i) => <i key={i} className={spots[value]?.includes(i) ? "pip" : ""} />)}
  </span>;
}

function CardBack({ count, development = false }: { count: number; development?: boolean }) {
  return <div className={`card-back-stack ${development ? "development-back" : "resource-back"} ${count === 0 ? "empty-stack" : ""}`}
    aria-label={`${development ? "発展" : "資源"}カード ${count}枚・非公開`}>
    <div className="card-back" aria-hidden="true"><span>{development ? "✦" : "⬡"}</span></div>
    <span className="card-quantity">{count}<small>枚</small></span>
    <span className="card-category">{development ? "発展" : "資源"}</span>
  </div>;
}

export function PlayerSeat({ player, position, isMe, revealHand = false, active, roll, rank, statusText, handOnly = false }: {
  player: Player; position: string; isMe: boolean; active: boolean;
  revealHand?: boolean;
  roll?: OrderRoll; rank?: number;
  statusText?: string;
  handOnly?: boolean;
}) {
  const showHand = isMe || revealHand;
  const score = isMe ? player.score ?? player.public_score : player.public_score;
  return <article className={`seat seat-${position} player-${player.color} ${isMe ? "own-seat" : ""} ${showHand ? "hand-visible-seat" : ""} ${active ? "active-seat" : ""} ${handOnly ? "hand-only-seat" : ""}`}
    aria-label={`${player.name}${isMe ? "・あなた" : ""}・${position}`}>
    {!handOnly && <header className="seat-header">
      <span className="seat-marker" aria-hidden="true">{player.id}</span>
      <div><h3>{player.name}{player.is_ai && <small>AI</small>}{isMe && <small>あなた</small>}
        {player.has_longest_road && <b className="award-mark" title="最長交易路" aria-label="最長交易路">⌁</b>}
        {player.has_largest_army && <b className="award-mark" title="最大騎士力" aria-label="最大騎士力">♞</b>}
      </h3>
        <p>{statusText ?? (rank ? `${rank}番手` : active ? "サイコロの番" : "順番決め待ち")}</p></div>
      <div className="seat-score" aria-label={`${isMe ? "あなたの得点" : "公開得点"} ${score}点`}>
        {score}<small>点</small>
      </div>
    </header>}
    {!handOnly && roll && <div className="seat-roll"><Dice value={roll.dice[0]} /><Dice value={roll.dice[1]} /><strong>{roll.total}</strong><small>{roll.round > 1 ? `再投 ${roll.round - 1}` : "初回"}</small></div>}
    {showHand ? <div className={`own-hand ${revealHand && !isMe ? "revealed-hand" : ""}`}>
      <div className="hand-label">資源カード</div>
      <div className="resource-hand">{RESOURCES.map(resource => {
        const count = player.resources?.[resource] ?? 0;
        return <div key={resource} className={`resource-face ${count === 0 ? "no-cards" : ""}`} aria-label={`${TERRAIN_INFO[resource].resource} ${count}枚`}>
          <svg viewBox="-30 -30 60 60" aria-hidden="true"><PortIcon resource={resource} /></svg>
          <span>{TERRAIN_INFO[resource].resource}</span><strong>{count}</strong>
        </div>;
      })}</div>
      <div className="hand-label">発展カード</div>
      <DevelopmentCards cards={player.development_cards ?? []} />
    </div> : <div className="hidden-hand"><CardBack count={player.resource_count} /><CardBack count={player.development_count} development /></div>}
    <div className="used-cards"><div className="hand-label">使用済み</div>
      <DevelopmentCards cards={player.used_development_cards ?? []} used />
    </div>
  </article>;
}
