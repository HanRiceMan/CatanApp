"use client";

import { TERRAIN_INFO, type BoardData, type Tile, type GameSetup, type PlacementSelection } from "../lib/types";
import { TerrainIcon } from "./TerrainIcon";
import { PortIcon } from "./PortIcon";
import { PlacementLayer } from "./PlacementLayer";
import { SIZE, round, boardPoint as point } from "../lib/boardGeometry";

export type BuildMode = "road" | "settlement" | "city" | null;

// domain-model.mdの整数座標を、ここで初めて画面用の座標へ変換する。
// 小数を丸め、Node.jsとブラウザの計算誤差で初期HTMLが食い違うのを防ぐ。
const tileCenter = (q: number, r: number) => point(3 * q, q + 2 * r);
const hexPoints = Array.from({ length: 6 }, (_, i) => {
  const angle = i * Math.PI / 3;
  return `${round(Math.cos(angle) * (SIZE - 1))},${round(Math.sin(angle) * (SIZE - 1))}`;
}).join(" ");
const blankTiles = Array.from({ length: 5 }, (_, q) => q - 2).flatMap(q =>
  Array.from({ length: 5 }, (_, r) => r - 2)
    .filter(r => Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r)) <= 2)
    .map(r => ({ q, r })),
);

export function Board({ board, selectedId, onSelect, game, selection, placementDisabled, onSelectPlacement, buildMode, buildDisabled, onSelectBuild }: {
  board: BoardData | null;
  selectedId: number | null;
  onSelect: (tile: Tile) => void;
  game: GameSetup | null;
  selection: PlacementSelection | null;
  placementDisabled: boolean;
  onSelectPlacement: (selection: PlacementSelection) => void;
  buildMode: BuildMode;
  buildDisabled: boolean;
  onSelectBuild: (kind: Exclude<BuildMode, null>, targetId: number) => void;
}) {
  const robberSelection = game?.phase === "robber_move" && game.viewer_id === game.current_player_id;
  return <svg className={`board-svg ${robberSelection ? "choosing-robber" : ""}`} viewBox="0 0 800 670" preserveAspectRatio="xMaxYMid meet" role="group" aria-label="カタンの島。地形を選択すると詳細を表示します。">
    <defs>
      <radialGradient id="ocean"><stop stopColor="#38666b" /><stop offset="1" stopColor="#163c44" /></radialGradient>
      <pattern id="waves" width="48" height="40" patternUnits="userSpaceOnUse"><path d="M8 20q8-6 16 0t16 0" fill="none" stroke="#c1d9c8" strokeWidth=".7" opacity=".12" /></pattern>
      <filter id="island-shadow" x="-30%" y="-30%" width="160%" height="160%"><feDropShadow dx="0" dy="7" stdDeviation="4" floodOpacity=".2" /></filter>
    </defs>
    <rect width="800" height="670" rx="12" fill="url(#ocean)" />
    <rect width="800" height="670" rx="12" fill="url(#waves)" />
    <circle cx="400" cy="333" r="308" className="sea-ring" />
    <circle cx="400" cy="333" r="293" className="sea-ring" strokeDasharray="2 9" />
    <g transform="translate(64 70)" className="compass" aria-hidden="true">
      <path d="M0-21 5-5 21 0 5 5 0 21-5 5-21 0-5-5Z" fill="none" stroke="currentColor" />
      <path d="M0-21 5-5 0 0Z" fill="currentColor" /><text y="-29" textAnchor="middle">N</text>
    </g>
    <text x="36" y="626" className="ocean-caption">THE ISLAND OF CATAN</text>
    <text x="764" y="626" textAnchor="end" className="ocean-caption">19 LANDS · 9 HARBORS</text>
    {board ? <>
      <g filter="url(#island-shadow)">
        {board.tiles.map(tile => {
          const center = tileCenter(tile.q, tile.r);
          const info = TERRAIN_INFO[tile.terrain];
          const red = tile.number === 6 || tile.number === 8;
          const pips = tile.number === null ? 0 : 6 - Math.abs(7 - tile.number);
          const hasRobber = tile.id === (game?.robber_tile_id ?? board.robber_tile_id);
          const robberPreview = robberSelection && selectedId === tile.id && game.robber.legal_tile_ids.includes(tile.id);
          return <g key={tile.id} transform={`translate(${center.x} ${center.y})`}
            className={`land-tile ${selectedId === tile.id ? "selected" : ""} ${robberSelection && !hasRobber ? "robber-candidate" : ""} ${robberPreview ? "robber-preview" : ""}`} role="button" tabIndex={0}
            aria-label={`${info.name}、${tile.number ?? "数字なし"}${hasRobber ? "、盗賊あり" : ""}${robberPreview ? "、盗賊の移動候補・未確定" : ""}`}
            aria-pressed={selectedId === tile.id} onClick={() => onSelect(tile)}
            onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(tile); } }}>
            <polygon className="land-shape" points={hexPoints} fill={info.color} stroke="#e9d9b5" strokeWidth="3" />
            <g transform="translate(0 -29) scale(.55)" color={info.ink} stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" fill="none"><TerrainIcon terrain={tile.terrain} /></g>
            <text y="-7" className="terrain-label" fill={info.ink} textAnchor="middle">{info.name}</text>
            {tile.number !== null ? <g transform="translate(0 19)" className={`number-token${hasRobber ? " robber-token" : robberPreview ? " robber-preview-token" : red ? " red-token" : ""}`}>
              <circle r="21" fill="#152b29" opacity=".2" cy="2" />
              <circle r="21" fill={hasRobber ? "#747975" : robberPreview ? "#8f9691" : "#fff3d9"} stroke={hasRobber || robberPreview ? "#535a58" : "#ead4a9"} strokeWidth="1.5" />
              <text y="4" textAnchor="middle">{tile.number}</text>
              {Array.from({ length: pips }, (_, i) => <circle key={i} cx={(i - (pips - 1) / 2) * 4.8} cy="12" r="1.5" />)}
            </g> : null}
            {(hasRobber || robberPreview) && <g
              transform={tile.number !== null ? "translate(31 15) scale(.65)" : "translate(0 22)"}
              aria-label={hasRobber ? "盗賊" : "盗賊の移動候補"} className={robberPreview ? "robber-preview-piece" : ""} fill="#3b3430">
              <circle cy="-7" r="6" /><path d="M-6-1h12l5 15h-22Z" /><ellipse cy="15" rx="13" ry="3" />
            </g>}
          </g>;
        })}
      </g>
      {board.ports.map(port => {
        const edge = board.edges.find(e => e.id === port.edge_id)!;
        const a = board.vertices.find(v => v.id === edge.vertex_ids[0])!;
        const b = board.vertices.find(v => v.id === edge.vertex_ids[1])!;
        const mid = point((a.x + b.x) / 2, (a.y + b.y) / 2);
        const length = Math.hypot(mid.x - 400, mid.y - 333);
        const x = mid.x + (mid.x - 400) / length * 43;
        const y = mid.y + (mid.y - 333) / length * 43;
        const pa = point(a.x, a.y), pb = point(b.x, b.y);
        return <g key={port.id} className="port" aria-label={`${port.resource ? TERRAIN_INFO[port.resource].resource : "一般"}港 ${port.ratio}対1`}>
          <path d={`M${pa.x} ${pa.y}L${x} ${y}L${pb.x} ${pb.y}`} fill="none" stroke="#c3cdb2" strokeWidth="1.5" opacity=".65" />
          <circle cx={pa.x} cy={pa.y} r="3" fill="#fff2ce" /><circle cx={pb.x} cy={pb.y} r="3" fill="#fff2ce" />
          <g transform={`translate(${x} ${y})`}>
            <title>{port.resource ? TERRAIN_INFO[port.resource].resource : "一般"}港 {port.ratio}:1</title>
            <circle cy="2" r="27" fill="#092e36" opacity=".3" />
            <circle cy="-3" r="25" fill={port.resource ? "#e5f1d2" : "#d9eef1"} stroke="#fff4d7" strokeWidth="3" />
            <g transform="translate(0 -7) scale(.68)"><PortIcon resource={port.resource === "desert" ? null : port.resource} /></g>
            <rect x="-21" y="12" width="42" height="20" rx="9" fill="#fff7df" stroke="#d4bc85" />
            <text y="26" textAnchor="middle" className="port-ratio">{port.ratio}:1</text>
          </g>
        </g>;
      })}
      {game && <PlacementLayer game={game} selection={selection} disabled={placementDisabled} onSelect={onSelectPlacement}
        buildMode={buildMode} buildDisabled={buildDisabled} onSelectBuild={onSelectBuild} />}
    </> : <>
      <g opacity=".15">{blankTiles.map(({ q, r }) => {
        const p = tileCenter(q, r);
        return <polygon key={`${q},${r}`} transform={`translate(${p.x} ${p.y})`} points={hexPoints} fill="#6f9590" stroke="#dcebd0" strokeWidth="2" />;
      })}</g>
      <g className="empty-map" textAnchor="middle">
        <text x="400" y="314" className="empty-map-title">まだ見ぬ島へ。</text>
        <text x="400" y="350">ゲームスタートで、新しい盤面をつくろう。</text>
      </g>
    </>}
  </svg>;
}
