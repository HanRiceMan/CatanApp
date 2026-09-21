import type { GameSetup, PlacementSelection } from "../lib/types";
import type { BuildMode } from "./Board";
import { boardPoint } from "../lib/boardGeometry";

export function PlacementLayer({ game, selection, disabled, onSelect, buildMode, buildDisabled, onSelectBuild }: {
  game: GameSetup; selection: PlacementSelection | null; disabled: boolean;
  onSelect: (selection: PlacementSelection) => void;
  buildMode: BuildMode; buildDisabled: boolean;
  onSelectBuild: (kind: Exclude<BuildMode, null>, targetId: number) => void;
}) {
  const vertexPoint = (id: number) => {
    const vertex = game.board.vertices[id];
    return boardPoint(vertex.x, vertex.y);
  };
  const colorClass = (playerId: number | null) => `player-${game.players.find(p => p.id === playerId)?.color ?? "red"}`;
  const ownerClass = colorClass(buildMode ? game.current_player_id : game.initial_placement.player_id);
  const select = (kind: PlacementSelection["kind"], targetId: number) => {
    if (!disabled) onSelect({ kind, targetId, revision: game.revision });
  };
  const actionCandidates = buildMode === "road" ? game.action.legal.road_ids
    : buildMode === "settlement" ? game.action.legal.settlement_ids
      : buildMode === "city" ? game.action.legal.city_ids : [];

  return <g className="placement-layer">
    {game.roads.map(road => {
      const edge = game.board.edges[road.edge_id];
      const a = vertexPoint(edge.vertex_ids[0]), b = vertexPoint(edge.vertex_ids[1]);
      return <g key={road.edge_id} className={colorClass(road.player_id)} role="img" aria-label={`プレイヤー ${road.player_id} の道・辺 ${road.edge_id}`}>
        <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="built-road-outline" />
        <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="built-road" />
      </g>;
    })}
    {game.initial_placement.legal_edge_ids.map(id => {
      const edge = game.board.edges[id];
      const a = vertexPoint(edge.vertex_ids[0]), b = vertexPoint(edge.vertex_ids[1]);
      const chosen = selection?.kind === "road" && selection.targetId === id;
      // 両端を少し短くし、開拓地や隣の候補とのクリック領域の重なりを避ける。
      const x1 = a.x * .8 + b.x * .2, y1 = a.y * .8 + b.y * .2;
      const x2 = a.x * .2 + b.x * .8, y2 = a.y * .2 + b.y * .8;
      const length = Math.hypot(x2 - x1, y2 - y1);
      const nx = -(y2 - y1) / length * 12, ny = (x2 - x1) / length * 12;
      const hitPoints = `${x1 + nx},${y1 + ny} ${x2 + nx},${y2 + ny} ${x2 - nx},${y2 - ny} ${x1 - nx},${y1 - ny}`;
      return <g key={id} className={`placement-target ${ownerClass} ${chosen ? "chosen" : ""}`}
        role="button" tabIndex={disabled ? -1 : 0} aria-disabled={disabled} aria-pressed={chosen}
        aria-label={`道の候補：辺 ${id}`} onClick={() => select("road", id)}
        onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select("road", id); } }}>
        <polygon points={hitPoints} className="road-hit-area" />
        <line x1={x1} y1={y1} x2={x2} y2={y2} className="road-candidate" />
      </g>;
    })}
    {game.initial_placement.legal_vertex_ids.map(id => {
      const point = vertexPoint(id);
      const chosen = selection?.kind === "settlement" && selection.targetId === id;
      return <g key={id} transform={`translate(${point.x} ${point.y})`}
        className={`placement-target ${ownerClass} ${chosen ? "chosen" : ""}`} role="button" tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled} aria-pressed={chosen} aria-label={`開拓地の候補：交差点 ${id}`}
        onClick={() => select("settlement", id)}
        onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select("settlement", id); } }}>
        <circle r="17" fill="transparent" />
        <circle r={chosen ? 11 : 7} className="settlement-candidate" />
        {chosen && <path d="M-6 4V-1L0-7 6-1V4Z" fill="white" pointerEvents="none" />}
      </g>;
    })}
    {buildMode === "road" && actionCandidates.map(id => {
      const edge = game.board.edges[id];
      const a = vertexPoint(edge.vertex_ids[0]), b = vertexPoint(edge.vertex_ids[1]);
      const x1 = a.x * .8 + b.x * .2, y1 = a.y * .8 + b.y * .2, x2 = a.x * .2 + b.x * .8, y2 = a.y * .2 + b.y * .8;
      const length = Math.hypot(x2 - x1, y2 - y1), nx = -(y2 - y1) / length * 12, ny = (x2 - x1) / length * 12;
      return <g key={`build-${id}`} className={`placement-target ${ownerClass}`} role="button" tabIndex={buildDisabled ? -1 : 0} aria-disabled={buildDisabled} aria-label={`道を建設：辺 ${id}`} onClick={() => !buildDisabled && onSelectBuild("road", id)}>
        <polygon points={`${x1 + nx},${y1 + ny} ${x2 + nx},${y2 + ny} ${x2 - nx},${y2 - ny} ${x1 - nx},${y1 - ny}`} className="road-hit-area" /><line x1={x1} y1={y1} x2={x2} y2={y2} className="road-candidate" />
      </g>;
    })}
    {buildMode !== "road" && actionCandidates.map(id => {
      const point = vertexPoint(id); const city = buildMode === "city";
      return <g key={`build-${id}`} transform={`translate(${point.x} ${point.y})`} className={`placement-target ${ownerClass} build-target`} role="button" tabIndex={buildDisabled ? -1 : 0} aria-disabled={buildDisabled} aria-label={`${city ? "都市へ" : "開拓地を"}建設：交差点 ${id}`} onClick={() => !buildDisabled && onSelectBuild(buildMode!, id)} onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); if (!buildDisabled) onSelectBuild(buildMode!, id); } }}>
        <circle r="30" className="building-hit-area" /><circle r={city ? 24 : 16} className="settlement-candidate" />{city ? <path d="M-7 5V-5H-3V-10H3V-5H7V5Z" fill="white" pointerEvents="none" /> : <path d="M-6 4V-1L0-7 6-1V4Z" fill="white" pointerEvents="none" />}
      </g>;
    })}
    {game.settlements.map(settlement => {
      const point = vertexPoint(settlement.vertex_id);
      return <g key={settlement.vertex_id} transform={`translate(${point.x} ${point.y})`}
        className={`built-settlement ${colorClass(settlement.player_id)}`} role="img"
        aria-label={`プレイヤー ${settlement.player_id} の開拓地・交差点 ${settlement.vertex_id}`}>
        <path d="M-11 8V-3L0-13 11-3V8Z" />
        <path d="M-3 8V0H3V8" className="house-door" />
      </g>;
    })}
    {game.cities.map(city => {
      const point = vertexPoint(city.vertex_id);
      return <g key={city.vertex_id} transform={`translate(${point.x} ${point.y})`}
        className={`built-settlement built-city ${colorClass(city.player_id)}`} role="img"
        aria-label={`プレイヤー ${city.player_id} の都市・交差点 ${city.vertex_id}`}>
        <path d="M-13 9V-5H-6V-13H0V-6H7V-13H13V9Z" />
        <path d="M-3 9V1H3V9" className="house-door" />
      </g>;
    })}
  </g>;
}
