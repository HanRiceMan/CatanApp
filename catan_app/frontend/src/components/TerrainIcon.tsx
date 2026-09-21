import type { Terrain } from "../lib/types";

// 外部の画像を使わず、地形を小さなSVGで描く。
export function TerrainIcon({ terrain }: { terrain: Terrain }) {
  switch (terrain) {
    case "wood": return <g><path d="M-19 9 -10-11 -1 9ZM-5 9 7-18 19 9Z" fill="currentColor" /><path d="M-10 9v6M7 9v6" /></g>;
    case "brick": return <g><path d="M-23 12-10-9 4 12ZM-2 12 11-15 25 12Z" fill="currentColor" opacity=".7" /><path d="M-13 6h16M-8 0h8M7 5h13" /></g>;
    case "sheep": return <g><path d="M-16 6c-8-7-2-15 5-13 0-9 12-11 16-4 10-4 17 7 9 14-2 7-23 12-30 3Z" fill="currentColor" /><ellipse cx="16" cy="0" rx="5" ry="6" fill="currentColor" /><path d="M-8 9v7M8 8v8" /></g>;
    case "wheat": return <g><path d="M0 17v-34M-13 17V-8M13 17V-8M0-7-7-14M0 1 8-7M0 10-8 2M-13 3-19-3M13 7 19 0" /></g>;
    case "ore": return <g><path d="M-26 14-8-17 11 14ZM-2 14 11-9 27 14Z" fill="currentColor" opacity=".85" /><path d="m-15-5 7 4 6-5M8-2l4 4 3-2" stroke="var(--mountain-line, #60737b)" /></g>;
    case "desert": return <g><path d="M-27 9q17-20 34 0M-3 12q16-18 29 0" /><circle cx="14" cy="-13" r="5" fill="currentColor" stroke="none" /></g>;
  }
}
