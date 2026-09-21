import type { Resource } from "../lib/types";

// 港と手札に共通で使う、自作の資源ピクトグラム。外部画像は不要。
export function PortIcon({ resource }: { resource: Resource | null }) {
  switch (resource) {
    case "wood": return <g><path d="M0-20-15 3h8l-11 12h36L7 3h8Z" fill="#228c57" stroke="#12633a" strokeWidth="1.5" /><path d="M-3 13h6v9h-6Z" fill="#8a5230" /></g>;
    case "sheep": return <g><path d="M-16 8c-10-6-4-17 3-16 1-10 13-13 18-5 13-4 21 9 11 18-4 9-22 12-32 3Z" fill="#fffef2" stroke="#496d43" strokeWidth="1.6" /><ellipse cx="17" cy="1" rx="6" ry="8" fill="#495546" /><circle cx="19" cy="-1" r="1.2" fill="white" /><path d="M-9 13v6M7 13v6" stroke="#495546" strokeWidth="3" /></g>;
    case "wheat": return <g stroke="#9d680d" strokeWidth="2.2" strokeLinecap="round"><path d="M0 21v-40M-10 20V-8M10 20V-8" /><path d="M0-10c-11 0-12-9-11-11 8 0 11 4 11 11M0 1c10-1 12-9 10-12C2-9 0-6 0 1M0 11c-10-1-12-8-10-11C-2 2 0 5 0 11" fill="#f3bd24" /><path d="m-10 2-6-7m26 13 6-8" /></g>;
    case "ore": return <g><path d="m-21 8 8-23 21-4 16 17-7 21-27 2Z" fill="#8eacbf" stroke="#435c74" strokeWidth="1.8" /><path d="m-13-15 10 17L8-19M-21 8l18-6 20 17M-3 2l27-4" fill="none" stroke="#dcecf5" strokeWidth="2" /></g>;
    case "brick": return <g stroke="#653b24" strokeWidth="1.7"><path d="M-23 1h22v13h-22ZM2 1h22v13H2ZM-12-14h24v13h-24Z" fill="#b76d35" /><path d="M-20 4h16M5 4h16M-9-11H9" stroke="#e0a069" /></g>;
    default: return <g><path d="M-24 9h48l-8 13h-29Z" fill="#d38442" stroke="#834b2d" strokeWidth="1.5" /><path d="M-2-24V10" stroke="#754a2c" strokeWidth="2.5" /><path d="M-5-20-22 5H-5Z" fill="#fef7db" /><path d="M2-22V5h20Z" fill="#f3bd35" /><path d="M-25 25q6-5 12 0t12 0t12 0t12 0" fill="none" stroke="#429aaa" strokeWidth="2" /></g>;
  }
}
