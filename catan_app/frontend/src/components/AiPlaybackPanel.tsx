import type { GameSetup } from "../lib/types";

export function AiPlaybackPanel({ game, playing, stepping, speed, onPlayingChange, onSpeedChange, onStep }: {
  game: GameSetup; playing: boolean; stepping: boolean; speed: number;
  onPlayingChange: (playing: boolean) => void; onSpeedChange: (speed: number) => void; onStep: () => void;
}) {
  const lastAction = game.ai.recent_log.at(-1) ?? "再生すると、順番決めから開始します。";
  const finished = game.phase === "game_over";
  const waitingHuman = !finished && !game.ai.can_step;
  return <section className="ai-playback" aria-label="AI観戦コントロール">
    <div className="ai-playback-heading"><div><p className="eyebrow">AI SPECTATOR</p><h2>コンピューター対局を観戦</h2></div><span className={playing ? "live" : ""}>{finished ? "終了" : playing ? "再生中" : "一時停止"}</span></div>
    <div className="ai-current-action" aria-live="polite"><small>操作 {game.ai.actions}</small><strong>{lastAction}</strong>{game.ai.actor_id && <span>次：プレイヤー {game.ai.actor_id}</span>}</div>
    <div className="ai-player-controls">
      <button type="button" disabled={finished || waitingHuman} onClick={() => onPlayingChange(!playing)}>{playing ? "⏸ 一時停止" : "▶ 再生"}</button>
      <button type="button" disabled={playing || stepping || finished || waitingHuman} onClick={onStep}>⏭ 1操作進める</button>
      <label>再生速度<select value={speed} onChange={event => onSpeedChange(Number(event.target.value))}>
        <option value={1500}>ゆっくり（1.5秒）</option><option value={750}>標準（0.75秒）</option><option value={300}>速い（0.3秒）</option><option value={100}>最速（0.1秒）</option>
      </select></label>
    </div>
    {waitingHuman && <p className="observer-note">人間プレイヤーの操作を待っています。その操作後に再生を押してください。</p>}
    <details className="ai-action-log" open><summary>直近の操作</summary><ol>{game.ai.recent_log.slice(-10).reverse().map((line, index) => <li key={`${game.ai.actions - index}-${line}`}>{line}</li>)}</ol></details>
  </section>;
}
