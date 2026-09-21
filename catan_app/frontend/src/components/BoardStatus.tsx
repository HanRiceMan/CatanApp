import type { GameSetup } from "../lib/types";
import { Dice } from "./PlayerSeat";

const PHASE_LABEL: Record<GameSetup["phase"], string> = {
  rolling_order: "順番決め", setup_ready: "初期配置", ready_to_play: "開始待ち",
  turn_pre_roll: "サイコロ待ち", action: "行動中", discarding: "資源を捨てる",
  robber_move: "盗賊を移動", robber_steal: "資源を奪う",
  trade_response: "交渉の返答待ち", trade_counter_offer: "逆交渉中", game_over: "ゲーム終了",
};
export function BoardStatus({ game }: { game: GameSetup }) {
  const activeId = game.phase === "rolling_order" ? game.next_roller_id : game.phase === "setup_ready" ? game.initial_placement.player_id : game.phase === "ready_to_play" ? game.first_player_id : game.current_player_id;
  const active = game.players.find(player => player.id === activeId);
  const responder = game.pending_trade?.recipient_id;
  return <div className={`board-turn-status player-${active?.color ?? "red"}`} aria-label="現在の手番と出目">
    <div><span className="turn-player-dot" /><strong>{game.phase === "game_over" ? `P${game.winner_id} の勝利` : `P${activeId ?? "—"} の番`}</strong>
      <small>{PHASE_LABEL[game.phase]}{responder ? ` · P${responder} が操作` : ""}</small></div>
    <div className="current-dice">{game.last_roll ? <><Dice value={game.last_roll.dice[0]} /><Dice value={game.last_roll.dice[1]} /><strong>{game.last_roll.total}</strong></> :
      <small>{game.turn_number ? "この手番はまだ振っていません" : "通常手番の出目"}</small>}</div>
  </div>;
}
