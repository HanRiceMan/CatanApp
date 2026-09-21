import type { GameSetup, PlacementSelection } from "../lib/types";

export function InitialPlacementPanel({ game, selection, busy, disconnected, onConfirm }: {
  game: GameSetup; selection: PlacementSelection | null; busy: boolean; disconnected: boolean;
  onConfirm: () => void;
}) {
  const setup = game.initial_placement;
  const isMe = game.viewer_id !== null && setup.player_id === game.viewer_id;
  const isSettlement = setup.piece === "settlement";

  return <>
    <p className="turn-message" role="status">プレイヤー {setup.player_id} {isSettlement ? "開拓地を置きます" : "道を置きます"}</p>
    {isMe ? <>
      <button className="start-button" disabled={!selection || busy || disconnected} onClick={onConfirm}>
        {busy ? "配置しています…" : isSettlement ? "ここに開拓地を置く" : "ここに道を置く"}
      </button>
    </> : null}
  </>;
}
