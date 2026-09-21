"use client";

import { useEffect, useRef, useState } from "react";
import { Board, type BuildMode } from "../components/Board";
import { PlayerSeat } from "../components/PlayerSeat";
import { InitialPlacementPanel } from "../components/InitialPlacementPanel";
import { NormalTurnPanel } from "../components/NormalTurnPanel";
import { AiPlaybackPanel } from "../components/AiPlaybackPanel";
import { PlayerRoster } from "../components/PlayerRoster";
import { DiceCutIn } from "../components/DiceCutIn";
import { DevelopmentCutIn } from "../components/DevelopmentCutIn";
import { BuildCutIn } from "../components/BuildCutIn";
import { CommerceCutIn } from "../components/CommerceCutIn";
import { ActivityHistory } from "../components/ActivityHistory";
import { DiceHistory } from "../components/DiceHistory";
import { buildUnavailableReason } from "../lib/actions";
import { ApiError, advanceAi, bankTrade, buildPiece, counterTrade, createGame, discardCards, endTurn, getGame, moveRobber, placeInitialPiece, proposeTrade, respondTrade, rollForOrder, rollTurnDice, setPlayerController, startPlay, stealCard, useDevelopment } from "../lib/api";
import type { TradeOffer } from "../lib/api";
import { createRequestId } from "../lib/requestId";
import { DEFAULT_RULES, TERRAIN_INFO, type BoardRules, type GameSetup, type Player, type PlayerAccess, type Tile, type PlacementSelection } from "../lib/types";

type Session = { id: string; token?: string };
const RULE_OPTIONS: { key: keyof BoardRules; label: string; description: string }[] = [
  { key: "desert_center", label: "砂漠を中央に固定", description: "中央のタイルID 9を砂漠にします。" },
  { key: "red_numbers_not_adjacent", label: "6・8を隣接させない", description: "6同士・8同士も含め、辺で隣り合わない配置。" },
  { key: "red_numbers_distinct_terrains", label: "6・8を4種類の地形に分散", description: "資源地形ごとに最大1枚。5種類のうち1種類には置きません。" },
  { key: "terrain_clusters_max_two", label: "同じ地形のつながりは2枚まで", description: "3枚以上の連結は禁止。別々の2枚組はOK。" },
];
const EMPTY_PLAYERS: Player[] = (["red", "cyan", "purple", "yellow"] as const).map((color, i) => ({
  id: i + 1, name: `プレイヤー ${i + 1}`, color, is_ai: false,
  resource_count: 0, development_count: 0, public_score: 0, played_knights: 0,
  has_longest_road: false, has_largest_army: false,
  settlement_count: 0, city_count: 0, road_count: 0,
}));
function errorMessage(error: unknown) {
  return error instanceof ApiError ? error.message : "Python側（8003番）に接続できません。起動状態を確認してください。";
}
function accessLink(id: string, token?: string) {
  const query = new URLSearchParams({ game: id });
  if (token) query.set("token", token);
  return `/#${query.toString()}`;
}

export default function Home() {
  const [session, setSession] = useState<Session | null>(null);
  const [game, setGame] = useState<GameSetup | null>(null);
  const [access, setAccess] = useState<PlayerAccess[]>([]);
  const [rules, setRules] = useState<BoardRules>(DEFAULT_RULES);
  const [aiPlayerIds, setAiPlayerIds] = useState<number[]>([]);
  const [watchAi, setWatchAi] = useState(true);
  const [aiPlaying, setAiPlaying] = useState(false);
  const [aiSpeed, setAiSpeed] = useState(750);
  const [aiStepping, setAiStepping] = useState(false);
  const [diceAnimating, setDiceAnimating] = useState(false);
  const [developmentAnimating, setDevelopmentAnimating] = useState(false);
  const [buildAnimating, setBuildAnimating] = useState(false);
  const [commerceAnimating, setCommerceAnimating] = useState(false);
  const cutInAnimating = diceAnimating || developmentAnimating || buildAnimating || commerceAnimating;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [selectedTile, setSelectedTile] = useState<Tile | null>(null);
  const [placementChoice, setPlacementChoice] = useState<PlacementSelection | null>(null);
  const [buildMode, setBuildMode] = useState<BuildMode>(null);
  const [boardExpanded, setBoardExpanded] = useState(false);
  const pending = useRef(false);
  const pendingRoll = useRef<{ revision: number; requestId: string } | null>(null);
  const pendingPlacement = useRef<(PlacementSelection & { requestId: string }) | null>(null);
  const aiStepPending = useRef(false);
  const aiAutoStarted = useRef(false);
  const aiControllerAuto = useRef(false);
  const aiPausedByUser = useRef(false);

  useEffect(() => {
    if (game?.phase === "robber_move") setSelectedTile(null);
  }, [game?.phase, game?.turn_number, game?.viewer_id]);

  useEffect(() => {
    function readLocation() {
      const params = new URLSearchParams(window.location.hash.slice(1));
      const id = params.get("game");
      const token = params.get("token") ?? undefined;
      setSession(id ? { id, token } : null);
      setGame(null);
      setSelectedTile(null);
      setBuildMode(null);
      setBoardExpanded(false);
      setError("");
      setConnectionError("");
      setAccess([]);
      setAiPlaying(false);
      aiAutoStarted.current = false;
      aiControllerAuto.current = false;
      aiPausedByUser.current = false;
      if (id && !token) {
        try { setAccess(JSON.parse(sessionStorage.getItem(`catan-access:${id}`) ?? "[]")); } catch { /* 保存なしでも観戦は可能 */ }
      }
    }
    readLocation();
    window.addEventListener("hashchange", readLocation);
    return () => window.removeEventListener("hashchange", readLocation);
  }, []);

  useEffect(() => {
    if (!game) return;
    const viewerIsAi = Boolean(game.viewer_id && game.players.find(player => player.id === game.viewer_id)?.is_ai);
    const autoEligible = game.ai.watch_mode || viewerIsAi || aiControllerAuto.current;
    // AIの手番になった瞬間（can_stepがfalse→true）も自動再生を再開する。
    // ただし、デバッグのために人が一時停止した場合だけは尊重する。
    if (autoEligible && game.ai.can_step && !aiPausedByUser.current) setAiPlaying(true);
    aiAutoStarted.current = true;
  }, [game?.id, game?.revision, game?.viewer_id, game?.ai.watch_mode, game?.ai.can_step]);

  useEffect(() => {
    if (!session) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    async function poll() {
      let fatal = false;
      try {
        const next = await getGame(session!.id, session!.token,
          AbortSignal.any([controller.signal, AbortSignal.timeout(10000)]));
        if (disposed) return;
        // 操作の応答より古いポーリング結果で画面を巻き戻さない。
        setGame(previous => !previous || previous.id !== next.id || previous.revision <= next.revision ? next : previous);
        setConnectionError("");
      } catch (failure) {
        if (disposed) return;
        setConnectionError(errorMessage(failure));
        fatal = failure instanceof ApiError && [403, 404].includes(failure.status);
      }
      if (!disposed && !fatal) timer = setTimeout(poll, 1000);
    }
    void poll();
    return () => { disposed = true; clearTimeout(timer); controller.abort(); };
  }, [session]);

  useEffect(() => {
    // 観戦モードだけでなく、個別プレイヤー画面でそのプレイヤーがAIの場合も
    // 「再生」を押さずに自動で次の操作へ進める。
    const currentGame = game;
    const viewerIsAi = Boolean(currentGame?.viewer_id && currentGame.players.find(player => player.id === currentGame.viewer_id)?.is_ai);
    if (!session || !currentGame || !(currentGame.ai.watch_mode || viewerIsAi || aiControllerAuto.current) || !aiPlaying || cutInAnimating || !currentGame.ai.can_step || currentGame.phase === "game_over") return;
    const timer = setTimeout(() => { void stepAi(); }, aiSpeed);
    return () => clearTimeout(timer);
  }, [session, game?.revision, game?.viewer_id, game?.players, game?.ai.watch_mode, game?.ai.can_step, game?.phase, aiPlaying, aiSpeed, cutInAnimating]);

  useEffect(() => {
    if (game && (!game.ai.can_step || game.phase === "game_over")) setAiPlaying(false);
  }, [game?.ai.can_step, game?.phase]);

  async function startGame() {
    if (pending.current) return;
    pending.current = true;
    setBusy(true);
    setError("");
    try {
      const created = await createGame(rules, aiPlayerIds, watchAi && aiPlayerIds.length > 0);
      setGame(created.game);
      const createdViewer = created.game.players.find(player => player.id === created.game.viewer_id);
      aiPausedByUser.current = false;
      setAiPlaying(created.game.ai.can_step && (created.game.ai.watch_mode || Boolean(createdViewer?.is_ai)));
      setAccess(created.player_access);
      setSelectedTile(null);
      try { sessionStorage.setItem(`catan-access:${created.game.id}`, JSON.stringify(created.player_access)); } catch { /* 保存不可でも今回の画面ではリンクを使える */ }
      window.history.replaceState(null, "", accessLink(created.game.id));
      setSession({ id: created.game.id });
    } catch (failure) { setError(errorMessage(failure)); }
    finally { pending.current = false; setBusy(false); }
  }

  async function stepAi() {
    if (!session || !game?.ai.can_step || aiStepPending.current || cutInAnimating || pending.current) return;
    aiStepPending.current = true;
    setAiStepping(true);
    setError("");
    try {
      const { game: publicView } = await advanceAi(session.id, 1);
      const next = session.token ? await getGame(session.id, session.token) : publicView;
      setGame(previous => !previous || previous.revision <= next.revision ? next : previous);
      if (!next.ai.can_step || next.phase === "game_over") setAiPlaying(false);
    } catch (failure) {
      setError(errorMessage(failure));
      setAiPlaying(false);
    } finally {
      aiStepPending.current = false;
      setAiStepping(false);
    }
  }

  async function rollDice() {
    if (!session?.token || !game || pending.current) return;
    pending.current = true;
    setBusy(true);
    setError("");
    if (!pendingRoll.current || pendingRoll.current.revision !== game.revision) {
      pendingRoll.current = { revision: game.revision, requestId: createRequestId() };
    }
    const attempt = pendingRoll.current;
    try {
      const next = await rollForOrder(session.id, session.token, attempt.revision, attempt.requestId);
      setGame(previous => !previous || previous.revision <= next.revision ? next : previous);
      pendingRoll.current = null;
    } catch (failure) {
      setError(errorMessage(failure));
      if (failure instanceof ApiError) pendingRoll.current = null;
    } finally { pending.current = false; setBusy(false); }
  }

  const players = game?.players ?? EMPTY_PLAYERS;
  const viewerId = game?.viewer_id ?? null;
  const viewerPlayer = players.find(player => player.id === viewerId);
  const manualControlsDisabled = busy || cutInAnimating || Boolean(viewerPlayer?.is_ai);
  const activeBuildMode = game && buildMode && !buildUnavailableReason(game, buildMode) ? buildMode : null;
  const seatOrder = game?.seat_order ?? [1, 2, 3, 4];
  const canRoll = game?.phase === "rolling_order" && viewerId !== null && game.next_roller_id === viewerId;
  const canPlace = game?.phase === "setup_ready" && viewerId !== null && game.initial_placement.player_id === viewerId;
  const placementSelection = game && canPlace && placementChoice?.revision === game.revision
    && placementChoice.kind === game.initial_placement.piece
    && (placementChoice.kind === "settlement" ? game.initial_placement.legal_vertex_ids : game.initial_placement.legal_edge_ids).includes(placementChoice.targetId)
    ? placementChoice : null;
  const selectedInfo = selectedTile ? TERRAIN_INFO[selectedTile.terrain] : null;

  async function confirmPlacement() {
    if (!session?.token || !game || !placementSelection || !canPlace || pending.current) return;
    pending.current = true;
    setBusy(true);
    setError("");
    const old = pendingPlacement.current;
    if (!old || old.revision !== placementSelection.revision || old.kind !== placementSelection.kind || old.targetId !== placementSelection.targetId) {
      pendingPlacement.current = { ...placementSelection, requestId: createRequestId() };
    }
    const attempt = pendingPlacement.current!;
    try {
      const next = await placeInitialPiece(session.id, session.token, attempt.revision, attempt.requestId, attempt.kind, attempt.targetId);
      setGame(previous => !previous || previous.revision <= next.revision ? next : previous);
      setPlacementChoice(null);
      pendingPlacement.current = null;
    } catch (failure) {
      setError(errorMessage(failure));
      if (failure instanceof ApiError) pendingPlacement.current = null;
    } finally { pending.current = false; setBusy(false); }
  }

  async function normalAction(kind: string, payload: Record<string, unknown> = {}) {
    if (kind === "build_mode") {
      const chosen = payload.kind as BuildMode;
      if (chosen && (!game || manualControlsDisabled || buildUnavailableReason(game, chosen))) return;
      setBuildMode(chosen); setError(""); return;
    }
    if (!session?.token || !game || pending.current) return;
    pending.current = true; setBusy(true); setError("");
    const requestId = createRequestId();
    try {
      let next: GameSetup;
      if (kind === "start") next = await startPlay(session.id, session.token, game.revision, requestId);
      else if (kind === "roll") next = await rollTurnDice(session.id, session.token, game.revision, requestId);
      else if (kind === "discard") next = await discardCards(session.id, session.token, game.revision, requestId, payload.resources as Record<"wood" | "brick" | "sheep" | "wheat" | "ore", number>);
      else if (kind === "robber") next = await moveRobber(session.id, session.token, game.revision, requestId, payload.tile_id as number);
      else if (kind === "steal") next = await stealCard(session.id, session.token, game.revision, requestId, payload.victim_id as number);
      else if (kind === "build") next = await buildPiece(session.id, session.token, game.revision, requestId, payload.kind as "road" | "settlement" | "city" | "development", payload.target_id as number | undefined);
      else if (kind === "bank") next = await bankTrade(session.id, session.token, game.revision, requestId, payload.give_resource as "wood" | "brick" | "sheep" | "wheat" | "ore", payload.receive_resource as "wood" | "brick" | "sheep" | "wheat" | "ore");
      else if (kind === "development") next = await useDevelopment(session.id, session.token, game.revision, requestId, payload.card as "knight" | "road_building" | "year_of_plenty" | "monopoly", payload.resources as ("wood" | "brick" | "sheep" | "wheat" | "ore")[] | undefined);
      else next = await endTurn(session.id, session.token, game.revision, requestId);
      setGame(previous => !previous || previous.revision <= next.revision ? next : previous);
      if (["build", "end", "development", "robber"].includes(kind)) setBuildMode(null);
      if (next.phase === "robber_move" || kind === "robber") setSelectedTile(null);
    } catch (failure) { setError(errorMessage(failure)); }
    finally { pending.current = false; setBusy(false); }
  }

  async function submitTrade(targetId: number, offers: TradeOffer[]) {
    if (!session?.token || !game || pending.current) return;
    pending.current = true; setBusy(true); setError("");
    try { const next = await proposeTrade(session.id, session.token, game.revision, createRequestId(), targetId, offers); setGame(previous => !previous || previous.revision <= next.revision ? next : previous); }
    catch (failure) { setError(errorMessage(failure)); } finally { pending.current = false; setBusy(false); }
  }
  async function answerTrade(decision: "accept" | "decline" | "counter", offerIndex?: number) {
    if (!session?.token || !game || pending.current) return;
    pending.current = true; setBusy(true); setError("");
    try { const next = await respondTrade(session.id, session.token, game.revision, createRequestId(), decision, offerIndex); setGame(previous => !previous || previous.revision <= next.revision ? next : previous); }
    catch (failure) { setError(errorMessage(failure)); } finally { pending.current = false; setBusy(false); }
  }
  async function submitCounterTrade(offers: TradeOffer[]) {
    if (!session?.token || !game || pending.current) return;
    pending.current = true; setBusy(true); setError("");
    try { const next = await counterTrade(session.id, session.token, game.revision, createRequestId(), offers); setGame(previous => !previous || previous.revision <= next.revision ? next : previous); }
    catch (failure) { setError(errorMessage(failure)); } finally { pending.current = false; setBusy(false); }
  }

  async function changeController(playerId: number, isAi: boolean, token?: string) {
    const playerToken = token ?? (viewerId === playerId ? session?.token : undefined);
    if (!session || !game || !playerToken || pending.current) return;
    // 観戦再生中なら、次のAI操作との競合を避けるため先に一時停止する。
    setAiPlaying(false);
    pending.current = true; setBusy(true); setError("");
    try {
      const changed = await setPlayerController(session.id, playerToken, playerId, game.revision, createRequestId(), isAi);
      const next = session.token ? changed : await getGame(session.id);
      setGame(previous => !previous || previous.revision <= next.revision ? next : previous);
      const nextViewer = next.players.find(player => player.id === next.viewer_id);
      // 操作画面からAIへ切り替えた瞬間は、観戦モードでなくても自動再生を開始する。
      aiControllerAuto.current = isAi;
      aiPausedByUser.current = false;
      setAiPlaying(next.ai.can_step && (next.ai.watch_mode || isAi || Boolean(nextViewer?.is_ai)));
    } catch (failure) { setError(errorMessage(failure)); }
    finally { pending.current = false; setBusy(false); }
  }

  return <main className={`table-app ${game ? "in-game" : ""} ${game?.ai.watch_mode ? "watch-mode" : ""} ${boardExpanded ? "board-expanded" : ""}`}>
    <header className="topbar">
      <a className="brand" href="/"><span className="brand-symbol">⬡</span> CATAN<span className="brand-divider" /><span className="brand-subtitle">LOCAL TABLE</span></a>
      <div className="topbar-actions">
        <span className="local-badge"><span />{viewerId ? `プレイヤー ${viewerId} の画面` : "全体表示・4人のテーブル"}</span>
        {game && session?.token && viewerPlayer && <button className="header-controller-toggle" type="button" disabled={busy}
          onClick={() => void changeController(viewerPlayer.id, !viewerPlayer.is_ai)}
          aria-label={`${viewerPlayer.name}を${viewerPlayer.is_ai ? "AIから人" : "人からAI"}の操作に切り替える`}>
          {viewerPlayer.is_ai ? "AI→人" : "人→AI"}
        </button>}
      </div>
    </header>
    <section className="page-heading">
      <div><p className="eyebrow">ONE ISLAND, FOUR EXPLORERS</p><h1>{!game ? "この島から、はじめよう。" : game.phase === "game_over" ? "島の覇者が、決まった。" : game.phase === "ready_to_play" ? "8つの開拓地から、はじまる。" : game.phase === "setup_ready" ? "島に、最初の一歩を。" : game.phase === "rolling_order" ? "サイコロで、席を決めよう。" : "島を、ひろげよう。"}</h1>
        <p className="intro">{game?.ai.watch_mode ? "観戦モード：どの視点でも4人全員の手札を公開しています。" : viewerId ? "あなたは6時の席。自分の手札だけが見える画面です。" : "12時から時計回りに着席。各プレイヤー用のタブで操作します。"}</p></div>
      {session && <a className="quiet-link" href="/">新しい対局を作る</a>}
    </section>

    <div className="game-layout">
      <section className="board-section" aria-label="4人のテーブル">
        <div className="board-heading">
          <div className="board-title-tools"><h2>カタン島</h2>{game && <><DiceHistory game={game} /><ActivityHistory game={game} /></>}</div>
          <span className="board-status"><i className={game ? "ready" : ""} />{game?.phase === "game_over" ? `ゲーム終了・P${game.winner_id}勝利` : game?.phase === "ready_to_play" ? "初期配置完了" : game?.phase === "setup_ready" ? `初期配置 ${game.initial_placement.completed_pairs}/8組` : game?.phase === "rolling_order" ? `順番決め・第${game.round_number}ラウンド` : game ? `第${game.turn_number}手番` : session ? "接続中…" : "スタート待ち"}</span>
        </div>
        <div className="island-stage">
          <PlayerRoster players={players} order={seatOrder} rankingPending={!game || game.phase === "rolling_order"} viewerId={viewerId}
            turnPlayerId={!game || game.phase === "game_over" ? null : game.phase === "rolling_order" ? game.next_roller_id : game.phase === "setup_ready" ? game.initial_placement.player_id : game.phase === "ready_to_play" ? game.first_player_id : game.current_player_id} />
          <div className={`board-wrap ${busy && !game ? "generating" : ""}`} aria-busy={busy && !game}>
            {game && <div className="board-corner-controls">
              <button className="board-zoom-toggle" type="button" aria-pressed={boardExpanded}
                aria-label={boardExpanded ? "盤面を縮小表示に戻す" : "盤面を拡大表示する"}
                onClick={() => setBoardExpanded(expanded => !expanded)}>
                <span aria-hidden="true">{boardExpanded ? "⊖" : "⊕"}</span>{boardExpanded ? "縮小" : "拡大"}
              </button>
            </div>}
            <Board board={game?.board ?? null} selectedId={selectedTile?.id ?? null} onSelect={setSelectedTile}
              game={game} selection={placementSelection} placementDisabled={manualControlsDisabled || !!connectionError || !canPlace}
              onSelectPlacement={choice => { setPlacementChoice(choice); setError(""); }} buildMode={activeBuildMode}
              buildDisabled={manualControlsDisabled || !!connectionError || game?.phase !== "action" || viewerId !== game.current_player_id}
              onSelectBuild={(kind, targetId) => void normalAction("build", { kind, target_id: targetId })} />
            {game && <DiceCutIn key={`dice-${game.id}`} game={game} onPlayingChange={setDiceAnimating} />}
            {game && <DevelopmentCutIn key={`development-${game.id}`} game={game} onPlayingChange={setDevelopmentAnimating} />}
            {game && <BuildCutIn key={`build-${game.id}`} game={game} onPlayingChange={setBuildAnimating} />}
            {game && <CommerceCutIn key={`commerce-${game.id}`} game={game} onPlayingChange={setCommerceAnimating} />}
          </div>
        </div>
        {viewerId !== null && (() => {
          const index = seatOrder.indexOf(viewerId);
          const player = players.find(p => p.id === viewerId)!;
          return <div className="own-hand-column" aria-label="自分の手札">
            <PlayerSeat player={player} position="south" isMe revealHand={Boolean(game?.ai.watch_mode)}
              handOnly
              active={(game?.phase === "setup_ready" ? game.initial_placement.player_id : game?.phase === "rolling_order" ? game?.next_roller_id : game?.phase === "game_over" ? null : game?.current_player_id) === viewerId}
              statusText={game?.phase === "setup_ready" && game.initial_placement.player_id === viewerId ? "初期配置の番" : game?.current_player_id === viewerId && !["rolling_order", "game_over"].includes(game.phase) ? "手番" : undefined}
              rank={game && game.phase !== "rolling_order" ? index + 1 : undefined} />
          </div>;
        })()}
      </section>

      <aside className="sidebar">
        {!session ? <section className="setup-card">
          <p className="eyebrow">NEW EXPEDITION</p><h2>島の準備</h2>
          <fieldset className="rule-options" disabled={busy}><legend>盤面の生成ルール</legend>
            {RULE_OPTIONS.map(option => <label key={option.key}>
              <input type="checkbox" checked={rules[option.key]} onChange={event => setRules(previous => ({ ...previous, [option.key]: event.target.checked }))} />
              <span>{option.label}<small>{option.description}</small></span>
            </label>)}
          </fieldset>
          <fieldset className="rule-options ai-options" disabled={busy}><legend>プレイヤーの操作担当</legend>
            {[1, 2, 3, 4].map(id => <label key={id}><input type="checkbox" checked={aiPlayerIds.includes(id)} onChange={event => setAiPlayerIds(previous => event.target.checked ? [...previous, id].sort() : previous.filter(value => value !== id))} /><span>プレイヤー {id} をコンピューターにする<small>手番・初期配置・交渉を自動で進めます。</small></span></label>)}
            <div className="ai-presets"><button type="button" onClick={() => setAiPlayerIds([2, 3, 4])}>P1だけ人間</button><button type="button" onClick={() => setAiPlayerIds([1, 2, 3, 4])}>4人全員AI</button><button type="button" onClick={() => setAiPlayerIds([])}>4人全員人間</button></div>
            <label className="spectator-toggle"><input type="checkbox" checked={watchAi} disabled={aiPlayerIds.length === 0} onChange={event => setWatchAi(event.target.checked)} /><span>AIの動きを観戦する<small>自動操作を一手ずつ画面に反映し、再生速度も変更できます。</small></span></label>
          </fieldset>
          <p className="muted">港の位置・種類は固定です。</p>
          <button className="start-button" disabled={busy} onClick={startGame}>{busy ? "盤面を生成中…" : "ゲームスタート"}</button>
        </section> : <section className="setup-card action-dock">
          <p className="eyebrow">{game?.phase === "rolling_order" ? "SEATING ROUND" : game?.phase === "setup_ready" ? "FIRST SETTLEMENTS" : "CATAN GAME"}</p><h2>{game?.phase === "ready_to_play" ? "初期配置が完了しました" : game?.phase === "setup_ready" ? "開拓地と道の初期配置" : game?.phase === "rolling_order" ? "サイコロで順番決め" : "通常手番"}</h2>
          {game?.phase === "rolling_order" && <>
            <p className="turn-message" role="status">プレイヤー {game.next_roller_id} の番です</p>
            <p className="muted">2個の合計が大きい順に着席。{game.round_number > 1 ? "同点の人だけ、もう一度振ります。" : "同点の場合は、その人たちだけ振り直します。"}</p>
            {viewerId !== null ? <button className="start-button" onClick={rollDice} disabled={!canRoll || manualControlsDisabled || !!connectionError}>
              {busy ? "サイコロを振っています…" : canRoll ? "サイコロを振る" : "他のプレイヤーの番です"}
            </button> : <p className="observer-note">下のリンクから、各プレイヤー用タブを開いて振ってください。</p>}
          </>}
          {game?.phase === "setup_ready" && <InitialPlacementPanel game={game} selection={placementSelection}
            busy={manualControlsDisabled} disconnected={!!connectionError} onConfirm={confirmPlacement} />}
          {game?.phase === "ready_to_play" && <>
            <p className="turn-message" role="status">全員が開拓地2つ・道2本を配置しました。</p>
            <p className="muted">2つ目の開拓地から開始資源を配りました。全員2点でスタートです。</p>
            {viewerId === game.first_player_id ? <button className="start-button" disabled={manualControlsDisabled} onClick={() => normalAction("start")}>プレイヤー {game.first_player_id} としてゲームを開始</button> : <p className="muted">最初の手番はプレイヤー {game.first_player_id}。本人のタブでゲームを開始します。</p>}
          </>}
          {game && ["turn_pre_roll", "discarding", "robber_move", "robber_steal", "action", "trade_response", "trade_counter_offer", "game_over"].includes(game.phase) && <NormalTurnPanel key={`${game.id}-${game.turn_number}-${game.viewer_id}`} game={game} selectedTile={selectedTile} busy={manualControlsDisabled || !!connectionError} onAction={normalAction} onPropose={submitTrade} onRespond={answerTrade} onCounter={submitCounterTrade} />}
        </section>}

        {(error || connectionError) && <p className="error" role="alert">{error || connectionError}</p>}

        {session && access.length > 0 && !session.token && <section className="access-card">
          <p className="eyebrow">PLAYER TABS</p><h2>4人の操作画面</h2>
          <p className="muted">AI担当を含む全員の個別画面を開けます。観戦再生は切り替え時に自動で一時停止します。</p>
          <div className="player-links">{access.map(item => {
            const player = game?.players.find(candidate => candidate.id === item.player_id) ?? EMPTY_PLAYERS[item.player_id - 1];
            return <div className="player-link-row" key={item.player_id}><a
              className={`player-${EMPTY_PLAYERS[item.player_id - 1].color}`} href={accessLink(session.id, item.token)}
              target="_blank" rel="noopener noreferrer" referrerPolicy="no-referrer">
              <i />プレイヤー {item.player_id}<small>{player.is_ai ? "AIデバッグ" : "人が操作"}</small><span>↗</span>
            </a><button type="button" disabled={busy} onClick={() => void changeController(item.player_id, !player.is_ai, item.token)}>{player.is_ai ? "人に切替" : "AIに切替"}</button></div>;
          })}</div>
          <small>個別リンクには手札を見る権限があります。本人またはデバッグ担当だけが使用してください。</small>
        </section>}
        {game?.ai.watch_mode && <AiPlaybackPanel game={game} playing={aiPlaying} stepping={aiStepping || cutInAnimating} speed={aiSpeed}
          onPlayingChange={playing => { aiPausedByUser.current = !playing; setAiPlaying(playing); }} onSpeedChange={setAiSpeed} onStep={() => { void stepAi(); }} />}
        {game && game.ai.player_ids.length > 0 && !game.ai.watch_mode && <details className="game-settings ai-log"><summary>コンピューターの自動操作記録（{game.ai.actions}操作）</summary><p>AI担当：{game.ai.player_ids.map(id => `P${id}`).join("・")}</p><ol>{game.ai.recent_log.map((line, index) => <li key={`${index}-${line}`}>{line}</li>)}</ol></details>}
        {session?.token && <a className="quiet-link" href={accessLink(session.id)} target="_blank" rel="noopener noreferrer">全体表示を別タブで開く ↗</a>}
        {session && !session.token && access.length === 0 && <p className="muted">観戦用の全体表示です。操作には作成者から受け取ったプレイヤー用リンクが必要です。</p>}

        {selectedTile && selectedInfo && <section className="detail-card" aria-live="polite">
          <p className="eyebrow">LAND DETAILS · TILE {selectedTile.id}</p>
          <h2>{selectedInfo.name}<span className="detail-number">{selectedTile.number ?? "—"}</span></h2>
          <p>資源：{selectedInfo.resource}</p>
          <p className="muted">{selectedTile.number ? `出る確率は ${6 - Math.abs(7 - selectedTile.number)}/36。` : "砂漠は資源を生産しません。"}</p>
        </section>}

        {game && <details className="game-settings"><summary>この対局の生成条件</summary>
          <ul>{RULE_OPTIONS.map(option => <li key={option.key}>{game.rules[option.key] ? "✓" : "−"} {option.label}</li>)}</ul>
          <p>生成番号：<code>{game.seed}</code></p>
        </details>}
        <div className="milestone-note"><span>現在の実装範囲</span><p>通常手番のダイス、生産、7と盗賊、建設、銀行交易、プレイヤー間の交渉・逆交渉、発展カード、10点勝利まで。</p></div>
      </aside>
    </div>
    <footer className="footer"><span>CATAN / 学習プロジェクト</span><span>Pythonを再起動すると対局は終了します</span></footer>
  </main>;
}
