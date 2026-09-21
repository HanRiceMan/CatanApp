export type Terrain = "wood" | "brick" | "sheep" | "wheat" | "ore" | "desert";
export type Tile = { id: number; q: number; r: number; terrain: Terrain; number: number | null };
export type Vertex = { id: number; x: number; y: number; hex_ids: number[]; edge_ids: number[] };
export type Edge = { id: number; vertex_ids: [number, number]; hex_ids: number[] };
export type Port = { id: number; edge_id: number; resource: Terrain | null; ratio: number };
export type BoardData = {
  tiles: Tile[];
  vertices: Vertex[];
  edges: Edge[];
  ports: Port[];
  robber_tile_id: number;
};
export type Resource = Exclude<Terrain, "desert">;
export type PlayerColor = "red" | "cyan" | "purple" | "yellow";
export type BoardRules = {
  desert_center: boolean;
  red_numbers_not_adjacent: boolean;
  red_numbers_distinct_terrains: boolean;
  terrain_clusters_max_two: boolean;
};
export const DEFAULT_RULES: BoardRules = {
  desert_center: false, red_numbers_not_adjacent: true,
  red_numbers_distinct_terrains: false, terrain_clusters_max_two: false,
};
export type Player = {
  id: number; name: string; color: PlayerColor; is_ai: boolean;
  resource_count: number; development_count: number; public_score: number;
  played_knights: number; has_longest_road: boolean; has_largest_army: boolean;
  settlement_count: number; city_count: number; road_count: number;
  used_development_cards?: string[];
  pending_action?: string | null;
  // 資源・発展カードの内訳は通常本人だけ、AI観戦では全員分。score は常に本人だけ。
  resources?: Record<Resource, number>; development_cards?: string[]; new_development_cards?: string[]; score?: number;
};
export type OrderRoll = { player_id: number; round: number; dice: [number, number]; total: number };
export type ActivityEntry = {
  sequence: number; turn_number: number; player_id: number; kind: string; text: string; card?: string;
  trade?: {
    proposer_id: number; recipient_id: number; is_counter: boolean;
    outcome: "proposed" | "accepted" | "declined" | "counter_requested";
    offers: { give: Partial<Record<Resource, number>>; want: Partial<Record<Resource, number>> }[];
    selected_offer_index?: number;
  };
  bank?: { give: Partial<Record<Resource, number>>; want: Partial<Record<Resource, number>> };
};
export type PieceKind = "settlement" | "road";
export type PlacementSelection = { kind: PieceKind; targetId: number; revision: number };
export type GameSetup = {
  id: string;
  seed: number;
  phase: "rolling_order" | "setup_ready" | "ready_to_play" | "turn_pre_roll" | "discarding" | "robber_move" | "robber_steal" | "action" | "trade_response" | "trade_counter_offer" | "game_over";
  revision: number;
  rules: BoardRules;
  board: BoardData;
  players: Player[];
  setup_order: number[];
  seat_order: number[];
  viewer_id: number | null;
  round_number: number;
  ranking_groups: number[][];
  next_roller_id: number | null;
  rolls: OrderRoll[];
  longest_road_holder_id: number | null;
  largest_army_holder_id: number | null;
  settlements: { vertex_id: number; player_id: number }[];
  cities: { vertex_id: number; player_id: number }[];
  roads: { edge_id: number; player_id: number }[];
  bank: Record<Resource, number>;
  robber_tile_id: number;
  first_player_id: number | null;
  current_player_id: number | null;
  turn_number: number;
  last_roll: { player_id: number; dice: [number, number]; total: number; turn_number: number } | null;
  dice_history: { player_id: number; dice: [number, number]; total: number; turn_number: number }[];
  latest_dice_event?: { sequence: number; kind: "order" | "turn"; player_id: number; dice: [number, number]; total: number; turn_number?: number } | null;
  activity_log?: ActivityEntry[];
  winner_id: number | null;
  trade_count: number;
  trade_limit: number;
  pending_discard_count: number;
  your_discard_count: number;
  robber: { legal_tile_ids: number[]; victim_ids: number[] };
  pending_trade: {
    active: boolean; proposer_id: number; recipient_id: number; is_counter: boolean; awaiting_counter: boolean;
    offers?: { give: Record<Resource, number>; want: Record<Resource, number> }[];
  } | null;
  action: {
    legal: { road_ids: number[]; settlement_ids: number[]; city_ids: number[] };
    free_road_remaining: number; development_remaining: number; used_development_this_turn: boolean;
    port_ratios: Partial<Record<Resource, number>>;
  };
  ai: { player_ids: number[]; actions: number; recent_log: string[]; watch_mode: boolean; actor_id: number | null; can_step: boolean };
  initial_placement: {
    completed_pairs: number; total_pairs: number; player_id: number | null;
    piece: PieceKind | null; pending_vertex_id: number | null;
    legal_vertex_ids: number[]; legal_edge_ids: number[];
  };
};
export type PlayerAccess = { player_id: number; token: string };
export type CreatedGame = { game: GameSetup; player_access: PlayerAccess[]; ai_result: { status: string; steps: number; phase: string; winner_id: number | null } };

export const TERRAIN_INFO: Record<Terrain, { name: string; resource: string; color: string; ink: string }> = {
  wood: { name: "森林", resource: "木材", color: "#527a59", ink: "#eef5da" },
  brick: { name: "丘陵", resource: "レンガ", color: "#895837", ink: "#fff0de" },
  sheep: { name: "牧草地", resource: "羊毛", color: "#a2b66c", ink: "#293e30" },
  wheat: { name: "畑", resource: "小麦", color: "#dbb755", ink: "#624820" },
  ore: { name: "山地", resource: "鉱石", color: "#85979d", ink: "#f4f4e6" },
  desert: { name: "砂漠", resource: "資源なし", color: "#d3bc8f", ink: "#665033" },
};
