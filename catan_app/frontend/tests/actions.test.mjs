import { test } from "node:test";
import assert from "node:assert/strict";
import { buildUnavailableReason } from "../src/lib/actions.ts";

function game() {
  return {
    viewer_id: 1, current_player_id: 1, phase: "action",
    players: [{ id: 1, resources: { wood: 1, brick: 1, sheep: 1, wheat: 2, ore: 3 } }],
    action: { legal: { road_ids: [0], settlement_ids: [0], city_ids: [0] }, free_road_remaining: 0, development_remaining: 10 },
  };
}
test("資源不足では建設不可、必要枚数がそろうと建設可能", () => {
  for (const [kind, resource] of [["road", "wood"], ["settlement", "sheep"], ["city", "ore"], ["development", "wheat"]]) {
    const g = game();
    assert.equal(buildUnavailableReason(g, kind), null);
    g.players[0].resources[resource] = 0;
    assert.equal(buildUnavailableReason(g, kind), "資源が不足しています");
  }
});
test("資源があっても候補・山札・手番を満たさない場合は不可", () => {
  const g = game();
  g.action.legal.city_ids = [];
  assert.ok(buildUnavailableReason(g, "city"));
  g.action.development_remaining = 0;
  assert.ok(buildUnavailableReason(g, "development"));
  g.current_player_id = 2;
  assert.ok(buildUnavailableReason(g, "road"));
});
test("街道建設の無料道路は資源0でも可能で他の建設は不可", () => {
  const g = game();
  g.players[0].resources = {};
  g.action.free_road_remaining = 2;
  assert.equal(buildUnavailableReason(g, "road"), null);
  assert.ok(buildUnavailableReason(g, "settlement"));
  g.players[0].road_count = 15;
  assert.ok(buildUnavailableReason(g, "road"));
});
