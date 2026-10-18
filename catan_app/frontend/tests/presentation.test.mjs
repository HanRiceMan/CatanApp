import assert from "node:assert/strict";
import test from "node:test";
import { presentationKinds } from "../src/lib/presentation.ts";

const shown = {
  id: "game-1", revision: 10, latest_dice_event: null,
  activity_log: [{ sequence: 10, kind: "turn_end" }],
};

test("dice result holds the old board and hand until its cut-in finishes", () => {
  const next = { ...shown, revision: 11, latest_dice_event: { sequence: 11, kind: "turn", dice: [3, 4], total: 7 } };
  assert.deepEqual(presentationKinds(shown, next), ["dice"]);
  assert.deepEqual(presentationKinds(next, next), []);
});

test("build, development, discard and hidden-resource steal are presentation boundaries", () => {
  for (const [kind, expected] of [["build", "build"], ["development", "development"],
    ["discard", "incident"], ["steal", "incident"]]) {
    assert.deepEqual(presentationKinds(shown, { ...shown, revision: 11,
      activity_log: [...shown.activity_log, { sequence: 11, kind }] }), [expected]);
  }
});

test("trade proposals remain interactive, but their results retain the longer trade cut-in", () => {
  const proposal = { ...shown, revision: 11,
    activity_log: [...shown.activity_log, { sequence: 11, kind: "trade" }] };
  assert.deepEqual(presentationKinds(shown, proposal), []);
  const result = { ...proposal, revision: 12,
    activity_log: [...proposal.activity_log, { sequence: 12, kind: "trade_accepted" }] };
  assert.deepEqual(presentationKinds(proposal, result), ["commerce"]);
});

test("stale and unrelated games never hold presentation", () => {
  assert.deepEqual(presentationKinds(shown, { ...shown, id: "game-2", revision: 11 }), []);
  assert.deepEqual(presentationKinds(shown, { ...shown, revision: 9 }), []);
});
