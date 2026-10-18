import assert from "node:assert/strict";
import test from "node:test";
import { lastOwnTradeOffers } from "../src/lib/tradeDraft.ts";

test("reuses the last own offer independently of the next recipient", () => {
  const first = [{ give: { wood: 1 }, want: { ore: 1 } }];
  const later = [{ give: { brick: 2 }, want: { wheat: 1 } }];
  const entries = [
    { kind: "trade", player_id: 1, trade: { recipient_id: 2, offers: first } },
    { kind: "trade", player_id: 3, trade: { recipient_id: 1, offers: [{ give: { ore: 2 }, want: { wood: 1 } }] } },
    { kind: "trade", player_id: 1, trade: { recipient_id: 4, offers: later } },
  ];
  assert.equal(lastOwnTradeOffers(entries, 1), later);
  assert.equal(lastOwnTradeOffers(entries, 3), entries[1].trade.offers);
  assert.equal(lastOwnTradeOffers(entries, null), undefined);
});
