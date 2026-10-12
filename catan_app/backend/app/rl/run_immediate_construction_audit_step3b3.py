"""Read-only Step 3B-3 construction/development boundary audit.

No plan is selected, no controller is changed, and no optimizer is created.
The legal-action list, rather than resource heuristics, defines buildability.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path

import torch

from app.agents.ppo import PPOAgent
from app.domain.actions import (BankTradeAction, BuildCityAction, BuildRoadAction,
                                BuildSettlementAction, BuyDevelopmentAction,
                                EndTurnAction, ProposeTradeAction,
                                UseDevelopmentAction)
from app.domain.game import RESOURCES, legal_actions

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .run_strategic_gate_step3a14 import CHAMPION_ID, ROBBER_ID, _strict_parity
from .strategic_gate_rollout_step3a14 import run_strategic_episode
from .strategic_gate_step3a13 import frozen_snapshot, resolve_immediate_win
from .reward import actual_score
from .runtime_context import runtime_context_definition
from .victory_plan_context import build_victory_plan_context
from .victory_plan_policy import (VictoryPlanShadowHeads,
                                  encode_victory_plan_features)


SCHEMA = "immediate_construction_audit_step3b3_v1"
SEED_BASES = {"rule": 34000000, "champion": 34001000, "mixed": 34002000}
CARD_ORDER = ("knight", "monopoly", "year_of_plenty", "road_building")
CLASSES = ("NONE", "SETTLEMENT_ONLY", "CITY_ONLY", "BOTH")
_CITY_GAIN_FIELD = next(field for field in runtime_context_definition()["field_order"]
                        if field["name"] == "city.best_production_gain")


def _head_digest(heads: VictoryPlanShadowHeads) -> str:
    digest = sha256()
    for name, tensor in sorted(heads.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _action_family(action) -> str:
    for cls, name in ((BuildSettlementAction, "BUILD_SETTLEMENT"),
                      (BuildCityAction, "BUILD_CITY"),
                      (BuildRoadAction, "BUILD_ROAD"),
                      (BuyDevelopmentAction, "BUY_DEVELOPMENT"),
                      (BankTradeAction, "BANK_TRADE"),
                      (ProposeTradeAction, "PLAYER_TRADE"),
                      (UseDevelopmentAction, "PLAY_DEVELOPMENT"),
                      (EndTurnAction, "END_TURN")):
        if isinstance(action, cls):
            return name
    return "OTHER"


def classify_action_phase(game, player_id: int, action, *, legal=None) -> dict | None:
    """Classify a normal learner action boundary from exact legal actions."""
    if game.phase != "action" or game.free_road_remaining or game.current_player_id != player_id:
        return None
    legal = legal_actions(game, player_id) if legal is None else legal
    settlements = [a.vertex_id for a in legal if isinstance(a, BuildSettlementAction)]
    cities = [a.vertex_id for a in legal if isinstance(a, BuildCityAction)]
    build_class = ("BOTH" if settlements and cities else
                   "SETTLEMENT_ONLY" if settlements else
                   "CITY_ONLY" if cities else "NONE")
    usable = {card: any(isinstance(a, UseDevelopmentAction) and a.card == card
                        for a in legal) for card in CARD_ORDER}
    player = game.players[player_id - 1]
    usable_counts = {card: max(0, player.development_cards.count(card)
                              - player.new_development_cards.count(card))
                     for card in CARD_ORDER}
    if game.used_development_this_turn and any(usable.values()):
        raise AssertionError("Development legal after already used this turn")
    if any(usable[card] and usable_counts[card] == 0 for card in CARD_ORDER):
        raise AssertionError("Newly bought or absent Development card is legal")
    if action not in legal and not isinstance(action, ProposeTradeAction):
        raise AssertionError(f"Selected Action is not in normal legal actions: {action!r}")
    resources = {name: player.resources[name] for name in RESOURCES}
    hand_size = sum(resources.values())
    return {
        "build_class": build_class,
        "settlement_vertices": settlements, "city_vertices": cities,
        "action_family": _action_family(action),
        "action": repr(action),
        "played_card": action.card if isinstance(action, UseDevelopmentAction) else None,
        "usable_development": usable,
        "usable_development_counts": usable_counts,
        "development_used_this_turn": game.used_development_this_turn,
        "resource_vector": resources, "hand_size": hand_size,
        "discard_exposure": hand_size > 7,
        "buy_development_legal": any(isinstance(a, BuyDevelopmentAction) for a in legal),
        "bank_trade_legal": any(isinstance(a, BankTradeAction) for a in legal),
    }


def _both_plan_shadow(game, player_id, champion, heads) -> dict:
    context = build_victory_plan_context(game, player_id)
    encoded = encode_victory_plan_features(context)
    snapshot = frozen_snapshot(champion, game, player_id)
    with torch.no_grad():
        output = heads(
            snapshot.actor_latent,
            torch.tensor(snapshot.runtime_context.copy()).unsqueeze(0),
            torch.tensor(encoded.global_features.copy()).unsqueeze(0),
            torch.tensor(encoded.candidate_features.copy()).unsqueeze(0),
            torch.tensor(encoded.candidate_mask.copy()).unsqueeze(0))
    n = len(encoded.candidate_ids)
    return {
        "current_score": context.current_score,
        "remaining_points": context.remaining_points,
        "candidate_ids": [list(item) for item in encoded.candidate_ids],
        "candidate_mask": encoded.candidate_mask[:n].tolist(),
        "plan_probabilities": output.probabilities[0, :n].tolist(),
        "plan_entropy": float(output.entropy.item()),
        "priority_shadow": output.priorities[0, :n].tolist(),
        "plan_available": bool(output.plan_available.item()),
        "city_upgradeable_count": context.city.upgradeable_count,
        "best_city_production_gain_pips": (
            float(snapshot.runtime_context[_CITY_GAIN_FIELD["vector_index"]]
                  * _CITY_GAIN_FIELD["scale"])
            if snapshot.runtime_context[_CITY_GAIN_FIELD["vector_index"] + 1] else None),
        "settlement_connected_count": context.settlement.currently_connected_count,
    }


def audit_observer(env, action, _decision, _trace_index, *, champion, heads,
                   seed: int, profile: str, seat: int, records: list[dict],
                   phase_counts: Counter) -> None:
    game = env.game
    player_id = env.learning_player_id
    phase = "free_road" if game.phase == "action" and game.free_road_remaining else game.phase
    phase_counts[phase] += 1
    if phase not in {"action", "turn_pre_roll"}:
        return
    before = deepcopy(game)
    if phase == "turn_pre_roll":
        legal = legal_actions(game, player_id)
        usable = {card: any(isinstance(a, UseDevelopmentAction) and a.card == card
                            for a in legal) for card in CARD_ORDER}
        row = {"phase": phase, "usable_development": usable,
               "action_family": _action_family(action), "action": repr(action)}
    else:
        legal = legal_actions(game, player_id)
        row = classify_action_phase(game, player_id, action, legal=legal)
        if row is None:
            raise AssertionError("Normal action phase unexpectedly unclassified")
        win = resolve_immediate_win(game, player_id)
        row["immediate_win"] = win.action is not None
        row["immediate_win_action"] = repr(win.action) if win.action else None
        row["immediate_win_family"] = _action_family(win.action) if win.action else None
        row["immediate_win_count"] = len(win.winning_action_ids)
        row["current_score"] = actual_score(game, player_id)
        row["remaining_points"] = max(0, 10 - row["current_score"])
        if row["build_class"] == "BOTH":
            row["plan_shadow"] = _both_plan_shadow(game, player_id, champion, heads)
    if game != before or game.random_source.counter != before.random_source.counter:
        raise AssertionError("Step 3B-3 observer mutated GameState or Game RNG")
    row.update({"seed": seed, "profile": profile, "seat": seat,
                "turn": game.turn_number, "revision": game.revision,
                "policy_step": env.policy_steps, "phase": phase})
    records.append(row)


def _write_gzip(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def _load_records(output: Path) -> tuple[list[dict], list[dict], Counter]:
    records, episodes, phases = [], [], Counter()
    for profile, base in SEED_BASES.items():
        for index in range(100):
            path = output / "episodes" / f"{profile}_{base + index}.json.gz"
            if not path.exists():
                continue
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload["schema"] != SCHEMA or payload["episode"]["seed"] != base + index:
                raise AssertionError("Episode checkpoint schema/seed mismatch")
            episodes.append(payload["episode"])
            records.extend(payload["records"])
            phases.update(payload["phase_counts"])
    return records, episodes, phases


def summarize(records: list[dict], episodes: list[dict], phases: Counter) -> dict:
    if len({(e["profile"], e["seed"]) for e in episodes}) != len(episodes):
        raise AssertionError("Duplicate game in Step 3B-3 evidence")
    if any(not e["parity"] or not e["terminal"] or e["truncated"] for e in episodes):
        raise AssertionError("Episode parity or terminal invariant failed")
    actions = [r for r in records if r["phase"] == "action"]
    regular = [r for r in actions if not r["immediate_win"]]
    buildable = [r for r in regular if r["build_class"] != "NONE"]
    class_counts = Counter(r["build_class"] for r in regular)
    class_counts_all = Counter(r["build_class"] for r in actions)
    behavior = {kind: dict(Counter(r["action_family"] +
                                   (":" + r["played_card"] if r["played_card"] else "")
                                   for r in regular if r["build_class"] == kind))
                for kind in CLASSES}
    dev_overlap = {kind: {
        "states": class_counts[kind],
        "any_usable": sum(any(r["usable_development"].values()) for r in regular
                          if r["build_class"] == kind),
        "by_card": {card: sum(r["usable_development"][card] for r in regular
                              if r["build_class"] == kind) for card in CARD_ORDER},
        "actual_play_by_card": dict(Counter(r["played_card"] for r in regular
                                            if r["build_class"] == kind and r["played_card"])),
    } for kind in CLASSES}
    behavior_by_dev_availability = {kind: {
        label: {"states": len(subset),
                "actions": dict(Counter(r["action_family"] for r in subset))}
        for label, subset in (
            ("usable_dev_absent", [r for r in regular if r["build_class"] == kind
                                   and not any(r["usable_development"].values())]),
            ("usable_dev_present", [r for r in regular if r["build_class"] == kind
                                    and any(r["usable_development"].values())]))}
        for kind in CLASSES}
    examples = {}
    for r in regular:
        key = (r["build_class"], r["action_family"], r["played_card"])
        if key not in examples:
            examples[key] = {name: r.get(name) for name in
                             ("seed", "profile", "seat", "turn", "revision", "policy_step",
                              "build_class", "action", "current_score", "remaining_points",
                              "resource_vector", "usable_development", "settlement_vertices",
                              "city_vertices", "plan_shadow")}
    summary = {
        "schema": SCHEMA, "seed_bases": SEED_BASES, "games": len(episodes),
        "profile_seat": dict(Counter(f"{e['profile']}/seat{e['seat']}" for e in episodes)),
        "learner_phase_counts": dict(phases), "normal_action_states": len(actions),
        "regular_action_states": len(regular),
        "immediate_win_excluded": len(actions) - len(regular),
        "immediate_win_by_family": dict(Counter(r["immediate_win_family"] for r in actions
                                                 if r["immediate_win"])),
        "immediate_win_by_class": dict(Counter(r["build_class"] for r in actions
                                                if r["immediate_win"])),
        "class_counts_all": dict(class_counts_all),
        "class_counts": dict(class_counts), "behavior": behavior,
        "class_by_profile": {profile: dict(Counter(r["build_class"] for r in regular
                                                  if r["profile"] == profile))
                             for profile in SEED_BASES},
        "behavior_by_profile_and_class": {profile: {
            kind: dict(Counter(r["action_family"] for r in regular
                               if r["profile"] == profile and r["build_class"] == kind))
            for kind in CLASSES} for profile in SEED_BASES},
        "development_overlap": dev_overlap,
        "behavior_by_dev_availability": behavior_by_dev_availability,
        "buildable_action_family": dict(Counter(r["action_family"] for r in buildable)),
        "buildable_actual_dev_by_card": dict(Counter(r["played_card"] for r in buildable
                                                     if r["played_card"])),
        "pre_roll_usable_by_card": {card: sum(r["usable_development"][card] for r in records
                                              if r["phase"] == "turn_pre_roll") for card in CARD_ORDER},
        "examples": {"/".join(str(x) for x in key): value for key, value in examples.items()},
        "episodes": episodes,
        "mutation_failures": 0, "rng_failures": 0, "parity_failures": 0,
        "optimizer_created": False,
    }
    return summary


def collect(output: Path, *, profile: str | None = None, limit: int = 100,
            start_index: int = 0, refresh_missing_city_gain: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)
    champion = PPOAgent(CHAMPION_ID, device="cpu")
    robber = PPOAgent(ROBBER_ID, device="cpu")
    champion.model.policy.requires_grad_(False)
    latent_dim = champion.model.policy.mlp_extractor.latent_dim_pi
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3302)
        heads = VictoryPlanShadowHeads(latent_dim)
    heads.eval()
    before = {"champion": _weight_digest(champion),
              "robber": _weight_digest(robber), "plan_priority_head": _head_digest(heads)}
    for name in ([profile] if profile else SEED_BASES):
        if name not in SEED_BASES:
            raise ValueError(f"Unknown profile: {name}")
        for index in range(start_index, min(limit, 100)):
            seed = SEED_BASES[name] + index
            path = output / "episodes" / f"{name}_{seed}.json.gz"
            if path.exists():
                if not refresh_missing_city_gain:
                    continue
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    previous = json.load(handle)
                if not any(row.get("build_class") == "BOTH" and
                           "best_city_production_gain_pips" not in row.get("plan_shadow", {})
                           for row in previous["records"]):
                    continue
            seat = index % 4 + 1
            baseline = run_strategic_episode(champion, robber, seed=seed, seat=seat,
                                             profile=name, mode="legacy")
            records: list[dict] = []
            phases: Counter = Counter()
            observer = lambda *args: audit_observer(
                *args, champion=champion, heads=heads, seed=seed, profile=name,
                seat=seat, records=records, phase_counts=phases)
            shadow = run_strategic_episode(champion, robber, seed=seed, seat=seat,
                                           profile=name, mode="legacy",
                                           pre_action_audit_observer=observer)
            if not _strict_parity(baseline, shadow) or not baseline.terminal or baseline.truncated:
                raise AssertionError(f"Behavior parity/terminal failed: {name}/{seed}/{seat}")
            episode = {"seed": seed, "profile": name, "seat": seat,
                       "winner": baseline.winner, "final_vp": baseline.final_vp,
                       "final_rank": baseline.final_rank, "policy_steps": baseline.policy_steps,
                       "rng_counter": baseline.rng_counter, "parity": True,
                       "terminal": baseline.terminal, "truncated": baseline.truncated}
            _write_gzip(path, {"schema": SCHEMA, "episode": episode,
                               "records": records, "phase_counts": dict(phases)})
            print(f"{name} {index + 1}/100 seed={seed} seat={seat} "
                  f"action_states={sum(r['phase'] == 'action' for r in records)} parity=ok",
                  flush=True)
    after = {"champion": _weight_digest(champion),
             "robber": _weight_digest(robber), "plan_priority_head": _head_digest(heads)}
    if before != after:
        raise AssertionError("Frozen/Shadow head weights changed")
    records, episodes, phases = _load_records(output)
    summary = summarize(records, episodes, phases)
    summary["weight_digests_before"] = before
    summary["weight_digests_after"] = after
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
        indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(SEED_BASES))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--refresh-missing-city-gain", action="store_true")
    args = parser.parse_args()
    result = collect(args.output, profile=args.profile, limit=args.limit,
                     start_index=args.start_index,
                     refresh_missing_city_gain=args.refresh_missing_city_gain)
    print(json.dumps({k: result[k] for k in
        ("games", "normal_action_states", "class_counts", "immediate_win_excluded")},
        ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
