"""Step 3A-12 small Shadow audit; no Gate execution or learning."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from app.agents.heuristic import HeuristicAgent
from app.agents.ppo import PPOAgent
from app.domain.game import GameState

from .action_space import get_action_mask
from .macro_teacher_dataset import OPPONENT_PROFILES, play_teacher_episode
from .settlement_planning_agent import SettlementPlanningAgent
from .strategic_action_surface import (FAMILIES, SURFACE_VERSION,
                                       detect_strategic_action_surface, family_of,
                                       summarize_strategic_surface)


CHAMPION_ID = "ppo_gnn_board_65k_s03_exp_v003"
ROBBER_ID = "ppo_robber_value_73k_s01_exp_v001"


class StrategicSurfaceAuditAgent(SettlementPlanningAgent):
    """Select the identical Champion action once; inspect candidates beforehand."""

    def __init__(self, policy, **planning):
        super().__init__(policy, **planning)
        self.records: list[dict] = []
        self.profile: str = ""
        self.seed: int = -1

    def select_action(self, game: GameState, player_id: int):
        if game.phase != "action":
            return super().select_action(game, player_id)
        rng_before = game.random_source.counter
        revision_before = game.revision
        mask = get_action_mask(game, player_id)
        surface = detect_strategic_action_surface(game, player_id, mask)
        action, _, trace = self._select_action_with_diagnostics(game, player_id)
        if (game.random_source.counter != rng_before
                or game.revision != revision_before):
            raise AssertionError("Shadow audit changed source RNG or revision")
        family = family_of(action)
        record = surface.to_dict()
        record.update({
            "game_seed": self.seed,
            "game_id": game.id,
            "profile": self.profile,
            "seat": game.seat_order.index(player_id) + 1,
            "player_id": player_id,
            "turn": game.turn_number,
            "round": game.round_number,
            "revision": game.revision,
            "rng_counter": rng_before,
            "champion_action_type": type(action).__name__,
            "champion_action_fields": asdict(action),
            "champion_family": family,
            "champion_execution_source": trace.execution_source.value,
            "champion_in_candidate_set": (
                family in FAMILIES and record["candidate_mask"][FAMILIES.index(family)]),
        })
        self.records.append(record)
        return action


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def collect(*, root: Path, games_per_profile: int, seed_start: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    champion = PPOAgent(CHAMPION_ID, device="cpu")
    robber = PPOAgent(ROBBER_ID, device="cpu")
    if champion.settlement_planning_config is None or robber.settlement_planning_config is None:
        raise ValueError("Both saved models need their existing Planner config")
    factories = {
        "rule": HeuristicAgent,
        "champion": lambda: SettlementPlanningAgent(
            champion, **champion.settlement_planning_config),
        "robber_ppo": lambda: SettlementPlanningAgent(
            robber, **robber.settlement_planning_config),
    }
    records: list[dict] = []
    games: list[dict] = []
    for profile_index, profile in enumerate(OPPONENT_PROFILES):
        for index in range(games_per_profile):
            seed = seed_start + profile_index * 1000 + index
            seat = index % 4 + 1
            checkpoint = root / f"game-{profile}-{seed}.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                records.extend(saved["records"])
                games.append(saved["game"])
                continue
            agent = StrategicSurfaceAuditAgent(
                champion, **champion.settlement_planning_config)
            agent.profile, agent.seed = profile, seed
            baseline = play_teacher_episode(
                SettlementPlanningAgent(champion, **champion.settlement_planning_config),
                seed, champion_seat=seat, opponent_profile=profile,
                opponent_factories=factories, collect_reports=False)
            shadow = play_teacher_episode(
                agent, seed, champion_seat=seat, opponent_profile=profile,
                opponent_factories=factories, collect_reports=False)
            parity_keys = ("winner", "final_vp", "final_rank", "final_scores",
                           "rng_counter", "all_action_count")
            parity = (baseline.action_trace == shadow.action_trace
                      and baseline.final_state == shadow.final_state
                      and baseline.final_views == shadow.final_views
                      and all(baseline.game[key] == shadow.game[key]
                              for key in parity_keys))
            if not parity:
                raise AssertionError(f"Champion/Shadow parity failed: {profile}/{seed}")
            game = {
                "profile": profile, "game_seed": seed, "seat": seat,
                "action_count": len(baseline.action_trace),
                "action_phase_decisions": len(agent.records),
                "gate_eligible_decisions": sum(r["gate_eligible"] for r in agent.records),
                "winner": baseline.game["winner"],
                "final_vp": baseline.game["final_vp"],
                "final_rank": baseline.game["final_rank"],
                "rng_counter": baseline.game["rng_counter"],
                "parity": parity,
            }
            _write_json(checkpoint, {"game": game, "records": agent.records})
            records.extend(agent.records)
            games.append(game)
            print(f"{profile} seed={seed} seat={seat} action={len(agent.records)} "
                  f"gate={game['gate_eligible_decisions']}", flush=True)
    summary = summarize_strategic_surface(records, games)
    summary.update({
        "seed_start": seed_start, "games_per_profile": games_per_profile,
        "champion_model_id": CHAMPION_ID, "robber_model_id": ROBBER_ID,
        "profile_seat_games": dict(Counter(
            f"{game['profile']}:{game['seat']}" for game in games)),
    })
    _write_json(root / "summary.json", summary)
    _write_json(root / "games.json", games)
    with (root / "decisions.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="strategic_action_step3a12_20261004")
    parser.add_argument("--games-per-profile", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=2920000)
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."}
            or args.games_per_profile < 1):
        parser.error("Invalid experiment ID or games per profile")
    root = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if (root / "summary.json").exists():
        parser.error("Experiment is already complete; use a new ID")
    summary = collect(root=root, games_per_profile=args.games_per_profile,
                      seed_start=args.seed_start)
    print(json.dumps({key: summary[key] for key in
                      ("game_count", "action_phase_decisions", "gate_eligible_decisions",
                       "parity_games")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
