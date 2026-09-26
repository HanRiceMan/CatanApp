"""同じ盤面でAgentの初期配置だけを再生し、選択の質を診断する。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import fmean

from app.agents.base import Agent
from app.agents.ppo import PPOAgent
from app.domain.actions import Action, PlaceInitialRoadAction, PlaceInitialSettlementAction
from app.domain.board import Board
from app.domain.game import GameState, legal_road_ids, legal_settlement_ids

from .action_space import action_to_id
from .audit import model_audit, runtime_versions, source_fingerprint
from .compare import PolicyCompatibleHeuristicAgent
from .env import CatanEnv, CatanEnvConfig
from .model_registry import ModelRegistry


NUMBER_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
RESOURCES = ("wood", "brick", "sheep", "wheat", "ore")


def vertex_production(board: Board, vertex_id: int) -> dict[str, int]:
    """交差点に隣接する数字の出やすさを資源別に集計する。"""
    production = dict.fromkeys(RESOURCES, 0)
    for tile_id in board.vertices[vertex_id].hex_ids:
        tile = board.tiles[tile_id]
        if tile.terrain in production:
            production[tile.terrain] += NUMBER_PIPS.get(tile.number, 0)
    return production


def vertex_pips(board: Board, vertex_id: int) -> int:
    return sum(vertex_production(board, vertex_id).values())


def road_extension_pips(game: GameState, settlement_id: int, edge_id: int) -> int:
    """その道路からあと1本道を伸ばした先の、空いた開拓地候補の最大pips。

    将来の資源・競合・建築費は予測しない。初期道路の向きの幾何学的指標。
    """
    edge = game.board.edges[edge_id]
    if settlement_id not in edge.vertex_ids:
        raise ValueError("初期道路は選んだ開拓地に接している必要があります。")
    far_id = next(vertex for vertex in edge.vertex_ids if vertex != settlement_id)
    candidate_pips: list[int] = []
    for next_edge_id in game.board.vertices[far_id].edge_ids:
        if next_edge_id == edge_id or next_edge_id in game.roads:
            continue
        other = next(vertex for vertex in game.board.edges[next_edge_id].vertex_ids if vertex != far_id)
        if other in game.settlements or other in game.cities:
            continue
        neighbors = {neighbor for adjacent_edge in game.board.vertices[other].edge_ids
                     for neighbor in game.board.edges[adjacent_edge].vertex_ids if neighbor != other}
        if neighbors.intersection(game.settlements) or neighbors.intersection(game.cities):
            continue
        candidate_pips.append(vertex_pips(game.board, other))
    return max(candidate_pips, default=0)


def _port_access(board: Board, vertex_ids: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(sorted({port.resource or "generic" for port in board.ports
                         if set(board.edges[port.edge_id].vertex_ids).intersection(vertex_ids)}))


class GreedyPipsSettlementAgent:
    """評価専用。初期開拓地だけpips最大、道路と通常ターンは既存Heuristic。"""

    def __init__(self) -> None:
        self._fallback = PolicyCompatibleHeuristicAgent()

    def select_action(self, game: GameState, player_id: int) -> Action:
        if game.phase == "setup_ready" and game.pending_settlement_id is None:
            candidates = legal_settlement_ids(game)
            vertex_id = min(candidates, key=lambda candidate: (-vertex_pips(game.board, candidate), candidate))
            return PlaceInitialSettlementAction(vertex_id)
        return self._fallback.select_action(game, player_id)


class GreedyPipsSetupAgent(GreedyPipsSettlementAgent):
    """評価専用。初期開拓地・道路とも単純なpips最大、通常ターンは既存Heuristic。"""

    def select_action(self, game: GameState, player_id: int) -> Action:
        if game.phase != "setup_ready" or game.pending_settlement_id is None:
            return super().select_action(game, player_id)
        settlement_id = game.pending_settlement_id
        candidates = legal_road_ids(game)
        edge_id = min(candidates, key=lambda candidate: (-road_extension_pips(game, settlement_id, candidate),
                                                         candidate))
        return PlaceInitialRoadAction(edge_id)


def probe_setup(agent: Agent, seed: int, index: int, *, observation_version: str = "v1") -> dict:
    """通常ターンに入る前まで再生。学習済みモデルも状態変更はしない。"""
    learner_id = index % 4 + 1
    env = CatanEnv(CatanEnvConfig(learning_player_id=learner_id,
                                 policy_compatible_opponents=True,
                                 observation_version=observation_version))
    settlements: list[int] = []
    roads: list[int] = []
    settlement_choices: list[dict] = []
    road_choices: list[dict] = []
    try:
        _, info = env.reset(seed=seed)
        for _ in range(24):
            game = env.game
            if game is None:
                raise RuntimeError("ゲームが初期化されていません。")
            action = agent.select_action(game, learner_id)
            action_id = action_to_id(action)
            if not info["action_mask"][action_id]:
                raise AssertionError(f"seed {seed}: Agentが合法手以外を選びました。")
            if isinstance(action, PlaceInitialSettlementAction):
                candidates = legal_settlement_ids(game)
                pips = vertex_pips(game.board, action.vertex_id)
                settlement_choices.append({"selected_pips": pips,
                                           "best_legal_pips": max(vertex_pips(game.board, v) for v in candidates),
                                           "legal_candidates": len(candidates)})
                settlements.append(action.vertex_id)
            elif isinstance(action, PlaceInitialRoadAction):
                settlement_id = game.pending_settlement_id
                if settlement_id is None:
                    raise AssertionError("初期道路に対応する開拓地がありません。")
                candidates = legal_road_ids(game)
                potential = road_extension_pips(game, settlement_id, action.edge_id)
                road_choices.append({"selected_extension_pips": potential,
                                     "best_legal_extension_pips": max(road_extension_pips(game, settlement_id, e)
                                                                       for e in candidates),
                                     "legal_candidates": len(candidates)})
                roads.append(action.edge_id)
            _, _, terminated, truncated, info = env.step(action_id)
            if len(settlements) == len(roads) == 2:
                break
            if terminated or truncated:
                raise AssertionError("初期配置完了前に対局が終了しました。")
        else:
            raise AssertionError("初期配置が所定の操作数で終わりませんでした。")
        game = env.game
        assert game is not None
        production = dict.fromkeys(RESOURCES, 0)
        for vertex_id in settlements:
            for resource, pips in vertex_production(game.board, vertex_id).items():
                production[resource] += pips
        first_resources = {name for name, pips in vertex_production(game.board, settlements[0]).items() if pips}
        second_resources = {name for name, pips in vertex_production(game.board, settlements[1]).items() if pips}
        return {"seed": seed, "learner_id": learner_id,
                "seat": game.seat_order.index(learner_id) + 1,
                "settlement_ids": settlements, "road_ids": roads,
                "settlement_choices": settlement_choices, "road_choices": road_choices,
                "first_pips": settlement_choices[0]["selected_pips"],
                "second_pips": settlement_choices[1]["selected_pips"],
                "total_pips": sum(production.values()),
                "resource_pips": production,
                "resource_kinds": sum(pips > 0 for pips in production.values()),
                "second_new_resource_kinds": len(second_resources - first_resources),
                "missing_road_resources": sum(production[name] == 0 for name in ("wood", "brick")),
                "ports": _port_access(game.board, tuple(settlements)),
                "road_extension_pips_after_setup": [road_extension_pips(game, v, e)
                                                     for v, e in zip(settlements, roads)]}
    finally:
        env.close()


def summarize(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        return fmean(row[key] for row in rows)

    settlement_counts = [Counter(row["settlement_ids"][i] for row in rows) for i in range(2)]

    return {"games": len(rows),
            "first_pips": mean("first_pips"), "second_pips": mean("second_pips"),
            "total_pips": mean("total_pips"), "resource_kinds": mean("resource_kinds"),
            "second_new_resource_kinds": mean("second_new_resource_kinds"),
            "missing_road_resources": mean("missing_road_resources"),
            "resource_pips": {resource: fmean(row["resource_pips"][resource] for row in rows)
                              for resource in RESOURCES},
            "settlement_pips_regret": [fmean(row["settlement_choices"][i]["best_legal_pips"]
                                              - row["settlement_choices"][i]["selected_pips"] for row in rows)
                                       for i in range(2)],
            "settlement_best_pips_rate": [fmean(row["settlement_choices"][i]["best_legal_pips"]
                                                == row["settlement_choices"][i]["selected_pips"] for row in rows)
                                          for i in range(2)],
            "settlement_unique_vertices": [len(counts) for counts in settlement_counts],
            "settlement_most_common": [{"vertex_id": counts.most_common(1)[0][0],
                                        "share": counts.most_common(1)[0][1] / len(rows)}
                                       for counts in settlement_counts],
            "road_extension_regret": [fmean(row["road_choices"][i]["best_legal_extension_pips"]
                                             - row["road_choices"][i]["selected_extension_pips"] for row in rows)
                                      for i in range(2)],
            "road_extension_after_setup": [fmean(row["road_extension_pips_after_setup"][i] for row in rows)
                                           for i in range(2)],
            "any_port_rate": fmean(bool(row["ports"]) for row in rows),
            "no_wood_rate": fmean(row["resource_pips"]["wood"] == 0 for row in rows),
            "no_brick_rate": fmean(row["resource_pips"]["brick"] == 0 for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=26001)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    if args.games <= 0:
        parser.error("gamesは正の整数です。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    if output.exists():
        parser.error("実験フォルダが既にあります。別IDを指定してください。")
    registry = ModelRegistry()
    agents: dict[str, Agent] = {"heuristic": PolicyCompatibleHeuristicAgent(),
                                "greedy_settlement": GreedyPipsSettlementAgent(),
                                "greedy_pips": GreedyPipsSetupAgent()}
    agent_versions = {name: "v1" for name in agents}
    audits = {}
    for model_id in args.models:
        agent = PPOAgent(model_id, models_root=registry.models_root, device="cpu")
        if agent.heuristic_initial_placement:
            parser.error("この診断はPPOが自分で初期配置を選ぶモデルだけを指定してください。")
        agents[model_id] = agent
        agent_versions[model_id] = agent.manifest.observation_version
        audits[model_id] = model_audit(model_id, agent.model, registry)
    output.mkdir(parents=True)
    definition = {"experiment_id": args.experiment_id, "seeds": list(range(args.seed_start, args.seed_start + args.games)),
                  "models": args.models, "source_sha256": source_fingerprint(),
                  "runtime_versions": runtime_versions(),
                  "model_sha256": {name: data["model_sha256"] for name, data in audits.items()},
                  "opponents": "heuristic_x3", "rotate_player_ids": True,
                  "policy_compatible_opponents": True, "setup_only": True}
    (output / "definition.json").write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = {}
    for name, agent in agents.items():
        rows = [probe_setup(agent, seed, index, observation_version=agent_versions[name])
                for index, seed in enumerate(definition["seeds"])]
        result = {"agent": name, "summary": summarize(rows), "episodes": rows}
        (output / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries[name] = result["summary"]
        print(json.dumps({"agent": name, **result["summary"]}, ensure_ascii=False), flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
