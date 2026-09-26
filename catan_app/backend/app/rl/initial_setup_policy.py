"""通常手番のPPOを変更せず、初期配置だけを学習する小さな方策。"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from app.agents.heuristic import initial_road_value, initial_vertex_value
from app.domain.actions import PlaceInitialRoadAction, PlaceInitialSettlementAction
from app.domain.board import create_topology

from .action_space import action_to_id
from .candidate_policy import (EDGE_START, PORT_START, PROGRESS_START, TILE_START,
                               VERTEX_START, CandidateMaskablePolicy)
from .compare import PolicyCompatibleHeuristicAgent
from .env import CatanEnv, CatanEnvConfig
from .pretrain_candidate_setup import (INITIAL_ROAD_START, INITIAL_START,
                                       _collect_full_setup_examples)


class InitialSetupPolicy(nn.Module):
    """PPOの候補表現を複製し、初期開拓地・道路だけを採点する。

    PPO本体とはパラメータを共有しないため、この方策の教師あり学習で
    通常ターンのAction確率やValueは変化しない。
    """

    def __init__(self, source_policy: CandidateMaskablePolicy):
        super().__init__()
        source = source_policy.mlp_extractor
        self.context = deepcopy(source.context)
        self.vertex_encoder = deepcopy(source.vertex_encoder)
        self.edge_encoder = deepcopy(source.edge_encoder)
        self.settlement_head = deepcopy(source.initial_settlement_head)
        self.road_head = deepcopy(source.initial_road_head)
        self.register_buffer("edge_vertices", source.edge_vertices.detach().clone())
        topology = create_topology()
        future_ids = []
        future_masks = []
        for edge in topology.edges:
            orientations = []
            orientation_masks = []
            a, b = edge.vertex_ids
            for far in (b, a):  # settlementがa側、b側にある場合の順。
                reachable = []
                for onward_edge in topology.vertices[far].edge_ids:
                    if onward_edge == edge.id:
                        continue
                    reachable.extend(vertex for vertex in topology.edges[onward_edge].vertex_ids
                                     if vertex != far)
                reachable = reachable[:2]
                orientations.append(reachable + [0] * (2 - len(reachable)))
                orientation_masks.append([True] * len(reachable) + [False] * (2 - len(reachable)))
            future_ids.append(orientations)
            future_masks.append(orientation_masks)
        self.register_buffer("future_vertex_ids", torch.tensor(future_ids, dtype=torch.long))
        self.register_buffer("future_vertex_masks", torch.tensor(future_masks, dtype=torch.bool))
        self.road_future_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.road_future_head[-1].weight)
        nn.init.zeros_(self.road_future_head[-1].bias)

    def forward(self, observations: torch.Tensor, kind: str) -> torch.Tensor:
        batch = observations.shape[0]
        vertices = observations[:, VERTEX_START:EDGE_START].reshape(batch, 54, 14)
        edges = observations[:, EDGE_START:PORT_START].reshape(batch, 72, 5)
        own_production = (vertices[:, :, 1:2] * vertices[:, :, 9:14]).sum(dim=1).clamp(0, 1)
        global_values = torch.cat((observations[:, :TILE_START],
                                   observations[:, PROGRESS_START:], own_production), dim=1)
        context = self.context(global_values)
        vertex_state = self.vertex_encoder(torch.cat(
            (vertices, context[:, None, :].expand(-1, 54, -1)), dim=2))
        if kind == "settlement":
            return self.settlement_head(vertex_state).squeeze(-1)
        if kind == "road":
            endpoints = vertices[:, self.edge_vertices]
            endpoint_features = torch.cat((endpoints.mean(dim=2), endpoints.amax(dim=2)), dim=2)
            state = self.edge_encoder(torch.cat(
                (edges, endpoint_features, context[:, None, :].expand(-1, 72, -1)), dim=2))
            future = vertex_state[:, self.future_vertex_ids]
            future_mask = self.future_vertex_masks[None, :, :, :, None]
            future = future.masked_fill(~future_mask, -1e9).amax(dim=3)
            has_future = self.future_vertex_masks.any(dim=2)[None, :, :, None]
            future = torch.where(has_future, future, torch.zeros_like(future))
            orientation_weights = endpoints[:, :, :, 1:2]
            denominator = orientation_weights.sum(dim=2).clamp_min(1.0)
            selected_future = (future * orientation_weights).sum(dim=2) / denominator
            return (self.road_head(state).squeeze(-1)
                    + self.road_future_head(torch.cat((state, selected_future), dim=2)).squeeze(-1))
        raise ValueError(f"未対応の初期配置種別です: {kind}")

    def select_action_id(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """初期配置maskから合法な開拓地または道路を1つ返す。"""
        device = next(self.parameters()).device
        for kind, start, count in (("settlement", INITIAL_START, 54),
                                   ("road", INITIAL_ROAD_START, 72)):
            local_mask = np.asarray(action_mask[start:start + count], dtype=np.bool_)
            if local_mask.any():
                with torch.no_grad():
                    logits = self(torch.from_numpy(observation[None, :]).to(device), kind)[0]
                    mask_tensor = torch.from_numpy(local_mask).to(device)
                    selected = int(logits.masked_fill(~mask_tensor, -1e9).argmax().item())
                return start + selected
        raise ValueError("初期配置の合法Actionがありません。")


def _masked_logits(model: InitialSetupPolicy, examples: dict[str, np.ndarray],
                   kind: str) -> torch.Tensor:
    device = next(model.parameters()).device
    observations = torch.from_numpy(examples["observations"]).to(device)
    masks = torch.from_numpy(examples["masks"]).to(device)
    return model(observations, kind).masked_fill(~masks, -1e9)


def _metrics(model: InitialSetupPolicy, dataset: dict[str, dict[str, np.ndarray]]) -> dict:
    model.eval()
    result = {}
    with torch.no_grad():
        for kind in ("settlement", "road"):
            examples = dataset[kind]
            selected = _masked_logits(model, examples, kind).argmax(dim=1).cpu().numpy()
            result[kind] = {
                "examples": len(selected),
                "teacher_agreement": float(np.mean(selected == examples["labels"])),
                "distinct_selected": int(len(np.unique(selected))),
            }
    return result


def _merge_datasets(parts: list[dict[str, dict[str, np.ndarray]]]) -> dict[str, dict[str, np.ndarray]]:
    return {kind: {
        key: np.concatenate([part[kind][key] for part in parts], axis=0)
        for key in ("observations", "masks", "labels")
    } for kind in ("settlement", "road")}


def collect_on_policy_setup_examples(seeds: range, model: InitialSetupPolicy) -> dict[str, dict[str, np.ndarray]]:
    """現在の方策で進んだ状態に対するHeuristicの正解を集める。"""
    rows = {"settlement": {"observations": [], "masks": [], "labels": []},
            "road": {"observations": [], "masks": [], "labels": []}}
    teacher = PolicyCompatibleHeuristicAgent()
    for index, seed in enumerate(seeds):
        player_id = index % 4 + 1
        env = CatanEnv(CatanEnvConfig(learning_player_id=player_id,
                                     policy_compatible_opponents=True,
                                     observation_version="v2"))
        try:
            observation, info = env.reset(seed=seed)
            counts = {"settlement": 0, "road": 0}
            for _ in range(24):
                game = env.game
                if game is None:
                    raise AssertionError("盤面がありません。")
                teacher_action = teacher.select_action(game, player_id)
                if isinstance(teacher_action, PlaceInitialSettlementAction):
                    kind, start, count = "settlement", INITIAL_START, 54
                    label = teacher_action.vertex_id
                elif isinstance(teacher_action, PlaceInitialRoadAction):
                    kind, start, count = "road", INITIAL_ROAD_START, 72
                    label = teacher_action.edge_id
                else:
                    kind = ""
                if kind:
                    local_mask = info["action_mask"][start:start + count].copy()
                    if not local_mask[label]:
                        raise AssertionError(f"教師の初期{kind}が合法候補にありません。")
                    rows[kind]["observations"].append(observation.copy())
                    rows[kind]["masks"].append(local_mask)
                    rows[kind]["labels"].append(label)
                    counts[kind] += 1
                    action_id = model.select_action_id(observation, info["action_mask"])
                else:
                    action_id = action_to_id(teacher_action)
                if not info["action_mask"][action_id]:
                    raise AssertionError(f"方策の非法Action: seed={seed}, action={action_id}")
                observation, _, terminated, truncated, info = env.step(action_id)
                if counts == {"settlement": 2, "road": 2}:
                    break
                if terminated or truncated:
                    raise AssertionError("初期配置の途中で対局が終了しました。")
            if counts != {"settlement": 2, "road": 2}:
                raise AssertionError(f"seed={seed}: 初期配置4操作を収集できませんでした。")
        finally:
            env.close()
    return {kind: {
        "observations": np.stack(values["observations"]).astype(np.float32),
        "masks": np.stack(values["masks"]).astype(np.bool_),
        "labels": np.asarray(values["labels"], dtype=np.int64),
    } for kind, values in rows.items()}


def _fit(model: InitialSetupPolicy, training: dict[str, dict[str, np.ndarray]],
         validation: dict[str, dict[str, np.ndarray]], *, training_seed: int,
         max_epochs: int, learning_rate: float = 5e-4) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(training_seed)
    best_loss = float("inf")
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for kind in ("settlement", "road"):
            examples = training[kind]
            groups = np.array_split(rng.permutation(len(examples["labels"])),
                                    int(np.ceil(len(examples["labels"]) / 128)))
            for indices in groups:
                batch = {key: value[indices] for key, value in examples.items()}
                labels = torch.from_numpy(batch["labels"]).to(next(model.parameters()).device)
                loss = nn.functional.cross_entropy(_masked_logits(model, batch, kind), labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(np.mean([
                nn.functional.cross_entropy(
                    _masked_logits(model, validation[kind], kind),
                    torch.from_numpy(validation[kind]["labels"]).to(
                        next(model.parameters()).device),
                ).item()
                for kind in ("settlement", "road")
            ]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 8:
            break
    if best_state is None:
        raise AssertionError("初期配置モデルを保存できませんでした。")
    model.load_state_dict(best_state)
    model.eval()
    return history


def train_initial_setup_policy(source_policy: CandidateMaskablePolicy, *, train_boards: int = 600,
                               validation_boards: int = 150, test_boards: int = 150,
                               seed_start: int = 70_001, training_seed: int = 42,
                               max_epochs: int = 80) -> tuple[InitialSetupPolicy, dict]:
    """初期配置専用コピーを学習し、PPO本体を変更せず返す。"""
    if min(train_boards, validation_boards, test_boards, max_epochs) <= 0:
        raise ValueError("盤面数とepochは正の整数にしてください。")
    torch.manual_seed(training_seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ranges = {
        "train": range(seed_start, seed_start + train_boards),
        "validation": range(seed_start + train_boards, seed_start + train_boards + validation_boards),
        "test": range(seed_start + train_boards + validation_boards,
                      seed_start + train_boards + validation_boards + test_boards),
    }
    datasets = {name: _collect_full_setup_examples(seeds) for name, seeds in ranges.items()}
    model = InitialSetupPolicy(source_policy).to(next(source_policy.parameters()).device)
    before = _metrics(model, datasets["test"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(training_seed)
    best_loss = float("inf")
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for kind in ("settlement", "road"):
            examples = datasets["train"][kind]
            groups = np.array_split(rng.permutation(len(examples["labels"])),
                                    int(np.ceil(len(examples["labels"]) / 128)))
            for indices in groups:
                batch = {key: value[indices] for key, value in examples.items()}
                labels = torch.from_numpy(batch["labels"]).to(next(model.parameters()).device)
                loss = nn.functional.cross_entropy(_masked_logits(model, batch, kind), labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(np.mean([
                nn.functional.cross_entropy(
                    _masked_logits(model, datasets["validation"][kind], kind),
                    torch.from_numpy(datasets["validation"][kind]["labels"]).to(
                        next(model.parameters()).device),
                ).item()
                for kind in ("settlement", "road")
            ]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    if best_state is None:
        raise AssertionError("初期配置モデルを保存できませんでした。")
    model.load_state_dict(best_state)
    model.eval()
    report = {
        "seed_ranges": {name: [seeds.start, seeds.stop - 1] for name, seeds in ranges.items()},
        "training_seed": training_seed,
        "best_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
        "before_test": before,
        "after_test": _metrics(model, datasets["test"]),
        "history": history,
    }
    return model, report


def train_initial_setup_policy_dagger(
        source_policy: CandidateMaskablePolicy, *, train_boards: int = 600,
        validation_boards: int = 150, test_boards: int = 150,
        seed_start: int = 80_001, training_seed: int = 42,
        dagger_rounds: int = 3, dagger_boards: int = 300,
        dagger_validation_boards: int = 100, dagger_seed_start: int = 81_001,
        dagger_test_boards: int = 200) -> tuple[InitialSetupPolicy, dict]:
    """教師軌跡で事前学習後、方策自身が訪れた状態を教師データへ集約する。"""
    if min(dagger_rounds, dagger_boards, dagger_validation_boards, dagger_test_boards) <= 0:
        raise ValueError("DAggerのround数と盤面数は正の整数にしてください。")
    model, initial_report = train_initial_setup_policy(
        source_policy, train_boards=train_boards, validation_boards=validation_boards,
        test_boards=test_boards, seed_start=seed_start, training_seed=training_seed)
    original_training = _collect_full_setup_examples(range(seed_start, seed_start + train_boards))
    aggregated = [original_training]
    rounds = []
    stride = dagger_boards + dagger_validation_boards
    for round_index in range(dagger_rounds):
        round_start = dagger_seed_start + round_index * stride
        on_policy = collect_on_policy_setup_examples(
            range(round_start, round_start + dagger_boards), model)
        validation = collect_on_policy_setup_examples(
            range(round_start + dagger_boards, round_start + stride), model)
        before = _metrics(model, validation)
        aggregated.append(on_policy)
        training = _merge_datasets(aggregated)
        history = _fit(model, training, validation,
                       training_seed=training_seed + round_index + 1, max_epochs=30)
        rounds.append({
            "round": round_index + 1,
            "training_boards_total": train_boards + dagger_boards * (round_index + 1),
            "rollout_seed_range": [round_start, round_start + dagger_boards - 1],
            "validation_seed_range": [round_start + dagger_boards, round_start + stride - 1],
            "before_validation": before,
            "after_validation": _metrics(model, validation),
            "best_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
            "history": history,
        })
    test_start = dagger_seed_start + dagger_rounds * stride
    on_policy_test = collect_on_policy_setup_examples(
        range(test_start, test_start + dagger_test_boards), model)
    report = {
        "method": "DAgger",
        "initial_supervised": initial_report,
        "rounds": rounds,
        "on_policy_test_seed_range": [test_start, test_start + dagger_test_boards - 1],
        "on_policy_test": _metrics(model, on_policy_test),
    }
    return model, report


def collect_ranked_setup_examples(seeds: range) -> dict[str, dict[str, np.ndarray]]:
    """全合法候補のHeuristic基礎評価値を教師として収集する。"""
    rows = {kind: {"observations": [], "masks": [], "labels": [], "scores": []}
            for kind in ("settlement", "road")}
    teacher = PolicyCompatibleHeuristicAgent()
    for index, seed in enumerate(seeds):
        player_id = index % 4 + 1
        env = CatanEnv(CatanEnvConfig(learning_player_id=player_id,
                                     policy_compatible_opponents=True,
                                     observation_version="v2"))
        try:
            observation, info = env.reset(seed=seed)
            counts = {"settlement": 0, "road": 0}
            for _ in range(24):
                game = env.game
                if game is None:
                    raise AssertionError("盤面がありません。")
                action = teacher.select_action(game, player_id)
                if isinstance(action, PlaceInitialSettlementAction):
                    kind, start, count, label = "settlement", INITIAL_START, 54, action.vertex_id
                    score_values = np.asarray([
                        initial_vertex_value(game, vertex_id, player_id)
                        for vertex_id in range(count)
                    ], dtype=np.float32)
                elif isinstance(action, PlaceInitialRoadAction):
                    kind, start, count, label = "road", INITIAL_ROAD_START, 72, action.edge_id
                    if game.pending_settlement_id is None:
                        raise AssertionError("初期道路の起点開拓地がありません。")
                    score_values = np.asarray([
                        initial_road_value(game, edge_id, game.pending_settlement_id, player_id)
                        for edge_id in range(count)
                    ], dtype=np.float32)
                else:
                    kind = ""
                if kind:
                    local_mask = info["action_mask"][start:start + count].copy()
                    rows[kind]["observations"].append(observation.copy())
                    rows[kind]["masks"].append(local_mask)
                    rows[kind]["labels"].append(label)
                    rows[kind]["scores"].append(score_values)
                    counts[kind] += 1
                observation, _, terminated, truncated, info = env.step(action_to_id(action))
                if counts == {"settlement": 2, "road": 2}:
                    break
                if terminated or truncated:
                    raise AssertionError("初期配置の途中で対局が終了しました。")
            if counts != {"settlement": 2, "road": 2}:
                raise AssertionError(f"seed={seed}: ランキング教師を収集できませんでした。")
        finally:
            env.close()
    return {kind: {
        "observations": np.stack(values["observations"]).astype(np.float32),
        "masks": np.stack(values["masks"]).astype(np.bool_),
        "labels": np.asarray(values["labels"], dtype=np.int64),
        "scores": np.stack(values["scores"]).astype(np.float32),
    } for kind, values in rows.items()}


def _ranking_loss(model: InitialSetupPolicy, examples: dict[str, np.ndarray], kind: str,
                  temperature: float) -> torch.Tensor:
    logits = _masked_logits(model, examples, kind)
    device = logits.device
    masks = torch.from_numpy(examples["masks"]).to(device)
    scores = torch.from_numpy(examples["scores"]).to(device)
    teacher_logits = (scores / temperature).masked_fill(~masks, -1e9)
    targets = torch.softmax(teacher_logits, dim=1)
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _ranking_metrics(model: InitialSetupPolicy,
                     dataset: dict[str, dict[str, np.ndarray]]) -> dict:
    model.eval()
    result = {}
    with torch.no_grad():
        for kind in ("settlement", "road"):
            examples = dataset[kind]
            selected = _masked_logits(model, examples, kind).argmax(dim=1).cpu().numpy()
            masked_scores = np.where(examples["masks"], examples["scores"], -np.inf)
            selected_scores = examples["scores"][np.arange(len(selected)), selected]
            best_scores = masked_scores.max(axis=1)
            result[kind] = {
                "examples": len(selected),
                "teacher_action_agreement": float(np.mean(selected == examples["labels"])),
                "optimal_value_rate": float(np.mean(selected_scores == best_scores)),
                "mean_value_regret": float(np.mean(best_scores - selected_scores)),
                "distinct_selected": int(len(np.unique(selected))),
            }
    return result


def train_ranked_initial_setup_policy(
        source_policy: CandidateMaskablePolicy, *, train_boards: int = 600,
        validation_boards: int = 150, test_boards: int = 150,
        seed_start: int = 90_001, training_seed: int = 42,
        max_epochs: int = 80, temperature: float = 5.0,
        validation_seed_start: int | None = None,
        test_seed_start: int | None = None) -> tuple[InitialSetupPolicy, dict]:
    """全候補のHeuristic評価順位をsoft targetとして初期配置方策を学習する。"""
    if min(train_boards, validation_boards, test_boards, max_epochs) <= 0 or temperature <= 0:
        raise ValueError("盤面数・epoch・temperatureは正の値にしてください。")
    torch.manual_seed(training_seed)
    validation_start = (seed_start + train_boards if validation_seed_start is None
                        else validation_seed_start)
    test_start = (validation_start + validation_boards if test_seed_start is None
                  else test_seed_start)
    ranges = {"train": range(seed_start, seed_start + train_boards),
              "validation": range(validation_start, validation_start + validation_boards),
              "test": range(test_start, test_start + test_boards)}
    datasets = {name: collect_ranked_setup_examples(seeds) for name, seeds in ranges.items()}
    model = InitialSetupPolicy(source_policy).to(next(source_policy.parameters()).device)
    before = _ranking_metrics(model, datasets["test"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(training_seed)
    best_loss = float("inf")
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for kind in ("settlement", "road"):
            examples = datasets["train"][kind]
            groups = np.array_split(rng.permutation(len(examples["labels"])),
                                    int(np.ceil(len(examples["labels"]) / 128)))
            for indices in groups:
                batch = {key: value[indices] for key, value in examples.items()}
                loss = _ranking_loss(model, batch, kind, temperature)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(np.mean([
                _ranking_loss(model, datasets["validation"][kind], kind, temperature).item()
                for kind in ("settlement", "road")
            ]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    if best_state is None:
        raise AssertionError("ランキング学習モデルを保存できませんでした。")
    model.load_state_dict(best_state)
    model.eval()
    report = {
        "method": "all-candidate soft ranking",
        "teacher_score": "pips*10 + terrain_kinds*4 + new_terrain_kinds*3; road=max reachable vertex",
        "temperature": temperature,
        "seed_ranges": {name: [seeds.start, seeds.stop - 1] for name, seeds in ranges.items()},
        "training_seed": training_seed,
        "best_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
        "before_test": before,
        "after_test": _ranking_metrics(model, datasets["test"]),
        "history": history,
    }
    return model, report
