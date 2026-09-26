"""初期配置候補を盤面Graph上で採点する、小規模な依存なしGNN。"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from app.domain.actions import Action
from app.domain.board import create_topology
from app.domain.game import GameState

from .action_space import get_action_mask, id_to_action
from .candidate_policy import (EDGE_START, PORT_START, PROGRESS_START, TILE_START,
                               VERTEX_START, BASE_OBSERVATION_SIZE,
                               CandidateMaskablePolicy)
from .initial_setup_policy import (INITIAL_ROAD_START, INITIAL_START,
                                   collect_ranked_setup_examples)
from .observation import encode_observation, get_observation


class GraphMessageLayer(nn.Module):
    """各交差点へ隣接交差点の平均特徴を伝えるGraph層。"""

    def __init__(self, size: int):
        super().__init__()
        self.self_projection = nn.Linear(size, size)
        self.neighbor_projection = nn.Linear(size, size, bias=False)
        self.normalization = nn.LayerNorm(size)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.einsum("ij,bjk->bik", adjacency, values)
        updated = self.self_projection(values) + self.neighbor_projection(neighbors)
        return torch.relu(self.normalization(updated))


class GraphInitialSetupPolicy(nn.Module):
    """54交差点の接続関係を2層伝播して開拓地・道路を採点する。"""

    def __init__(self, source_policy: CandidateMaskablePolicy):
        super().__init__()
        source = source_policy.mlp_extractor
        topology = create_topology()
        adjacency = torch.zeros((54, 54), dtype=torch.float32)
        for edge in topology.edges:
            a, b = edge.vertex_ids
            adjacency[a, b] = adjacency[b, a] = 1.0
        degrees = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("adjacency", adjacency / degrees)
        self.register_buffer("edge_vertices", torch.tensor(
            [edge.vertex_ids for edge in topology.edges], dtype=torch.long))

        self.context = deepcopy(source.context)
        if self.context[0].in_features > 142:
            expanded = self.context[0]
            compact = nn.Linear(142, expanded.out_features)
            with torch.no_grad():
                compact.weight.copy_(expanded.weight[:, :142])
                compact.bias.copy_(expanded.bias)
            self.context[0] = compact
        self.vertex_input = nn.Sequential(nn.Linear(14 + 64, 64), nn.ReLU())
        self.message_layers = nn.ModuleList([GraphMessageLayer(64), GraphMessageLayer(64)])
        self.settlement_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.edge_encoder = nn.Sequential(nn.Linear(5 + 64 * 2 + 64, 96), nn.ReLU(),
                                          nn.Linear(96, 64), nn.ReLU())
        self.road_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def _states(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = observations.shape[0]
        vertices = observations[:, VERTEX_START:EDGE_START].reshape(batch, 54, 14)
        edges = observations[:, EDGE_START:PORT_START].reshape(batch, 72, 5)
        own_production = (vertices[:, :, 1:2] * vertices[:, :, 9:14]).sum(dim=1).clamp(0, 1)
        global_values = torch.cat((
            observations[:, :TILE_START],
            observations[:, PROGRESS_START:BASE_OBSERVATION_SIZE],
            own_production,
        ), dim=1)
        context = self.context(global_values)
        vertex_state = self.vertex_input(torch.cat(
            (vertices, context[:, None, :].expand(-1, 54, -1)), dim=2))
        for layer in self.message_layers:
            vertex_state = layer(vertex_state, self.adjacency)
        endpoints = vertex_state[:, self.edge_vertices]
        edge_state = self.edge_encoder(torch.cat(
            (edges, endpoints.mean(dim=2), endpoints.amax(dim=2),
             context[:, None, :].expand(-1, 72, -1)), dim=2))
        return vertex_state, edge_state, context

    def forward(self, observations: torch.Tensor, kind: str) -> torch.Tensor:
        vertices, edges, _ = self._states(observations)
        if kind == "settlement":
            return self.settlement_head(vertices).squeeze(-1)
        if kind == "road":
            return self.road_head(edges).squeeze(-1)
        raise ValueError(f"未対応の初期配置種別です: {kind}")

    def select_action_id(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        device = next(self.parameters()).device
        for kind, start, count in (("settlement", INITIAL_START, 54),
                                   ("road", INITIAL_ROAD_START, 72)):
            local_mask = np.asarray(action_mask[start:start + count], dtype=np.bool_)
            if local_mask.any():
                with torch.no_grad():
                    logits = self(torch.from_numpy(observation[None, :]).to(device), kind)[0]
                    selected = int(logits.masked_fill(
                        ~torch.from_numpy(local_mask).to(device), -1e9).argmax().item())
                return start + selected
        raise ValueError("初期配置の合法Actionがありません。")


class GraphInitialSetupAgent:
    """学習済みGNNをCatanEnvと評価器から共通利用するAdapter。"""

    def __init__(self, policy: GraphInitialSetupPolicy, observation_version: str = "v2"):
        self.policy = policy
        self.observation_version = observation_version

    def select_action(self, game: GameState, player_id: int) -> Action:
        observation = encode_observation(get_observation(
            game, player_id, version=self.observation_version))
        action_mask = get_action_mask(game, player_id)
        return id_to_action(self.policy.select_action_id(observation, action_mask))


def _masked_logits(model: GraphInitialSetupPolicy, examples: dict[str, np.ndarray],
                   kind: str) -> torch.Tensor:
    device = next(model.parameters()).device
    observations = torch.from_numpy(examples["observations"]).to(device)
    masks = torch.from_numpy(examples["masks"]).to(device)
    return model(observations, kind).masked_fill(~masks, -1e9)


def _loss(model: GraphInitialSetupPolicy, examples: dict[str, np.ndarray], kind: str,
          temperature: float) -> torch.Tensor:
    logits = _masked_logits(model, examples, kind)
    masks = torch.from_numpy(examples["masks"]).to(logits.device)
    scores = torch.from_numpy(examples["scores"]).to(logits.device)
    targets = torch.softmax((scores / temperature).masked_fill(~masks, -1e9), dim=1)
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _metrics(model: GraphInitialSetupPolicy,
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


def train_graph_initial_setup_policy(
        source_policy: CandidateMaskablePolicy, *, train_boards: int = 600,
        validation_boards: int = 300, test_boards: int = 500,
        train_seed_start: int = 100_001, validation_seed_start: int = 106_001,
        test_seed_start: int = 106_501, training_seed: int = 42,
        max_epochs: int = 80, temperature: float = 5.0) -> tuple[GraphInitialSetupPolicy, dict]:
    """MLPランキング実験と同じ教師値でGNNだけを学習する。"""
    torch.manual_seed(training_seed)
    ranges = {"train": range(train_seed_start, train_seed_start + train_boards),
              "validation": range(validation_seed_start, validation_seed_start + validation_boards),
              "test": range(test_seed_start, test_seed_start + test_boards)}
    datasets = {name: collect_ranked_setup_examples(seeds) for name, seeds in ranges.items()}
    model = GraphInitialSetupPolicy(source_policy).to(next(source_policy.parameters()).device)
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
                loss = _loss(model, batch, kind, temperature)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(np.mean([
                _loss(model, datasets["validation"][kind], kind, temperature).item()
                for kind in ("settlement", "road")]))
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
        raise AssertionError("GNN初期配置モデルを保存できませんでした。")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "method": "two-layer vertex message passing + all-candidate soft ranking",
        "temperature": temperature,
        "seed_ranges": {name: [values.start, values.stop - 1] for name, values in ranges.items()},
        "training_seed": training_seed,
        "best_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
        "before_test": before,
        "after_test": _metrics(model, datasets["test"]),
        "history": history,
    }
