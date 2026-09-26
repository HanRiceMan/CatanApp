"""固定Action IDを維持したまま、盤面候補を共有重みで採点するMaskablePPO方策。"""

from __future__ import annotations

import torch
from torch import nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from app.domain.actions import (BuildCityAction, BuildRoadAction, BuildSettlementAction,
                                MoveRobberAction, PlaceInitialRoadAction,
                                PlaceInitialSettlementAction, StealResourceAction)
from app.domain.board import create_topology

from .action_space import ACTION_SPACE_SIZE, action_to_id
from .action_families import ACTION_FAMILY_INDEX
from .hierarchical_distribution import (FAMILY_COUNT,
                                        HierarchicalMaskableCategoricalDistribution)
from .observation import OBSERVATION_VECTOR_SIZES
from .goal_context import GOAL_CONTEXT_SIZE
from .belief_context import BELIEF_CONTEXT_SIZE, BELIEF_FEATURES_PER_OPPONENT


TILE_START = 29 + 42
VERTEX_START = TILE_START + 19 * 18
EDGE_START = VERTEX_START + 54 * 14
PORT_START = EDGE_START + 72 * 5
PROGRESS_START = PORT_START + 9 * 8
assert PROGRESS_START + 66 == OBSERVATION_VECTOR_SIZES["v2"]
BASE_OBSERVATION_SIZE = OBSERVATION_VECTOR_SIZES["v2"]


def _block_start(action) -> int:
    return action_to_id(action)


SPATIAL_BLOCKS = (
    (_block_start(PlaceInitialSettlementAction(0)), 54),
    (_block_start(PlaceInitialRoadAction(0)), 72),
    (_block_start(MoveRobberAction(0)), 19),
    (_block_start(BuildRoadAction(0)), 72),
    (_block_start(BuildSettlementAction(0)), 54),
    (_block_start(BuildCityAction(0)), 54),
)
for (start, count), action_type in zip(
    SPATIAL_BLOCKS,
    (PlaceInitialSettlementAction, PlaceInitialRoadAction, MoveRobberAction,
     BuildRoadAction, BuildSettlementAction, BuildCityAction),
):
    if any(action_to_id(action_type(index)) != start + index for index in range(count)):
        raise AssertionError("空間Action IDが連続していません。候補方策の対応を更新してください。")


class CandidateActorCritic(nn.Module):
    """位置ごとの重みを持たない空間Actorと、通常の全盤面Critic。"""

    latent_dim_pi = ACTION_SPACE_SIZE
    latent_dim_vf = 128

    def __init__(self, observation_size: int):
        super().__init__()
        if observation_size not in {
            OBSERVATION_VECTOR_SIZES["v2"], OBSERVATION_VECTOR_SIZES["v3"],
            OBSERVATION_VECTOR_SIZES["v4"], OBSERVATION_VECTOR_SIZES["v5"],
        }:
            raise ValueError("候補方式はObservation v2/v3/v4/v5専用です。")
        topology = create_topology()
        self.register_buffer("edge_vertices", torch.tensor([edge.vertex_ids for edge in topology.edges],
                                                           dtype=torch.long))
        self.register_buffer("tile_vertices", torch.tensor([
            [vertex.id for vertex in topology.vertices if tile_id in vertex.hex_ids]
            for tile_id in range(len(topology.hexes))], dtype=torch.long))
        if self.tile_vertices.shape != (19, 6):
            raise AssertionError("タイルと交差点の対応が不正です。")

        goal_size = observation_size - BASE_OBSERVATION_SIZE
        self.context = nn.Sequential(nn.Linear(29 + 42 + 66 + 5 + goal_size, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU())
        self.vertex_encoder = nn.Sequential(nn.Linear(14 + 64, 64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.ReLU())
        self.edge_encoder = nn.Sequential(nn.Linear(5 + 14 * 2 + 64, 64), nn.ReLU(),
                                          nn.Linear(64, 64), nn.ReLU())
        self.tile_encoder = nn.Sequential(nn.Linear(18 + 14 + 64, 64), nn.ReLU(),
                                          nn.Linear(64, 64), nn.ReLU())
        self.initial_settlement_head = nn.Linear(64, 1)
        self.settlement_head = nn.Linear(64, 1)
        self.city_head = nn.Linear(64, 1)
        self.initial_road_head = nn.Linear(64, 1)
        self.road_head = nn.Linear(64, 1)
        self.robber_head = nn.Linear(64, 1)
        self.nonspatial_head = nn.Linear(64, ACTION_SPACE_SIZE)
        self.critic = nn.Sequential(nn.Linear(observation_size, 256), nn.ReLU(),
                                    nn.Linear(256, self.latent_dim_vf), nn.ReLU())

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        vertices = observations[:, VERTEX_START:EDGE_START].reshape(batch, 54, 14)
        edges = observations[:, EDGE_START:PORT_START].reshape(batch, 72, 5)
        tiles = observations[:, TILE_START:VERTEX_START].reshape(batch, 19, 18)
        own_mask = vertices[:, :, 1:2]
        own_production = (own_mask * vertices[:, :, 9:14]).sum(dim=1).clamp(0, 1)
        global_values = torch.cat((
            observations[:, :TILE_START],
            observations[:, PROGRESS_START:BASE_OBSERVATION_SIZE],
            own_production,
            observations[:, BASE_OBSERVATION_SIZE:],
        ), dim=1)
        context = self.context(global_values)

        vertex_state = self.vertex_encoder(torch.cat(
            (vertices, context[:, None, :].expand(-1, 54, -1)), dim=2))
        endpoints = vertices[:, self.edge_vertices]
        endpoint_features = torch.cat((endpoints.mean(dim=2), endpoints.amax(dim=2)), dim=2)
        edge_state = self.edge_encoder(torch.cat(
            (edges, endpoint_features, context[:, None, :].expand(-1, 72, -1)), dim=2))
        tile_neighbors = vertices[:, self.tile_vertices].mean(dim=2)
        tile_state = self.tile_encoder(torch.cat(
            (tiles, tile_neighbors, context[:, None, :].expand(-1, 19, -1)), dim=2))

        logits = self.nonspatial_head(context)
        (initial_settlement, initial_road, robber, road, settlement, city) = SPATIAL_BLOCKS
        logits[:, initial_settlement[0]:initial_settlement[0] + initial_settlement[1]] = (
            self.initial_settlement_head(vertex_state).squeeze(-1))
        logits[:, initial_road[0]:initial_road[0] + initial_road[1]] = (
            self.initial_road_head(edge_state).squeeze(-1))
        logits[:, robber[0]:robber[0] + robber[1]] = self.robber_head(tile_state).squeeze(-1)
        logits[:, road[0]:road[0] + road[1]] = self.road_head(edge_state).squeeze(-1)
        logits[:, settlement[0]:settlement[0] + settlement[1]] = (
            self.settlement_head(vertex_state).squeeze(-1))
        logits[:, city[0]:city[0] + city[1]] = self.city_head(vertex_state).squeeze(-1)
        return logits

    def forward_critic(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(observations), self.forward_critic(observations)


class CandidateMaskablePolicy(MaskableActorCriticPolicy):
    """SB3-contribのmask処理を再利用し、Actorの空間出力だけを候補方式へ変更。"""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CandidateActorCritic(self.features_dim)

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        # CandidateActorCriticが既に377個のlogitを返す。ID別線形層を重ねない。
        self.action_net = nn.Identity()
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)


class HierarchicalCandidateActorCritic(CandidateActorCritic):
    """候補logitに加え、状態依存のAction Family logitを出力する。"""

    latent_dim_pi = ACTION_SPACE_SIZE + FAMILY_COUNT

    def __init__(self, observation_size: int):
        super().__init__(observation_size)
        self.family_head = nn.Sequential(nn.Linear(observation_size, 128), nn.ReLU(),
                                         nn.Linear(128, FAMILY_COUNT))

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        candidates = super().forward_actor(observations)
        return torch.cat((candidates, self.family_head(observations)), dim=1)


class HierarchicalCandidateMaskablePolicy(CandidateMaskablePolicy):
    """P(Action Family) × P(候補|Family)を直接PPOで学習する実験用方策。"""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HierarchicalCandidateActorCritic(self.features_dim)

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        self.action_net = nn.Identity()
        self.action_dist = HierarchicalMaskableCategoricalDistribution()
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)


class ResidualGraphMessageLayer(nn.Module):
    """交差点Graph上で隣接状態を伝播する残差Message Passing層。"""

    def __init__(self, feature_size: int = 64):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size),
        )
        self.normalization = nn.LayerNorm(feature_size)

    def forward(self, states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.einsum("ij,bjf->bif", adjacency, states)
        update = self.update(torch.cat((states, neighbors), dim=2))
        return torch.relu(self.normalization(states + update))


class GraphHierarchicalCandidateActorCritic(HierarchicalCandidateActorCritic):
    """既存階層方策へ、通常ターン用Graph補正Headを追加する。

    既存Headはそのまま残し、Graph補正Headの最終層を0初期化する。これにより
    既存モデルから重みを移した直後のAction logitは完全に一致し、学習による
    差だけを比較できる。
    """

    def __init__(self, observation_size: int):
        super().__init__(observation_size)
        topology = create_topology()
        adjacency = torch.zeros((54, 54), dtype=torch.float32)
        for edge in topology.edges:
            first, second = edge.vertex_ids
            adjacency[first, second] = 1.0
            adjacency[second, first] = 1.0
        adjacency /= adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("vertex_adjacency", adjacency)

        self.graph_layers = nn.ModuleList([
            ResidualGraphMessageLayer(64),
            ResidualGraphMessageLayer(64),
        ])
        self.graph_initial_settlement_head = nn.Linear(64, 1)
        self.graph_settlement_head = nn.Linear(64, 1)
        self.graph_city_head = nn.Linear(64, 1)
        self.graph_initial_road_head = nn.Sequential(
            nn.Linear(64 * 3, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.graph_road_head = nn.Sequential(
            nn.Linear(64 * 3, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.graph_robber_head = nn.Sequential(
            nn.Linear(64 * 2, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        for head in (
            self.graph_initial_settlement_head,
            self.graph_settlement_head,
            self.graph_city_head,
            self.graph_initial_road_head[-1],
            self.graph_road_head[-1],
            self.graph_robber_head[-1],
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _encoded_spatial_states(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = observations.shape[0]
        vertices = observations[:, VERTEX_START:EDGE_START].reshape(batch, 54, 14)
        edges = observations[:, EDGE_START:PORT_START].reshape(batch, 72, 5)
        tiles = observations[:, TILE_START:VERTEX_START].reshape(batch, 19, 18)
        own_mask = vertices[:, :, 1:2]
        own_production = (own_mask * vertices[:, :, 9:14]).sum(dim=1).clamp(0, 1)
        global_values = torch.cat((
            observations[:, :TILE_START],
            observations[:, PROGRESS_START:BASE_OBSERVATION_SIZE],
            own_production,
            observations[:, BASE_OBSERVATION_SIZE:],
        ), dim=1)
        context = self.context(global_values)
        vertex_state = self.vertex_encoder(torch.cat(
            (vertices, context[:, None, :].expand(-1, 54, -1)), dim=2))
        for layer in self.graph_layers:
            vertex_state = layer(vertex_state, self.vertex_adjacency)

        endpoints = vertex_state[:, self.edge_vertices]
        endpoint_features = torch.cat((endpoints.mean(dim=2), endpoints.amax(dim=2)), dim=2)
        edge_state = torch.cat((
            self.edge_encoder(torch.cat((
                edges,
                torch.cat((vertices[:, self.edge_vertices].mean(dim=2),
                           vertices[:, self.edge_vertices].amax(dim=2)), dim=2),
                context[:, None, :].expand(-1, 72, -1),
            ), dim=2)),
            endpoint_features,
        ), dim=2)

        tile_base = self.tile_encoder(torch.cat((
            tiles,
            vertices[:, self.tile_vertices].mean(dim=2),
            context[:, None, :].expand(-1, 19, -1),
        ), dim=2))
        tile_state = torch.cat((tile_base, vertex_state[:, self.tile_vertices].mean(dim=2)), dim=2)
        return vertex_state, edge_state, tile_state

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        base_logits = super().forward_actor(observations)
        candidates = base_logits[:, :ACTION_SPACE_SIZE]
        family_logits = base_logits[:, ACTION_SPACE_SIZE:]
        vertex_state, edge_state, tile_state = self._encoded_spatial_states(observations)
        correction = torch.zeros_like(candidates)
        (initial_settlement, initial_road, robber, road, settlement, city) = SPATIAL_BLOCKS
        correction[:, initial_settlement[0]:initial_settlement[0] + initial_settlement[1]] = (
            self.graph_initial_settlement_head(vertex_state).squeeze(-1))
        correction[:, initial_road[0]:initial_road[0] + initial_road[1]] = (
            self.graph_initial_road_head(edge_state).squeeze(-1))
        correction[:, robber[0]:robber[0] + robber[1]] = (
            self.graph_robber_head(tile_state).squeeze(-1))
        correction[:, road[0]:road[0] + road[1]] = self.graph_road_head(edge_state).squeeze(-1)
        correction[:, settlement[0]:settlement[0] + settlement[1]] = (
            self.graph_settlement_head(vertex_state).squeeze(-1))
        correction[:, city[0]:city[0] + city[1]] = self.graph_city_head(vertex_state).squeeze(-1)
        family_correction = self._graph_family_correction(
            vertex_state, edge_state, tile_state, family_logits
        )
        return torch.cat((candidates + correction, family_logits + family_correction), dim=1)

    def _graph_family_correction(
        self,
        vertex_state: torch.Tensor,
        edge_state: torch.Tensor,
        tile_state: torch.Tensor,
        family_logits: torch.Tensor,
    ) -> torch.Tensor:
        """旧GraphモデルではFamily logitを変更しない。"""
        return torch.zeros_like(family_logits)


class GraphHierarchicalCandidateMaskablePolicy(HierarchicalCandidateMaskablePolicy):
    """既存階層分布を保ったまま、空間候補へGraph補正を加える方策。"""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GraphHierarchicalCandidateActorCritic(self.features_dim)


class GraphFamilyHierarchicalCandidateActorCritic(GraphHierarchicalCandidateActorCritic):
    """盤面Graphの全体要約をAction Family選択にも残差入力する。"""

    def __init__(self, observation_size: int):
        super().__init__(observation_size)
        # vertex mean/max=128、edge mean=192、tile mean=128。
        self.graph_family_head = nn.Sequential(
            nn.Linear(448, 64), nn.ReLU(), nn.Linear(64, FAMILY_COUNT),
        )
        nn.init.zeros_(self.graph_family_head[-1].weight)
        nn.init.zeros_(self.graph_family_head[-1].bias)
        # 教師あり残差を元PPOへ混ぜる比率。state_dict外なので旧artifactと互換。
        self.graph_family_scale = 1.0

    def _graph_family_correction(
        self,
        vertex_state: torch.Tensor,
        edge_state: torch.Tensor,
        tile_state: torch.Tensor,
        family_logits: torch.Tensor,
    ) -> torch.Tensor:
        summary = torch.cat((
            vertex_state.mean(dim=1),
            vertex_state.amax(dim=1),
            edge_state.mean(dim=1),
            tile_state.mean(dim=1),
        ), dim=1)
        return self.graph_family_head(summary) * self.graph_family_scale


class GraphFamilyHierarchicalCandidateMaskablePolicy(GraphHierarchicalCandidateMaskablePolicy):
    """空間候補とAction Familyの両方をGraphで補正する実験方策。"""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GraphFamilyHierarchicalCandidateActorCritic(self.features_dim)


class BeliefGraphFamilyHierarchicalCandidateActorCritic(
    GraphFamilyHierarchicalCandidateActorCritic
):
    """共同ベイズ手札推定だけを既存GNNへ残差入力するv4方策。"""

    def __init__(self, observation_size: int):
        if observation_size != OBSERVATION_VECTOR_SIZES["v4"]:
            raise ValueError("Belief Graph方策はObservation v4専用です。")
        super().__init__(observation_size)
        self.belief_action_head = nn.Sequential(
            nn.Linear(BELIEF_CONTEXT_SIZE, 128), nn.ReLU(),
            nn.Linear(128, ACTION_SPACE_SIZE),
        )
        self.belief_family_head = nn.Sequential(
            nn.Linear(BELIEF_CONTEXT_SIZE, 64), nn.ReLU(),
            nn.Linear(64, FAMILY_COUNT),
        )
        for head in (self.belief_action_head[-1], self.belief_family_head[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        logits = super().forward_actor(observations)
        belief = observations[:, BASE_OBSERVATION_SIZE:]
        candidates = logits[:, :ACTION_SPACE_SIZE] + self.belief_action_head(belief)
        families = logits[:, ACTION_SPACE_SIZE:] + self.belief_family_head(belief)
        return torch.cat((candidates, families), dim=1)


class BeliefGraphFamilyHierarchicalCandidateMaskablePolicy(
    GraphFamilyHierarchicalCandidateMaskablePolicy
):
    """現行GNNへ、学習済みではないベイズ推定値を入力するv4方策。"""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = BeliefGraphFamilyHierarchicalCandidateActorCritic(
            self.features_dim
        )


class ContextualBeliefGraphFamilyHierarchicalCandidateActorCritic(
    GraphFamilyHierarchicalCandidateActorCritic
):
    """ベイズ推定と現行PPOの盤面候補を組み合わせるv4残差方策。"""

    def __init__(self, observation_size: int):
        if observation_size != OBSERVATION_VECTOR_SIZES["v4"]:
            raise ValueError("Contextual Belief Graph方策はObservation v4専用です。")
        super().__init__(observation_size)
        contextual_size = BELIEF_CONTEXT_SIZE + ACTION_SPACE_SIZE + FAMILY_COUNT
        self.belief_action_head = nn.Sequential(
            nn.Linear(contextual_size, 192), nn.ReLU(),
            nn.Linear(192, ACTION_SPACE_SIZE),
        )
        self.belief_family_head = nn.Sequential(
            nn.Linear(contextual_size, 96), nn.ReLU(),
            nn.Linear(96, FAMILY_COUNT),
        )
        for head in (self.belief_action_head[-1], self.belief_family_head[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        logits = super().forward_actor(observations)
        features = torch.cat((
            observations[:, BASE_OBSERVATION_SIZE:], logits.detach(),
        ), dim=1)
        candidates = logits[:, :ACTION_SPACE_SIZE] + self.belief_action_head(features)
        families = logits[:, ACTION_SPACE_SIZE:] + self.belief_family_head(features)
        return torch.cat((candidates, families), dim=1)


class ContextualBeliefGraphFamilyHierarchicalCandidateMaskablePolicy(
    GraphFamilyHierarchicalCandidateMaskablePolicy
):
    """ベイズ推定を現行PPOの盤面意図へ条件付けするv4方策。"""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ContextualBeliefGraphFamilyHierarchicalCandidateActorCritic(
            self.features_dim
        )


class RobberBeliefGraphFamilyHierarchicalCandidateActorCritic(
    GraphFamilyHierarchicalCandidateActorCritic
):
    """ベイズ残差を盗賊移動19候補・被害者4候補だけへ限定するv5方策。"""

    ROBBER_START = _block_start(MoveRobberAction(0))
    ROBBER_COUNT = 19
    VICTIM_START = _block_start(StealResourceAction(1))
    VICTIM_COUNT = 4
    ROBBER_CONTEXT_SIZE = BELIEF_CONTEXT_SIZE + 4

    def __init__(self, observation_size: int):
        if observation_size != OBSERVATION_VECTOR_SIZES["v5"]:
            raise ValueError("Robber Belief Graph方策はObservation v5専用です。")
        super().__init__(observation_size)
        # タイルはGNNの盤面状態と全相手の推定、被害者は対応する相手1人分の
        # 推定・公開状態・既存logitから採点する。
        self.robber_tile_head = nn.Sequential(
            nn.Linear(128 + self.ROBBER_CONTEXT_SIZE + 1, 96), nn.ReLU(),
            nn.Linear(96, 1),
        )
        self.robber_victim_head = nn.Sequential(
            nn.Linear(BELIEF_FEATURES_PER_OPPONENT + 14 + 1, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        for head in (self.robber_tile_head[-1], self.robber_victim_head[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        logits = super().forward_actor(observations)
        context = observations[:, BASE_OBSERVATION_SIZE:]
        _, _, tile_state = self._encoded_spatial_states(observations)
        base_candidates = logits[:, :ACTION_SPACE_SIZE].detach()
        tile_base = base_candidates[
            :, self.ROBBER_START:self.ROBBER_START + self.ROBBER_COUNT
        ]
        tile_features = torch.cat((
            tile_state,
            context[:, None, :].expand(-1, self.ROBBER_COUNT, -1),
            tile_base[:, :, None],
        ), dim=2)
        tile_correction = self.robber_tile_head(tile_features).squeeze(-1)

        batch = observations.shape[0]
        relative_beliefs = context[:, :BELIEF_CONTEXT_SIZE].reshape(
            batch, 3, BELIEF_FEATURES_PER_OPPONENT
        )
        relative_public = observations[:, 29:TILE_START].reshape(batch, 3, 14)
        self_seat = context[:, BELIEF_CONTEXT_SIZE:].argmax(dim=1)
        absolute = torch.arange(4, device=observations.device)[None, :]
        offsets = (absolute - self_seat[:, None]) % 4
        source = (offsets - 1).clamp(0, 2)
        belief_index = source[:, :, None].expand(
            -1, -1, BELIEF_FEATURES_PER_OPPONENT
        )
        public_index = source[:, :, None].expand(-1, -1, 14)
        victim_beliefs = relative_beliefs.gather(1, belief_index)
        victim_public = relative_public.gather(1, public_index)
        is_opponent = (offsets != 0)[:, :, None]
        victim_beliefs = victim_beliefs * is_opponent
        victim_public = victim_public * is_opponent
        victim_base = base_candidates[
            :, self.VICTIM_START:self.VICTIM_START + self.VICTIM_COUNT
        ]
        victim_correction = self.robber_victim_head(torch.cat((
            victim_beliefs, victim_public, victim_base[:, :, None],
        ), dim=2)).squeeze(-1)

        candidates = logits[:, :ACTION_SPACE_SIZE].clone()
        candidates[:, self.ROBBER_START:self.ROBBER_START + self.ROBBER_COUNT] += (
            tile_correction
        )
        candidates[:, self.VICTIM_START:self.VICTIM_START + self.VICTIM_COUNT] += (
            victim_correction
        )
        return torch.cat((candidates, logits[:, ACTION_SPACE_SIZE:]), dim=1)


class RobberBeliefGraphFamilyHierarchicalCandidateMaskablePolicy(
    GraphFamilyHierarchicalCandidateMaskablePolicy
):
    """計画層から盗賊判断だけを移管するためのv4方策。"""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = RobberBeliefGraphFamilyHierarchicalCandidateActorCritic(
            self.features_dim
        )


class GoalGraphFamilyHierarchicalCandidateActorCritic(
    GraphFamilyHierarchicalCandidateActorCritic
):
    """v3の明示的な目標Contextを、候補・Action Familyの残差へ接続する。"""

    def __init__(self, observation_size: int):
        if observation_size != OBSERVATION_VECTOR_SIZES["v3"]:
            raise ValueError("Goal Graph方策はObservation v3専用です。")
        super().__init__(observation_size)
        self.goal_action_head = nn.Sequential(
            nn.Linear(GOAL_CONTEXT_SIZE, 128), nn.ReLU(),
            nn.Linear(128, ACTION_SPACE_SIZE),
        )
        self.goal_family_head = nn.Sequential(
            nn.Linear(GOAL_CONTEXT_SIZE, 64), nn.ReLU(),
            nn.Linear(64, FAMILY_COUNT),
        )
        for head in (self.goal_action_head[-1], self.goal_family_head[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        logits = super().forward_actor(observations)
        goal = observations[:, BASE_OBSERVATION_SIZE:]
        candidates = logits[:, :ACTION_SPACE_SIZE] + self.goal_action_head(goal)
        families = logits[:, ACTION_SPACE_SIZE:] + self.goal_family_head(goal)
        return torch.cat((candidates, families), dim=1)


class GoalGraphFamilyHierarchicalCandidateMaskablePolicy(
    GraphFamilyHierarchicalCandidateMaskablePolicy
):
    """現行GNNへ、明示的な可変目標を追加した教師模倣用方策。"""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GoalGraphFamilyHierarchicalCandidateActorCritic(
            self.features_dim
        )


class GatedGoalGraphFamilyHierarchicalCandidateActorCritic(
    GoalGraphFamilyHierarchicalCandidateActorCritic
):
    """計画補正が必要な局面だけ、目標残差を有効にする。"""

    def __init__(self, observation_size: int):
        super().__init__(observation_size)
        self.goal_override_gate = nn.Sequential(
            nn.Linear(GOAL_CONTEXT_SIZE + FAMILY_COUNT, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        nn.init.zeros_(self.goal_override_gate[-1].weight)
        nn.init.zeros_(self.goal_override_gate[-1].bias)
        self.register_buffer("goal_gate_threshold", torch.tensor(0.5))

    def override_gate_logits(
        self, observations: torch.Tensor,
        base_family_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if base_family_logits is None:
            base = GraphFamilyHierarchicalCandidateActorCritic.forward_actor(
                self, observations
            )
            base_family_logits = base[:, ACTION_SPACE_SIZE:]
        return self.goal_override_gate(torch.cat((
            observations[:, BASE_OBSERVATION_SIZE:],
            base_family_logits.detach(),
        ), dim=1)).squeeze(1)

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        # Goal親クラスの常時補正は使わず、その一つ上のGraph Familyまでを土台にする。
        logits = GraphFamilyHierarchicalCandidateActorCritic.forward_actor(
            self, observations
        )
        goal = observations[:, BASE_OBSERVATION_SIZE:]
        probability = torch.sigmoid(self.override_gate_logits(
            observations, logits[:, ACTION_SPACE_SIZE:]
        ))
        gate = (probability if self.training else
                (probability >= self.goal_gate_threshold).to(probability.dtype))[:, None]
        candidates = logits[:, :ACTION_SPACE_SIZE] + gate * self.goal_action_head(goal)
        families = logits[:, ACTION_SPACE_SIZE:] + gate * self.goal_family_head(goal)
        return torch.cat((candidates, families), dim=1)


class GatedGoalGraphFamilyHierarchicalCandidateMaskablePolicy(
    GoalGraphFamilyHierarchicalCandidateMaskablePolicy
):
    """目標残差の適用可否も教師ありで学ぶv3方策。"""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GatedGoalGraphFamilyHierarchicalCandidateActorCritic(
            self.features_dim
        )


class HeteroGraphMessageLayer(nn.Module):
    """交差点・辺・タイル間を双方向に伝播する異種Graph層。"""

    def __init__(self, feature_size: int = 64):
        super().__init__()
        self.vertex_update = nn.Sequential(
            nn.Linear(feature_size * 3, feature_size), nn.ReLU(),
            nn.Linear(feature_size, feature_size),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size), nn.ReLU(),
            nn.Linear(feature_size, feature_size),
        )
        self.tile_update = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size), nn.ReLU(),
            nn.Linear(feature_size, feature_size),
        )
        self.vertex_norm = nn.LayerNorm(feature_size)
        self.edge_norm = nn.LayerNorm(feature_size)
        self.tile_norm = nn.LayerNorm(feature_size)

    def forward(
        self,
        vertices: torch.Tensor,
        edges: torch.Tensor,
        tiles: torch.Tensor,
        vertex_edge_incidence: torch.Tensor,
        vertex_tile_incidence: torch.Tensor,
        edge_vertices: torch.Tensor,
        tile_vertices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_messages = torch.einsum("ve,bef->bvf", vertex_edge_incidence, edges)
        tile_messages = torch.einsum("vt,btf->bvf", vertex_tile_incidence, tiles)
        vertex_delta = self.vertex_update(torch.cat(
            (vertices, edge_messages, tile_messages), dim=2
        ))
        endpoint_messages = vertices[:, edge_vertices].mean(dim=2)
        edge_delta = self.edge_update(torch.cat((edges, endpoint_messages), dim=2))
        corner_messages = vertices[:, tile_vertices].mean(dim=2)
        tile_delta = self.tile_update(torch.cat((tiles, corner_messages), dim=2))
        return (
            torch.relu(self.vertex_norm(vertices + vertex_delta)),
            torch.relu(self.edge_norm(edges + edge_delta)),
            torch.relu(self.tile_norm(tiles + tile_delta)),
        )


class HeteroGraphHierarchicalCandidateActorCritic(GraphHierarchicalCandidateActorCritic):
    """現行GNNの上にTile/Vertex/Edge異種Graph残差を追加する。"""

    def __init__(self, observation_size: int):
        super().__init__(observation_size)
        topology = create_topology()
        vertex_edge = torch.zeros((54, 72), dtype=torch.float32)
        for edge in topology.edges:
            for vertex_id in edge.vertex_ids:
                vertex_edge[vertex_id, edge.id] = 1.0
        vertex_edge /= vertex_edge.sum(dim=1, keepdim=True).clamp_min(1.0)
        vertex_tile = torch.zeros((54, 19), dtype=torch.float32)
        for tile_id, vertex_ids in enumerate(self.tile_vertices.tolist()):
            for vertex_id in vertex_ids:
                vertex_tile[vertex_id, tile_id] = 1.0
        vertex_tile /= vertex_tile.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("hetero_vertex_edge_incidence", vertex_edge)
        self.register_buffer("hetero_vertex_tile_incidence", vertex_tile)
        self.hetero_edge_projection = nn.Sequential(nn.Linear(192, 64), nn.ReLU())
        self.hetero_tile_projection = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.hetero_layers = nn.ModuleList([
            HeteroGraphMessageLayer(64),
            HeteroGraphMessageLayer(64),
        ])
        self.hetero_settlement_head = nn.Linear(64, 1)
        self.hetero_city_head = nn.Linear(64, 1)
        self.hetero_road_head = nn.Sequential(
            nn.Linear(64 * 3, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.hetero_robber_head = nn.Sequential(
            nn.Linear(64 * 2, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        for head in (
            self.hetero_settlement_head,
            self.hetero_city_head,
            self.hetero_road_head[-1],
            self.hetero_robber_head[-1],
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _hetero_states(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vertices, graph_edges, graph_tiles = self._encoded_spatial_states(observations)
        edges = self.hetero_edge_projection(graph_edges)
        tiles = self.hetero_tile_projection(graph_tiles)
        for layer in self.hetero_layers:
            vertices, edges, tiles = layer(
                vertices, edges, tiles,
                self.hetero_vertex_edge_incidence,
                self.hetero_vertex_tile_incidence,
                self.edge_vertices,
                self.tile_vertices,
            )
        return vertices, edges, tiles

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        base_logits = super().forward_actor(observations)
        candidate_logits = base_logits[:, :ACTION_SPACE_SIZE]
        family_logits = base_logits[:, ACTION_SPACE_SIZE:]
        vertices, edges, tiles = self._hetero_states(observations)
        correction = torch.zeros_like(candidate_logits)
        _, _, robber, road, settlement, city = SPATIAL_BLOCKS
        edge_endpoints = vertices[:, self.edge_vertices]
        edge_state = torch.cat((
            edges, edge_endpoints.mean(dim=2), edge_endpoints.amax(dim=2)
        ), dim=2)
        tile_state = torch.cat((tiles, vertices[:, self.tile_vertices].mean(dim=2)), dim=2)
        correction[:, robber[0]:robber[0] + robber[1]] = (
            self.hetero_robber_head(tile_state).squeeze(-1)
        )
        correction[:, road[0]:road[0] + road[1]] = (
            self.hetero_road_head(edge_state).squeeze(-1)
        )
        correction[:, settlement[0]:settlement[0] + settlement[1]] = (
            self.hetero_settlement_head(vertices).squeeze(-1)
        )
        correction[:, city[0]:city[0] + city[1]] = (
            self.hetero_city_head(vertices).squeeze(-1)
        )
        return torch.cat((candidate_logits + correction, family_logits), dim=1)


class HeteroGraphHierarchicalCandidateMaskablePolicy(GraphHierarchicalCandidateMaskablePolicy):
    """現行GNNへ異種Graph残差を追加したv2方策。"""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HeteroGraphHierarchicalCandidateActorCritic(self.features_dim)


class ExpansionGraphHierarchicalCandidateActorCritic(
    GraphHierarchicalCandidateActorCritic
):
    """資源補完・港・接続先を明示し、拡張行動だけを残差補正するGNN。"""

    _EXPANSION_FAMILIES = (
        ACTION_FAMILY_INDEX["road"],
        ACTION_FAMILY_INDEX["settlement"],
        ACTION_FAMILY_INDEX["development_purchase"],
        ACTION_FAMILY_INDEX["end_turn"],
    )

    def __init__(self, observation_size: int):
        super().__init__(observation_size)
        topology = create_topology()
        vertex_port = torch.zeros((54, 9), dtype=torch.float32)
        # PortObservationはboard.portsの順で9件。各港の両端へ同じ公開特徴を渡す。
        from app.domain.board import FIXED_PORTS
        edge_by_vertices = {
            tuple(sorted(edge.vertex_ids)): edge.id for edge in topology.edges
        }
        port_edges = [edge_by_vertices[tuple(sorted(vertices))]
                      for vertices, _ in FIXED_PORTS]
        for port_id, edge_id in enumerate(port_edges):
            for vertex_id in topology.edges[edge_id].vertex_ids:
                vertex_port[vertex_id, port_id] = 1.0
        self.register_buffer("expansion_vertex_port_incidence", vertex_port)

        # graph vertex 64 + raw vertex 14 + port 8 + self resource 5
        # + self production 5 + global context 64 = 160
        self.expansion_vertex_encoder = nn.Sequential(
            nn.Linear(160, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU(),
        )
        self.expansion_graph_layers = nn.ModuleList([
            ResidualGraphMessageLayer(64), ResidualGraphMessageLayer(64),
        ])
        self.expansion_edge_encoder = nn.Sequential(
            nn.Linear(64 * 3 + 5 + 64, 96), nn.ReLU(),
            nn.Linear(96, 64), nn.ReLU(),
        )
        self.expansion_settlement_head = nn.Linear(64, 1)
        self.expansion_road_head = nn.Linear(64, 1)
        # vertex mean/max + edge mean + context + 資源保有フラグ5 + 拠点数2
        self.expansion_family_head = nn.Sequential(
            nn.Linear(128 + 64 + 64 + 5 + 2, 96), nn.ReLU(),
            nn.Linear(96, len(self._EXPANSION_FAMILIES)),
        )
        for head in (
            self.expansion_settlement_head,
            self.expansion_road_head,
            self.expansion_family_head[-1],
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _expansion_states(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = observations.shape[0]
        raw_vertices = observations[:, VERTEX_START:EDGE_START].reshape(batch, 54, 14)
        raw_edges = observations[:, EDGE_START:PORT_START].reshape(batch, 72, 5)
        ports = observations[:, PORT_START:PROGRESS_START].reshape(batch, 9, 8)
        graph_vertices, _, _ = self._encoded_spatial_states(observations)
        own_resources = observations[:, :5]
        own_mask = raw_vertices[:, :, 1:2]
        own_production = (own_mask * raw_vertices[:, :, 9:14]).sum(dim=1).clamp(0, 1)
        global_values = torch.cat((observations[:, :TILE_START],
                                   observations[:, PROGRESS_START:], own_production), dim=1)
        context = self.context(global_values)
        vertex_ports = torch.einsum(
            "vp,bpf->bvf", self.expansion_vertex_port_incidence, ports
        )
        vertices = self.expansion_vertex_encoder(torch.cat((
            graph_vertices,
            raw_vertices,
            vertex_ports,
            own_resources[:, None, :].expand(-1, 54, -1),
            own_production[:, None, :].expand(-1, 54, -1),
            context[:, None, :].expand(-1, 54, -1),
        ), dim=2))
        for layer in self.expansion_graph_layers:
            vertices = layer(vertices, self.vertex_adjacency)
        endpoints = vertices[:, self.edge_vertices]
        edges = self.expansion_edge_encoder(torch.cat((
            endpoints.mean(dim=2), endpoints.amax(dim=2),
            # 方向ごとの候補比較に、両端の差も残す。
            torch.abs(endpoints[:, :, 0] - endpoints[:, :, 1]),
            raw_edges,
            context[:, None, :].expand(-1, 72, -1),
        ), dim=2))
        return vertices, edges, context, own_resources

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        base_logits = super().forward_actor(observations)
        candidate_logits = base_logits[:, :ACTION_SPACE_SIZE]
        family_logits = base_logits[:, ACTION_SPACE_SIZE:]
        vertices, edges, context, own_resources = self._expansion_states(observations)

        correction = torch.zeros_like(candidate_logits)
        _, _, _, road, settlement, _ = SPATIAL_BLOCKS
        correction[:, road[0]:road[0] + road[1]] = (
            self.expansion_road_head(edges).squeeze(-1)
        )
        correction[:, settlement[0]:settlement[0] + settlement[1]] = (
            self.expansion_settlement_head(vertices).squeeze(-1)
        )

        # 0/1の資源保有と拠点数を明示し、資源温存と3拠点目をFamily判断へ渡す。
        resource_presence = (own_resources > 0).to(observations.dtype)
        site_counts = observations[:, 17:19]
        family_input = torch.cat((
            vertices.mean(dim=1), vertices.amax(dim=1), edges.mean(dim=1),
            context, resource_presence, site_counts,
        ), dim=1)
        expansion_family = self.expansion_family_head(family_input)
        family_correction = torch.zeros_like(family_logits)
        for column, family_id in enumerate(self._EXPANSION_FAMILIES):
            family_correction[:, family_id] = expansion_family[:, column]
        return torch.cat((candidate_logits + correction,
                          family_logits + family_correction), dim=1)


class ExpansionGraphHierarchicalCandidateMaskablePolicy(
    GraphHierarchicalCandidateMaskablePolicy
):
    """現行GNNへ拡張戦略用の候補・Family残差を追加した方策。"""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ExpansionGraphHierarchicalCandidateActorCritic(
            self.features_dim
        )


def initialize_graph_policy_from_hierarchical(
    graph_policy: GraphHierarchicalCandidateMaskablePolicy,
    source_policy: HierarchicalCandidateMaskablePolicy,
) -> tuple[str, ...]:
    """既存階層方策の互換パラメータをGraph方策へ移し、追加キーを返す。"""
    incompatible = graph_policy.load_state_dict(source_policy.state_dict(), strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"移行元に未知のパラメータがあります: {incompatible.unexpected_keys}")
    allowed_prefixes = (
        "mlp_extractor.vertex_adjacency",
        "mlp_extractor.graph_layers.",
        "mlp_extractor.graph_family_head.",
        "mlp_extractor.graph_initial_settlement_head.",
        "mlp_extractor.graph_settlement_head.",
        "mlp_extractor.graph_city_head.",
        "mlp_extractor.graph_initial_road_head.",
        "mlp_extractor.graph_road_head.",
        "mlp_extractor.graph_robber_head.",
    )
    unknown_missing = [key for key in incompatible.missing_keys
                       if not key.startswith(allowed_prefixes)]
    if unknown_missing:
        raise ValueError(f"移行できない既存パラメータがあります: {unknown_missing}")
    return tuple(incompatible.missing_keys)


def initialize_goal_policy_from_graph(
    goal_policy: GoalGraphFamilyHierarchicalCandidateMaskablePolicy,
    source_policy: GraphHierarchicalCandidateMaskablePolicy,
) -> tuple[str, ...]:
    """v2現行GNNをv3目標方策へ移し、追加直後の全logitを一致させる。"""
    source = source_policy.state_dict()
    target = goal_policy.state_dict()
    copied: list[str] = []
    expanded = {
        "mlp_extractor.context.0.weight",
        "mlp_extractor.family_head.0.weight",
        "mlp_extractor.critic.0.weight",
    }
    for key, value in source.items():
        if key not in target:
            raise ValueError(f"移行先に既存パラメータがありません: {key}")
        if target[key].shape == value.shape:
            target[key].copy_(value)
            copied.append(key)
        elif key in expanded and target[key].shape[0] == value.shape[0]:
            # v3はv2ベクトルの末尾へ目標特徴を追加するため、既存列をそのまま保てる。
            target[key].zero_()
            target[key][:, :value.shape[1]].copy_(value)
            copied.append(key)
        else:
            raise ValueError(
                f"目標方策へ移行できないshapeです: {key} {value.shape}->{target[key].shape}"
            )
    goal_policy.load_state_dict(target)
    return tuple(key for key in target if key not in copied)


def initialize_belief_policy_from_graph(
    belief_policy: (BeliefGraphFamilyHierarchicalCandidateMaskablePolicy
                    | ContextualBeliefGraphFamilyHierarchicalCandidateMaskablePolicy
                    | RobberBeliefGraphFamilyHierarchicalCandidateMaskablePolicy),
    source_policy: GraphHierarchicalCandidateMaskablePolicy,
) -> tuple[str, ...]:
    """v2現行GNNをv4手札推定方策へ移し、初期logitを完全一致させる。"""
    source = source_policy.state_dict()
    target = belief_policy.state_dict()
    copied: list[str] = []
    expanded = {
        "mlp_extractor.context.0.weight",
        "mlp_extractor.family_head.0.weight",
        "mlp_extractor.critic.0.weight",
    }
    for key, value in source.items():
        if key not in target:
            raise ValueError(f"移行先に既存パラメータがありません: {key}")
        if target[key].shape == value.shape:
            target[key].copy_(value)
            copied.append(key)
        elif key in expanded and target[key].shape[0] == value.shape[0]:
            target[key].zero_()
            target[key][:, :value.shape[1]].copy_(value)
            copied.append(key)
        else:
            raise ValueError(
                f"手札推定方策へ移行できないshapeです: "
                f"{key} {value.shape}->{target[key].shape}"
            )
    belief_policy.load_state_dict(target)
    return tuple(key for key in target if key not in copied)


def initialize_hetero_policy_from_graph(
    hetero_policy: HeteroGraphHierarchicalCandidateMaskablePolicy,
    source_policy: GraphHierarchicalCandidateMaskablePolicy,
) -> tuple[str, ...]:
    """現行GNNを異種GNN v2へ移し、追加直後のlogit一致を維持する。"""
    incompatible = hetero_policy.load_state_dict(source_policy.state_dict(), strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"移行元に未知のパラメータがあります: {incompatible.unexpected_keys}")
    allowed_prefixes = (
        "mlp_extractor.hetero_vertex_edge_incidence",
        "mlp_extractor.hetero_vertex_tile_incidence",
        "mlp_extractor.hetero_edge_projection.",
        "mlp_extractor.hetero_tile_projection.",
        "mlp_extractor.hetero_layers.",
        "mlp_extractor.hetero_settlement_head.",
        "mlp_extractor.hetero_city_head.",
        "mlp_extractor.hetero_road_head.",
        "mlp_extractor.hetero_robber_head.",
    )
    unknown = [key for key in incompatible.missing_keys
               if not key.startswith(allowed_prefixes)]
    if unknown:
        raise ValueError(f"異種GNNへ移行できないパラメータがあります: {unknown}")
    return tuple(incompatible.missing_keys)


def initialize_expansion_policy_from_graph(
    expansion_policy: ExpansionGraphHierarchicalCandidateMaskablePolicy,
    source_policy: GraphHierarchicalCandidateMaskablePolicy,
) -> tuple[str, ...]:
    """現行GNNを拡張GNNへ移し、追加直後のlogitを完全一致させる。"""
    incompatible = expansion_policy.load_state_dict(
        source_policy.state_dict(), strict=False
    )
    if incompatible.unexpected_keys:
        raise ValueError(
            f"移行元に未知のパラメータがあります: {incompatible.unexpected_keys}"
        )
    allowed_prefixes = (
        "mlp_extractor.expansion_vertex_port_incidence",
        "mlp_extractor.expansion_vertex_encoder.",
        "mlp_extractor.expansion_graph_layers.",
        "mlp_extractor.expansion_edge_encoder.",
        "mlp_extractor.expansion_settlement_head.",
        "mlp_extractor.expansion_road_head.",
        "mlp_extractor.expansion_family_head.",
    )
    unknown = [key for key in incompatible.missing_keys
               if not key.startswith(allowed_prefixes)]
    if unknown:
        raise ValueError(f"拡張GNNへ移行できないパラメータがあります: {unknown}")
    return tuple(incompatible.missing_keys)


def configure_graph_residual_finetune(
    policy: GraphHierarchicalCandidateMaskablePolicy,
) -> dict[str, int]:
    """既存Actorを固定し、Graph補正とValue推定だけを学習対象にする。"""
    for parameter in policy.parameters():
        parameter.requires_grad = False
    trainable_modules = (
        policy.mlp_extractor.graph_layers,
        policy.mlp_extractor.graph_initial_settlement_head,
        policy.mlp_extractor.graph_settlement_head,
        policy.mlp_extractor.graph_city_head,
        policy.mlp_extractor.graph_initial_road_head,
        policy.mlp_extractor.graph_road_head,
        policy.mlp_extractor.graph_robber_head,
        policy.mlp_extractor.critic,
        policy.value_net,
    )
    family_head = getattr(policy.mlp_extractor, "graph_family_head", None)
    if family_head is not None:
        trainable_modules += (family_head,)
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in policy.parameters()
                    if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in policy.parameters()
                 if not parameter.requires_grad)
    if trainable <= 0 or frozen <= 0:
        raise AssertionError("Graph限定学習のパラメータ分離に失敗しました。")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def configure_graph_family_pretrain(
    policy: GraphFamilyHierarchicalCandidateMaskablePolicy,
) -> dict[str, int]:
    """教師あり事前学習ではGraph EncoderとFamily残差Headだけを更新する。"""
    for parameter in policy.parameters():
        parameter.requires_grad = False
    for module in (policy.mlp_extractor.graph_layers,
                   policy.mlp_extractor.graph_family_head):
        for parameter in module.parameters():
            parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in policy.parameters()
                    if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in policy.parameters()
                 if not parameter.requires_grad)
    if trainable <= 0 or frozen <= 0:
        raise AssertionError("Graph Family事前学習のパラメータ分離に失敗しました。")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def configure_graph_candidate_pretrain(
    policy: GraphHierarchicalCandidateMaskablePolicy,
) -> dict[str, int]:
    """Family選択を固定し、通常手番のGraph候補比較だけを更新する。"""
    for parameter in policy.parameters():
        parameter.requires_grad = False
    for module in (
        policy.mlp_extractor.graph_layers,
        policy.mlp_extractor.graph_robber_head,
        policy.mlp_extractor.graph_road_head,
        policy.mlp_extractor.graph_settlement_head,
        policy.mlp_extractor.graph_city_head,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in policy.parameters()
                    if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in policy.parameters()
                 if not parameter.requires_grad)
    if trainable <= 0 or frozen <= 0:
        raise AssertionError("Graph候補事前学習のパラメータ分離に失敗しました。")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def configure_hetero_graph_finetune(
    policy: HeteroGraphHierarchicalCandidateMaskablePolicy,
) -> dict[str, int]:
    """現行Actorを固定し、異種Graph残差とValue推定だけをPPO更新する。"""
    for parameter in policy.parameters():
        parameter.requires_grad = False
    for module in (
        policy.mlp_extractor.hetero_edge_projection,
        policy.mlp_extractor.hetero_tile_projection,
        policy.mlp_extractor.hetero_layers,
        policy.mlp_extractor.hetero_settlement_head,
        policy.mlp_extractor.hetero_city_head,
        policy.mlp_extractor.hetero_road_head,
        policy.mlp_extractor.hetero_robber_head,
        policy.mlp_extractor.critic,
        policy.value_net,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in policy.parameters()
                    if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in policy.parameters()
                 if not parameter.requires_grad)
    if trainable <= 0 or frozen <= 0:
        raise AssertionError("異種GNN限定学習のパラメータ分離に失敗しました。")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def configure_expansion_graph_finetune(
    policy: ExpansionGraphHierarchicalCandidateMaskablePolicy,
) -> dict[str, int]:
    """現行Actorを守り、拡張戦略残差とValue推定だけを更新する。"""
    for parameter in policy.parameters():
        parameter.requires_grad = False
    for module in (
        policy.mlp_extractor.expansion_vertex_encoder,
        policy.mlp_extractor.expansion_graph_layers,
        policy.mlp_extractor.expansion_edge_encoder,
        policy.mlp_extractor.expansion_settlement_head,
        policy.mlp_extractor.expansion_road_head,
        policy.mlp_extractor.expansion_family_head,
        policy.mlp_extractor.critic,
        policy.value_net,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in policy.parameters()
                    if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in policy.parameters()
                 if not parameter.requires_grad)
    if trainable <= 0 or frozen <= 0:
        raise AssertionError("拡張GNN限定学習のパラメータ分離に失敗しました。")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def configure_expansion_graph_pretrain(
    policy: ExpansionGraphHierarchicalCandidateMaskablePolicy,
) -> dict[str, int]:
    """既存PPOを教師にするKL事前調整用の、拡張残差だけを選択する。

    Criticや既存Actorを更新しないため、騎士・盗賊・取引などの既存方針を
    目標分布へ直接合わせ直すことはない。
    """
    for parameter in policy.parameters():
        parameter.requires_grad = False
    for module in (
        policy.mlp_extractor.expansion_vertex_encoder,
        policy.mlp_extractor.expansion_graph_layers,
        policy.mlp_extractor.expansion_edge_encoder,
        policy.mlp_extractor.expansion_settlement_head,
        policy.mlp_extractor.expansion_road_head,
        policy.mlp_extractor.expansion_family_head,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in policy.parameters()
                    if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in policy.parameters()
                 if not parameter.requires_grad)
    if trainable <= 0 or frozen <= 0:
        raise AssertionError("拡張GNN事前調整のパラメータ分離に失敗しました。")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}
