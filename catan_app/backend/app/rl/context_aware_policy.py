"""Champion v2 GNNを固定したままv7 Contextを受け取る実験方策。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.agents.heuristic import HeuristicAgent
from app.agents.ppo import PPOAgent
from app.domain.actions import Action
from app.domain.game import GameActionError, GameState

from .action_space import ACTION_SPACE_SIZE, get_action_mask, id_to_action
from .candidate_policy import (BASE_OBSERVATION_SIZE,
                               GraphHierarchicalCandidateActorCritic,
                               GraphHierarchicalCandidateMaskablePolicy)
from .env import CatanEnv, CatanEnvConfig
from .hierarchical_distribution import FAMILY_COUNT
from .observation import OBSERVATION_VECTOR_SIZES, encode_observation, get_observation
from .runtime_context import (RUNTIME_CONTEXT_SIZE, RUNTIME_CONTEXT_VERSION,
                              runtime_context_definition)


CONTEXT_ARTIFACT_VERSION = "context_aware_ppo_v1"


def _context_schema_hash() -> str:
    canonical = json.dumps(runtime_context_definition(), ensure_ascii=False,
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ContextAwareGraphActorCritic(GraphHierarchicalCandidateActorCritic):
    """旧GNN/Actor/Criticをv2 sliceで実行し、小残差だけを別枝で追加。"""

    def __init__(self, observation_size: int):
        if observation_size != BASE_OBSERVATION_SIZE + RUNTIME_CONTEXT_SIZE:
            raise ValueError("Context-aware PolicyはObservation v7専用です。")
        super().__init__(BASE_OBSERVATION_SIZE)
        self.context_encoder = nn.Sequential(
            nn.Linear(RUNTIME_CONTEXT_SIZE, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.context_actor_projection = nn.Linear(
            64, ACTION_SPACE_SIZE + FAMILY_COUNT,
        )
        self.context_critic_projection = nn.Linear(64, self.latent_dim_vf)
        for head in (self.context_actor_projection, self.context_critic_projection):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _context_latent(self, observations: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(observations[:, BASE_OBSERVATION_SIZE:])

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        old_logits = super().forward_actor(observations[:, :BASE_OBSERVATION_SIZE])
        return old_logits + self.context_actor_projection(self._context_latent(observations))

    def forward_critic(self, observations: torch.Tensor) -> torch.Tensor:
        old_latent = super().forward_critic(observations[:, :BASE_OBSERVATION_SIZE])
        return old_latent + self.context_critic_projection(self._context_latent(observations))


class ContextAwareGraphMaskablePolicy(GraphHierarchicalCandidateMaskablePolicy):
    """階層Action分布と377 maskを保持したv7専用の実験Policy。"""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ContextAwareGraphActorCritic(self.features_dim)


def initialize_context_policy_from_champion(
    target: ContextAwareGraphMaskablePolicy,
    source: GraphHierarchicalCandidateMaskablePolicy,
) -> tuple[str, ...]:
    """全旧パラメータをbitwiseでコピーし、新残差だけ0初期化のまま残す。"""
    if type(source) is not GraphHierarchicalCandidateMaskablePolicy:
        raise TypeError("移行元は現行Championのv2 Graph方策に限定します。")
    result = target.load_state_dict(source.state_dict(), strict=False)
    if result.unexpected_keys:
        raise ValueError(f"移行元に未知のパラメータ: {result.unexpected_keys}")
    allowed = ("mlp_extractor.context_encoder.",
               "mlp_extractor.context_actor_projection.",
               "mlp_extractor.context_critic_projection.")
    if not result.missing_keys or any(
        not key.startswith(allowed) for key in result.missing_keys
    ):
        raise ValueError(f"既存重みの移行漏れ: {result.missing_keys}")
    for old_name, old_value in source.state_dict().items():
        if not torch.equal(old_value, target.state_dict()[old_name]):
            raise AssertionError(f"Champion重みが変化しました: {old_name}")
    for head in (target.mlp_extractor.context_actor_projection,
                 target.mlp_extractor.context_critic_projection):
        if torch.count_nonzero(head.weight) or torch.count_nonzero(head.bias):
            raise AssertionError("Context残差がzero初期化されていません。")
    return tuple(result.missing_keys)


@dataclass(frozen=True)
class ContextExperimentManifest:
    source_model_id: str
    artifact_version: str = CONTEXT_ARTIFACT_VERSION
    observation_version: str = "v7"
    observation_size: int = OBSERVATION_VECTOR_SIZES["v7"]
    action_space_size: int = ACTION_SPACE_SIZE
    runtime_context_version: str = RUNTIME_CONTEXT_VERSION
    runtime_context_dimension: int = RUNTIME_CONTEXT_SIZE
    runtime_context_schema_sha256: str = field(default_factory=_context_schema_hash)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["policy_class"] = ContextAwareGraphMaskablePolicy.__name__
        return result


class ContextAwarePPOAgent:
    """モデルRegistry/UIに登録しない、Champion比較専用in-memory Agent。"""

    def __init__(self, champion: PPOAgent, model):
        self.champion = champion
        self.model = model
        self.manifest = ContextExperimentManifest(champion.manifest.model_id)
        self.initial_setup_policy = champion.initial_setup_policy
        self.settlement_planning_config = champion.settlement_planning_config
        self.deterministic = champion.deterministic

    def select_action(self, game: GameState, player_id: int) -> Action:
        if game.phase in {"setup_ready", "trade_response", "trade_counter_offer"}:
            # 初期配置専用GNNと履歴依存交渉は今回移管しない。
            return self.champion.select_action(game, player_id)
        observation = encode_observation(get_observation(game, player_id, version="v7"))
        mask = get_action_mask(game, player_id)
        if not mask.any():
            raise GameActionError("Context-aware PPOが操作できる合法Actionがありません。")
        action_id, _ = self.model.predict(
            observation, deterministic=self.deterministic, action_masks=mask,
        )
        selected = int(np.asarray(action_id).item())
        if selected < 0 or selected >= len(mask) or not mask[selected]:
            raise GameActionError("Context-aware PPOが非法Actionを返しました。")
        return id_to_action(selected)


def create_context_aware_experiment_agent(champion: PPOAgent) -> ContextAwarePPOAgent:
    """学習・モデル保存なしでv7方策を構築し、Champion重みを移す。"""
    from sb3_contrib import MaskablePPO

    if champion.manifest.observation_version != "v2":
        raise ValueError("Champion Observationはv2である必要があります。")
    environment = CatanEnv(CatanEnvConfig(observation_version="v7"))
    source = champion.model
    model = MaskablePPO(
        ContextAwareGraphMaskablePolicy, environment,
        learning_rate=source.learning_rate,
        n_steps=source.n_steps, batch_size=source.batch_size,
        n_epochs=source.n_epochs, gamma=source.gamma,
        gae_lambda=source.gae_lambda, ent_coef=source.ent_coef,
        vf_coef=source.vf_coef, max_grad_norm=source.max_grad_norm,
        normalize_advantage=source.normalize_advantage,
        policy_kwargs={"net_arch": [], "ortho_init": False},
        seed=champion.manifest.training_seed, device="cpu", verbose=0,
    )
    initialize_context_policy_from_champion(model.policy, source.policy)
    model.policy.eval()
    return ContextAwarePPOAgent(champion, model)


def save_context_aware_artifact(agent: ContextAwarePPOAgent, directory: Path) -> None:
    """新規ディレクトリだけへ実験artifactを保存。Championには書き込まない。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    agent.model.save(str(directory / "model"))
    (directory / "manifest.json").write_text(
        json.dumps(agent.manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_context_aware_artifact(directory: Path, champion: PPOAgent) -> ContextAwarePPOAgent:
    """parentとschema一致を確認し、v7モデルだけを別artifactから読む。"""
    from sb3_contrib import MaskablePPO

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected = ContextExperimentManifest(champion.manifest.model_id).to_dict()
    if manifest != expected:
        raise ValueError("Context artifactの親モデルまたはschemaが一致しません。")
    model = MaskablePPO.load(
        str(directory / "model.zip"),
        env=CatanEnv(CatanEnvConfig(observation_version="v7")),
        device="cpu",
    )
    if (type(model.policy) is not ContextAwareGraphMaskablePolicy
            or model.observation_space.shape != (OBSERVATION_VECTOR_SIZES["v7"],)
            or model.action_space.n != ACTION_SPACE_SIZE):
        raise ValueError("保存モデルのPolicyまたはAction/Observation次元が不正です。")
    model.policy.eval()
    return ContextAwarePPOAgent(champion, model)
