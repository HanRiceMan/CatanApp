"""強化学習向けのアダプター層。

このパッケージはGame Engineを変更せず、Observation・固定Action Space・
将来のEnvironmentを提供する。PPOやGymnasiumにはまだ依存しない。
"""

from .action_space import (ACTION_SPACE_SIZE, ACTION_SPACE_VERSION, action_to_id,
                           get_action_mask, id_to_action, legal_policy_actions)
from .observation import (OBSERVATION_VECTOR_SIZE, OBSERVATION_VERSION,
                          encode_observation, get_observation)
from .env import (CatanEnv, CatanEnvConfig, ExternalActionRequired, IllegalPolicyAction)
from .evaluate import EvaluationConfig, EvaluationReport, evaluate_agent
from .model_registry import ModelManifest, ModelRegistry
from .reward import RewardConfig

__all__ = [
    "ACTION_SPACE_SIZE", "ACTION_SPACE_VERSION", "OBSERVATION_VECTOR_SIZE",
    "OBSERVATION_VERSION", "CatanEnv", "CatanEnvConfig", "ExternalActionRequired",
    "IllegalPolicyAction", "EvaluationConfig", "EvaluationReport", "ModelManifest", "ModelRegistry",
    "RewardConfig", "action_to_id", "encode_observation", "evaluate_agent", "get_action_mask",
    "get_observation", "id_to_action", "legal_policy_actions",
]
