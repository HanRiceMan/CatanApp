"""最初のMaskablePPO学習と、モデル・設定・評価結果の保存。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from time import monotonic
from typing import Any

from app.domain.board import BoardRules

from .env import CatanEnv, CatanEnvConfig
from .evaluate import EvaluationConfig, EvaluationReport, evaluate_agent
from .model_registry import ModelManifest, ModelRegistry
from .observation import OBSERVATION_VERSION, OBSERVATION_VECTOR_SIZES
from .reward import RewardConfig


CANDIDATE_POLICY_ARCHITECTURES = frozenset({
    "candidate", "hierarchical_candidate", "gnn_hierarchical_candidate",
})


@dataclass(frozen=True)
class TrainConfig:
    model_id: str
    total_timesteps: int = 100_000
    seed: int = 42
    learning_player_id: int = 1
    opponent_controller: str = "heuristic"
    heuristic_opponents: int | None = None
    heuristic_initial_placement: bool = False
    observation_version: str = OBSERVATION_VERSION
    policy_architecture: str = "mlp"
    pretrain_initial_settlement: bool = False
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    policy_net_arch: tuple[int, ...] = (256, 128)
    device: str = "auto"
    board_rules: BoardRules = field(default_factory=BoardRules)
    reward: RewardConfig = field(default_factory=RewardConfig)
    evaluation_seeds: tuple[int, ...] = (1001, 1002, 1003, 1004, 1005)
    checkpoint_interval_steps: int = 25_000
    resume_checkpoint_steps: int | None = None
    resume_from_model_id: str | None = None
    models_root: Path | None = None


@dataclass(frozen=True)
class TrainingResult:
    manifest: ModelManifest
    evaluation: EvaluationReport
    model_directory: Path


def _default_models_root() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _validate_config(config: TrainConfig) -> None:
    if Path(config.model_id).name != config.model_id:
        raise ValueError("model_idが不正です。")
    if config.total_timesteps <= 0 or config.n_steps <= 0 or config.batch_size <= 0 or config.n_epochs <= 0:
        raise ValueError("学習step・batch_size・epochは正の整数にしてください。")
    if config.batch_size > config.n_steps:
        raise ValueError("batch_sizeはn_steps以下にしてください。")
    if config.opponent_controller not in {"heuristic", "random"}:
        raise ValueError("opponent_controllerはheuristicまたはrandomです。")
    if config.heuristic_opponents is not None and config.heuristic_opponents not in range(4):
        raise ValueError("heuristic_opponentsは0〜3です。")
    if config.observation_version not in OBSERVATION_VECTOR_SIZES:
        raise ValueError(f"未対応のObservation版です: {config.observation_version}")
    if config.policy_architecture not in {"mlp", *CANDIDATE_POLICY_ARCHITECTURES}:
        raise ValueError("policy_architectureはmlp、candidate、hierarchical_candidate、"
                         "gnn_hierarchical_candidateです。")
    if config.policy_architecture in CANDIDATE_POLICY_ARCHITECTURES and config.observation_version != "v2":
        raise ValueError("候補方式の方策にはObservation v2が必要です。")
    if config.pretrain_initial_settlement and config.policy_architecture != "candidate":
        raise ValueError("初期開拓地の事前学習は候補方式のみ対応です。")
    if config.pretrain_initial_settlement and config.resume_checkpoint_steps is not None:
        raise ValueError("チェックポイント再開時に事前学習を繰り返すことはできません。")
    if not config.evaluation_seeds:
        raise ValueError("評価seedを1つ以上指定してください。")
    if config.checkpoint_interval_steps <= 0:
        raise ValueError("checkpoint_interval_stepsは正の整数にしてください。")
    if config.resume_checkpoint_steps is not None and config.resume_checkpoint_steps <= 0:
        raise ValueError("resume_checkpoint_stepsは正の整数にしてください。")
    if config.resume_from_model_id is not None and (
        Path(config.resume_from_model_id).name != config.resume_from_model_id
        or config.resume_checkpoint_steps is None
    ):
        raise ValueError("別モデルから再開するには安全なモデルIDとチェックポイントstep数を指定してください。")


def _manifest(config: TrainConfig, evaluation: dict[str, Any], training_steps: int) -> ModelManifest:
    from .action_space import ACTION_SPACE_SIZE, ACTION_SPACE_VERSION
    return ModelManifest(
        model_id=config.model_id,
        algorithm="MaskablePPO",
        artifact_filename="model.zip",
        observation_version=config.observation_version,
        observation_size=OBSERVATION_VECTOR_SIZES[config.observation_version],
        action_space_version=ACTION_SPACE_VERSION,
        action_space_size=ACTION_SPACE_SIZE,
        training_steps=training_steps,
        training_seed=config.seed,
        created_at=datetime.now(timezone.utc).isoformat(),
        evaluation=evaluation,
    )


def _config_dict(config: TrainConfig) -> dict[str, Any]:
    result = asdict(config)
    result["models_root"] = str(config.models_root) if config.models_root is not None else None
    return result


def train_maskable_ppo(config: TrainConfig) -> TrainingResult:
    """PPOを学習し、UI互換性を検証可能なローカルモデルとして公開する。"""
    _validate_config(config)
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as error:
        raise RuntimeError("PPO学習にはrequirements-rl.txtの仮想環境が必要です。") from error
    from .candidate_policy import CandidateMaskablePolicy

    models_root = config.models_root or _default_models_root()
    model_directory = models_root / config.model_id
    continuing_same_directory = config.resume_checkpoint_steps is not None and config.resume_from_model_id is None
    if model_directory.exists() and not continuing_same_directory:
        raise FileExistsError(f"同名のモデルディレクトリが既にあります: {model_directory}")
    checkpoint = None
    if config.resume_checkpoint_steps is not None:
        source_directory = (models_root / config.resume_from_model_id
                            if config.resume_from_model_id else model_directory)
        checkpoint = source_directory / f"checkpoint_{config.resume_checkpoint_steps}_steps.zip"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"再開用チェックポイントが見つかりません: {checkpoint}")
        if continuing_same_directory and (model_directory / "model.zip").exists():
            raise FileExistsError("完成済みモデルのディレクトリからは再開できません。")
    if not model_directory.exists():
        model_directory.mkdir(parents=True)
    tensorboard_directory = model_directory / "tensorboard"

    class ProgressCheckpointCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.last_checkpoint = config.resume_checkpoint_steps or 0
            self.started_at = monotonic()

        def _on_step(self) -> bool:
            if self.num_timesteps - self.last_checkpoint >= config.checkpoint_interval_steps:
                self.model.save(str(model_directory / f"checkpoint_{self.num_timesteps}_steps.zip"))
                self.last_checkpoint = self.num_timesteps
                elapsed = monotonic() - self.started_at
                current_run_steps = self.num_timesteps - (config.resume_checkpoint_steps or 0)
                print(f"学習中: {self.num_timesteps:,}/{config.total_timesteps:,} steps "
                      f"({elapsed:.0f}秒、{current_run_steps / max(elapsed, 1):.1f} steps/秒)", flush=True)
            return True

    environment = CatanEnv(CatanEnvConfig(
        learning_player_id=config.learning_player_id,
        opponent_controller=config.opponent_controller,
        heuristic_opponents=config.heuristic_opponents,
        heuristic_initial_placement=config.heuristic_initial_placement,
        observation_version=config.observation_version,
        board_rules=config.board_rules,
        reward=config.reward,
    ))
    if checkpoint is not None:
        model = MaskablePPO.load(str(checkpoint), env=environment, device=config.device,
                                 tensorboard_log=str(tensorboard_directory))
        if model.num_timesteps != config.resume_checkpoint_steps:
            raise ValueError("チェックポイントの学習step数がファイル名と一致しません。")
        from .candidate_policy import (GraphHierarchicalCandidateMaskablePolicy,
                                       HierarchicalCandidateMaskablePolicy)
        expected_policy = {"candidate": CandidateMaskablePolicy,
                           "hierarchical_candidate": HierarchicalCandidateMaskablePolicy,
                           "gnn_hierarchical_candidate": GraphHierarchicalCandidateMaskablePolicy}.get(
                               config.policy_architecture)
        architecture_matches = ((expected_policy is None and not isinstance(
            model.policy, (CandidateMaskablePolicy, HierarchicalCandidateMaskablePolicy,
                           GraphHierarchicalCandidateMaskablePolicy)))
            or (expected_policy is not None and type(model.policy) is expected_policy))
        net_arch_matches = (config.policy_architecture in CANDIDATE_POLICY_ARCHITECTURES or
                            list(model.policy.net_arch["pi"] if isinstance(model.policy.net_arch, dict)
                                 else model.policy.net_arch) == list(config.policy_net_arch))
        if (not architecture_matches or not net_arch_matches
                or model.n_steps != config.n_steps or model.batch_size != config.batch_size
                or model.n_epochs != config.n_epochs or model.gamma != config.gamma
                or model.gae_lambda != config.gae_lambda or model.ent_coef != config.ent_coef
                or model.learning_rate != config.learning_rate):
            raise ValueError("指定ハイパーパラメータが再開元チェックポイントと一致しません。")
        if model.num_timesteps >= config.total_timesteps:
            raise ValueError("再開目標step数はチェックポイントより大きくしてください。")
        # load() は保存時の seed でモデルを初期化する。継続実験の seed を
        # Python / NumPy / PyTorch / Environment 全てへ適用し直す。
        model.set_random_seed(config.seed)
        model.learn(total_timesteps=config.total_timesteps - model.num_timesteps,
                    reset_num_timesteps=False, callback=ProgressCheckpointCallback(), progress_bar=False)
    else:
        from .candidate_policy import (GraphHierarchicalCandidateMaskablePolicy,
                                       HierarchicalCandidateMaskablePolicy)
        policy = {"candidate": CandidateMaskablePolicy,
                  "hierarchical_candidate": HierarchicalCandidateMaskablePolicy,
                  "gnn_hierarchical_candidate": GraphHierarchicalCandidateMaskablePolicy}.get(
                      config.policy_architecture, "MlpPolicy")
        policy_kwargs = ({"net_arch": [], "ortho_init": False}
                         if config.policy_architecture in CANDIDATE_POLICY_ARCHITECTURES
                         else {"net_arch": list(config.policy_net_arch)})
        model = MaskablePPO(
            policy,
            environment,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tensorboard_directory),
            seed=config.seed,
            device=config.device,
            verbose=0,
        )
        if config.pretrain_initial_settlement:
            from .pretrain_candidate_setup import pretrain_initial_settlement
            pretraining = pretrain_initial_settlement(model.policy)
            _json_dump(model_directory / "pretraining.json", pretraining)
            print(f"初期配置の事前学習: test教師一致率 "
                  f"{pretraining['after_test']['teacher_agreement']:.1%}", flush=True)
            # 教師ありデータ収集・更新が使った乱数状態をPPO開始前に揃え直す。
            model.set_random_seed(config.seed)
        model.learn(total_timesteps=config.total_timesteps, callback=ProgressCheckpointCallback(), progress_bar=False)
    model.save(str(model_directory / "model.zip"))
    environment.close()

    # Registryを通せる状態にしてから、PPOAgentで評価する。
    preliminary = _manifest(config, evaluation={}, training_steps=model.num_timesteps)
    _json_dump(model_directory / "metadata.json", preliminary.to_dict())
    _json_dump(model_directory / "config.json", _config_dict(config))
    from app.agents.ppo import PPOAgent
    evaluation = evaluate_agent(
        PPOAgent(config.model_id, models_root=models_root),
        config.evaluation_seeds,
        config=EvaluationConfig(
            learning_player_id=config.learning_player_id,
            opponent_controller=config.opponent_controller,
            heuristic_opponents=config.heuristic_opponents,
            board_rules=config.board_rules,
            reward=config.reward,
            observation_version=config.observation_version,
        ),
        agent_name=f"PPOAgent:{config.model_id}",
    )
    manifest = _manifest(config, evaluation=evaluation.to_dict(), training_steps=model.num_timesteps)
    _json_dump(model_directory / "metadata.json", manifest.to_dict())
    # 最終的にRegistryで実行時互換性を検証する。
    ModelRegistry(models_root).load(config.model_id)
    return TrainingResult(manifest, evaluation, model_directory)


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Catan MaskablePPOを学習する")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--eval-seed-start", type=int, default=7001)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=25_000)
    parser.add_argument("--resume-checkpoint-steps", type=int, default=None)
    parser.add_argument("--resume-from-model-id", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--heuristic-opponents", type=int, choices=range(4), default=None,
                        help="相手3人のうちルールベースAIの人数。未指定なら従来通り全員heuristic")
    parser.add_argument("--heuristic-initial-placement", action="store_true",
                        help="学習者の初期配置だけHeuristicへ委譲する実験")
    parser.add_argument("--observation-version", choices=tuple(OBSERVATION_VECTOR_SIZES),
                        default=OBSERVATION_VERSION, help="学習・保存するObservation版")
    parser.add_argument("--policy-architecture", choices=("mlp", "candidate", "hierarchical_candidate",
                                                           "gnn_hierarchical_candidate"), default="mlp",
                        help="候補方式はObservation v2と組み合わせる")
    parser.add_argument("--pretrain-initial-settlement", action="store_true",
                        help="候補方式の初期開拓地HeadをHeuristicの選択で事前学習する")
    values = parser.parse_args()
    if values.eval_games <= 0:
        parser.error("--eval-gamesは正の整数にしてください。")
    evaluation_seeds = (tuple(values.eval_seeds) if values.eval_seeds is not None else
                        tuple(range(values.eval_seed_start, values.eval_seed_start + values.eval_games)))
    return TrainConfig(model_id=values.model_id, total_timesteps=values.timesteps, seed=values.seed,
                       heuristic_opponents=values.heuristic_opponents,
                       heuristic_initial_placement=values.heuristic_initial_placement,
                       observation_version=values.observation_version,
                       policy_architecture=values.policy_architecture,
                       pretrain_initial_settlement=values.pretrain_initial_settlement,
                       evaluation_seeds=evaluation_seeds, checkpoint_interval_steps=values.checkpoint_interval,
                       resume_checkpoint_steps=values.resume_checkpoint_steps,
                       resume_from_model_id=values.resume_from_model_id, device=values.device)


if __name__ == "__main__":
    result = train_maskable_ppo(_parse_args())
    print(json.dumps({"model_directory": str(result.model_directory),
                      "evaluation": result.evaluation.to_dict()["summary"]}, ensure_ascii=False, indent=2))
