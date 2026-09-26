"""現行最強ハイブリッドAIを教師・対戦相手にした交渉込みの追加学習。"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from app.agents.heuristic import HeuristicAgent
from app.agents.ppo import PPOAgent
from app.domain.actions import (BuildCityAction, BuildSettlementAction,
                                BuyDevelopmentAction, EndTurnAction,
                                ProposeCounterTradeAction, ProposeTradeAction,
                                RespondToTradeAction)
from app.domain.game import apply_action, create_game, pending_ai_player_id

from .action_space import ACTION_SPACE_SIZE, action_to_id
from .candidate_policy import (GraphHierarchicalCandidateMaskablePolicy,
                               GraphFamilyHierarchicalCandidateMaskablePolicy,
                               BeliefGraphFamilyHierarchicalCandidateMaskablePolicy,
                               ContextualBeliefGraphFamilyHierarchicalCandidateMaskablePolicy,
                               GoalGraphFamilyHierarchicalCandidateMaskablePolicy,
                               GatedGoalGraphFamilyHierarchicalCandidateMaskablePolicy,
                               initialize_belief_policy_from_graph,
                               initialize_goal_policy_from_graph,
                               initialize_graph_policy_from_hierarchical)
from .env import CatanEnv, CatanEnvConfig
from .hierarchical_distribution import ACTION_FAMILY_IDS, FAMILY_COUNT
from .model_registry import ModelManifest
from .observation import OBSERVATION_VECTOR_SIZES
from .reward import actual_score
from .settlement_planning_agent import SettlementPlanningAgent
from .trade_strategy import (choose_counter_trade, choose_proactive_trade,
                             choose_trade_response)


TRADE_ACTIONS = (ProposeTradeAction, RespondToTradeAction, ProposeCounterTradeAction)


class SeededMixedLeagueAgent:
    """75%の局は最強AI 2人＋ルールAI 1人、25%はルールAI 3人にする。"""

    def __init__(self, strongest, *, learning_player_id: int = 1,
                 strong_episode_numerator: int = 3,
                 strong_episode_denominator: int = 4):
        if learning_player_id not in range(1, 5):
            raise ValueError("learning_player_idは1〜4です。")
        if not 0 <= strong_episode_numerator <= strong_episode_denominator:
            raise ValueError("リーグ混合率が不正です。")
        self.strongest = strongest
        self.heuristic = HeuristicAgent()
        self.learning_player_id = learning_player_id
        self.numerator = strong_episode_numerator
        self.denominator = strong_episode_denominator

    def strong_player_ids(self, seed: int) -> frozenset[int]:
        opponents = [pid for pid in range(1, 5) if pid != self.learning_player_id]
        if seed % self.denominator >= self.numerator:
            return frozenset()
        offset = seed % len(opponents)
        rotated = opponents[offset:] + opponents[:offset]
        return frozenset(rotated[:2])

    def select_action(self, game, player_id):
        delegate = (self.strongest if player_id in self.strong_player_ids(game.seed)
                    else self.heuristic)
        return delegate.select_action(game, player_id)


class StrategicTradeAuxiliaryAgent:
    """最強AIと同じ交渉層だけを、PPO推論なしで実行する。"""

    def __init__(self, planning: SettlementPlanningAgent):
        self.planning = planning

    def _options(self) -> dict[str, int | float]:
        return {
            "target_sites": self.planning.target_sites,
            "max_plan_roads": self.planning.max_plan_roads,
            "max_wait_rounds": self.planning.max_wait_rounds,
            "max_city_wait_rounds": self.planning.max_city_wait_rounds,
        }

    def select_action(self, game, player_id):
        options = self._options()
        if game.phase == "action":
            if not self.planning.proactive_player_trade:
                return EndTurnAction()
            return choose_proactive_trade(
                game, player_id,
                max_scarcity_deficit=(
                    self.planning.proactive_trade_max_scarcity_deficit
                ),
                strategic_surplus_trade=self.planning.strategic_surplus_trade,
                opponent_score_limit=(
                    self.planning.proactive_trade_opponent_score_limit
                ),
                **options,
            ) or EndTurnAction()
        if game.phase == "trade_response":
            if not self.planning.strategic_trade_response:
                return HeuristicAgent().select_action(game, player_id)
            return choose_trade_response(
                game, player_id,
                opponent_score_limit=(
                    self.planning.strategic_trade_opponent_score_limit
                ),
                **options,
            )
        if game.phase == "trade_counter_offer":
            counter = choose_counter_trade(game, player_id, **options)
            return counter or HeuristicAgent().select_action(game, player_id)
        return EndTurnAction()


@dataclass(frozen=True)
class TeacherExamples:
    observations: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    summary: dict[str, object]
    override_labels: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def _strongest(model_id: str, models_root: Path, device: str):
    base = PPOAgent(model_id, models_root=models_root, device=device)
    if base.settlement_planning_config is None:
        raise ValueError("現行最強AIにはsettlement_planning設定が必要です。")
    return base, SettlementPlanningAgent(base, **base.settlement_planning_config)


def collect_teacher_examples(teacher, seeds: Iterable[int], *,
                             learning_player_id: int = 1,
                             observation_version: str = "v2",
                             planner_overrides_only: bool = False,
                             include_strategic_actions: bool = False,
                             collect_override_gate: bool = False) -> TeacherExamples:
    league = SeededMixedLeagueAgent(teacher, learning_player_id=learning_player_id)
    opponents = {pid: league for pid in range(1, 5) if pid != learning_player_id}
    environment = CatanEnv(
        CatanEnvConfig(
            learning_player_id=learning_player_id,
            allow_player_trades=True,
            observation_version=observation_version,
        ),
        initial_placement_agent=teacher,
        opponent_agents=opponents,
        learner_auxiliary_agent=StrategicTradeAuxiliaryAgent(teacher),
    )
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    override_labels: list[bool] = []
    families: Counter[str] = Counter()
    auxiliary_actions = 0
    planner_overrides = 0
    override_families: Counter[str] = Counter()
    strategic_additions: Counter[str] = Counter()
    games = 0
    try:
        for seed in seeds:
            observation, info = environment.reset(seed=int(seed))
            terminated = truncated = False
            while not (terminated or truncated):
                game = environment.game
                if game is None:
                    raise RuntimeError("Environmentにゲームがありません。")
                mask = np.asarray(info["action_mask"], dtype=np.bool_)
                base_action = (teacher.policy.select_action(game, learning_player_id)
                               if (planner_overrides_only or collect_override_gate)
                               and game.phase == "action"
                               else None)
                action = teacher.select_action(game, learning_player_id)
                if isinstance(action, TRADE_ACTIONS):
                    raise AssertionError("交渉ActionがPPO境界へ漏れています。")
                action_id = action_to_id(action)
                if not mask[action_id]:
                    raise AssertionError("教師ActionがAction Mask外です。")
                is_planner_override = bool(
                    game.phase == "action" and base_action is not None
                    and action != base_action
                )
                if is_planner_override:
                    planner_overrides += 1
                    override_families[type(action).__name__] += 1
                is_strategic_action = bool(
                    include_strategic_actions
                    and isinstance(action, (
                        BuildSettlementAction, BuildCityAction, BuyDevelopmentAction,
                    ))
                )
                if is_strategic_action and not is_planner_override:
                    strategic_additions[type(action).__name__] += 1
                selected_for_gate = collect_override_gate and game.phase == "action"
                if mask.sum() >= 2 and (
                    selected_for_gate
                    or not planner_overrides_only
                    or is_planner_override
                    or is_strategic_action
                ):
                    observations.append(np.asarray(observation, dtype=np.float32).copy())
                    masks.append(mask.copy())
                    labels.append(action_id)
                    override_labels.append(is_planner_override)
                    families[type(action).__name__] += 1
                observation, _, terminated, truncated, info = environment.step(action_id)
            if truncated:
                raise AssertionError(f"教師局が打ち切られました: seed={seed}")
            auxiliary_actions += int(info["learner_auxiliary_actions"])
            games += 1
    finally:
        environment.close()
    if not observations:
        raise AssertionError("教師データを収集できませんでした。")
    return TeacherExamples(
        observations=np.stack(observations),
        masks=np.stack(masks),
        labels=np.asarray(labels, dtype=np.int64),
        summary={
            "games": games,
            "examples": len(labels),
            "learner_trade_actions": auxiliary_actions,
            "planner_overrides_only": planner_overrides_only,
            "planner_overrides": planner_overrides,
            "override_label_counts": dict(override_families),
            "include_strategic_actions": include_strategic_actions,
            "strategic_addition_counts": dict(strategic_additions),
            "collect_override_gate": collect_override_gate,
            "gate_positive_examples": int(sum(override_labels)),
            "gate_negative_examples": int(len(override_labels) - sum(override_labels)),
            "label_counts": dict(families),
        },
        override_labels=np.asarray(override_labels, dtype=np.bool_),
    )


def _probabilities(policy, observations: torch.Tensor,
                   masks: torch.Tensor) -> torch.Tensor:
    distribution = policy.get_distribution(observations, action_masks=masks)
    return distribution.distribution.probs.clamp_min(1e-12)


def _family_probabilities(policy, observations: torch.Tensor,
                          action_masks: torch.Tensor) -> torch.Tensor:
    logits = policy.mlp_extractor.forward_actor(observations)[:, ACTION_SPACE_SIZE:]
    family_ids = torch.as_tensor(ACTION_FAMILY_IDS, dtype=torch.long,
                                 device=action_masks.device)
    family_masks = torch.stack([
        action_masks[:, family_ids == family].any(dim=1)
        for family in range(FAMILY_COUNT)
    ], dim=1)
    return torch.softmax(logits.masked_fill(~family_masks, -1e8), dim=1).clamp_min(1e-12)


def _imitation_metrics(policy, examples: TeacherExamples) -> dict[str, float | int]:
    device = next(policy.parameters()).device
    policy.eval()
    with torch.no_grad():
        probabilities = _probabilities(
            policy,
            torch.as_tensor(examples.observations, device=device),
            torch.as_tensor(examples.masks, dtype=torch.bool, device=device),
        )
        labels = torch.as_tensor(examples.labels, dtype=torch.long, device=device)
        selected = probabilities.gather(1, labels[:, None]).squeeze(1)
        predictions = probabilities.argmax(dim=1)
        family_ids = torch.as_tensor(ACTION_FAMILY_IDS, dtype=torch.long, device=device)
        predicted_families = family_ids[predictions]
        label_families = family_ids[labels]
        family_probabilities = _family_probabilities(
            policy,
            torch.as_tensor(examples.observations, device=device),
            torch.as_tensor(examples.masks, dtype=torch.bool, device=device),
        )
        selected_families = family_probabilities.gather(
            1, label_families[:, None]
        ).squeeze(1)
    result = {
        "examples": len(examples),
        "agreement": float((predictions == labels).float().mean()),
        "family_agreement": float(
            (predicted_families == label_families).float().mean()
        ),
        "family_negative_log_likelihood": float(-selected_families.log().mean()),
        "negative_log_likelihood": float(-selected.log().mean()),
    }
    if (isinstance(policy, GatedGoalGraphFamilyHierarchicalCandidateMaskablePolicy)
            and examples.override_labels is not None):
        with torch.no_grad():
            gate = torch.sigmoid(policy.mlp_extractor.override_gate_logits(
                torch.as_tensor(examples.observations, device=device)
            ))
        truth = torch.as_tensor(examples.override_labels, dtype=torch.bool, device=device)
        predicted = gate >= 0.5
        positives = truth.sum().clamp_min(1)
        negatives = (~truth).sum().clamp_min(1)
        true_positive = (predicted & truth).sum()
        false_positive = (predicted & ~truth).sum()
        result.update({
            "override_rate": float(truth.float().mean()),
            "gate_recall": float(true_positive / positives),
            "gate_specificity": float(((~predicted) & (~truth)).sum() / negatives),
            "gate_precision": float(true_positive / (true_positive + false_positive).clamp_min(1)),
            "gate_mean_probability": float(gate.mean()),
        })
    return result


def configure_league_family_finetune(
    policy: GraphFamilyHierarchicalCandidateMaskablePolicy,
) -> dict[str, int]:
    """既存候補選択を固定し、Family残差と価値推定だけを更新する。"""
    if not isinstance(policy, GraphFamilyHierarchicalCandidateMaskablePolicy):
        raise TypeError("GraphFamilyHierarchicalCandidateMaskablePolicyが必要です。")
    for parameter in policy.parameters():
        parameter.requires_grad = False
    modules = (policy.mlp_extractor.critic, policy.value_net)
    if isinstance(policy, GoalGraphFamilyHierarchicalCandidateMaskablePolicy):
        modules += (
            policy.mlp_extractor.goal_action_head,
            policy.mlp_extractor.goal_family_head,
        )
        if isinstance(policy, GatedGoalGraphFamilyHierarchicalCandidateMaskablePolicy):
            modules += (policy.mlp_extractor.goal_override_gate,)
    elif isinstance(policy, (
        BeliefGraphFamilyHierarchicalCandidateMaskablePolicy,
        ContextualBeliefGraphFamilyHierarchicalCandidateMaskablePolicy,
    )):
        modules += (
            policy.mlp_extractor.belief_action_head,
            policy.mlp_extractor.belief_family_head,
        )
    else:
        modules += (policy.mlp_extractor.graph_family_head,)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in policy.parameters()
                    if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in policy.parameters()
                 if not parameter.requires_grad)
    if trainable <= 0 or frozen <= 0:
        raise AssertionError("リーグFamily限定学習のパラメータ分離に失敗しました。")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def calibrate_goal_gate(
    policy: GatedGoalGraphFamilyHierarchicalCandidateMaskablePolicy,
    validation: TeacherExamples,
    *, min_specificity: float = 0.8,
) -> dict[str, float]:
    """誤補正を抑える条件で、検証データから推論閾値を選ぶ。"""
    if validation.override_labels is None:
        raise ValueError("Gate較正にはoverride labelが必要です。")
    device = next(policy.parameters()).device
    policy.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(policy.mlp_extractor.override_gate_logits(
            torch.as_tensor(validation.observations, device=device)
        ))
    truth = torch.as_tensor(validation.override_labels, dtype=torch.bool, device=device)
    positives = truth.sum().clamp_min(1)
    negatives = (~truth).sum().clamp_min(1)
    candidates = []
    for threshold in torch.linspace(0.05, 1.0, 96, device=device):
        predicted = probabilities >= threshold
        recall = float((predicted & truth).sum() / positives)
        specificity = float(((~predicted) & (~truth)).sum() / negatives)
        precision = float(
            (predicted & truth).sum() / predicted.sum().clamp_min(1)
        )
        candidates.append((specificity >= min_specificity,
                           (recall + specificity) / 2, precision,
                           recall, specificity, float(threshold)))
    eligible = [row for row in candidates if row[0]] or candidates
    selected = max(eligible, key=lambda row: (row[1], row[2], row[3], -row[5]))
    policy.mlp_extractor.goal_gate_threshold.fill_(selected[5])
    return {
        "threshold": selected[5], "recall": selected[3],
        "specificity": selected[4], "precision": selected[2],
        "balanced_accuracy": selected[1],
    }


def distill_strongest_policy(
    policy: GraphFamilyHierarchicalCandidateMaskablePolicy,
    train: TeacherExamples,
    validation: TeacherExamples,
    *,
    learning_rate: float = 1e-5,
    batch_size: int = 256,
    max_epochs: int = 8,
    patience: int = 2,
    anchor_weight: float = 1.0,
    seed: int = 20260926,
) -> dict[str, object]:
    """既存Actorを固定し、GNN残差を最強ハイブリッドAIへ近づける。"""
    if not isinstance(policy, GraphFamilyHierarchicalCandidateMaskablePolicy):
        raise TypeError("GraphFamilyHierarchicalCandidateMaskablePolicyが必要です。")
    if min(learning_rate, batch_size, max_epochs, patience) <= 0 or anchor_weight < 0:
        raise ValueError("模倣学習設定が不正です。")
    device = next(policy.parameters()).device
    anchor = deepcopy(policy).to(device).eval()
    for parameter in anchor.parameters():
        parameter.requires_grad = False
    selection = configure_league_family_finetune(policy)
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    before = {
        "train": _imitation_metrics(policy, train),
        "validation": _imitation_metrics(policy, validation),
    }
    best_state = {key: value.detach().cpu().clone()
                  for key, value in policy.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    train_obs = torch.as_tensor(train.observations)
    train_masks = torch.as_tensor(train.masks, dtype=torch.bool)
    train_labels = torch.as_tensor(train.labels, dtype=torch.long)
    family_ids = np.asarray(ACTION_FAMILY_IDS, dtype=np.int64)
    train_families = family_ids[train.labels]
    counts = np.bincount(train_families, minlength=int(family_ids.max()) + 1)
    family_weights = np.zeros_like(counts, dtype=np.float32)
    present = counts > 0
    family_weights[present] = 1.0 / np.sqrt(counts[present])
    family_weights[present] /= family_weights[present].mean()
    train_weights = torch.as_tensor(family_weights[train_families])
    train_family_labels = torch.as_tensor(train_families, dtype=torch.long)
    train_override_labels = (torch.as_tensor(train.override_labels, dtype=torch.float32)
                             if train.override_labels is not None else None)
    gate_positive_weight = None
    if isinstance(policy, GatedGoalGraphFamilyHierarchicalCandidateMaskablePolicy):
        if train_override_labels is None or validation.override_labels is None:
            raise ValueError("Gate学習にはoverride label付き教師データが必要です。")
        positives = float(train_override_labels.sum())
        negatives = float(len(train_override_labels) - positives)
        gate_positive_weight = torch.tensor(
            max(1.0, negatives / max(positives, 1.0)), device=device
        )
    for epoch in range(1, max_epochs + 1):
        policy.train()
        permutation = torch.randperm(len(train), generator=generator)
        losses = []
        for start in range(0, len(train), batch_size):
            indices = permutation[start:start + batch_size]
            observations = train_obs[indices].to(device)
            masks = train_masks[indices].to(device)
            labels = train_labels[indices].to(device)
            family_labels = train_family_labels[indices].to(device)
            weights = train_weights[indices].to(device)
            family_probabilities = _family_probabilities(policy, observations, masks)
            per_example = -family_probabilities.gather(
                1, family_labels[:, None]
            ).log().squeeze(1)
            teacher_loss = (per_example * weights).sum() / weights.sum()
            with torch.no_grad():
                anchor_probabilities = _family_probabilities(anchor, observations, masks)
            anchor_loss = F.kl_div(
                family_probabilities.log(), anchor_probabilities,
                reduction="batchmean"
            )
            if isinstance(policy, (
                GoalGraphFamilyHierarchicalCandidateMaskablePolicy,
                BeliefGraphFamilyHierarchicalCandidateMaskablePolicy,
                ContextualBeliefGraphFamilyHierarchicalCandidateMaskablePolicy,
            )):
                action_probabilities = _probabilities(policy, observations, masks)
                action_per_example = -action_probabilities.gather(
                    1, labels[:, None]
                ).log().squeeze(1)
                action_teacher_loss = (
                    action_per_example * weights
                ).sum() / weights.sum()
                with torch.no_grad():
                    anchor_actions = _probabilities(anchor, observations, masks)
                action_anchor_loss = F.kl_div(
                    action_probabilities.log(), anchor_actions,
                    reduction="batchmean",
                )
                teacher_loss = teacher_loss + 0.25 * action_teacher_loss
                anchor_loss = anchor_loss + 0.5 * action_anchor_loss
            loss = teacher_loss + anchor_weight * anchor_loss
            if isinstance(policy, GatedGoalGraphFamilyHierarchicalCandidateMaskablePolicy):
                gate_labels = train_override_labels[indices].to(device)
                gate_logits = policy.mlp_extractor.override_gate_logits(observations)
                gate_loss = F.binary_cross_entropy_with_logits(
                    gate_logits, gate_labels, pos_weight=gate_positive_weight,
                )
                loss = loss + gate_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_metrics = _imitation_metrics(policy, validation)
        validation_loss = float(validation_metrics["family_negative_log_likelihood"])
        if isinstance(policy, (
            GoalGraphFamilyHierarchicalCandidateMaskablePolicy,
            BeliefGraphFamilyHierarchicalCandidateMaskablePolicy,
            ContextualBeliefGraphFamilyHierarchicalCandidateMaskablePolicy,
        )):
            validation_loss += 0.25 * float(
                validation_metrics["negative_log_likelihood"]
            )
        if isinstance(policy, GatedGoalGraphFamilyHierarchicalCandidateMaskablePolicy):
            policy.eval()
            with torch.no_grad():
                validation_observations = torch.as_tensor(
                    validation.observations, device=device
                )
                validation_gate_labels = torch.as_tensor(
                    validation.override_labels, dtype=torch.float32, device=device
                )
                validation_loss += float(F.binary_cross_entropy_with_logits(
                    policy.mlp_extractor.override_gate_logits(validation_observations),
                    validation_gate_labels,
                    pos_weight=gate_positive_weight,
                ))
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "validation_agreement": validation_metrics["agreement"],
        })
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in policy.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    policy.load_state_dict(best_state)
    gate_calibration = None
    if isinstance(policy, GatedGoalGraphFamilyHierarchicalCandidateMaskablePolicy):
        gate_calibration = calibrate_goal_gate(policy, validation)
    return {
        "parameter_selection": selection,
        "before": before,
        "after": {
            "train": _imitation_metrics(policy, train),
            "validation": _imitation_metrics(policy, validation),
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "anchor_weight": anchor_weight,
        "gate_calibration": gate_calibration,
    }


def _match(agent, opponents, seeds: Iterable[int]) -> dict[str, object]:
    episodes = []
    for index, seed in enumerate(seeds):
        learner_id = index % 4 + 1
        game = create_game(int(seed), ai_player_ids=(1, 2, 3, 4))
        agents = {pid: (agent if pid == learner_id else opponents)
                  for pid in range(1, 5)}
        for step in range(100_000):
            if game.phase == "game_over":
                break
            actor = pending_ai_player_id(game)
            if actor is None:
                raise RuntimeError(f"操作担当がありません: {game.phase}")
            action = agents[actor].select_action(game, actor)
            apply_action(game, actor, action, game.revision,
                         f"league-eval-{step:06d}")
            game.ai_action_number += 1
        else:
            raise RuntimeError(f"評価局が終了しません: seed={seed}")
        scores = {pid: actual_score(game, pid) for pid in range(1, 5)}
        learner_score = scores[learner_id]
        rank = 1 + sum(score > learner_score for pid, score in scores.items()
                       if pid != learner_id)
        episodes.append({
            "seed": int(seed), "learner_id": learner_id,
            "won": game.winner_id == learner_id,
            "score": learner_score, "rank": rank,
            "winner_id": game.winner_id,
        })
    return {
        "games": len(episodes),
        "win_rate": float(np.mean([item["won"] for item in episodes])),
        "score": float(np.mean([item["score"] for item in episodes])),
        "rank": float(np.mean([item["rank"] for item in episodes])),
        "episodes": episodes,
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source-model-id", default="ppo_gnn_board_65k_s03_exp_v003")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--train-games", type=int, default=120)
    parser.add_argument("--validation-games", type=int, default=30)
    parser.add_argument("--rl-steps", type=int, default=25_000)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=2_300_001)
    parser.add_argument("--eval-seed-start", type=int, default=2_400_001)
    parser.add_argument("--training-seed", type=int, default=20260927)
    parser.add_argument("--distill-epochs", type=int, default=8)
    parser.add_argument("--distill-learning-rate", type=float, default=1e-5)
    parser.add_argument("--ppo-learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset-artifact", type=Path, default=None,
                        help="既存のteacher_examples.npzを再利用する。")
    parser.add_argument("--skip-ppo", action="store_true",
                        help="模倣直後の候補を保存・評価し、PPO追加学習を行わない。")
    parser.add_argument("--goal-context", action="store_true",
                        help="v3の目標交差点・道路・不足資源・待ち時間をPPOへ入力する。")
    parser.add_argument("--belief-context", action="store_true",
                        help="v4の共同ベイズ相手手札推定をPPOへ入力する。")
    parser.add_argument("--focus-planner-overrides", action="store_true",
                        help="現行PPOを計画層が変更したaction局面だけを教師にする。")
    parser.add_argument("--include-strategic-actions", action="store_true",
                        help="重点収集時も開拓地・都市・発展購入は一致局面を教材へ残す。")
    parser.add_argument("--goal-gate", action="store_true",
                        help="全action局面にoverrideラベルを付け、目標残差の適用Gateを学ぶ。")
    args = parser.parse_args()
    if (Path(args.experiment_id).name != args.experiment_id
            or args.experiment_id in {".", ".."}
            or min(args.train_games, args.validation_games, args.rl_steps,
                   args.eval_games, args.distill_epochs) <= 0):
        parser.error("実験名と学習量は正の安全な値で指定してください。")
    if args.goal_gate and not args.goal_context:
        parser.error("goal-gateにはgoal-contextが必要です。")
    if args.goal_context and args.belief_context:
        parser.error("goal-contextとbelief-contextは同時指定できません。")

    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / args.experiment_id
    if output.exists():
        parser.error(f"実験フォルダが既にあります: {output}")
    output.mkdir(parents=True)
    source, learner_teacher = _strongest(
        args.source_model_id, args.models_root, args.device
    )
    frozen_base, frozen_teacher = _strongest(
        args.source_model_id, args.models_root, args.device
    )
    if not isinstance(source.model.policy, GraphHierarchicalCandidateMaskablePolicy):
        raise TypeError("移行元がGNN階層候補PPOではありません。")
    observation_version = (
        "v4" if args.belief_context else "v3" if args.goal_context else "v2"
    )

    if args.dataset_artifact is None:
        train_end = args.seed_start + args.train_games
        validation_end = train_end + args.validation_games
        train = collect_teacher_examples(
            learner_teacher, range(args.seed_start, train_end),
            observation_version=observation_version,
            planner_overrides_only=args.focus_planner_overrides,
            include_strategic_actions=args.include_strategic_actions,
            collect_override_gate=args.goal_gate,
        )
        validation = collect_teacher_examples(
            learner_teacher, range(train_end, validation_end),
            observation_version=observation_version,
            planner_overrides_only=args.focus_planner_overrides,
            include_strategic_actions=args.include_strategic_actions,
            collect_override_gate=args.goal_gate,
        )
        np.savez_compressed(
            output / "teacher_examples.npz",
            train_observations=train.observations, train_masks=train.masks,
            train_labels=train.labels,
            train_override_labels=train.override_labels,
            validation_observations=validation.observations,
            validation_masks=validation.masks,
            validation_labels=validation.labels,
            validation_override_labels=validation.override_labels,
        )
    else:
        artifact = args.dataset_artifact.resolve()
        if not artifact.is_file():
            parser.error(f"教師データがありません: {artifact}")
        with np.load(artifact, allow_pickle=False) as values:
            train = TeacherExamples(
                values["train_observations"], values["train_masks"],
                values["train_labels"],
                {"games": args.train_games,
                 "examples": int(values["train_labels"].shape[0]),
                 "dataset_artifact": str(artifact)},
                values["train_override_labels"] if "train_override_labels" in values else None,
            )
            validation = TeacherExamples(
                values["validation_observations"], values["validation_masks"],
                values["validation_labels"],
                {"games": args.validation_games,
                 "examples": int(values["validation_labels"].shape[0]),
                 "dataset_artifact": str(artifact)},
                (values["validation_override_labels"]
                 if "validation_override_labels" in values else None),
            )

    league = SeededMixedLeagueAgent(frozen_teacher)
    environment = CatanEnv(
        CatanEnvConfig(
            learning_player_id=1,
            allow_player_trades=True,
            observation_version=observation_version,
        ),
        initial_placement_agent=frozen_teacher,
        opponent_agents={2: league, 3: league, 4: league},
        learner_auxiliary_agent=StrategicTradeAuxiliaryAgent(frozen_teacher),
    )
    from sb3_contrib import MaskablePPO
    policy_class = (
        GatedGoalGraphFamilyHierarchicalCandidateMaskablePolicy
        if args.goal_gate else
        GoalGraphFamilyHierarchicalCandidateMaskablePolicy
        if args.goal_context else
        ContextualBeliefGraphFamilyHierarchicalCandidateMaskablePolicy
        if args.belief_context else
        GraphFamilyHierarchicalCandidateMaskablePolicy
    )
    model = MaskablePPO(
        policy_class,
        environment,
        learning_rate=args.ppo_learning_rate,
        n_steps=source.model.n_steps,
        batch_size=source.model.batch_size,
        n_epochs=source.model.n_epochs,
        gamma=source.model.gamma,
        gae_lambda=source.model.gae_lambda,
        ent_coef=source.model.ent_coef,
        vf_coef=source.model.vf_coef,
        max_grad_norm=source.model.max_grad_norm,
        normalize_advantage=source.model.normalize_advantage,
        policy_kwargs={"net_arch": [], "ortho_init": False},
        seed=args.training_seed,
        device=args.device,
        verbose=0,
    )
    if args.belief_context:
        initialize_belief_policy_from_graph(model.policy, source.model.policy)
    elif args.goal_context:
        initialize_goal_policy_from_graph(model.policy, source.model.policy)
    else:
        initialize_graph_policy_from_hierarchical(model.policy, source.model.policy)
    model.num_timesteps = int(source.model.num_timesteps)
    distillation = distill_strongest_policy(
        model.policy, train, validation,
        learning_rate=args.distill_learning_rate,
        max_epochs=args.distill_epochs,
        seed=args.training_seed,
    )
    _write(output / "distillation.json", {
        "train_collection": train.summary,
        "validation_collection": validation.summary,
        **distillation,
    })

    source_steps = int(model.num_timesteps)
    model.save(str(output / "distilled_model.zip"))
    if not args.skip_ppo:
        model.set_random_seed(args.training_seed)
        model.learning_rate = args.ppo_learning_rate
        model.lr_schedule = lambda _: args.ppo_learning_rate
        for group in model.policy.optimizer.param_groups:
            group["lr"] = args.ppo_learning_rate
        model.learn(total_timesteps=args.rl_steps, reset_num_timesteps=False,
                    progress_bar=False)
    environment.close()

    candidate_id = f"{args.experiment_id}_candidate"
    models_root = output / "models"
    model_directory = models_root / candidate_id
    model_directory.mkdir(parents=True)
    model.save(str(model_directory / "model.zip"))
    if frozen_base.manifest.initial_setup is not None:
        setup_name = frozen_base.manifest.initial_setup["artifact_filename"]
        shutil.copy2(args.models_root / args.source_model_id / setup_name,
                     model_directory / setup_name)
    manifest = ModelManifest(
        model_id=candidate_id,
        algorithm="MaskablePPO",
        artifact_filename="model.zip",
        observation_version=observation_version,
        observation_size=OBSERVATION_VECTOR_SIZES[observation_version],
        action_space_version=frozen_base.manifest.action_space_version,
        action_space_size=ACTION_SPACE_SIZE,
        training_steps=int(model.num_timesteps),
        training_seed=args.training_seed,
        created_at=datetime.now(timezone.utc).isoformat(),
        evaluation={},
        initial_setup=frozen_base.manifest.initial_setup,
    )
    _write(model_directory / "metadata.json", manifest.to_dict())
    source_config = json.loads(
        (args.models_root / args.source_model_id / "config.json").read_text(encoding="utf-8")
    )
    source_config.update({
        "model_id": candidate_id,
        "observation_version": observation_version,
        "observation_size": OBSERVATION_VECTOR_SIZES[observation_version],
        "policy_architecture": (
            "gated_goal_gnn_family_hierarchical_candidate"
            if args.goal_gate else
            "goal_gnn_family_hierarchical_candidate"
            if args.goal_context else
            "contextual_belief_gnn_family_hierarchical_candidate"
            if args.belief_context else
            "gnn_family_hierarchical_candidate"
        ),
        "league_training": {
            "source_model_id": args.source_model_id,
            "source_steps": source_steps,
            "additional_rl_steps": 0 if args.skip_ppo else args.rl_steps,
            "player_trades_enabled": True,
            "goal_context_enabled": args.goal_context,
            "belief_context_enabled": args.belief_context,
            "planner_overrides_only": args.focus_planner_overrides,
            "include_strategic_actions": args.include_strategic_actions,
            "goal_gate_enabled": args.goal_gate,
            "learner_trade_controller": args.source_model_id,
            "opponent_mix": "75%: strongest_x2+heuristic_x1; 25%: heuristic_x3",
            "distillation": {
                "train_games": args.train_games,
                "validation_games": args.validation_games,
                "learning_rate": args.distill_learning_rate,
                "epochs": args.distill_epochs,
            },
            "ppo_learning_rate": args.ppo_learning_rate,
        },
    })
    _write(model_directory / "config.json", source_config)

    candidate_base = PPOAgent(candidate_id, models_root=models_root, device=args.device)
    candidate = SettlementPlanningAgent(
        candidate_base, **candidate_base.settlement_planning_config
    )
    eval_seeds = range(args.eval_seed_start, args.eval_seed_start + args.eval_games)
    evaluation = {
        "source_vs_strongest_x3": _match(
            frozen_teacher, frozen_teacher,
            range(args.eval_seed_start, args.eval_seed_start + args.eval_games),
        ),
        "candidate_vs_strongest_x3": _match(candidate, frozen_teacher, eval_seeds),
        "source_vs_heuristic_x3": _match(
            frozen_teacher, HeuristicAgent(),
            range(args.eval_seed_start, args.eval_seed_start + args.eval_games),
        ),
        "candidate_vs_heuristic_x3": _match(
            candidate, HeuristicAgent(),
            range(args.eval_seed_start, args.eval_seed_start + args.eval_games),
        ),
    }
    _write(output / "evaluation.json", evaluation)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    _write(output / "definition.json", serializable_args | {
        "models_root": str(args.models_root),
        "output_model": str(model_directory),
    })
    print(json.dumps({
        "model_directory": str(model_directory),
        "distillation": distillation["after"],
        "evaluation": {key: {name: value for name, value in report.items()
                              if name != "episodes"}
                       for key, report in evaluation.items()},
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
