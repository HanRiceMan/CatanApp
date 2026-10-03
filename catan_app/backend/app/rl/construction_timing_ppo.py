"""One guarded semi-MDP PPO update for the Construction Timing Gate only."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch
from torch import nn

from .construction_timing import (BUILD_NOW, ConstructionTimingGate,
                                  SurfaceCritic, SurfaceTransition,
                                  timing_trainable_parameters)
from .construction_timing_rollout import TimingEpisode


GAMMA = 0.99
GAE_LAMBDA = 0.95


def macro_gae(transitions: tuple[SurfaceTransition, ...], *,
              gamma: float = GAMMA, gae_lambda: float = GAE_LAMBDA,
              ) -> tuple[np.ndarray, np.ndarray]:
    """Variable gamma**duration; lambda traces over successive gate decisions."""
    if not transitions:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    if transitions[-1].truncated:
        raise ValueError("Time-limit truncation is censored, not terminal training data")
    if not transitions[-1].done:
        raise ValueError("Complete terminal rollout required for first update")
    advantages = np.zeros(len(transitions), np.float64)
    carry = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        item = transitions[index]
        if item.truncated:
            raise ValueError("Truncated transition cannot enter first update")
        if item.done:
            next_value, carry = 0.0, 0.0
        else:
            if index + 1 >= len(transitions):
                raise ValueError("Nonterminal transition has no next delegated state")
            next_item = transitions[index + 1]
            if item.next_surface_type != next_item.surface_type:
                raise ValueError("Bootstrap target is not next actually delegated surface")
            if (item.next_surface_observation is not None
                    and not np.array_equal(item.next_surface_observation,
                                           next_item.surface_observation)):
                raise ValueError("Bootstrap observation differs from next delegated state")
            if (item.next_runtime_context is not None
                    and not np.array_equal(item.next_runtime_context,
                                           next_item.runtime_context)):
                raise ValueError("Bootstrap context differs from next delegated state")
            next_value = next_item.value
        discount = gamma ** item.macro_duration
        delta = item.reward_accumulated + discount * next_value - item.value
        carry = delta + discount * gae_lambda * carry
        advantages[index] = carry
    returns = advantages + np.array([item.value for item in transitions])
    return advantages.astype(np.float32), returns.astype(np.float32)


def _statistics(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max())}


def _frozen_features(champion_policy, transitions: list[SurfaceTransition]):
    observation = np.stack([item.surface_observation for item in transitions]).astype(np.float32)
    context = torch.as_tensor(np.stack([item.runtime_context for item in transitions]),
                              dtype=torch.float32)
    subtype = torch.tensor([[1.0, 0.0] if item.surface_type == "SETTLEMENT_ONLY"
                            else [0.0, 1.0] for item in transitions], dtype=torch.float32)
    policy = champion_policy.model.policy
    with torch.no_grad():
        tensor, _ = policy.obs_to_tensor(observation)
        latent = policy.mlp_extractor.forward_actor(tensor).detach()
        base_value = policy.predict_values(tensor).detach()
    return latent, context.to(latent.device), subtype.to(latent.device), base_value


@dataclass(frozen=True)
class UpdateResult:
    accepted: bool
    reason: str
    metrics: dict


def one_timing_ppo_update(
    champion_policy, gate: ConstructionTimingGate, critic: SurfaceCritic,
    episodes: list[TimingEpisode], *, actor_lr: float = 1e-4,
    critic_lr: float = 3e-4, batch_size: int = 32, epochs: int = 2,
    clip_range: float = 0.1, target_kl: float = 0.01,
    entropy_coef: float = 0.005, training_seed: int = 2430001,
) -> UpdateResult:
    """Exactly one guarded update; rejection restores both new modules."""
    if any(item.integration_forced for item in episodes):
        raise ValueError("Forced integration examples are not on-policy data")
    if any(item.truncated or not item.terminal for item in episodes):
        raise ValueError("Truncated/incomplete episodes are quarantined")
    transitions = [item for episode in episodes for item in episode.transitions]
    if len(transitions) < 2:
        raise ValueError("Too few delegated transitions")
    if any(not item.delegated for item in transitions):
        raise ValueError("Non-delegated Action in Gate Actor buffer")
    advantages, returns = zip(*(macro_gae(episode.transitions)
                                for episode in episodes if episode.transitions), strict=False)
    advantage = np.concatenate(advantages)
    return_values = np.concatenate(returns)
    if not np.isfinite(advantage).all() or not np.isfinite(return_values).all():
        raise ValueError("NaN/inf advantage or return")
    raw_stats = _statistics(advantage)
    if raw_stats["std"] < 1e-6:
        raise ValueError("Nearly constant advantage; update refused")
    normalized = (advantage - raw_stats["mean"]) / (raw_stats["std"] + 1e-8)
    normalized_stats = _statistics(normalized)
    if np.max(np.abs(normalized)) > 10:
        raise ValueError("Extreme advantage outlier; update refused")

    optimizer_ids = {id(parameter) for parameter in timing_trainable_parameters(gate, critic)}
    frozen_ids = {id(parameter) for parameter in champion_policy.model.policy.parameters()}
    if optimizer_ids & frozen_ids:
        raise AssertionError("Champion parameter leaked into Gate optimizer")
    champion_weights = {key: value.detach().clone()
                        for key, value in champion_policy.model.policy.state_dict().items()}
    latent, context, subtype, base_value = _frozen_features(champion_policy, transitions)
    action = torch.tensor([item.gate_action for item in transitions],
                          dtype=torch.long, device=latent.device)
    old_log_prob = torch.tensor([item.gate_log_prob for item in transitions],
                                dtype=torch.float32, device=latent.device)
    normalized_tensor = torch.as_tensor(normalized, device=latent.device)
    target = torch.as_tensor(return_values, device=latent.device)
    with torch.no_grad():
        old_logits = gate(latent, context, subtype).detach()
        old_distribution = torch.distributions.Categorical(logits=old_logits)
        if not torch.allclose(old_distribution.log_prob(action), old_log_prob, atol=1e-5):
            raise ValueError("Stored on-policy log probability differs from current Gate")
        pre_value = critic(base_value, latent, context, subtype).reshape(-1)
        stored_value = torch.tensor([item.value for item in transitions], device=latent.device)
        if not torch.allclose(pre_value, stored_value, atol=1e-4):
            raise ValueError("Stored on-policy value differs from current Surface Critic")
    before_gate = deepcopy(gate.state_dict())
    before_critic = deepcopy(critic.state_dict())
    actor_optimizer = torch.optim.Adam(gate.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    generator = torch.Generator().manual_seed(training_seed)
    actor_losses, critic_losses, entropies = [], [], []
    stopped_early = False
    for _ in range(epochs):
        permutation = torch.randperm(len(transitions), generator=generator).tolist()
        for start in range(0, len(transitions), batch_size):
            indices = permutation[start:start + batch_size]
            distribution = torch.distributions.Categorical(
                logits=gate(latent[indices], context[indices], subtype[indices]))
            ratio = torch.exp(distribution.log_prob(action[indices])
                              - old_log_prob[indices])
            unclipped = ratio * normalized_tensor[indices]
            clipped = torch.clamp(ratio, 1 - clip_range,
                                  1 + clip_range) * normalized_tensor[indices]
            entropy = distribution.entropy().mean()
            actor_loss = -torch.minimum(unclipped, clipped).mean() - entropy_coef * entropy
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(gate.parameters(), 0.5)
            actor_optimizer.step()
            value = critic(base_value[indices], latent[indices],
                           context[indices], subtype[indices]).reshape(-1)
            critic_loss = torch.mean((value - target[indices]) ** 2)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_optimizer.step()
            actor_losses.append(float(actor_loss.detach()))
            critic_losses.append(float(critic_loss.detach()))
            entropies.append(float(entropy.detach()))
            with torch.no_grad():
                new_distribution = torch.distributions.Categorical(
                    logits=gate(latent, context, subtype))
                kl = torch.distributions.kl_divergence(old_distribution,
                                                        new_distribution)
                if float(kl.mean()) > target_kl:
                    stopped_early = True
                    break
        if stopped_early:
            break
    with torch.no_grad():
        new_distribution = torch.distributions.Categorical(
            logits=gate(latent, context, subtype))
        kl = torch.distributions.kl_divergence(old_distribution, new_distribution).cpu().numpy()
        before_probability = old_distribution.probs[:, BUILD_NOW].cpu().numpy()
        after_probability = new_distribution.probs[:, BUILD_NOW].cpu().numpy()
    kl_by_subtype = {}
    probabilities = {}
    for name in ("SETTLEMENT_ONLY", "CITY_ONLY"):
        indices = np.array([item.surface_type == name for item in transitions])
        if indices.any():
            kl_by_subtype[name] = float(kl[indices].mean())
            probabilities[name] = {
                "count": int(indices.sum()),
                "before_p_build": float(before_probability[indices].mean()),
                "after_p_build": float(after_probability[indices].mean()),
                "after_p_defer": float(1 - after_probability[indices].mean()),
            }
    collapse = bool((after_probability > 0.95).mean() > 0.8
                    or (after_probability < 0.05).mean() > 0.8)
    reason = ("KL_EXCEEDED" if kl.mean() > 2 * target_kl or kl.max() > 0.1
              else "GATE_COLLAPSE" if collapse else "ACCEPTED")
    accepted = reason == "ACCEPTED"
    if not accepted:
        gate.load_state_dict(before_gate)
        critic.load_state_dict(before_critic)
    if any(not torch.equal(value, champion_policy.model.policy.state_dict()[key])
           for key, value in champion_weights.items()):
        raise AssertionError("Frozen Champion weights changed")
    metrics = {
        "transitions": len(transitions), "advantage_raw": raw_stats,
        "advantage_normalized": normalized_stats,
        "actor_loss_mean": float(np.mean(actor_losses)),
        "critic_loss_mean": float(np.mean(critic_losses)),
        "entropy_mean": float(np.mean(entropies)),
        "mean_kl": float(kl.mean()), "max_kl": float(kl.max()),
        "kl_by_subtype": kl_by_subtype, "probability_by_subtype": probabilities,
        "build_collapse_fraction": float((after_probability > 0.95).mean()),
        "defer_collapse_fraction": float((after_probability < 0.05).mean()),
        "optimizer_parameter_count": sum(item.numel() for item in
                                           timing_trainable_parameters(gate, critic)),
        "frozen_champion_unchanged": True,
        "minibatches": len(actor_losses), "early_stopped": stopped_early,
    }
    return UpdateResult(accepted, reason, metrics)
