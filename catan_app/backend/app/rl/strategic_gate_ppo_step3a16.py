"""One guarded on-policy semi-MDP PPO update for the six-way Strategic Gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch import nn

from .collect_strategic_gate_shadow_step3a13 import _weight_digest
from .strategic_action_surface import FAMILIES
from .strategic_gate_step3a13 import StrategicActionGate, StrategicSurfaceCritic
from .runtime_context import RUNTIME_CONTEXT_VERSION


TEMPERATURE = 2.0
GAMMA = 0.99
GAE_LAMBDA = 0.95


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    return {"n": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
            "min": float(values.min()) if values.size else None,
            "p5": float(np.percentile(values, 5)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None,
            "max": float(values.max()) if values.size else None}


def strategic_macro_gae(transitions: list[dict[str, Any]], *,
                        gamma: float = GAMMA, gae_lambda: float = GAE_LAMBDA,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Step 3A-9 definition: gamma**duration; lambda over successive Gate calls."""
    if not transitions:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    if transitions[-1]["truncated"] or not transitions[-1]["done"]:
        raise ValueError("Complete terminal episode required for first Strategic update")
    advantages = np.zeros(len(transitions), np.float64)
    carry = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        item = transitions[index]
        if item["truncated"] or item["macro_duration"] < 1:
            raise ValueError("Invalid or censored Strategic transition")
        if item["done"]:
            next_value, carry = 0.0, 0.0
        else:
            if index + 1 >= len(transitions) or not item["next_actual_delegated"]:
                raise ValueError("Bootstrap must target next actually delegated Gate")
            next_item = transitions[index + 1]
            if item["next_candidate_mask"] != next_item["candidate_mask"]:
                raise ValueError("Next delegated Gate candidate mask mismatch")
            if item["close_policy_step"] != next_item["start_policy_step"]:
                raise ValueError("Delegated macro boundary is discontinuous")
            next_value = next_item["value"]
        discount = gamma ** item["macro_duration"]
        delta = item["reward_accumulated"] + discount * next_value - item["value"]
        carry = delta + discount * gae_lambda * carry
        advantages[index] = carry
    returns = advantages + np.asarray([item["value"] for item in transitions])
    return advantages.astype(np.float32), returns.astype(np.float32)


def rollout_diagnostics(episodes: list[dict[str, Any]], *,
                        gamma: float = GAMMA) -> dict[str, Any]:
    decisions = [decision for episode in episodes for decision in episode["decisions"]]
    transitions = [item for episode in episodes for item in episode["transitions"]]
    if len(decisions) != len(transitions):
        raise ValueError("Off-policy or missing Strategic transitions")
    if any(episode["truncated"] or not episode["terminal"] for episode in episodes):
        raise ValueError("Censored episode cannot enter first update")
    resolver_crossed = sum(
        transition["start_policy_step"] <= event["pre_policy_step"] <
        transition["close_policy_step"]
        for episode in episodes for transition in episode["transitions"]
        for event in episode["resolver_events"])
    family_selected = Counter(item["selected_family"] for item in decisions)
    family_available = Counter(FAMILIES[index] for item in decisions
                               for index, enabled in enumerate(item["candidate_mask"])
                               if enabled)
    count_buckets = Counter("4+" if item["candidate_count"] >= 4
                            else str(item["candidate_count"]) for item in decisions)
    non_top = Counter(item["selected_family"] for item in decisions
                      if not item["selected_is_native_top"])
    for episode in episodes:
        if len(episode["decisions"]) != len(episode["transitions"]):
            raise ValueError("Off-policy episode boundary")
        for index, (decision, transition) in enumerate(zip(
                episode["decisions"], episode["transitions"], strict=True)):
            if (transition["decision_index"] != index or
                    transition["family"] != decision["selected_family"] or
                    decision["forced_integration"] or
                    decision["protected_player_trade_selected"] or
                    decision["prior_temperature"] != TEMPERATURE):
                raise ValueError("Nondelegated / forced / protected data in Actor buffer")
            if (abs(transition["gate_log_prob"] - decision["gate_log_prob"]) > 1e-8
                    or abs(decision["gate_log_prob"] - np.log(
                        decision["selected_probability"])) > 1e-5):
                raise ValueError("Stored on-policy log probability mismatch")
            expected_reward = sum(gamma ** step * reward for step, reward in enumerate(
                transition["reward_steps"]))
            if (abs(expected_reward - transition["reward_accumulated"]) > 1e-8 or
                    transition["macro_duration"] != len(transition["reward_steps"])):
                raise ValueError("Macro return or duration mismatch")
            if transition["done"] and transition["discount"] != 0:
                raise ValueError("Terminal bootstrap must be zero")
            if not transition["done"] and abs(
                    transition["discount"] - gamma ** transition["macro_duration"]) > 1e-8:
                raise ValueError("Nonterminal macro discount mismatch")
            if len(decision.get("observation_v2", [])) != 1667 or len(
                    decision.get("runtime_context", [])) != 79:
                raise ValueError("Missing or malformed pre-action Student input")
    return {
        "games": len(episodes), "transitions": len(transitions),
        "profile_games": dict(Counter(item["profile"] for item in episodes)),
        "seat_games": dict(Counter(str(item["seat"]) for item in episodes)),
        "family_selected": dict(family_selected),
        "family_available": dict(family_available),
        "candidate_count": dict(count_buckets),
        "native_top_selected": len(decisions) - sum(non_top.values()),
        "native_non_top_selected": sum(non_top.values()),
        "native_non_top_by_family": dict(non_top),
        "entropy": distribution(np.asarray([item["entropy"] for item in decisions])),
        "macro_duration": distribution(np.asarray([item["macro_duration"]
                                                   for item in transitions])),
        "macro_reward": distribution(np.asarray([item["reward_accumulated"]
                                                 for item in transitions])),
        "terminal_reward": distribution(np.asarray([item["reward_accumulated"]
                                                    for item in transitions if item["done"]])),
        "nonterminal_reward": distribution(np.asarray([item["reward_accumulated"]
                                                       for item in transitions if not item["done"]])),
        "terminal_transitions": sum(item["done"] for item in transitions),
        "protected_trades_crossed": sum(item["protected_trades_crossed"]
                                        for item in transitions),
        "nondelegated_surfaces_crossed": sum(item["nondelegated_surfaces_crossed"]
                                             for item in transitions),
        "resolver_events": sum(len(item["resolver_events"]) for item in episodes),
        "resolver_crossed": resolver_crossed,
    }


def _frozen_features(champion, decisions: list[dict[str, Any]]):
    observation = np.asarray([item["observation_v2"] for item in decisions],
                             dtype=np.float32)
    policy = champion.model.policy
    with torch.no_grad():
        tensor, _ = policy.obs_to_tensor(observation)
        latent = policy.mlp_extractor.forward_actor(tensor).detach()
        base_value = policy.predict_values(tensor).detach()
    context = torch.as_tensor(np.asarray([item["runtime_context"]
                                         for item in decisions], dtype=np.float32),
                              device=latent.device)
    mask = torch.as_tensor(np.asarray([item["candidate_mask"] for item in decisions],
                                     dtype=bool), device=latent.device)
    native = torch.as_tensor(np.asarray([item["native_family_logits"]
                                       for item in decisions], dtype=np.float32),
                             device=latent.device)
    return latent, context, mask, native, base_value


def preflight_strategic_update(
    champion, gate: StrategicActionGate, critic: StrategicSurfaceCritic,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read-only report and hard gates, deliberately before any optimizer exists."""
    diagnostics = rollout_diagnostics(episodes)
    decisions = [item for episode in episodes for item in episode["decisions"]]
    if diagnostics["transitions"] < 256:
        raise ValueError("Fewer than 256 delegated transitions")
    per_episode = [strategic_macro_gae(episode["transitions"])
                   for episode in episodes if episode["transitions"]]
    advantages = np.concatenate([item[0] for item in per_episode])
    returns = np.concatenate([item[1] for item in per_episode])
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
        raise ValueError("Nonfinite advantage or return")
    raw = distribution(advantages)
    if raw["std"] is None or raw["std"] < 1e-6:
        raise ValueError("Nearly constant advantage")
    normalized = (advantages - raw["mean"]) / (raw["std"] + 1e-8)
    if np.max(np.abs(normalized)) > 10:
        raise ValueError("Extreme single-outlier advantage")
    normalized_stats = distribution(normalized)
    family_advantage = {family: {"n": len(values), "mean": mean(values)}
                        for family, values in _group_values(
                            decisions, advantages, "selected_family").items()}
    done = np.asarray([item["done"] for episode in episodes
                       for item in episode["transitions"]])
    terminal_advantage = {label: distribution(advantages[done == flag])
                          for label, flag in (("terminal", True),
                                              ("nonterminal", False))}
    gate_ids = {id(parameter) for parameter in gate.parameters()}
    critic_ids = {id(parameter) for parameter in critic.parameters()}
    frozen_ids = {id(parameter) for parameter in champion.model.policy.parameters()}
    if gate_ids & critic_ids or (gate_ids | critic_ids) & frozen_ids:
        raise AssertionError("Frozen Champion parameter would enter optimizer")
    latent, context, mask, native, base_value = _frozen_features(champion, decisions)
    with torch.no_grad():
        computed_logits = gate(latent, context, mask, native,
                               temperature=TEMPERATURE)
        policy = torch.distributions.Categorical(logits=computed_logits)
        stored_logits = torch.as_tensor(np.asarray(
            [item["old_calibrated_logits"] for item in decisions], dtype=np.float32),
            device=latent.device)
        logits_error = (computed_logits - stored_logits).abs().max().item()
        action = torch.as_tensor([FAMILIES.index(item["selected_family"])
                                  for item in decisions], device=latent.device)
        saved_log_prob = torch.as_tensor([item["gate_log_prob"] for item in decisions],
                                         device=latent.device)
        predicted_value = critic(base_value, latent, context, mask).reshape(-1)
        saved_value = torch.as_tensor([item["value"] for item in decisions],
                                      device=latent.device)
        saved_base_value = torch.as_tensor([item["frozen_base_value"]
                                            for item in decisions], device=latent.device)
        log_error = (policy.log_prob(action) - saved_log_prob).abs().max().item()
        value_error = (predicted_value - saved_value).abs().max().item()
        base_error = (base_value.reshape(-1) - saved_base_value).abs().max().item()
        stored_probability = torch.as_tensor(np.asarray(
            [item["masked_probability"] for item in decisions], dtype=np.float32),
            device=latent.device)
        probability_error = (policy.probs - stored_probability).abs().max().item()
    if (logits_error > 1e-5 or log_error > 1e-5 or
            value_error > 1e-4 or base_error > 1e-4
            or probability_error > 1e-6):
        raise ValueError("Stored old T=2 Policy / Critic does not reproduce")
    return {"rollout": diagnostics, "advantage_raw": raw,
            "advantage_normalized": normalized_stats,
            "advantage_by_family": family_advantage,
            "advantage_by_terminal": terminal_advantage,
            "stored_log_prob_max_error": log_error,
            "stored_calibrated_logits_max_error": logits_error,
            "stored_probability_max_error": probability_error,
            "stored_value_max_error": value_error,
            "stored_base_value_max_error": base_error,
            "trainable_gate_parameters": sum(p.numel() for p in gate.parameters()),
            "trainable_critic_parameters": sum(p.numel() for p in critic.parameters()),
            "optimizer_parameter_ids_disjoint_from_champion": True,
            "frozen_champion_digest": _weight_digest(champion)}


@dataclass(frozen=True)
class StrategicUpdateResult:
    accepted: bool
    reason: str
    metrics: dict[str, Any]


def save_strategic_artifact(path: Path, gate: StrategicActionGate,
                            critic: StrategicSurfaceCritic,
                            manifest: dict[str, Any]) -> None:
    """Independent experiment artifact; never writes to Champion model paths."""
    path.mkdir(parents=True, exist_ok=False)
    torch.save({"gate": gate.state_dict(), "critic": critic.state_dict()},
               path / "weights.pt")
    (path / "manifest.json").write_text(json.dumps(
        manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_strategic_artifact(path: Path, latent_dim: int,
                            ) -> tuple[StrategicActionGate, StrategicSurfaceCritic, dict]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["temperature"] != TEMPERATURE or
            tuple(manifest["strategic_family_order"]) != FAMILIES or
            manifest.get("runtime_context_version", RUNTIME_CONTEXT_VERSION)
            != RUNTIME_CONTEXT_VERSION):
        raise ValueError("Strategic artifact schema or temperature mismatch")
    weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    gate = StrategicActionGate(latent_dim)
    critic = StrategicSurfaceCritic(latent_dim)
    gate.load_state_dict(weights["gate"])
    critic.load_state_dict(weights["critic"])
    return gate.eval(), critic.eval(), manifest


def one_strategic_ppo_update(
    champion, gate: StrategicActionGate, critic: StrategicSurfaceCritic,
    episodes: list[dict[str, Any]], *, actor_lr: float = 1e-4,
    critic_lr: float = 3e-4, batch_size: int = 64, epochs: int = 2,
    clip_range: float = .1, entropy_coef: float = .005,
    training_seed: int = 316001, target_mean_kl: float = .01,
    max_kl_limit: float = .10,
) -> StrategicUpdateResult:
    """At most one update; rejection restores both experiment-only modules."""
    if epochs != 2 or batch_size != 64:
        raise ValueError("Step 3A-16 fixes one two-epoch, batch-64 update")
    diagnostics = rollout_diagnostics(episodes)
    decisions = [item for episode in episodes for item in episode["decisions"]]
    if len(decisions) < 2:
        raise ValueError("Too few actually delegated Strategic transitions")
    per_episode = [strategic_macro_gae(episode["transitions"])
                   for episode in episodes if episode["transitions"]]
    advantages = np.concatenate([item[0] for item in per_episode])
    returns = np.concatenate([item[1] for item in per_episode])
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
        raise ValueError("Nonfinite advantage or return")
    raw = distribution(advantages)
    if raw["std"] is None or raw["std"] < 1e-6:
        raise ValueError("Advantage standard deviation is nearly zero")
    normalized = (advantages - raw["mean"]) / (raw["std"] + 1e-8)
    normal = distribution(normalized)
    if np.max(np.abs(normalized)) > 10:
        raise ValueError("Extreme single-outlier advantage; update refused")
    by_family = {family: {"n": len(values), "mean": mean(values)}
                 for family, values in _group_values(
                     decisions, advantages, "selected_family").items()}
    done_flags = [item["done"] for episode in episodes for item in episode["transitions"]]
    by_terminal = {label: distribution(advantages[np.asarray(done_flags) == wanted])
                   for label, wanted in (("terminal", True), ("nonterminal", False))}
    gate_ids = {id(parameter) for parameter in gate.parameters()}
    critic_ids = {id(parameter) for parameter in critic.parameters()}
    frozen_ids = {id(parameter) for parameter in champion.model.policy.parameters()}
    if gate_ids & critic_ids or (gate_ids | critic_ids) & frozen_ids:
        raise AssertionError("Frozen Champion or cross-head parameter in optimizer")
    champion_digest = _weight_digest(champion)
    latent, context, mask, native, base_value = _frozen_features(champion, decisions)
    action = torch.as_tensor([FAMILIES.index(item["selected_family"])
                              for item in decisions], device=latent.device)
    old_log_prob = torch.as_tensor([item["gate_log_prob"] for item in decisions],
                                   dtype=torch.float32, device=latent.device)
    target = torch.as_tensor(returns, device=latent.device)
    normalized_tensor = torch.as_tensor(normalized, device=latent.device)
    with torch.no_grad():
        old_logits = gate(latent, context, mask, native, temperature=TEMPERATURE)
        old_dist = torch.distributions.Categorical(logits=old_logits)
        old_value = critic(base_value, latent, context, mask).reshape(-1)
        if not torch.allclose(old_dist.log_prob(action), old_log_prob, atol=1e-5):
            raise ValueError("Stored old T=2 log_prob differs from current Gate")
        if not torch.allclose(old_dist.probs, torch.as_tensor(np.asarray(
                [item["masked_probability"] for item in decisions], dtype=np.float32),
                device=latent.device), atol=1e-6):
            raise ValueError("Stored old T=2 distribution differs from current Gate")
        if not torch.allclose(old_value, torch.as_tensor([item["value"]
                  for item in decisions], device=latent.device), atol=1e-4):
            raise ValueError("Stored Strategic value differs from current Critic")
        if not torch.allclose(base_value.reshape(-1), torch.as_tensor(
                [item["frozen_base_value"] for item in decisions],
                device=latent.device), atol=1e-4):
            raise ValueError("Stored frozen Champion value differs from observation")
    before_gate = deepcopy(gate.state_dict())
    before_critic = deepcopy(critic.state_dict())
    actor_optimizer = torch.optim.Adam(gate.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    optimizer_ids = {id(parameter) for group in actor_optimizer.param_groups +
                     critic_optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != gate_ids | critic_ids:
        raise AssertionError("Optimizer parameter IDs do not match Gate and Critic")
    generator = torch.Generator().manual_seed(training_seed)
    actor_losses, critic_losses, entropy_losses, clip_fractions = [], [], [], []
    for _ in range(epochs):
        order = torch.randperm(len(decisions), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            current = torch.distributions.Categorical(logits=gate(
                latent[indices], context[indices], mask[indices], native[indices],
                temperature=TEMPERATURE))
            ratio = torch.exp(current.log_prob(action[indices]) - old_log_prob[indices])
            clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            entropy = current.entropy().mean()
            actor_loss = -torch.minimum(ratio * normalized_tensor[indices],
                                         clipped_ratio * normalized_tensor[indices]).mean()
            entropy_loss = -entropy_coef * entropy
            actor_optimizer.zero_grad(set_to_none=True)
            (actor_loss + entropy_loss).backward()
            nn.utils.clip_grad_norm_(gate.parameters(), .5)
            actor_optimizer.step()
            prediction = critic(base_value[indices], latent[indices],
                                context[indices], mask[indices]).reshape(-1)
            critic_loss = torch.mean((prediction - target[indices]) ** 2)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), .5)
            critic_optimizer.step()
            actor_losses.append(float(actor_loss.detach()))
            critic_losses.append(float(critic_loss.detach()))
            entropy_losses.append(float(entropy_loss.detach()))
            clip_fractions.append(float(((ratio < 1 - clip_range) |
                                         (ratio > 1 + clip_range)).float().mean()))
    with torch.no_grad():
        new_dist = torch.distributions.Categorical(logits=gate(
            latent, context, mask, native, temperature=TEMPERATURE))
        kl = torch.distributions.kl_divergence(old_dist, new_dist).cpu().numpy()
        ratios = torch.exp(new_dist.log_prob(action) - old_log_prob).cpu().numpy()
        old_probs = old_dist.probs.cpu().numpy()
        new_probs = new_dist.probs.cpu().numpy()
        old_entropy = old_dist.entropy().cpu().numpy()
        new_entropy = new_dist.entropy().cpu().numpy()
    if not np.isfinite(kl).all() or not np.isfinite(ratios).all():
        raise ValueError("Nonfinite post-update KL or ratio")
    candidate_counts = np.asarray([item["candidate_count"] for item in decisions])
    kl_by_count = {bucket: distribution(kl[indices])
                   for bucket, indices in _bucket_indices(candidate_counts).items()}
    entropy_by_count = {bucket: {"before": distribution(old_entropy[indices]),
                                 "after": distribution(new_entropy[indices])}
                        for bucket, indices in _bucket_indices(candidate_counts).items()}
    sampled_by_family = {family: distribution(kl[np.asarray(
        [item["selected_family"] == family for item in decisions])])
        for family in FAMILIES}
    family_shift = {family: {"available": int(mask[:, index].sum()),
                             "prob_before": float(old_probs[mask[:, index].cpu().numpy(), index].mean())
                             if bool(mask[:, index].any()) else None,
                             "prob_after": float(new_probs[mask[:, index].cpu().numpy(), index].mean())
                             if bool(mask[:, index].any()) else None,
                             "top_before": int((old_probs.argmax(axis=1) == index).sum()),
                             "top_after": int((new_probs.argmax(axis=1) == index).sum()),
                             "above_095_before": int((old_probs[:, index] > .95).sum()),
                             "above_095_after": int((new_probs[:, index] > .95).sum())}
                    for index, family in enumerate(FAMILIES)}
    peaked_before = int((old_probs.max(axis=1) > .95).sum())
    peaked_after = int((new_probs.max(axis=1) > .95).sum())
    collapsed = bool(peaked_after > max(peaked_before + len(decisions) * .25,
                                      len(decisions) * .8) or
                     max((values["top_after"] for values in family_shift.values()), default=0)
                     > len(decisions) * .95)
    reason = ("KL_EXCEEDED" if kl.mean() > target_mean_kl or kl.max() > max_kl_limit
              else "FAMILY_COLLAPSE" if collapsed else "ACCEPTED")
    if reason != "ACCEPTED":
        gate.load_state_dict(before_gate)
        critic.load_state_dict(before_critic)
    if _weight_digest(champion) != champion_digest:
        raise AssertionError("Frozen Champion weights changed")
    metrics = {
        "rollout": diagnostics,
        "advantage_raw": raw, "advantage_normalized": normal,
        "advantage_by_terminal": by_terminal, "advantage_by_family": by_family,
        "trainable_parameters": sum(p.numel() for p in gate.parameters()) +
                                sum(p.numel() for p in critic.parameters()),
        "optimizer_parameter_ids_verified": True, "champion_digest": champion_digest,
        "actor_loss_mean": mean(actor_losses), "critic_loss_mean": mean(critic_losses),
        "entropy_loss_mean": mean(entropy_losses),
        "clip_fraction_mean": mean(clip_fractions), "ratio": distribution(ratios),
        "kl": distribution(kl), "kl_by_candidate_count": kl_by_count,
        "kl_by_sampled_family": sampled_by_family,
        "entropy_before": distribution(old_entropy),
        "entropy_after": distribution(new_entropy),
        "entropy_by_candidate_count": entropy_by_count,
        "family_probability_shift": family_shift,
        "peaked_state_count_before": peaked_before,
        "peaked_state_count_after": peaked_after,
        "family_collapse": collapsed,
        "minibatches": len(actor_losses), "epochs": epochs,
    }
    return StrategicUpdateResult(reason == "ACCEPTED", reason, metrics)


def _group_values(decisions: list[dict], values: np.ndarray, key: str):
    groups: dict[str, list[float]] = defaultdict(list)
    for decision, value in zip(decisions, values, strict=True):
        groups[decision[key]].append(float(value))
    return groups


def _bucket_indices(counts: np.ndarray) -> dict[str, np.ndarray]:
    return {"2": counts == 2, "3": counts == 3, "4+": counts >= 4}
