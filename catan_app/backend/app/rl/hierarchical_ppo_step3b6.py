"""Investment-only hierarchical PPO primitives (Step 3B-6).

The four-family rollout mask is a *training* mask, distinct from the seven-
family diagnostic mask. No Champion logit, Action, Intent or Planner score is
used as a learning target.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch
from torch import nn

from .hierarchical_l3_shadow_step3b5 import (
    L3ShadowActor, L3DecisionFacts, ROUTE_ORDER, differentiable_soft_plan,
)
from .hierarchical_shadow_controller_step3b4 import FAMILY_ORDER, HierarchicalRoute
from .strategic_gate_ppo_step3a16 import distribution
from .victory_plan_policy import GLOBAL_FEATURE_DIM, VictoryPlanShadowHeads
from .runtime_context import RUNTIME_CONTEXT_SIZE


SCHEMA = "hierarchical_investment_ppo_step3b6_v1"
INVESTMENT_FAMILIES = ("BUILD_ROAD", "BUY_DEVELOPMENT", "BANK_TRADE", "END_TURN")
GAMMA = .99
GAE_LAMBDA = .95


class HierarchicalInvestmentPolicy(nn.Module):
    """Actor-only gradient path: L3 -> soft Plan/Priority -> L1/L2."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.plan_heads = VictoryPlanShadowHeads(latent_dim)
        self.l3_actor = L3ShadowActor(latent_dim)

    def forward(self, latent: torch.Tensor, runtime: torch.Tensor,
                global_features: torch.Tensor, candidates: torch.Tensor,
                plan_mask: torch.Tensor, facts: L3DecisionFacts):
        plan = self.plan_heads(latent.detach(), runtime, global_features,
                               candidates, plan_mask)
        bridge = differentiable_soft_plan(plan, candidates)
        l3 = self.l3_actor(latent.detach(), runtime, global_features, bridge, facts)
        return plan, bridge, l3


class HierarchicalInvestmentCritic(nn.Module):
    """Independent critic; it never backpropagates through Actor L1/L2."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        fixed = RUNTIME_CONTEXT_SIZE + GLOBAL_FEATURE_DIM + 6 + 4 + 1 + len(FAMILY_ORDER)
        self.encoder = nn.Sequential(nn.Linear(latent_dim + fixed, hidden_dim), nn.Tanh())
        self.value_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, latent: torch.Tensor, runtime: torch.Tensor,
                global_features: torch.Tensor, bridge, family_mask: torch.Tensor):
        scales = bridge.weighted_features.new_tensor((10., 10., 1., 1., 11., 1.))
        features = torch.cat((latent.detach(), runtime, global_features,
                              bridge.weighted_features.detach() / scales,
                              bridge.expected_priorities.detach(),
                              bridge.available.to(runtime.dtype).unsqueeze(-1),
                              family_mask.to(runtime.dtype)), dim=-1)
        return self.value_head(self.encoder(features)).squeeze(-1)


def investment_training_mask(route: HierarchicalRoute, diagnostic_mask,
                             executable_mask) -> tuple[bool, ...]:
    """Only legal, executable Investment families can enter an Actor sample."""
    if len(diagnostic_mask) != len(FAMILY_ORDER) or len(executable_mask) != len(FAMILY_ORDER):
        raise ValueError("Seven-family mask required")
    if route is not HierarchicalRoute.INVESTMENT:
        return (False,) * len(FAMILY_ORDER)
    allowed = set(INVESTMENT_FAMILIES)
    return tuple(bool(diagnostic_mask[i] and executable_mask[i]
                      and family in allowed) for i, family in enumerate(FAMILY_ORDER))


def masked_probabilities(raw_logits: torch.Tensor, mask: torch.Tensor):
    """Strict zeros for excluded families, including all-invalid rows."""
    mask = mask.bool()
    available = mask.any(dim=-1)
    masked = raw_logits.masked_fill(~mask, -1e8)
    probabilities = torch.softmax(masked, dim=-1) * mask.to(raw_logits.dtype)
    probabilities = torch.where(available.unsqueeze(-1), probabilities,
                                torch.zeros_like(probabilities))
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
    return masked, probabilities, entropy, available


def macro_gae(transitions: list[dict], gamma: float = GAMMA,
              gae_lambda: float = GAE_LAMBDA) -> tuple[np.ndarray, np.ndarray]:
    """Lambda is per successive delegated decision; gamma uses policy-step duration."""
    if not transitions:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    if not transitions[-1]["done"] or transitions[-1]["truncated"]:
        raise ValueError("Only complete terminal episodes enter PPO")
    advantages = np.zeros(len(transitions), np.float64)
    carry = 0.
    for i in range(len(transitions) - 1, -1, -1):
        item = transitions[i]
        k = item["macro_duration"]
        if k < 1 or item["truncated"] or not isfinite(item["reward_accumulated"]):
            raise ValueError("Invalid semi-MDP macro")
        if item["done"]:
            next_value, carry = 0., 0.
        else:
            if i + 1 >= len(transitions) or not item["next_actual_delegated"]:
                raise ValueError("Nonterminal macro must close at next delegated state")
            if item["close_policy_step"] != transitions[i + 1]["start_policy_step"]:
                raise ValueError("Discontinuous delegated boundary")
            next_value = transitions[i + 1]["value"]
        delta = item["reward_accumulated"] + gamma ** k * next_value - item["value"]
        carry = delta + gamma ** k * gae_lambda * carry
        advantages[i] = carry
    values = np.asarray([t["value"] for t in transitions], np.float64)
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


@dataclass(frozen=True)
class UpdateResult:
    accepted: bool
    reason: str
    metrics: dict


def _batch(decisions: list[dict]) -> dict[str, torch.Tensor | L3DecisionFacts]:
    def tensor(key: str, dtype=torch.float32):
        return torch.as_tensor(np.asarray([item[key] for item in decisions]), dtype=dtype)

    mask = tensor("family_mask", torch.bool)
    route = torch.zeros((len(decisions), len(ROUTE_ORDER)), dtype=torch.float32)
    route[:, ROUTE_ORDER.index(HierarchicalRoute.INVESTMENT)] = 1
    # INVESTMENT has no buildable S/C, so its diagnostic mask is exactly the
    # L3 legal input used at rollout; the four-way comparison mask is narrower.
    legal = tensor("diagnostic_family_mask", torch.bool)
    facts = L3DecisionFacts(route, legal, mask,
                            torch.zeros((len(decisions), 1)),
                            torch.ones(len(decisions), dtype=torch.bool))
    return {
        "latent": tensor("actor_latent"), "runtime": tensor("runtime_context"),
        "global": tensor("global_plan_features"),
        "candidates": tensor("candidate_plan_features"),
        "plan_mask": tensor("candidate_plan_mask", torch.bool),
        "facts": facts, "family_mask": mask,
        "action": tensor("selected_index", torch.long),
        "old_log_prob": tensor("old_log_prob"),
        "old_probability": tensor("old_probability"),
        "old_value": tensor("old_value"),
    }


def _select(batch: dict, indices: list[int]) -> dict:
    facts = batch["facts"]
    return {**{key: value[indices] for key, value in batch.items()
               if key != "facts"},
            "facts": L3DecisionFacts(
                facts.route_one_hot[indices], facts.legal_family_mask[indices],
                facts.comparison_family_mask[indices],
                facts.construction_commit[indices], facts.l3_route_available[indices])}


def _forward(policy: HierarchicalInvestmentPolicy,
             critic: HierarchicalInvestmentCritic, batch: dict):
    plan, bridge, l3 = policy(
        batch["latent"], batch["runtime"], batch["global"],
        batch["candidates"], batch["plan_mask"], batch["facts"])
    value = critic(batch["latent"], batch["runtime"], batch["global"],
                   bridge, batch["family_mask"])
    return plan, bridge, l3, value


def preflight(champion, policy: HierarchicalInvestmentPolicy,
              critic: HierarchicalInvestmentCritic, episodes: list[dict],
              *, minimum_transitions: int = 256) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """No optimizer exists until all episode, reward and stored-policy checks pass."""
    if not episodes or any(not item["terminal"] or item["truncated"] for item in episodes):
        raise ValueError("All training episodes must terminate without truncation")
    decisions = [d for episode in episodes for d in episode["decisions"]]
    transitions = [t for episode in episodes for t in episode["transitions"]]
    if len(decisions) != len(transitions) or len(decisions) < minimum_transitions:
        raise ValueError("Missing/off-policy or insufficient delegated transitions")
    if any(item["delegation_probability"] != .10 for item in episodes):
        raise ValueError("Training delegation rate changed")
    for episode in episodes:
        if len(episode["decisions"]) != len(episode["transitions"]):
            raise ValueError("Decision/transition mismatch")
        for i, (decision, transition) in enumerate(zip(
                episode["decisions"], episode["transitions"], strict=True)):
            mask = decision["family_mask"]
            diagnostic = decision["diagnostic_family_mask"]
            if (decision["route"] != HierarchicalRoute.INVESTMENT.value or
                    decision["phase"] != "action" or
                    decision["selected_family"] not in INVESTMENT_FAMILIES or
                    not mask[decision["selected_index"]] or
                    len(mask) != len(FAMILY_ORDER) or
                    len(diagnostic) != len(FAMILY_ORDER) or
                    any(diagnostic[index] for index in (0, 1)) or
                    any(enabled and not diagnostic[index]
                        for index, enabled in enumerate(mask)) or
                    mask[FAMILY_ORDER.index("PLAYER_TRADE")] or
                    transition["decision_index"] != i or
                    transition["selected_family"] != decision["selected_family"] or
                    transition["family_mask"] != mask or
                    transition["old_log_prob"] != decision["old_log_prob"]):
                raise ValueError("Protected, nondelegated or mask-mismatched Actor sample")
            expected = sum(GAMMA ** j * reward for j, reward in enumerate(
                transition["reward_steps"]))
            if (len(transition["reward_steps"]) != transition["macro_duration"] or
                    abs(expected - transition["reward_accumulated"]) > 1e-8 or
                    transition["truncated"] or
                    (transition["done"] and transition["discount"] != 0.) or
                    (not transition["done"] and abs(
                        transition["discount"] - GAMMA ** transition["macro_duration"]) > 1e-8)):
                raise ValueError("Macro reward / bootstrap mismatch")
            if any(not isfinite(v) for v in transition["reward_steps"]):
                raise ValueError("Nonfinite reward")
    per_episode = [macro_gae(e["transitions"]) for e in episodes if e["transitions"]]
    advantages = np.concatenate([pair[0] for pair in per_episode])
    returns = np.concatenate([pair[1] for pair in per_episode])
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
        raise ValueError("Nonfinite Advantage/return")
    if np.std(advantages) < 1e-6:
        raise ValueError("Near-zero Advantage variance")
    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    if np.max(np.abs(normalized)) > 10:
        raise ValueError("Extreme Advantage outlier")
    batch = _batch(decisions)
    champion.model.policy.requires_grad_(False)
    with torch.no_grad():
        observations = np.asarray([d["observation_v2"] for d in decisions], np.float32)
        observation_tensor, _ = champion.model.policy.obs_to_tensor(observations)
        latent = champion.model.policy.mlp_extractor.forward_actor(observation_tensor)
        latent_error = float((latent - batch["latent"]).abs().max())
        plan, bridge, l3, values = _forward(policy, critic, batch)
        selected = l3.probabilities.gather(1, batch["action"].unsqueeze(-1)).squeeze(-1)
        log_error = float((selected.log() - batch["old_log_prob"]).abs().max())
        probability_error = float((l3.probabilities - batch["old_probability"]).abs().max())
        value_error = float((values - batch["old_value"]).abs().max())
        masked_error = float(l3.probabilities[~batch["family_mask"]].abs().max())
        plan_entropy_error = max(abs(float(plan.entropy[i]) - d["plan_entropy"])
                                 for i, d in enumerate(decisions))
    if (latent_error > 1e-5 or log_error > 1e-5 or probability_error > 1e-6 or
            value_error > 1e-5 or masked_error != 0 or plan_entropy_error > 1e-5):
        raise ValueError("Stored frozen latent, old Actor or Critic did not replay")
    actor_ids = {id(p) for p in policy.parameters()}
    critic_ids = {id(p) for p in critic.parameters()}
    frozen_ids = {id(p) for p in champion.model.policy.parameters()}
    if actor_ids & critic_ids or (actor_ids | critic_ids) & frozen_ids:
        raise AssertionError("Trainable/Frozen parameter ID overlap")
    done = np.asarray([t["done"] for t in transitions], dtype=bool)
    family_counts = {name: sum(d["selected_family"] == name for d in decisions)
                     for name in INVESTMENT_FAMILIES}
    report = {
        "games": len(episodes), "transitions": len(transitions),
        "family_selected": family_counts,
        "family_available": {name: sum(d["family_mask"][FAMILY_ORDER.index(name)]
                                      for d in decisions) for name in INVESTMENT_FAMILIES},
        "candidate_count": {str(n): sum(sum(d["family_mask"]) == n for d in decisions)
                            for n in range(1, 5)},
        "l3_entropy": distribution(np.asarray([d["l3_entropy"] for d in decisions])),
        "plan_entropy": distribution(np.asarray([d["plan_entropy"] for d in decisions])),
        "macro_duration": distribution(np.asarray([t["macro_duration"] for t in transitions])),
        "macro_reward": distribution(np.asarray([t["reward_accumulated"] for t in transitions])),
        "terminal_transitions": int(done.sum()),
        "nonterminal_transitions": int((~done).sum()),
        "protected_crossed": sum(t["protected_crossed"] for t in transitions),
        "resolver_crossed": sum(t["resolver_crossed"] for t in transitions),
        "nondelegated_investment_crossed": sum(
            t["nondelegated_investment_crossed"] for t in transitions),
        "advantage": distribution(advantages),
        "normalized_advantage": distribution(normalized),
        "terminal_advantage": distribution(advantages[done]),
        "nonterminal_advantage": distribution(advantages[~done]),
        "terminal_absolute_advantage_share": float(
            np.abs(advantages[done]).sum() / np.abs(advantages).sum()),
        "family_advantage": {name: distribution(advantages[np.asarray(
            [d["selected_family"] == name for d in decisions])])
            for name in INVESTMENT_FAMILIES},
        "critic_pre_mse": float(np.mean((returns - batch["old_value"].numpy()) ** 2)),
        "stored_replay_error": {"latent": latent_error, "log_prob": log_error,
                                "probability": probability_error, "value": value_error,
                                "plan_entropy": plan_entropy_error},
        "actor_parameters": sum(p.numel() for p in policy.parameters()),
        "critic_parameters": sum(p.numel() for p in critic.parameters()),
        "frozen_parameter_ids_disjoint": True,
    }
    return report, batch, advantages, returns


def one_guarded_update(
    champion, policy: HierarchicalInvestmentPolicy,
    critic: HierarchicalInvestmentCritic, episodes: list[dict], *,
    minibatch_seed: int, actor_lr: float = 1e-4, critic_lr: float = 3e-4,
    batch_size: int = 64, epochs: int = 2, clip_range: float = .1,
    entropy_coef: float = .005, minimum_transitions: int = 256,
) -> UpdateResult:
    """Exactly one two-epoch PPO pass; rejected candidates roll back in memory."""
    report, batch, advantages, returns = preflight(
        champion, policy, critic, episodes, minimum_transitions=minimum_transitions)
    old_actor, old_critic = deepcopy(policy.state_dict()), deepcopy(critic.state_dict())
    old_probability = batch["old_probability"].clone()
    normalized = torch.as_tensor((advantages - advantages.mean()) /
                                 (advantages.std() + 1e-8), dtype=torch.float32)
    target = torch.as_tensor(returns, dtype=torch.float32)
    actor_ids = {id(p) for p in policy.parameters()}
    critic_ids = {id(p) for p in critic.parameters()}
    actor_optimizer = torch.optim.Adam(policy.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    optimizer_ids = {id(p) for group in actor_optimizer.param_groups + critic_optimizer.param_groups
                     for p in group["params"]}
    if optimizer_ids != actor_ids | critic_ids:
        raise AssertionError("Optimizer trainable boundary mismatch")
    generator = torch.Generator().manual_seed(minibatch_seed)
    losses, value_losses, entropy_losses, clips = [], [], [], []
    gradients = {"l3": [], "priority": [], "plan_scorer": []}
    for _ in range(epochs):
        order = torch.randperm(len(advantages), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            minibatch = _select(batch, indices)
            plan, bridge, l3, _ = _forward(policy, critic, minibatch)
            selected = l3.probabilities.gather(1, minibatch["action"].unsqueeze(-1)).squeeze(-1)
            ratio = torch.exp(selected.clamp_min(1e-30).log() - minibatch["old_log_prob"])
            clipped = ratio.clamp(1 - clip_range, 1 + clip_range)
            actor_loss = -torch.minimum(ratio * normalized[indices],
                                         clipped * normalized[indices]).mean()
            entropy_loss = -entropy_coef * l3.entropy.mean()
            actor_optimizer.zero_grad(set_to_none=True)
            (actor_loss + entropy_loss).backward()
            for name, module in (("l3", policy.l3_actor),
                                 ("priority", policy.plan_heads.priority_head),
                                 ("plan_scorer", policy.plan_heads.plan_scorer)):
                gradients[name].append(float(torch.sqrt(sum(
                    (p.grad.detach() ** 2).sum() for p in module.parameters()
                    if p.grad is not None))))
            nn.utils.clip_grad_norm_(policy.parameters(), .5)
            actor_optimizer.step()
            # Critic sees detached L1/L2 representation; actor loss alone trains those heads.
            with torch.no_grad():
                _, detached_bridge, _ = policy(
                    minibatch["latent"], minibatch["runtime"], minibatch["global"],
                    minibatch["candidates"], minibatch["plan_mask"], minibatch["facts"])
            prediction = critic(minibatch["latent"], minibatch["runtime"],
                                minibatch["global"], detached_bridge,
                                minibatch["family_mask"])
            value_loss = ((prediction - target[indices]) ** 2).mean()
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), .5)
            critic_optimizer.step()
            losses.append(float(actor_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            entropy_losses.append(float(entropy_loss.detach()))
            clips.append(float(((ratio < 1 - clip_range) |
                                (ratio > 1 + clip_range)).float().mean()))
    with torch.no_grad():
        _, _, updated, values = _forward(policy, critic, batch)
        new_probability = updated.probabilities
        kl = (old_probability * (
            old_probability.clamp_min(1e-30).log() -
            new_probability.clamp_min(1e-30).log())).sum(-1).numpy()
        selected_new = new_probability.gather(1, batch["action"].unsqueeze(-1)).squeeze(-1)
        ratios = torch.exp(selected_new.clamp_min(1e-30).log() - batch["old_log_prob"]).numpy()
        entropy_after = updated.entropy.numpy()
    old_np, new_np = old_probability.numpy(), new_probability.numpy()
    mask_np = batch["family_mask"].numpy()
    counts = mask_np.sum(axis=1)
    family_shift = {}
    for index, family in enumerate(FAMILY_ORDER):
        available = mask_np[:, index]
        family_shift[family] = {
            "available": int(available.sum()),
            "probability_before": float(old_np[available, index].mean())
            if available.any() else None,
            "probability_after": float(new_np[available, index].mean())
            if available.any() else None,
            "top_before": int((old_np.argmax(axis=1) == index).sum()),
            "top_after": int((new_np.argmax(axis=1) == index).sum()),
            "above_095_before": int((old_np[:, index] > .95).sum()),
            "above_095_after": int((new_np[:, index] > .95).sum()),
        }
    peaked_before = int((old_np.max(axis=1) > .95).sum())
    peaked_after = int((new_np.max(axis=1) > .95).sum())
    family_collapse = (peaked_after > max(peaked_before + .25 * len(counts),
                                         .8 * len(counts)) or
                       max(item["top_after"] for item in family_shift.values())
                       > .95 * len(counts))
    if not np.isfinite(kl).all() or not np.isfinite(ratios).all():
        reason = "NONFINITE_POST_UPDATE"
    elif float(kl.mean()) > .01 or float(kl.max()) > .10:
        reason = "KL_EXCEEDED"
    elif np.mean(entropy_after) < .5 * report["l3_entropy"]["mean"]:
        reason = "ENTROPY_COLLAPSE"
    elif family_collapse:
        reason = "FAMILY_COLLAPSE"
    else:
        reason = "ACCEPTED"
    def module_delta(module, old, prefix):
        return float(torch.sqrt(sum(((value.detach() - old[f"{prefix}{name}"]) ** 2).sum()
                                    for name, value in module.named_parameters())))
    # Deltas are measured from the trained candidate before any rollback.
    deltas = {
        "l3_delta": module_delta(policy.l3_actor, old_actor, "l3_actor."),
        "priority_delta": module_delta(policy.plan_heads.priority_head,
                                       old_actor, "plan_heads.priority_head."),
        "plan_scorer_delta": module_delta(policy.plan_heads.plan_scorer,
                                          old_actor, "plan_heads.plan_scorer."),
    }
    if reason != "ACCEPTED":
        policy.load_state_dict(old_actor)
        critic.load_state_dict(old_critic)
    metrics = {**report, "actor_loss_mean": float(np.mean(losses)),
               "critic_loss_mean": float(np.mean(value_losses)),
               "entropy_loss_mean": float(np.mean(entropy_losses)),
               "clip_fraction": float(np.mean(clips)),
               "ratio": distribution(ratios), "kl": distribution(kl),
               "kl_by_candidate_count": {str(n): distribution(kl[counts == n])
                                         for n in range(1, 5)},
               "l3_entropy_after": distribution(entropy_after),
               "entropy_by_candidate_count": {str(n): {
                   "before": distribution(np.asarray([
                       d["l3_entropy"] for d, count in zip(
                           [item for episode in episodes for item in episode["decisions"]],
                           counts, strict=True) if count == n])),
                   "after": distribution(entropy_after[counts == n])}
                   for n in range(1, 5)},
               "family_probability_shift": family_shift,
               "peaked_before": peaked_before, "peaked_after": peaked_after,
               "family_collapse": bool(family_collapse),
               "gradient_norm": {name: distribution(np.asarray(values))
                                 for name, values in gradients.items()},
               **deltas,
               "critic_post_mse": float(np.mean((returns - values.numpy()) ** 2)),
               "minibatches": len(losses), "epochs": epochs,
               "optimizer_parameter_ids_verified": True}
    return UpdateResult(reason == "ACCEPTED", reason, metrics)
