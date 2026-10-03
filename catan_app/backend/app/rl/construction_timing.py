"""Step 3A-8: non-invasive construction-timing delegation primitives.

Only BUILD_NOW versus DEFER belongs to the gate. Concrete actions are produced
from the pre-action state by frozen executors; neither uses the Champion's
already-selected action as a label or a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import inspect
from typing import Callable

import numpy as np
import torch
from torch import nn

from app.domain.actions import Action, BuildCityAction, BuildSettlementAction
from app.domain.game import GameActionError, GameState, apply_action

from .action_space import ACTION_CATALOG, get_action_mask, id_to_action
from .construction_shadow import ConstructionSurface, detect_construction_surface
from .observation import encode_observation, get_observation
from .runtime_context import RUNTIME_CONTEXT_SIZE, encode_runtime_context
from .settlement_planning_agent import SettlementPlanningAgent


BUILD_NOW = 0
DEFER = 1
TIMING_SUBTYPES = ("SETTLEMENT_ONLY", "CITY_ONLY")


def timing_surface(game: GameState, player_id: int) -> ConstructionSurface:
    """Retain all Step 3A-7 hard exclusions; do not delegate dual-build states."""
    surface = detect_construction_surface(game, player_id)
    if surface.eligible and surface.subtype not in TIMING_SUBTYPES:
        return ConstructionSurface(surface.subtype, "DUAL_BUILD_DEFERRED",
                                   surface.legal_settlement_count,
                                   surface.legal_city_count)
    return surface


def _planning_settings(champion: SettlementPlanningAgent) -> dict:
    names = inspect.signature(SettlementPlanningAgent.__init__).parameters
    return {name: getattr(champion, name) for name in names
            if name not in {"self", "policy"}}


def _nonconstruction_mask(game: GameState, player_id: int) -> np.ndarray:
    mask = get_action_mask(game, player_id).copy()
    for item in ACTION_CATALOG:
        if isinstance(item.action, (BuildSettlementAction, BuildCityAction)):
            mask[item.id] = False
    return mask


def _validate_on_copy(game: GameState, player_id: int, action: Action) -> None:
    clone = deepcopy(game)
    apply_action(clone, player_id, action, clone.revision, "timing-candidate-check")


class _MaskedFrozenPolicy:
    def __init__(self, policy):
        self._policy = policy

    def __getattr__(self, name):
        return getattr(self._policy, name)

    def select_action(self, game: GameState, player_id: int) -> Action:
        mask = _nonconstruction_mask(game, player_id)
        if not mask.any():
            raise ValueError("NO_NONCONSTRUCTION_ACTION")
        observation = encode_observation(get_observation(
            game, player_id, version=self._policy.manifest.observation_version))
        action_id, _ = self._policy.model.predict(
            observation, deterministic=True, action_masks=mask)
        selected_id = int(np.asarray(action_id).item())
        if not mask[selected_id]:
            raise AssertionError("Masked frozen PPO selected an illegal action")
        return id_to_action(selected_id)


class _NonConstructionPlanner(SettlementPlanningAgent):
    """Run the frozen planner under a build-only prohibition, not a posthoc action rewrite."""

    def _best_city(self, game, player_id):
        return None

    def _best_settlement(self, game, player_id):
        return None

    def _city_action(self, game, player_id, base_action):
        return base_action

    def _construction_choice_mask(self, game, player_id):
        return _nonconstruction_mask(game, player_id)


@dataclass(frozen=True)
class TimingCandidates:
    surface: ConstructionSurface
    build_action: Action | None
    defer_action: Action | None
    build_error: str | None = None
    defer_error: str | None = None

    @property
    def delegatable(self) -> bool:
        return (self.surface.eligible and self.build_action is not None
                and self.defer_action is not None)


class FrozenConstructionExecutors:
    """Use the Champion's family-internal vertex choice, never its family decision."""

    def __init__(self, champion: SettlementPlanningAgent):
        self.champion = champion
        self.defer_planner = _NonConstructionPlanner(
            _MaskedFrozenPolicy(champion.policy), **_planning_settings(champion))

    def candidates(self, game: GameState, player_id: int,
                   surface: ConstructionSurface | None = None) -> TimingCandidates:
        surface = surface or timing_surface(game, player_id)
        if not surface.eligible:
            return TimingCandidates(surface, None, None, surface.exclusion,
                                    surface.exclusion)
        before = game.random_source.counter
        build_action = (self.champion._best_settlement(game, player_id)
                        if surface.subtype == "SETTLEMENT_ONLY"
                        else self.champion._best_city(game, player_id))
        build_error = None if build_action is not None else "NO_BUILD_CANDIDATE"
        if build_action is not None:
            try:
                _validate_on_copy(game, player_id, build_action)
            except (GameActionError, ValueError) as error:
                build_action, build_error = None, f"ILLEGAL_BUILD: {error}"
        try:
            defer_action = self.defer_planner.select_action(game, player_id)
            if isinstance(defer_action, (BuildSettlementAction, BuildCityAction)):
                raise AssertionError("DEFER returned construction")
            # Player-trade proposals are outside the fixed 377 catalog, but
            # the Planner's own generation path validates them separately.
            from app.domain.actions import ProposeTradeAction
            from .action_space import action_to_id
            if not isinstance(defer_action, ProposeTradeAction):
                mask = get_action_mask(game, player_id)
                if not mask[action_to_id(defer_action)]:
                    raise AssertionError("DEFER returned an illegal action")
            _validate_on_copy(game, player_id, defer_action)
            defer_error = None
        except (GameActionError, ValueError, AssertionError) as error:
            defer_action, defer_error = None, str(error)
        if game.random_source.counter != before:
            raise AssertionError("Candidate generation consumed game RNG")
        return TimingCandidates(surface, build_action, defer_action,
                                build_error, defer_error)


class ConstructionTimingGate(nn.Module):
    """Independent two-action actor; never touches the 377-way context residual."""

    def __init__(self, frozen_latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(frozen_latent_dim + RUNTIME_CONTEXT_SIZE + 2,
                                               hidden_dim), nn.Tanh())
        self.actor = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, frozen_latent: torch.Tensor, context: torch.Tensor,
                subtype_one_hot: torch.Tensor) -> torch.Tensor:
        return self.actor(self.encoder(torch.cat(
            (frozen_latent.detach(), context, subtype_one_hot), dim=-1)))


class SurfaceCritic(nn.Module):
    """Separate trainable residual around the frozen Champion value estimate."""

    def __init__(self, frozen_latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(frozen_latent_dim + RUNTIME_CONTEXT_SIZE + 2,
                                               hidden_dim), nn.Tanh())
        self.value_residual = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.value_residual.weight)
        nn.init.zeros_(self.value_residual.bias)

    def forward(self, frozen_value: torch.Tensor, frozen_latent: torch.Tensor,
                context: torch.Tensor, subtype_one_hot: torch.Tensor) -> torch.Tensor:
        return frozen_value.detach() + self.value_residual(self.encoder(torch.cat(
            (frozen_latent.detach(), context, subtype_one_hot), dim=-1)))


def timing_trainable_parameters(gate: ConstructionTimingGate,
                                critic: SurfaceCritic) -> list[nn.Parameter]:
    """Explicit optimizer allowlist; frozen Champion parameters are never returned."""
    return list(gate.parameters()) + list(critic.parameters())


def frozen_gate_input(champion, game: GameState, player_id: int,
                      subtype: str, *, max_plan_roads: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                             torch.Tensor]:
    """Read frozen v2 actor latent/value plus deterministic v7 context."""
    if subtype not in TIMING_SUBTYPES:
        raise ValueError("Timing gate supports only single-build surfaces")
    observation = encode_observation(get_observation(game, player_id, version="v2"))
    policy = champion.model.policy
    with torch.no_grad():
        tensor, _ = policy.obs_to_tensor(observation)
        latent = policy.mlp_extractor.forward_actor(tensor).detach()
        value = policy.predict_values(tensor).detach()
    context = torch.as_tensor(encode_runtime_context(
        game, player_id, max_plan_roads=max_plan_roads),
                              dtype=latent.dtype, device=latent.device).unsqueeze(0)
    subtype_tensor = torch.tensor([[1.0, 0.0] if subtype == "SETTLEMENT_ONLY"
                                   else [0.0, 1.0]], dtype=latent.dtype,
                                  device=latent.device)
    return latent, context, subtype_tensor, value


@dataclass(frozen=True)
class SurfaceTransition:
    surface_observation: np.ndarray
    runtime_context: np.ndarray
    surface_type: str
    gate_action: int
    gate_log_prob: float
    reward_accumulated: float
    next_surface_or_terminal: str
    done: bool
    macro_duration: int
    value: float
    delegated: bool = True
    next_surface_observation: np.ndarray | None = None
    next_runtime_context: np.ndarray | None = None
    next_surface_type: str | None = None
    truncated: bool = False
    player_trades_crossed: int = 0
    vp_delta_contribution: float = 0.0
    terminal_contribution: float = 0.0
    nondelegated_surfaces_crossed: int = 0


class SurfaceRolloutBuffer:
    """Semi-MDP transitions only for decisions actually controlled by the gate."""

    def __init__(self, gamma: float):
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma
        self.transitions: list[SurfaceTransition] = []

    def append(self, transition: SurfaceTransition) -> None:
        if not transition.delegated:
            raise ValueError("Planner-controlled action is not on-policy gate data")
        if transition.surface_type not in TIMING_SUBTYPES:
            raise ValueError("Dual-build and non-surface decisions are excluded")
        if transition.gate_action not in (BUILD_NOW, DEFER):
            raise ValueError("Invalid gate action")
        if transition.macro_duration < 1:
            raise ValueError("macro_duration is counted in learner decision steps")
        self.transitions.append(transition)

    def discounted_reward(self, step_rewards: list[float]) -> float:
        return sum((self.gamma ** i) * reward for i, reward in enumerate(step_rewards))

    def bootstrap_discount(self, duration: int) -> float:
        if duration < 1:
            raise ValueError("duration must be positive")
        return self.gamma ** duration


@dataclass(frozen=True)
class DelegationChoice:
    action: Action
    delegated: bool
    gate_action: int | None
    gate_log_prob: float | None
    surface_value: float | None
    candidates: TimingCandidates
    runtime_context: np.ndarray | None = None


class ConstructionTimingDelegator:
    """Disabled by default; the experimental gate never changes Champion games."""

    def __init__(self, champion: SettlementPlanningAgent,
                 gate: ConstructionTimingGate, critic: SurfaceCritic,
                 *, delegation_probability: float = 0.0):
        self.champion = champion
        self.gate = gate
        self.critic = critic
        self.executors = FrozenConstructionExecutors(champion)
        self.delegation_probability = delegation_probability

    def select(self, game: GameState, player_id: int, *,
               random_uniform: Callable[[], float],
               stochastic: bool = True) -> DelegationChoice:
        surface = timing_surface(game, player_id)
        candidates = self.executors.candidates(game, player_id, surface)
        if (not candidates.delegatable
                or not choose_delegation(self.delegation_probability, random_uniform)):
            return DelegationChoice(self.champion.select_action(game, player_id),
                                    False, None, None, None, candidates)
        latent, context, subtype, base_value = frozen_gate_input(
            self.champion.policy, game, player_id, surface.subtype,
            max_plan_roads=self.champion.max_plan_roads)
        with torch.no_grad():
            distribution = torch.distributions.Categorical(
                logits=self.gate(latent, context, subtype))
            if stochastic:
                threshold = float(distribution.probs[0, BUILD_NOW].item())
                selected = BUILD_NOW if random_uniform() < threshold else DEFER
                choice = torch.tensor([selected], device=latent.device)
            else:
                choice = distribution.probs.argmax(dim=-1)
            log_prob = distribution.log_prob(choice)
            value = self.critic(base_value, latent, context, subtype)
        selected = int(choice.item())
        action = candidates.build_action if selected == BUILD_NOW else candidates.defer_action
        return DelegationChoice(action, True, selected, float(log_prob.item()),
                                float(value.item()), candidates,
                                context[0].detach().cpu().numpy().copy())


class SurfaceTransitionBuilder:
    """Aggregate base CatanEnv learner-decision rewards until next surface/terminal.

    Call start only after real delegation, then add_step once per CatanEnv.step()
    (including Planner-controlled learner decisions). Opponent and automatic
    phase actions inside that step do not increment duration separately.
    """

    def __init__(self, gamma: float):
        self.gamma = SurfaceRolloutBuffer(gamma).gamma
        self.pending: dict | None = None
        self.rewards: list[float] = []
        self.vp_rewards: list[float] = []
        self.terminal_rewards: list[float] = []
        self.player_trades_crossed = 0
        self.nondelegated_surfaces_crossed = 0

    def start(self, *, surface_observation: np.ndarray,
              runtime_context: np.ndarray, surface_type: str,
              gate_action: int, gate_log_prob: float, value: float) -> None:
        if self.pending is not None:
            raise ValueError("Previous delegated transition is not closed")
        if surface_type not in TIMING_SUBTYPES or gate_action not in (BUILD_NOW, DEFER):
            raise ValueError("Invalid delegated surface")
        self.pending = dict(surface_observation=surface_observation.copy(),
                            runtime_context=runtime_context.copy(),
                            surface_type=surface_type, gate_action=gate_action,
                            gate_log_prob=gate_log_prob, value=value)
        self.rewards = []
        self.vp_rewards = []
        self.terminal_rewards = []
        self.player_trades_crossed = 0
        self.nondelegated_surfaces_crossed = 0

    def record_nondelegated_surface(self) -> None:
        if self.pending is not None:
            self.nondelegated_surfaces_crossed += 1

    def add_step(self, reward: float, *, vp_delta: float = 0.0,
                 terminal: float = 0.0, player_trade: bool = False) -> None:
        if self.pending is not None:
            self.rewards.append(float(reward))
            self.vp_rewards.append(float(vp_delta))
            self.terminal_rewards.append(float(terminal))
            self.player_trades_crossed += int(player_trade)

    def finish(self, *, next_surface_or_terminal: str, done: bool,
               truncated: bool = False,
               next_surface_observation: np.ndarray | None = None,
               next_runtime_context: np.ndarray | None = None,
               next_surface_type: str | None = None) -> SurfaceTransition:
        if self.pending is None or not self.rewards:
            raise ValueError("No delegated base step to finish")
        if done and truncated:
            raise ValueError("Terminal and truncation must remain distinct")
        transition = SurfaceTransition(
            **self.pending,
            reward_accumulated=sum(self.gamma ** i * reward
                                   for i, reward in enumerate(self.rewards)),
            next_surface_or_terminal=next_surface_or_terminal,
            done=done, macro_duration=len(self.rewards), delegated=True,
            next_surface_observation=(next_surface_observation.copy()
                                      if next_surface_observation is not None else None),
            next_runtime_context=(next_runtime_context.copy()
                                  if next_runtime_context is not None else None),
            next_surface_type=next_surface_type,
            truncated=truncated,
            player_trades_crossed=self.player_trades_crossed,
            vp_delta_contribution=sum(self.gamma ** i * reward
                                      for i, reward in enumerate(self.vp_rewards)),
            terminal_contribution=sum(self.gamma ** i * reward
                                      for i, reward in enumerate(self.terminal_rewards)),
            nondelegated_surfaces_crossed=self.nondelegated_surfaces_crossed)
        self.pending = None
        self.rewards = []
        self.vp_rewards = []
        self.terminal_rewards = []
        self.player_trades_crossed = 0
        self.nondelegated_surfaces_crossed = 0
        return transition


def choose_delegation(probability: float, random_uniform: Callable[[], float]) -> bool:
    """External RNG only; never consume GameState RNG for a delegation schedule."""
    if not 0 <= probability <= 1:
        raise ValueError("Delegation probability must be in [0,1]")
    return random_uniform() < probability
