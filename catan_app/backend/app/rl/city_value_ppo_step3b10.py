"""City-only value comparison boundary and independent City critic (3B-10)."""

from __future__ import annotations

import torch
from torch import nn

from .hierarchical_l3_shadow_step3b5 import ROUTE_ORDER
from .hierarchical_shadow_controller_step3b4 import FAMILY_ORDER, HierarchicalRoute
from .runtime_context import RUNTIME_CONTEXT_SIZE
from .victory_plan_policy import GLOBAL_FEATURE_DIM


SCHEMA = "city_value_ppo_step3b10_v1"
CITY_FAMILIES = ("BUILD_CITY", "BUILD_ROAD", "BUY_DEVELOPMENT",
                 "BANK_TRADE", "END_TURN")


def city_value_training_mask(route: HierarchicalRoute, diagnostic_mask,
                             executable_mask) -> tuple[bool, ...]:
    """Rule-legal and Frozen-Executor-executable City comparison only."""
    if len(diagnostic_mask) != len(FAMILY_ORDER) or len(executable_mask) != len(FAMILY_ORDER):
        raise ValueError("Seven-family mask required")
    if route is not HierarchicalRoute.CITY_VALUE_COMPARISON:
        return (False,) * len(FAMILY_ORDER)
    allowed = set(CITY_FAMILIES)
    return tuple(bool(family in allowed and diagnostic_mask[index]
                      and executable_mask[index])
                 for index, family in enumerate(FAMILY_ORDER))


class CityValueCritic(nn.Module):
    """City-specific value; detached Actor context and explicit route one-hot."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        fixed = (RUNTIME_CONTEXT_SIZE + GLOBAL_FEATURE_DIM + 6 + 4 + 1
                 + len(FAMILY_ORDER) + len(ROUTE_ORDER))
        self.encoder = nn.Sequential(nn.Linear(latent_dim + fixed, hidden_dim), nn.Tanh())
        self.value_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, latent: torch.Tensor, runtime: torch.Tensor,
                global_features: torch.Tensor, bridge, family_mask: torch.Tensor,
                *, route_one_hot: torch.Tensor | None = None):
        if route_one_hot is None or route_one_hot.shape != (latent.shape[0], len(ROUTE_ORDER)):
            raise ValueError("City critic requires an explicit route one-hot")
        if not torch.all(route_one_hot[:, ROUTE_ORDER.index(
                HierarchicalRoute.CITY_VALUE_COMPARISON)] == 1):
            raise ValueError("City critic cannot value another route")
        scales = bridge.weighted_features.new_tensor((10., 10., 1., 1., 11., 1.))
        features = torch.cat((latent.detach(), runtime, global_features,
                              bridge.weighted_features.detach() / scales,
                              bridge.expected_priorities.detach(),
                              bridge.available.to(runtime.dtype).unsqueeze(-1),
                              family_mask.to(runtime.dtype), route_one_hot), dim=-1)
        return self.value_head(self.encoder(features)).squeeze(-1)
