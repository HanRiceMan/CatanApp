"""発展購入の機会費用。入力は通常対局と同じObservation v2のみ。

待ち時間は期待生産を使う予測指標であり、実際の完成ターンではない。
交換用資源を建設用に予約してから余剰だけを交換し、二重計上を避ける。
"""

from __future__ import annotations

import torch
from torch import nn

from app.domain.board import create_topology
from .candidate_policy import TILE_START, VERTEX_START, EDGE_START, PORT_START, PROGRESS_START
from .observation import NUMBER_ORDER, NUMBER_PIPS, PHASE_ORDER, RESOURCE_ORDER


WAIT_CAP = 12.0
FEATURE_SIZE = 50


def budget_wait(stock, production, ratios, costs, bank, *, cap=WAIT_CAP):
    """資源別期待生産/ラウンドを共有予算として使う。返り値は[0, cap]。

現在の銀行在庫は即時交換にだけ使う。将来の銀行在庫・他人の行動・7の廃棄は
予測できないため将来推定には含めず、値は楽観的な目安として扱う。
    """
    def affordable(rounds):
        available = stock + rounds[:, None] * production
        deficits = (costs - available).clamp_min(0)
        surplus = (available - costs).clamp_min(0)
        trades = torch.floor(surplus / ratios + 1e-5).sum(1)
        needed = torch.ceil(deficits - 1e-5).clamp_min(0)
        return trades >= needed.sum(1)

    zero = torch.zeros(stock.shape[0], device=stock.device)
    now_deficits = (costs - stock).clamp_min(0)
    now = affordable(zero) & (bank >= now_deficits).all(1)
    low, high = zero.clone(), torch.full_like(zero, cap)
    for _ in range(10):
        mid = (low + high) / 2
        feasible = affordable(mid)
        high = torch.where(feasible, mid, high)
        low = torch.where(feasible, low, mid)
    return torch.where(now, zero, high)


class DevelopmentBudgetFeatures(nn.Module):
    """学習・対局双方で同じ公開盤面/本人手札から特徴を計算する。"""

    def __init__(self):
        super().__init__()
        topology = create_topology()
        adjacency = torch.zeros(54, 54)
        incident = torch.zeros(72, 54)
        endpoints = []
        for edge in topology.edges:
            a, b = edge.vertex_ids
            endpoints.append((a, b))
            adjacency[a, b] = adjacency[b, a] = 1
            incident[edge.id, a] = incident[edge.id, b] = 1
        tile_incident = torch.zeros(19, 54)
        for vertex in topology.vertices:
            for tile in vertex.hex_ids:
                tile_incident[tile, vertex.id] = 1
        for name, tensor in (("adjacency", adjacency), ("incident", incident),
                             ("tile_incident", tile_incident),
                             ("endpoints", torch.tensor(endpoints)),
                             ("pip_values", torch.tensor([NUMBER_PIPS.get(n, 0)
                                                          for n in NUMBER_ORDER], dtype=torch.float32))):
            self.register_buffer(name, tensor)

    def forward(self, observation):
        batch = len(observation)
        stock = torch.round(observation[:, :5] * 19)
        vertices = observation[:, VERTEX_START:EDGE_START].reshape(batch, 54, 14)
        edges = observation[:, EDGE_START:PORT_START].reshape(batch, 72, 5)
        tiles = observation[:, TILE_START:VERTEX_START].reshape(batch, 19, 18)
        own = vertices[:, :, 1]
        multiplier = own * (vertices[:, :, 6] + 2 * vertices[:, :, 7])
        pip_production = (vertices[:, :, 9:14] * 15 * multiplier[:, :, None]).sum(1)
        tile_buildings = multiplier @ self.tile_incident.T
        tile_pips = tiles[:, :, 6:17] @ self.pip_values
        robber_production = (tiles[:, :, :5] * (
            tiles[:, :, 17] * tile_pips * tile_buildings
        )[:, :, None]).sum(1)
        production = (pip_production - robber_production).clamp_min(0) * (4 / 36)
        ratios = torch.full_like(stock, 4)
        ratios = torch.where(observation[:, 23:24] > 0.5, 3.0, ratios)
        ratios = torch.where(observation[:, 24:29] > 0.5, 2.0, ratios)
        bank = torch.round(observation[:, PROGRESS_START + 33:PROGRESS_START + 38] * 19)

        occupied = 1 - vertices[:, :, 0]
        space = (occupied < 0.5) & ((occupied @ self.adjacency) < 0.5)
        enemy = (occupied - own) > 0.5
        reachable = ((edges[:, :, 1] @ self.incident) > 0) | (own > 0.5)
        reachable &= ~enemy
        distances = torch.full((batch, 54), 99.0, device=observation.device)
        distances[reachable] = 0
        free_edges = edges[:, :, 0] > 0.5
        a, b = self.endpoints[:, 0], self.endpoints[:, 1]
        for distance in range(1, 4):
            touching = free_edges & (reachable[:, a] | reachable[:, b])
            expanded = (touching.float() @ self.incident) > 0
            expanded &= ~enemy
            newly = expanded & ~reachable
            distances[newly] = distance
            reachable |= expanded
        remaining_roads = 15 - torch.round(observation[:, 19] * 15)
        valid = space & (distances <= remaining_roads[:, None])
        nearest = torch.where(valid, distances, 99.0).amin(1)
        settlements = torch.round(observation[:, 17] * 5)
        cities = torch.round(observation[:, 18] * 4)
        settlement_possible = (nearest <= 3) & (settlements < 5)
        city_possible = (settlements > 0) & (cities < 4)
        settlement_cost = torch.tensor([1, 1, 1, 1, 0], device=stock.device,
                                       dtype=stock.dtype).expand_as(stock).clone()
        settlement_cost[:, :2] += nearest.clamp_max(3)[:, None]
        city_cost = torch.tensor([0, 0, 0, 2, 3], device=stock.device).expand_as(stock)
        dev_cost = torch.tensor([0, 0, 1, 1, 1], device=stock.device)
        after = (stock - dev_cost).clamp_min(0)
        waits = []
        deficits = []
        for cost, possible in ((settlement_cost, settlement_possible), (city_cost, city_possible)):
            for resources in (stock, after):
                wait = budget_wait(resources, production, ratios, cost, bank)
                waits.append(torch.where(possible, wait, WAIT_CAP))
                deficits.append((cost - resources).clamp_min(0))
        settlement_before, settlement_after, city_before, city_after = waits
        best_before = torch.minimum(settlement_before, city_before)
        best_after = torch.minimum(settlement_after, city_after)
        delay = (best_after - best_before).clamp_min(0)
        connected = (nearest == 0) & settlement_possible
        played = torch.round(observation[:, 20] * 14)
        held = torch.round(observation[:, 5] * 14)
        others = observation[:, 29:71].reshape(batch, 3, 14)
        rival_knights = torch.round(others[:, :, 6].amax(1) * 14)
        army_holder = observation[:, 22] > 0.5
        # 騎士の在庫がある時は追加購入を自動優遇しない。
        army_need = (held == 0) & (played >= 2) & (
            (~army_holder & (rival_knights <= played + 1))
            | (army_holder & (rival_knights >= played - 1))
        )
        robber_need = (held == 0) & (robber_production.sum(1) >= 2)
        endgame = observation[:, 16] >= 0.8
        stalled = best_before >= 3.0
        # 購入で建設が0.5ラウンド以上遅れる近い局面だけを対象にする。
        reserve = (best_before < 3.0) & (delay >= 0.5)
        action_phase = observation[:, PROGRESS_START + PHASE_ORDER.index("action")] > 0.5
        reserve &= action_phase & ~(army_need | robber_need | endgame)
        # 生産不足の原因と、購入が消費する資源を別々に観測する。
        scalars = torch.stack((
            settlement_before / WAIT_CAP, settlement_after / WAIT_CAP,
            city_before / WAIT_CAP, city_after / WAIT_CAP,
            nearest.clamp_max(4) / 4, connected.float(),
            settlements / 5, cities / 4, observation[:, 16],
            held / 14, played / 14, rival_knights / 14, army_holder.float(),
            robber_production.sum(1) / 30, pip_production.sum(1) / 60,
        ), 1)
        features = torch.cat((stock / 19, production / 4,
                              *[value / 6 for value in deficits], ratios / 4, scalars), 1)
        assert features.shape[1] == FEATURE_SIZE
        return {
            "features": features,
            "reserve": reserve,
            "connected": connected,
            "settlement_before": settlement_before,
            "settlement_after": settlement_after,
            "city_before": city_before,
            "city_after": city_after,
            "best_before": best_before,
            "best_after": best_after,
            "delay": delay,
            "army_exception": army_need,
            "robber_exception": robber_need,
            "endgame_exception": endgame,
            "stalled": stalled,
        }
