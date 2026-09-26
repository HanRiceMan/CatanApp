"""公開資源イベントから相手手札の確率分布を逐次推定する。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log2

from app.domain.game import GameState, RESOURCES


Hand = tuple[int, int, int, int, int]
Distribution = dict[Hand, float]
JointHand = tuple[Hand, ...]
JointDistribution = dict[JointHand, float]
ZERO_HAND: Hand = (0, 0, 0, 0, 0)
RESOURCE_INDEX = {resource: index for index, resource in enumerate(RESOURCES)}
RESOURCE_SUPPLY = 19
MAX_JOINT_HYPOTHESES = 4096


def _normalize(distribution: Distribution, *, limit: int = 4096) -> Distribution:
    positive = [(hand, probability) for hand, probability in distribution.items()
                if probability > 0]
    if len(positive) > limit:
        positive = sorted(positive, key=lambda item: item[1], reverse=True)[:limit]
    total = sum(probability for _, probability in positive)
    return ({hand: probability / total for hand, probability in positive}
            if total else {ZERO_HAND: 1.0})


def _normalize_joint(distribution: JointDistribution, *,
                     limit: int = MAX_JOINT_HYPOTHESES) -> JointDistribution:
    positive = [(hands, probability) for hands, probability in distribution.items()
                if probability > 0]
    if len(positive) > limit:
        positive = sorted(
            positive, key=lambda item: item[1], reverse=True,
        )[:limit]
    total = sum(probability for _, probability in positive)
    return ({hands: probability / total for hands, probability in positive}
            if total else {})


def _apply_joint_deltas(distribution: JointDistribution,
                        player_indexes: dict[int, int],
                        deltas: dict) -> JointDistribution:
    """公開された複数人分の増減を、同じ仮説へ原子的に適用する。"""
    result: dict[JointHand, float] = defaultdict(float)
    for hands, probability in distribution.items():
        updated_hands = list(hands)
        valid = True
        for player_key, cards in deltas.items():
            player_id = int(player_key)
            if player_id not in player_indexes:
                valid = False
                break
            hand = list(updated_hands[player_indexes[player_id]])
            for resource, count in cards.items():
                hand[RESOURCE_INDEX[resource]] += int(count)
            if min(hand) < 0:
                valid = False
                break
            updated_hands[player_indexes[player_id]] = tuple(hand)
        if valid:
            result[tuple(updated_hands)] += probability
    return _normalize_joint(result)


def _joint_random_remove_one(distribution: JointDistribution,
                             player_index: int) -> JointDistribution:
    """種類が非公開の1枚を銀行へ戻す、旧イベントとの互換処理。"""
    result: dict[JointHand, float] = defaultdict(float)
    for hands, probability in distribution.items():
        hand = hands[player_index]
        total = sum(hand)
        if total <= 0:
            result[hands] += probability
            continue
        for resource_index, count in enumerate(hand):
            if not count:
                continue
            updated_hand = list(hand)
            updated_hand[resource_index] -= 1
            updated_hands = list(hands)
            updated_hands[player_index] = tuple(updated_hand)
            result[tuple(updated_hands)] += probability * count / total
    return _normalize_joint(result)


def _joint_hidden_transfer(distribution: JointDistribution, source_index: int,
                           target_index: int) -> JointDistribution:
    """盗賊の1枚を、送り手と受け手の相関を保ったまま分岐する。"""
    result: dict[JointHand, float] = defaultdict(float)
    for hands, probability in distribution.items():
        source = hands[source_index]
        total = sum(source)
        if total <= 0:
            result[hands] += probability
            continue
        for resource_index, count in enumerate(source):
            if not count:
                continue
            updated_source = list(source)
            updated_target = list(hands[target_index])
            updated_source[resource_index] -= 1
            updated_target[resource_index] += 1
            updated_hands = list(hands)
            updated_hands[source_index] = tuple(updated_source)
            updated_hands[target_index] = tuple(updated_target)
            result[tuple(updated_hands)] += probability * count / total
    return _normalize_joint(result)


def _joint_known_transfer(distribution: JointDistribution, source_index: int,
                          target_index: int, resource: str) -> JointDistribution:
    result: dict[JointHand, float] = defaultdict(float)
    resource_index = RESOURCE_INDEX[resource]
    for hands, probability in distribution.items():
        if not hands[source_index][resource_index]:
            continue
        updated_source = list(hands[source_index])
        updated_target = list(hands[target_index])
        updated_source[resource_index] -= 1
        updated_target[resource_index] += 1
        updated_hands = list(hands)
        updated_hands[source_index] = tuple(updated_source)
        updated_hands[target_index] = tuple(updated_target)
        result[tuple(updated_hands)] += probability
    return _normalize_joint(result)


def _condition_on_public_inventory(game: GameState, player_ids: tuple[int, ...],
                                   distribution: JointDistribution
                                   ) -> JointDistribution:
    """公開の銀行在庫・各人の総枚数と一致する共同仮説だけを残す。"""
    players = {player.id: player for player in game.players}
    public_totals = tuple(
        sum(players[player_id].resources.values())
        for player_id in player_ids
    )
    public_by_resource = tuple(
        RESOURCE_SUPPLY - game.bank[resource] for resource in RESOURCES
    )
    filtered = {
        hands: probability
        for hands, probability in distribution.items()
        if tuple(sum(hand) for hand in hands) == public_totals
        and tuple(sum(hand[index] for hand in hands)
                  for index in range(len(RESOURCES))) == public_by_resource
    }
    # 手作りの単体テストや旧保存データが銀行と同期していない場合は、
    # 情報を全消去せずイベント列だけの共同分布へ退避する。
    return _normalize_joint(filtered) if filtered else distribution


def _marginalize(distribution: JointDistribution,
                 player_index: int) -> Distribution:
    result: dict[Hand, float] = defaultdict(float)
    for hands, probability in distribution.items():
        result[hands[player_index]] += probability
    return _normalize(result)


@dataclass(frozen=True)
class HandBelief:
    distribution: Distribution

    @property
    def expected_total(self) -> float:
        return sum(sum(hand) * probability
                   for hand, probability in self.distribution.items())

    def expected(self, resource: str) -> float:
        index = RESOURCE_INDEX[resource]
        return sum(hand[index] * probability
                   for hand, probability in self.distribution.items())

    def probability_has(self, resource: str) -> float:
        index = RESOURCE_INDEX[resource]
        return sum(probability for hand, probability in self.distribution.items()
                   if hand[index] > 0)

    def random_steal_probability(self, resource: str) -> float:
        index = RESOURCE_INDEX[resource]
        return sum((hand[index] / sum(hand)) * probability
                   for hand, probability in self.distribution.items() if sum(hand))

    def to_dict(self) -> dict:
        entropy = -sum(probability * log2(probability)
                       for probability in self.distribution.values() if probability)
        return {
            "expected": {resource: self.expected(resource) for resource in RESOURCES},
            "probability_has": {
                resource: self.probability_has(resource) for resource in RESOURCES
            },
            "random_steal_probability": {
                resource: self.random_steal_probability(resource) for resource in RESOURCES
            },
            "expected_total": self.expected_total,
            "support_size": len(self.distribution),
            "entropy_bits": entropy,
        }


class BayesianHandEstimator:
    """公開イベントと公開在庫から、全盗賊履歴の相関を保って手札を推定する。"""

    def estimate(self, game: GameState,
                 viewer_id: int | None) -> dict[int, HandBelief]:
        player_ids = tuple(player.id for player in game.players)
        player_indexes = {player_id: index
                          for index, player_id in enumerate(player_ids)}
        zero_joint = tuple(ZERO_HAND for _ in player_ids)
        distribution: JointDistribution = {zero_joint: 1.0}
        for event in game.resource_events:
            deltas = event.get("deltas", {})
            if deltas:
                updated = _apply_joint_deltas(distribution, player_indexes, deltas)
                if updated:
                    distribution = updated

            unknown = event.get("unknown_loss")
            if unknown is not None:
                player_id = int(unknown["player_id"])
                for _ in range(int(unknown["count"])):
                    distribution = _joint_random_remove_one(
                        distribution, player_indexes[player_id]
                    )

            transfer = event.get("hidden_transfer")
            if transfer is None:
                continue
            source = int(transfer["from_player_id"])
            target = int(transfer["to_player_id"])
            known_to = {int(player_id) for player_id in transfer.get("known_to", [])}
            private_resource = transfer.get("private_resource")
            if viewer_id in known_to and private_resource in RESOURCE_INDEX:
                distribution = _joint_known_transfer(
                    distribution, player_indexes[source], player_indexes[target],
                    private_resource,
                )
            else:
                distribution = _joint_hidden_transfer(
                    distribution, player_indexes[source], player_indexes[target],
                )
        distribution = _condition_on_public_inventory(
            game, player_ids, distribution,
        )
        return {
            player_id: HandBelief(_marginalize(distribution, player_index))
            for player_id, player_index in player_indexes.items()
        }
