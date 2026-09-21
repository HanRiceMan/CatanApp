"""対局ごとの再現可能な確率イベント用乱数源。"""

from dataclasses import dataclass
from random import Random
from typing import MutableSequence, Sequence, TypeVar

T = TypeVar("T")


@dataclass
class GameRandomSource:
    """seed と消費回数だけを状態として持つ、比較・複製しやすい乱数源。"""

    seed: int
    counter: int = 0

    def _next(self) -> Random:
        # Random インスタンス自体は状態に持たず、deepcopy と対局比較を安定させる。
        # 1回目は Random(seed) に合わせ、既存の発展カード山札の並びを維持する。
        rng = Random(self.seed if self.counter == 0 else (self.seed << 32) ^ (self.counter * 0x9E3779B1))
        self.counter += 1
        return rng

    def roll_dice(self) -> tuple[int, int]:
        rng = self._next()
        return rng.randint(1, 6), rng.randint(1, 6)

    def choice(self, values: Sequence[T]) -> T:
        if not values:
            raise IndexError("choice from an empty sequence")
        return self._next().choice(values)

    def shuffle(self, values: MutableSequence[T]) -> None:
        self._next().shuffle(values)
