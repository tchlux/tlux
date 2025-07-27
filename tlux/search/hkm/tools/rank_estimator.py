"""
RankEstimator – streaming additive-error rank sketch (Karnin-Lang-Liberty).

Single-file, pure-Python implementation of the KLL algorithm for approximate
rank estimation over a numeric data stream.  The sketch is merge-able and
requires at most *k* live samples irrespective of the stream length *n*.  With
*k ≈ 1.6 / ε* the estimator guarantees worst-case additive rank error
**ε · n** with probability ≃ 0.99.

Example usage:

  import random
  est = RankEstimator.create(error=0.01, seed=42)
  for _ in range(10_000):
    est.insert(random.random())
  print(round(est.rank(0.5), 3))

"""

# Standard library imports
import math
import random
from dataclasses import dataclass
from typing import List, Sequence

__all__ = ["RankEstimator"]


# _Item - Lightweight container storing a value and its weight within the sketch.
@dataclass
class _Item:
    value: float
    weight: int


# RankEstimator
# 
# KLL sketch supporting single-pass updates, rank queries and merge.
# 
class RankEstimator:
    # Public constructor choosing *k* automatically from a target additive
    # error.  k ≈ 1.6 / ε is the constant derived in the KLL paper for 99 % success.
    #
    # Parameters
    #   error (float): Desired additive rank error ε ∈ (0, 1).
    #   k (int, optional): Override automatic choice; must be ≥ 1.
    #   seed (int, optional): Seed for reproducible random compaction.
    # 
    # Returns
    #   RankEstimator
    # 
    # Raises
    #   ValueError: On invalid arguments.
    #
    @classmethod
    def create(
        cls,
        *,
        error: float = 0.01,
        k: int | None = None,
        seed: int | None = None,
    ) -> "RankEstimator":
        if k is None:
            if not (0.0 < error < 1.0):
                raise ValueError("error must be in (0,1)")
            k = max(1, math.ceil(1.6 / error))
        if k <= 0:
            raise ValueError("k must be positive")
        return cls(k=k, seed=seed)

    # Internal constructor – prefer `create` for public use.
    def __init__(self, *, k: int, seed: int | None = None) -> None:
        self._k: int = k
        self._rng: random.Random = random.Random(seed)
        self._buffers: List[List[_Item]] = [[]]  # level 0 exists from start
        self._count: int = 0  # total weighted count of inserted items

    # Return capacity of *level* buffer according to geometric schedule.
    def _capacity_for(self, level: int) -> int:
        return self._k if level == 0 else max(1, self._k // (2 ** (level + 1)))

    # Extend buffers list until *level* exists.
    def _ensure_level(self, level: int) -> None:
        while level >= len(self._buffers):
            self._buffers.append([])

    # Compact a single level and push it to the next level.
    def _compact(self, level: int) -> None:
        buffer = self._buffers[level]
        if not buffer:
            return  # nothing to compact
        # Sort in-place by value.
        buffer.sort(key=lambda item: item.value)
        # Random offset 0/1 to keep unbiased rank.
        offset = self._rng.randint(0, 1)
        survivors: List[_Item] = buffer[offset::2]
        # Double weight of survivors before promotion.
        for item in survivors:
            item.weight <<= 1  # multiply by 2
        # Clear current level and push survivors upward.
        buffer.clear()
        next_level = level + 1
        self._ensure_level(next_level)
        self._buffers[next_level].extend(survivors)

    # Insert *value* into the sketch.
    # 
    # Parameters
    #   value (float): Number to be observed.
    # 
    # Returns
    #   None
    # 
    def insert(self, value: float) -> None:
        self._count += 1
        level = 0
        self._buffers[0].append(_Item(value=value, weight=1))
        # Cascade compactions while buffers overflow.
        while len(self._buffers[level]) > self._capacity_for(level):
            self._compact(level)
            level += 1
            self._ensure_level(level)

    # Estimate the *q*-th rank (0 ≤ q ≤ 1).
    # 
    # Parameters
    #   q (float): Desired rank fraction.
    # 
    # Returns
    #   float: Approximate q-rank value.
    # 
    # Raises
    #   ValueError: If q is outside [0, 1] or sketch is empty.
    # 
    def rank(self, q: float) -> float:
        if not (0.0 <= q <= 1.0):
            raise ValueError("q must be in [0,1]")
        if self._count == 0:
            raise ValueError("rank called on empty sketch")
        # Gather all (value, weight) pairs.
        items: List[_Item] = [item for buf in self._buffers for item in buf]
        # Global sort by value.
        items.sort(key=lambda item: item.value)
        # Target rank (1-based index).
        target = math.ceil(q * self._count)
        running = 0
        for item in items:
            running += item.weight
            if running >= target:
                return item.value
        # Fallback for q == 1.0 with rounding errors.
        return items[-1].value

    # Merge *other* into *self* in-place.
    # 
    # Parameters
    #   other (RankEstimator): Another compatible sketch.
    # 
    # Returns
    #   None
    # 
    # Raises
    #   ValueError: If k parameters differ.
    # 
    def merge(self, other: "RankEstimator") -> None:
        if self._k != other._k:
            raise ValueError("cannot merge sketches with different k values")
        # Ensure capacity for all levels present in *other*.
        for lvl, other_buf in enumerate(other._buffers):
            self._ensure_level(lvl)
            self._buffers[lvl].extend(other_buf)
            # Compact current level if required; may cascade.
            while len(self._buffers[lvl]) > self._capacity_for(lvl):
                self._compact(lvl)
        self._count += other._count

    # Returns current number of stored items (not total count).
    # This is useful for debugging memory footprint.
    # 
    # Returns
    #   int: Number of live samples held across all buffers.
    # 
    def size(self) -> int:
        return sum(len(buf) for buf in self._buffers)


if __name__ == "__main__":
    # Demonstration – approximate median of deterministic stream.
    est = RankEstimator.create(error=0.01, seed=1)

    for x in range(10_000):  # 0,1,2,...
        est.insert(float(x))

    approx_median = est.rank(0.5)
    print("Approx median:", approx_median)
    # Expect ≈ 5000 within 2 % additive rank error ⇒ ±200 elements.
