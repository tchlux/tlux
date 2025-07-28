"""
RankEstimator - streaming additive-error rank sketch (Karnin-Lang-Liberty).

Pure-Python, standard library and NumPy only. Implements the KLL algorithm
for approximate rank estimation over numeric data streams. Maintains <= k
live samples regardless of stream length n. With k ~= 2/eps, guarantees
worst-case additive rank error eps*n with probability >= 0.99.

Example
  import random
  est = RankEstimator.create(error=0.01, seed=42)
  for _ in range(10_000):
      est.add(random.random())
  print(round(est.rank(0.5), 3))
"""

import math
import random
import struct
from dataclasses import dataclass
from typing import List, Optional

__all__ = ["RankEstimator"]

# _Item
# Lightweight container for a value and its weight in the sketch.
#
# Parameters
#   value (float): Data value.
#   weight (int): Weight of this value.
#
@dataclass
class _Item:
    value: float
    weight: int


# RankEstimator
#
# KLL sketch supporting single-pass updates, quantile queries, and merge.
#
class RankEstimator:
    # Public constructor, computes k if not given, validates parameters.
    #
    # Parameters
    #   error (float): Desired additive rank error (eps) in (0,1).
    #   k (int, optional): Override for sketch size. Must be >= 1.
    #   seed (int, optional): Seed for reproducible randomness.
    #
    # Returns
    #   RankEstimator: New sketch instance.
    #
    # Raises
    #   ValueError: If parameters are invalid.
    #
    def __init__(
        self,
        *,
        error: float = 1 / 1024,
        k: Optional[int] = None,
        seed: Optional[int] = None
    ) -> "RankEstimator":
        if k is None:
            if not (0.0 < error < 1.0):
                raise ValueError("error must be in (0,1)")
            k = max(1, math.ceil(2.0 / error))
        if k < 1:
            raise ValueError("k must be >= 1")
        self._k: int = k
        self._rng: random.Random = random.Random(seed)
        self._buffers: List[List[_Item]] = [[]]
        self._count: int = 0

    # Compact the buffer at start_level recursively if overflowed.
    #
    # Parameters
    #   start_level (int): Buffer level to compact from.
    #
    # Returns
    #   None
    #
    def _compact(self, start_level: int = 0) -> None:
        level: int = start_level
        while True:
            if len(self._buffers) <= level:
                self._buffers.append([])
            buf = self._buffers[level]
            if len(buf) <= self._k:
                break
            buf.sort(key=lambda item: item.value)
            offset: int = self._rng.randint(0, 1)
            survivors: List[_Item] = buf[offset::2]
            for item in survivors:
                item.weight <<= 1
            buf.clear()
            if len(self._buffers) <= level + 1:
                self._buffers.append([])
            self._buffers[level + 1].extend(survivors)
            level += 1

    # Add a value into the sketch.
    #
    # Parameters
    #   value (float): Number to insert.
    #
    # Returns
    #   None
    #
    def add(self, value: float) -> None:
        self._count += 1
        self._buffers[0].append(_Item(value=value, weight=1))
        self._compact(0)

    # Return the value at quantile q in [0,1].
    #
    # Parameters
    #   q (float): Desired quantile.
    #
    # Returns
    #   float: Value at the approximate q-th quantile.
    #
    # Raises
    #   ValueError: If q is out of [0,1] or sketch is empty.
    #
    def rank(self, q: float) -> float:
        if not (0.0 <= q <= 1.0):
            raise ValueError("q must be in [0,1]")
        items: List[_Item] = [item for buf in self._buffers for item in buf]
        if not items:
            raise ValueError("rank called on empty sketch")
        items.sort(key=lambda item: item.value)
        total_weight: int = sum(item.weight for item in items)
        target: int = math.ceil(q * total_weight)
        running: int = 0
        for item in items:
            running += item.weight
            if running >= target:
                return item.value
        return items[-1].value  # Only for q==1.0

    # Merge another compatible sketch into this one.
    #
    # Parameters
    #   other (RankEstimator): Another RankEstimator with same k.
    #
    # Returns
    #   None
    #
    # Raises
    #   ValueError: If k values do not match.
    #
    def merge(self, other: "RankEstimator") -> None:
        if self._k != other._k:
            raise ValueError("cannot merge sketches with different k values")
        for level, other_buf in enumerate(other._buffers):
            while len(self._buffers) <= level:
                self._buffers.append([])
            self._buffers[level].extend(other_buf)
            self._compact(level)
        self._count += other._count

    # Return number of items stored in all buffers.
    #
    # Returns
    #   int: Current live sample count.
    #
    def size(self) -> int:
        return sum(len(buf) for buf in self._buffers)

    # Serialize this sketch to bytes (deterministic, versioned).
    #
    # Returns
    #   bytes: Serialized binary form.
    #
    def to_bytes(self) -> bytes:
        parts: List[bytes] = []
        # Header
        parts.append(b"RKER")
        # k (int32 LE)
        parts.append(self._k.to_bytes(4, "little", signed=True))
        # count (int64 LE)
        parts.append(self._count.to_bytes(8, "little", signed=True))
        # number of levels (uint32 LE)
        num_levels = len(self._buffers)
        parts.append(num_levels.to_bytes(4, "little", signed=False))
        # For each level, write length then items
        for buf in self._buffers:
            parts.append(len(buf).to_bytes(4, "little", signed=False))
            for item in buf:
                parts.append(struct.pack("<d", item.value))
                parts.append(item.weight.to_bytes(4, "little", signed=True))
        # RNG state (use getstate, encode as repr ASCII)
        rng_state = repr(self._rng.getstate()).encode("ascii")
        parts.append(len(rng_state).to_bytes(4, "little", signed=False))
        parts.append(rng_state)
        return b"".join(parts)

    # Deserialize sketch from bytes produced by to_bytes.
    #
    # Parameters
    #   data (bytes): Output from to_bytes.
    #
    # Returns
    #   RankEstimator: Restored sketch.
    #
    # Raises
    #   ValueError: If header/version is wrong or data corrupt.
    #
    @classmethod
    def from_bytes(cls, data: bytes) -> "RankEstimator":
        mv = memoryview(data)
        pos = 0
        # Header
        if mv[:4].tobytes() != b"RKER":
            raise ValueError("Invalid header")
        pos += 4
        k = int.from_bytes(mv[pos:pos+4], "little", signed=True)
        pos += 4
        count = int.from_bytes(mv[pos:pos+8], "little", signed=True)
        pos += 8
        num_levels = int.from_bytes(mv[pos:pos+4], "little", signed=False)
        pos += 4
        buffers: List[List[_Item]] = []
        for _ in range(num_levels):
            n_items = int.from_bytes(mv[pos:pos+4], "little", signed=False)
            pos += 4
            buf: List[_Item] = []
            for _ in range(n_items):
                value = struct.unpack("<d", mv[pos:pos+8])[0]
                pos += 8
                weight = int.from_bytes(mv[pos:pos+4], "little", signed=True)
                pos += 4
                buf.append(_Item(value=value, weight=weight))
            buffers.append(buf)
        rng_len = int.from_bytes(mv[pos:pos+4], "little", signed=False)
        pos += 4
        rng_state = eval(mv[pos:pos+rng_len].tobytes().decode("ascii"))
        pos += rng_len
        est = cls(k=k)
        est._count = count
        est._buffers = buffers
        est._rng.setstate(rng_state)
        return est


if __name__ == "__main__":
    # Demonstration: estimate the 95th percentile of a deterministic stream.
    n = 100_000
    error = 1 / 1024
    rank = 0.95
    for seed in range(1, 11):
        est = RankEstimator(error=error, seed=seed)
        for x in range(n):
            est.add(float(x))
        est = RankEstimator.from_bytes(est.to_bytes())
        approx = est.rank(rank)
        diff = round(approx - rank * n)
        print(f"{approx} ({diff:+03d} : {100 * diff / n:+5.3f}%)")
