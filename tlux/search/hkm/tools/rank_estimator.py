"""
RankEstimator - streaming additive-error rank sketch (Karnin-Lang-Liberty).

Pure-Python, standard library. Implements the KLL algorithm
for approximate rank estimation over numeric data streams. Maintains 
<= max_samples live samples regardless of stream length n. Guarantees
a worst case error bound with (tunable) specified probability.

Example
  import random
  est = RankEstimator(max_size=max_size(error=0.01, delta=0.01) seed=42)
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

# In the paper this number is 1.7 but here we adjust it to ensure
#  the probabilities work as expected (on monotonic and random sequences).
ERROR_CONSTANT: float = 2.0

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


# Compute how many levels to preallocate given k_H and c.
#
# Parameters:
#   k (int): The size of the largest level (top most, *last* to be filled).
#   c (float): The "shrink factor" between levels from top-most to bottom "entry level".
# 
# Returns:
#   int: Number of levels (H + 1), counting from [0,H), where
#        ceil(k_H * c**i) >= 1 for all i in [0, H).
#
def num_levels(k: int, c: float) -> int:
    i = 0
    cap = float(k)
    while cap > 1.0:
        i += 1
        cap *= c
    return i


# Return the total number of stored items (all levels combined) required
# to achieve additive rank error epsilon with failure probability delta,
# under the calibration used in this module.
#
# Parameters:
#   error (float): Additive rank error epsilon in (0,1).
#   delta (float): Failure probability in (0,1).
#   c (float): Geometric level shrink factor in (0,1), default 2/3.
#
# Returns:
#   int: Total max size across all levels (worst-case live sample bound).
#
# Raises:
#   ValueError: If arguments are out of range.
#
def max_size(error: float, delta: float, c: float = 2.0 / 3.0) -> int:
    if not (0.0 < error < 1.0):
        raise ValueError("error must be in (0,1)")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    if not (0.0 < c < 1.0):
        raise ValueError("c must be in (0,1)")
    k = (ERROR_CONSTANT / error) * math.log2(1.0 / delta)
    return num_levels(k=k, c=c)


# RankEstimator
#
# KLL sketch supporting single-pass updates, quantile queries, and merge.
#
class RankEstimator:
    # Public constructor, computes k if not given, validates parameters.
    #
    # Parameters
    #   error (float, optional): Desired additive rank error (eps) in (0,1).
    #   max_size (int, optional): Capped total estimator size.
    #   delta (float): Desired probability of exceeding error in (0,1).
    #   c (float): 
    #   seed (int, optional): Seed for reproducible randomness.
    #
    # Returns
    #   RankEstimator: New sketch instance.
    #
    # Raises
    #   ValueError: If parameters are invalid.
    #
    def __init__(self, *, max_size: int = 801, c: float = 2 / 3, seed: Optional[int] = None) -> "RankEstimator":
        # Parameter validation.
        if not (isinstance(max_size, int) and max_size >= 1):
            raise ValueError("max_size must be an integer >= 1")
        if not (0.0 < c < 1.0):
            raise ValueError("c must be in (0,1)")
        # Determine `k` given the maximum size.
        k = max(1, int(math.floor(max_size * (1.0 - c))))
        self._k: int = k
        self._c: float = c
        self._seed: Optional[int] = seed
        self._rng: random.Random = random.Random(seed)
        self._buffers: List[List[_Item]] = [[] for _ in range(num_levels(self._k, self._c))]
        self._count: int = 0
        self._min: _Item = _Item(value=float('inf'), weight=0)
        self._max: _Item = _Item(value=-float('inf'), weight=0)

    # Compute the maximum allowed items for a given level.
    def _capacity(self, level: int) -> int:
        # Let _k denote k_H (capacity of the top level).
        H = len(self._buffers) - 1
        level_size = self._k * (self._c ** (H - level))
        return 0 if level_size < 1 else math.ceil(level_size)

    # Compact the buffer at start_level recursively if overflowed.
    #
    # Parameters
    #   start_level (int): Buffer level to compact from.
    #
    # Returns
    #   None
    #
    def _compact(self, start_level: int = 0) -> None:
        for level in range(start_level, len(self._buffers)):
            buf = self._buffers[level]
            # If the buffer is not overflowing, we are done.
            if len(buf) <= self._capacity(level):
                break
            # Sort the buffer (before doing the compaction).
            buf.sort(key=lambda item: item.value)
            # If len(buf) is odd, leave one random item behind unchanged.
            if len(buf) & 1:
                keep_idx = self._rng.randint(0, len(buf) - 1)
                keep = buf.pop(keep_idx)
            else:
                keep = None
            # Pick an offset randomly, drop half (conserving weight), push into next level.
            offset = self._rng.randint(0, 1)
            survivors: List[_Item] = buf[offset::2]
            for item in survivors:
                item.weight *= 2
            buf.clear()
            # Normally, compact and send up.
            if (level + 1 < len(self._buffers)):
                if keep is not None:
                    buf.append(keep)
                nbuf = self._buffers[level + 1]
                nbuf.extend(survivors)
            # Top level: compact in place (no higher level).
            elif level == len(self._buffers) - 1:
                buf.extend(survivors)

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
        new_item: _Item = _Item(value=value, weight=1)
        self._buffers[0].append(new_item)
        self._compact(0)
        if value < self._min.value:
            self._min = new_item
        if value > self._max.value:
            self._max = new_item

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
        items.extend([self._min, self._max])
        items.sort(key=lambda item: item.value)
        total_weight: int = sum(item.weight for item in items)
        target: int = math.ceil(q * total_weight)
        running: int = 0
        for item in items:
            running += item.weight
            if running >= target:
                return item.value
        return items[-1].value  # Only for q==1.0

    # Confidence interval on the q-quantile value, with additive-rank error
    # calibrated via epsilon(delta). Returns a (lo, hi) tuple.
    #
    # Parameters:
    #   q (float): Desired quantile in [0,1].
    #   delta (float, optional): Failure probability in (0,1).
    #
    # Returns:
    #   (float, float): Lower/upper bounds on the q-quantile value.
    #
    # Raises:
    #   ValueError: If inputs are invalid or the sketch is empty.
    #
    def rank_bounds(self, q: float, delta: float = 2**(-7)) -> (float, float):
        if not (0.0 <= q <= 1.0):
            raise ValueError("q must be in [0,1]")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0,1)")
        items: List[_Item] = [item for buf in self._buffers for item in buf]
        if not items:
            raise ValueError("rank_bounds called on empty sketch")
        items.extend([self._min, self._max])
        items.sort(key=lambda item: item.value)
        n: int = sum(item.weight for item in items)
        eps: float = self.epsilon(delta)
        err: int = max(0, int(math.ceil(eps * n)))
        target: int = max(1, int(math.ceil(q * n)))
        lo_rank: int = max(1, target - err)
        hi_rank: int = min(n, target + err)
        # Walk cumulative weights once to read off lo/hi values.
        running: int = 0
        lo_val: Optional[float] = None
        hi_val: Optional[float] = None
        for item in items:
            running += item.weight
            if (lo_val is None) and (running >= lo_rank):
                lo_val = item.value
            if (running >= hi_rank):
                hi_val = item.value
                break
        # Fallback guards (should not trigger).
        if lo_val is None:
            lo_val = items[0].value
        if hi_val is None:
            hi_val = items[-1].value
        return (lo_val, hi_val)

    # Return EDF step points derived from the entire sketch.
    # Values are unique sorted sample values; quantiles are right-continuous
    # cumulative proportions F(x) = P[X <= x] computed from item weights.
    #
    # Returns:
    #   (List[float], List[float]): (values, quantiles) where quantiles[i] in [0,1].
    #
    # Raises:
    #   ValueError: If the sketch is empty.
    #
    def edf_points(self) -> (List[float], List[float]):
        # Get all items into a single list.
        items: List[_Item] = [it for buf in self._buffers for it in buf]
        if not items:
            raise ValueError("edf_points called on empty sketch")
        items.extend([self._min, self._max])
        items.sort(key=lambda it: it.value)
        # Aggregate weights for duplicate values.
        values: List[float] = []
        grouped_weights: List[int] = []
        cur_val: float = items[0].value
        acc_w: int = 0
        for it in items:
            if it.value != cur_val:
                values.append(cur_val)
                grouped_weights.append(acc_w)
                cur_val = it.value
                acc_w = 0
            acc_w += it.weight
        # Flush last group.
        values.append(cur_val)
        grouped_weights.append(acc_w)
        # Compute quantiles from weights.
        total_w: int = sum(grouped_weights)
        cum: int = 0
        quantiles: List[float] = []
        for w in grouped_weights:
            cum += w
            quantiles.append(cum / float(total_w))
        # Return the values and quantiles.
        return values, quantiles

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
        if self._k != other._k or self._c != other._c:
            raise ValueError("cannot merge sketches with different k or c values")
        for level, other_buf in enumerate(other._buffers):
            if level >= len(self._buffers):
                raise ValueError("cannot merge sketches with different level counts")
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

    # Return maximum possible number of live items given current parameters.
    #
    # Returns
    #   int: Worst-case live sample bound.
    #
    def max_size(self, max_iters=128) -> int:
        # Worst-case live sample bound as H -> infinity:
        # sum_{i=0..infty} k_H * c^i = k_H / (1 - c).
        if not (0.0 < self._c < 1.0):
            raise RuntimeError("c must be in (0,1) to bound max_size")
        return int(math.ceil(self._k / (1.0 - self._c)))

    # Return the additive rank error epsilon whose violation probability
    # is at most delta, using the same calibration as max_size().
    #
    # Parameters:
    #   delta (float): Failure probability in (0,1).
    #
    # Returns:
    #   float: Epsilon satisfying the bound for the current sketch size.
    #
    # Raises:
    #   ValueError: If delta is not in (0,1).
    #
    def epsilon(self, delta: float) -> float:
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0,1)")
        return (ERROR_CONSTANT / float(self._k)) * math.log2(1.0 / delta)

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
        # c (float64 LE)
        parts.append(struct.pack("<d", self._c))
        # count (int64 LE)
        parts.append(self._count.to_bytes(8, "little", signed=True))
        # number of levels (uint32 LE)
        num_levels = len(self._buffers)
        parts.append(num_levels.to_bytes(4, "little", signed=False))
        # minval (float32 LE)
        parts.append(struct.pack("<f", self._min.value))
        # maxval (float32 LE)
        parts.append(struct.pack("<f", self._max.value))
        # For each level, write length then items
        for l, buf in enumerate(self._buffers):
            cap = self._capacity(l)
            parts.append(cap.to_bytes(4, "little", signed=False))
            parts.append(len(buf).to_bytes(4, "little", signed=False))
            for item in buf:
                parts.append(struct.pack("<f", item.value))
                parts.append(item.weight.to_bytes(6, "little", signed=True))
            parts.append(bytes(10 * (cap - len(buf))))
        return b"".join(parts)

    # Deserialize sketch from bytes produced by to_bytes.
    #
    # Parameters
    #   data (bytes): Output from to_bytes.
    #   seed (int, optional): Seed for newly loaded estimator.
    #
    # Returns
    #   RankEstimator: Restored sketch.
    #
    # Raises
    #   ValueError: If header/version is wrong or data corrupt.
    #
    @classmethod
    def from_bytes(cls, data: bytes, seed: Optional[int] = None) -> "RankEstimator":
        mv = memoryview(data)
        pos = 0
        # header
        if mv[:4].tobytes() != b"RKER":
            raise ValueError("Invalid header")
        pos += 4
        # k
        k = int.from_bytes(mv[pos:pos+4], "little", signed=True)
        pos += 4
        # c
        c = struct.unpack("<d", mv[pos:pos+8])[0]
        pos += 8
        # count
        count = int.from_bytes(mv[pos:pos+8], "little", signed=True)
        pos += 8
        # num levels
        num_levels = int.from_bytes(mv[pos:pos+4], "little", signed=False)
        pos += 4
        # min val
        minval = struct.unpack("<f", mv[pos:pos+4])[0]
        pos += 4
        # max val
        maxval = struct.unpack("<f", mv[pos:pos+4])[0]
        pos += 4
        # buffers
        buffers: List[List[_Item]] = []
        for _ in range(num_levels):
            cap = int.from_bytes(mv[pos:pos+4], "little", signed=False)
            pos += 4
            n_items = int.from_bytes(mv[pos:pos+4], "little", signed=False)
            pos += 4
            buf: List[_Item] = []
            for _ in range(n_items):
                value = struct.unpack("<f", mv[pos:pos+4])[0]
                pos += 4
                weight = int.from_bytes(mv[pos:pos+6], "little", signed=True)
                pos += 6
                buf.append(_Item(value=value, weight=weight))
            buffers.append(buf)
            pos += 10 * (cap - len(buf))
        # create new estimator
        est = cls(max_size=int(math.ceil(k / (1.0 - c))), c=c, seed=seed)
        est._count = count
        est._buffers = buffers
        est._min = _Item(value=minval, weight=(1 if minval <  float('inf') else 0))
        est._max = _Item(value=maxval, weight=(1 if maxval > -float('inf') else 0))
        return est


if __name__ == "__main__":
    # Demonstration: estimate the 95th percentile of a deterministic stream.
    n = 100_000
    ntest = 20
    s = None

    # Pick a delta such that we expect a 50% chance of seeing *one* failure in `ntest` examples.
    delta = 1 - 2**(-1/ntest) # ~ 0.0340636711

    print()
    print(f"N = {n}")
    print(f" {len(RankEstimator().to_bytes())} bytes")
    print()
    print("Rnk -- Estmt  (abs err:  rel % )  |  size   k     c   max-size")
    import random
    from tlux.random import random_range
    for seed in range(1, ntest+1):
        est = RankEstimator(seed=seed)
        random.seed(s)
        for x in random_range(n):
        # for x in range(n):
            est.add(float(x))
        est_bytes = est.to_bytes()
        est = RankEstimator.from_bytes(est_bytes)
        rank = seed / (ntest+1)
        approx = round(est.rank(rank))
        diff = round(approx - rank * n)
        print(f"{100*rank:3.0f} -- {approx:6.0f} ({diff:+6d} : {100 * diff / n:+5.3f}%)  |  {est.size()} {est._k} {est._c:.4f} {est.max_size()}")
        lo, hi = est.rank_bounds(rank, delta)
        if lo < (rank * n) < hi:
            mark = "  "
        else:
            mark = "**"
            print(f"     {mark} [{lo:.6g}, {hi:.6g}] {100*(1-delta):.1f}% CI")
    print()

    edf_x, edf_y = est.edf_points()
    from tlux.plot import Plot
    p = Plot()
    p.add("EDF", edf_x, edf_y, color=1)
    p.show()
