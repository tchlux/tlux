"""
RankEstimator - streaming additive-error rank sketch (Karnin-Lang-Liberty).

Pure-Python standard library plus numpy. Implements the KLL algorithm
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
import numpy as np

__all__ = ["RankEstimator"]

# In the paper this number is 1.7 but here we adjust it to ensure
#  the probabilities work as expected (on monotonic and random sequences).
ERROR_CONSTANT: float = 6.0

# Add a module-level dtype for compact storage (float32 values, int64 weights).
ITEM_DTYPE = np.dtype([("value", np.float32), ("weight", np.int64)])

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
    return int(math.ceil(k / (1.0 - c)))


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
    def __init__(self, *, max_size: int = 2**12, c: float = 2 / 3, seed: Optional[int] = None) -> "RankEstimator":
        # Parameter validation.
        if not (isinstance(max_size, int) and max_size >= 1):
            raise ValueError("max_size must be an integer >= 1")
        if not (0.0 < c < 1.0):
            raise ValueError("c must be in (0,1)")
        # Determine `k` given the maximum size.
        k = max(1, int(math.floor(max_size * (1.0 - c))))
        self._k: int = k
        self._c: float = c
        self._rng: random.Random = random.Random(seed)
        self._min = np.float32(float('inf'))
        self._max = np.float32(-float('inf'))
        # Initialize buffers.
        caps_top = []
        cap = float(self._k)
        while cap >= 1.0:
            caps_top.append(int(math.ceil(cap)))
            cap *= self._c
        self._caps = list(reversed(caps_top))
        self._buffers = [np.zeros(2*cap, dtype=ITEM_DTYPE) for cap in self._caps]
        self._counts: List[int] = [0 for _ in self._caps]
        self._count: int = 0
        self._all_items = None
        self._total_weight = None


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
            # If the buffer is not overflowing, we are done.
            cap = self._caps[level]
            cnt = self._counts[level]
            if cnt <= cap:
                break
            # Get the sorted elements in the buffer.
            buf = self._buffers[level]
            order = np.argsort(buf["value"][:cnt])
            buf[:cnt] = buf[order]
            arr = buf[:cnt]
            # Decide which one to *not* compact (if the size is odd). Swap to back.
            keep = None
            if cnt & 1:
                keep = cnt - 1
                picked_idx = self._rng.randint(0, cnt - 1)
                arr[picked_idx], arr[keep] = arr[keep], arr[picked_idx]
                cnt -= 1
                arr = arr[:cnt]
            # Pick an offset randomly, drop half (conserving weight), push into next level.
            offset = self._rng.randint(0, 1)
            arr[:cnt//2] = arr[offset::2]
            survivors = arr[:cnt//2]
            survivors["weight"] *= 2
            # Normally, compact and send up.
            if (level + 1 < len(self._buffers)):
                nbuf = self._buffers[level + 1]
                ncnt = self._counts[level + 1]
                ncap = self._caps[level + 1]
                ns = len(survivors)
                nbuf[ncnt:ncnt + ns] = survivors
                self._counts[level + 1] += ns
                if keep is not None:
                    buf[0] = buf[keep]
                    self._counts[level] = 1
                else:
                    self._counts[level] = 0
            # Top level: compact in place (no higher level).
            elif level == len(self._buffers) - 1:
                self._counts[level] = len(survivors)
                if keep is not None:
                    buf[len(survivors)] = buf[keep]
                    self._counts[level] += 1

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
        value = np.float32(value)
        # Ensure space at level 0 by compacting if needed.
        if self._counts[0] >= self._caps[0]:
            self._compact(0)
        i = self._counts[0]
        new_item = np.zeros(1, dtype=ITEM_DTYPE)[0]
        new_item["value"] = value
        new_item["weight"] = 1
        self._buffers[0][i] = new_item
        self._counts[0] += 1
        # Trigger compaction if necessary.
        if self._counts[0] > self._caps[0]:
            self._compact(0)
        # Track global min/max as _Item for downstream compatibility.
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value
        # Reset the rank-estimation concatenation of all items in this estimator.
        self._all_items = None
        self._total_weight = None

        
    # Convenience property for getting all items currently in the estimator
    # and caching the result for faster repeated rank estimations.
    @property
    def all_items(self):
        if self._all_items is None:
            item_lists = [b[:c] for (b,c) in zip(self._buffers, self._counts)]
            self._all_items = np.concatenate(item_lists)
            order = np.argsort(self._all_items["value"])
            self._all_items = self._all_items[order]
            self._total_weight = self._all_items["weight"].sum()
        return self._all_items

    # Convenience property for getting the total weight of all items currently
    # in the estimator and caching the result for faster repeated rank estimations.
    @property
    def total_weight(self):
        items = self.all_items
        return self._total_weight

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
        if self._count < 1:
            raise ValueError("rank called on empty sketch")
        items = self.all_items
        total_weight = self.total_weight
        target: int = math.ceil(q * total_weight)
        running: int = 0
        for item in items:
            running += item["weight"]
            if running >= target:
                return item["value"]
        return items[-1]["value"]  # Only for q==1.0

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
        n: int = self.total_weight
        eps: float = self.epsilon(delta)
        err: int = max(0, int(math.ceil(eps * n)))
        target: int = max(1, int(math.ceil(q * n)))
        lo_rank: float = max(1, target - err)
        hi_rank: float = min(n, target + err)
        arr = self.all_items
        cum = np.cumsum(arr["weight"].astype(np.int64, copy=False))
        lo_idx = int(np.searchsorted(cum, lo_rank, side="left"))
        hi_idx = int(np.searchsorted(cum, hi_rank, side="left"))
        return float(arr["value"][lo_idx]), float(arr["value"][hi_idx])

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
        items = self.all_items
        v = items["value"]
        w = items["weight"]
        # Find the change points in the values.
        change = np.empty(v.size, dtype=bool)
        change[0] = True
        change[1:] = v[1:] != v[:-1]
        # Compute the cumulative weight at each change point.
        idx = np.nonzero(change)[0]
        group_w = np.add.reduceat(w, idx)
        cumulative = np.cumsum(group_w).astype(np.float64, copy=False)
        # Return the distribution points.
        return v[idx], (cumulative / cumulative[-1])

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
        # Merge all levels.
        for level in range(len(self._buffers)):
            n_other = other._counts[level]
            if n_other == 0:
                continue
            while self._counts[level] + n_other > self._caps[level]:
                self._compact(level)
            i = self._counts[level]
            self._buffers[level][i:i + n_other] = other._buffers[level][:n_other]
            self._counts[level] += n_other
        # Update the total count.
        self._count += other._count
        # Update global min/max from the other sketch.
        if other._min < self._min:
            self._min = other._min
        if other._max > self._max:
            self._max = other._max
        # Reset the rank-estimation concatenation of all items in this estimator.
        self._all_items = None
        self._total_weight = None

    # Return number of items stored in all buffers.
    #
    # Returns
    #   int: Current live sample count.
    #
    def size(self) -> int:
        return sum(self._counts)

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
    def to_bytes(self, version: int = 0) -> bytes:
        parts: List[bytes] = []
        # Header
        parts.append(b"RKER")
        # version
        parts.append(version.to_bytes(2, "little", signed=False))
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
        parts.append(struct.pack("<f", self._min))
        # maxval (float32 LE)
        parts.append(struct.pack("<f", self._max))
        # For each level, write length then items
        for i, (buf, cnt, cap) in enumerate(zip(self._buffers, self._counts, self._caps)):
            parts.append(cap.to_bytes(4, "little", signed=False))
            parts.append(cnt.to_bytes(4, "little", signed=False))
            buf["value"][cnt:] = 0.0
            buf["weight"][cnt:] = 0
            parts.append(buf[:cap].tobytes(order="C"))
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
    def from_bytes(cls, data: bytes, version: int = 0) -> "RankEstimator":
        mv = memoryview(data)
        pos = 0
        # Header
        if mv[:4].tobytes() != b"RKER":
            raise ValueError("Invalid header")
        pos += 4
        v = int.from_bytes(mv[pos:pos+2], "little", signed=False);    pos += 2
        if (v > version):
            raise RuntimeError(f"Encountered unexpected serialization version {v}, while this method supports {version}.")
        k = int.from_bytes(mv[pos:pos+4], "little", signed=True);           pos += 4
        c = struct.unpack("<d", mv[pos:pos+8])[0];                          pos += 8
        count = int.from_bytes(mv[pos:pos+8], "little", signed=True);       pos += 8
        num_levels = int.from_bytes(mv[pos:pos+4], "little", signed=False); pos += 4
        minval = struct.unpack("<f", mv[pos:pos+4])[0];                     pos += 4
        maxval = struct.unpack("<f", mv[pos:pos+4])[0];                     pos += 4
        buffers: List[np.ndarray] = []
        for _ in range(num_levels):
            cap = int.from_bytes(mv[pos:pos+4], "little", signed=False);            pos += 4
            cnt = int.from_bytes(mv[pos:pos+4], "little", signed=False);            pos += 4
            items = np.frombuffer(mv[pos:pos+12*cap], dtype=ITEM_DTYPE, count=cap); pos += 12*cap
            buffers.append((items[:cnt], cap))
        rng_len = int.from_bytes(mv[pos:pos+4], "little", signed=False); pos += 4
        rng_state = eval(mv[pos:pos+rng_len].tobytes().decode("ascii")); pos += rng_len
        ms = int(math.ceil(k / (1.0 - c)))
        est = cls(max_size=ms, c=c); 
        est._count = count
        for i, (vw, c) in enumerate(buffers):
            est._buffers[i][:len(vw)] = vw
            est._counts[i] = len(vw)
            if (est._caps[i] != c):
                raise ValueError(f"Buffer size mismatch. Level {i}, new estimator cap {est._caps[i]} expected {c}.")
        est._rng.setstate(rng_state)
        est._min = minval
        est._max = maxval
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
            mark = "✅"
        else:
            mark = "❌"
        print(f"      {mark} [{lo:.6g}, {hi:.6g}] {100*(1-delta):.1f}% CI")
    print()
