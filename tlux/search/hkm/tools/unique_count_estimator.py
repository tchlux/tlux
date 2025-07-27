"""
Compact cardinality estimator

Generalized fixed-memory cardinality estimation ("HyperLogLog sketch") with deterministic, vectorized,
and mergeable registers. Tracks unique byte sequences by hashing to N registers and recording the count
of leading zero bits plus one per register. Memory, speed, and error are all tunable via `precision`.

Accuracy: RSE ≈ 1.04 / sqrt(M). Ten-fold more registers reduces error by sqrt(10).
Memory:   1 byte per register; total size = 2**precision bytes.
Speed:    Hashing dominates. Each add() is constant-time.
Merge:    Supports register-wise OR-merge for sketches with equal precision.
Seeded:   User supplies hash function; default is deterministic (sha256).

Example
  from unique_count_estimator import UniqueCounter
  uc = UniqueCounter(precision=13)
  uc.add_batch((i.to_bytes(4, "big") for i in range(1000000)))
  est, lb, ub = uc.estimate()
  print(f"Estimate: {est:.0f} unique, 95% CI: [{lb:.0f}, {ub:.0f}]")
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Callable, Iterable, Tuple

__all__ = ["UniqueCounter"]

# Correction factor lookup table for HyperLogLog
_HLL_ALPHA = {
    8: 0.635,
    16: 0.673,
    32: 0.697,
    64: 0.709,
}

# UniqueCounter implements fixed-memory cardinality estimation using the HyperLogLog algorithm.
#
# Parameters:
#   precision (int): Number of register-index bits (3..18); total registers = 2**precision.
#   hash_function (Callable[[bytes], "hashlib._Hash"]): Hash function to use; default is sha256.
#
# Attributes:
#   precision (int): As above.
#   number_of_registers (int): 2**precision, number of registers.
#   correction_factor (float): Algorithm-specific correction constant.
#   registers (list[int]): Per-register maximum leading zero count.
#
# Example:
#   uc = UniqueCounter(precision=12)
#   uc.add(b"foo")
#   uc.add_batch([b"bar", b"baz"])
#   n, low, high = uc.estimate()
# 
class UniqueCounter:
    # Compute correction factor for given register count.
    #
    # Parameters:
    #   m (int): Number of registers.
    #
    # Returns:
    #   (float): Correction factor.
    #
    # Raises:
    #   ValueError: If m < 8 or not a power of two.
    # 
    @staticmethod
    def _correction_factor(m: int) -> float:
        if m in _HLL_ALPHA:
            return _HLL_ALPHA[m]
        if m >= 128 and (m & (m - 1)) == 0:
            return 0.7213 / (1 + 1.079 / m)
        raise ValueError("Number of registers must be power of two, >= 8.")

    # Return position of first '1' from left, counting from 1. Zero returns max+1.
    #
    # Parameters:
    #   n (int): Value.
    #   bits (int): Bit width.
    #
    # Returns:
    #   (int): Position of first 1 (1 = MSB), or bits+1 if n == 0.
    # 
    @staticmethod
    def _first_one_position(n: int, bits: int) -> int:
        if n == 0:
            return bits + 1
        return bits - n.bit_length() + 1

    # Lookup z-score for confidence interval.
    #
    # Parameters:
    #   confidence (float): Confidence in (0, 1).
    #
    # Returns:
    #   (float): z-score for confidence.
    #
    # Raises:
    #   ValueError: If confidence out of range.
    # 
    @staticmethod
    def _z_score(confidence: float) -> float:
        table = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.98: 2.326, 0.99: 2.576}
        if confidence in table:
            return table[confidence]
        if not 0.0 < confidence < 1.0:
            raise ValueError("Confidence level must be in (0,1).")
        try:
            # Python >= 3.8
            return math.sqrt(2) * math.erfinv(confidence)
        except AttributeError:
            # Rough approximation
            return math.sqrt(-2 * math.log(1 - confidence))

    # Initialize the estimator.
    #
    # Parameters:
    #   precision (int): Register-index bits [3, 18].
    #   hash_function (Callable): Function returning a hash object (default: sha256).
    #
    # Raises:
    #   ValueError: On invalid precision or hash function.
    # 
    def __init__(
        self,
        precision: int = 13,
        hash_function: Callable[[bytes], "hashlib._Hash"] = hashlib.sha256
    ):
        if not (3 <= precision <= 18):
            raise ValueError("precision must be in [3, 18].")
        self.precision: int = precision
        self.number_of_registers: int = 1 << precision
        self.correction_factor: float = self._correction_factor(self.number_of_registers)
        self.registers: list[int] = [0] * self.number_of_registers
        self._hash_function = hash_function
        sample_hash = hash_function()
        self.hash_bit_length: int = sample_hash.digest_size * 8
        if self.hash_bit_length < precision + 64:
            raise ValueError(
                f"Hash function must yield at least {precision+64} bits, got {self.hash_bit_length}."
            )

    # Add a single element to the counter.
    #
    # Parameters:
    #   data (bytes): Data to add.
    # 
    def add(self, data: bytes) -> None:
        h = self._hash_function(data).digest()
        hval = int.from_bytes(h, "big")
        idx = hval >> (self.hash_bit_length - self.precision)
        rem_mask = (1 << (self.hash_bit_length - self.precision)) - 1
        rem = hval & rem_mask
        pos = self._first_one_position(rem, self.hash_bit_length - self.precision)
        if pos > self.registers[idx]:
            self.registers[idx] = pos

    # Add multiple elements from iterable.
    #
    # Parameters:
    #   items (Iterable[bytes]): Stream of byte elements.
    # 
    def add_batch(self, items: Iterable[bytes]) -> None:
        for item in items:
            self.add(item)

    # Estimate unique count with confidence interval.
    #
    # Parameters:
    #   confidence_level (float): CI in (0,1); if <=0, only point estimate.
    #
    # Returns:
    #   (float, float, float): (estimate, lower_bound, upper_bound)
    # 
    def estimate(self, confidence_level: float = 0.95) -> Tuple[float, float, float]:
        est = self._raw_estimate()
        if confidence_level <= 0.0:
            return est, est, est
        rse = 1.04 / math.sqrt(self.number_of_registers)
        z = self._z_score(confidence_level)
        margin = z * est * rse
        return est, max(0.0, est - margin), est + margin

    # Raw (bias-corrected) cardinality estimate.
    #
    # Returns:
    #   (float): Estimated count.
    # 
    def _raw_estimate(self) -> float:
        m = self.number_of_registers
        cf = self.correction_factor
        sum_inv = sum(2.0 ** (-v) for v in self.registers)
        raw = cf * m * m / sum_inv
        zeros = self.registers.count(0)
        if zeros and raw <= 2.5 * m:
            return m * math.log(m / zeros)
        return raw

    # Merge another estimator (must have same precision).
    #
    # Parameters:
    #   other (UniqueCounter): Estimator to merge.
    #
    # Raises:
    #   ValueError: If precisions mismatch.
    # 
    def merge(self, other: "UniqueCounter") -> None:
        if self.precision != other.precision:
            raise ValueError("Cannot merge: precisions differ.")
        self.registers = [
            max(a, b) for a, b in zip(self.registers, other.registers)
        ]

    # Serialize estimator state to bytes.
    #
    # Returns:
    #   (bytes): 4 bytes precision, followed by registers.
    # 
    def to_bytes(self) -> bytes:
        b = struct.pack(">I", self.precision)
        b += bytes(self.registers)
        return b

    # Deserialize estimator from bytes.
    #
    # Parameters:
    #   serialized (bytes): State as produced by to_bytes().
    #   hash_function (Callable): Must match original.
    #
    # Returns:
    #   (UniqueCounter): New estimator.
    #
    # Raises:
    #   ValueError: On corrupt input.
    # 
    @classmethod
    def from_bytes(
        cls,
        serialized: bytes,
        hash_function: Callable[[bytes], "hashlib._Hash"] = hashlib.sha256
    ) -> "UniqueCounter":
        if len(serialized) < 4:
            raise ValueError("Serialized data too short.")
        precision = struct.unpack(">I", serialized[:4])[0]
        nreg = 1 << precision
        if len(serialized) != 4 + nreg:
            raise ValueError(f"Expected length {4+nreg}, got {len(serialized)}.")
        obj = cls(precision, hash_function)
        obj.registers = list(serialized[4:])
        return obj

    # Support in-place merge with +=
    def __iadd__(self, other: "UniqueCounter") -> "UniqueCounter":
        self.merge(other)
        return self

    # Debugging representation.
    def __repr__(self) -> str:
        rse = 1.04 / math.sqrt(self.number_of_registers)
        return (
            f"UniqueCounter(precision={self.precision}, RSE~{rse:.3%}, "
            f"registers={self.registers[:8]}...)"
        )


if __name__ == "__main__":
    # Quick demonstration; not a test.
    import time

    uc = UniqueCounter(precision=13)
    n = 1_000_000
    # Each unique: i in [0, n) as 4 bytes.
    data = (i.to_bytes(4, "big") for i in range(n))
    t0 = time.time()
    uc.add_batch(data)
    t1 = time.time()
    est, lo, hi = uc.estimate()
    print(f"\nEstimate: {est:,.0f} unique. [{lo:,.0f}, {hi:,.0f}] @ 95% CI")
    print(f"Processed {n / (t1 - t0):,.0f}/sec\n")
    print(uc)
