"""Bit-array bloom filter implementation.

This implementation is *pure-Python* and relies only on the standard
library.  The hash family is produced via **double hashing** on the
SHA-256 digest, which offers reproducible cross-platform behaviour while
avoiding extra dependencies.  A convenience constructor
:meth:`BloomFilter.create` picks a bit-array length and number of hash
functions from a desired capacity / target false-positive rate, matching
the analytical optimum *k = m/n ln 2*.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable


__all__ = ["BloomFilter"]


@dataclass
class BloomFilter:
    """Simple Bloom filter backed by a *bytearray* bit-array.

    Parameters
    ----------
    bits:
        Underlying bit-array (least-significant bit of ``bits[0]`` is bit 0).
    hash_count:
        Number of hash functions (``k``).  Must be ``>= 1``.
    """

    bits: bytearray
    hash_count: int

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls, capacity: int, fp_rate: float = 0.01, align: int = 8
    ) -> "BloomFilter":
        """Return a *new* filter sized for *capacity* and *fp_rate*.

        The bit-array length ``m`` is rounded *up* to the nearest multiple
        of *align* bits (default 8, i.e. whole bytes).
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (0 < fp_rate < 1):
            raise ValueError("fp_rate must be in (0,1)")

        # m = -(n ln p) / (ln 2)^2
        m = int(-capacity * math.log(fp_rate) / (math.log(2) ** 2))
        m = max(1, m)
        # round up to byte boundary (or *align*)
        m = (m + align - 1) // align * align

        # k = m/n ln 2
        k = max(1, int(round((m / capacity) * math.log(2))))
        return cls(bits=bytearray(m // 8), hash_count=k)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def _hashes(self, item: bytes) -> Iterable[int]:
        """Yield *k* distinct hash bucket indices via double hashing."""
        if not isinstance(item, (bytes, bytearray, memoryview)):
            raise TypeError("item for BloomFilter must be bytes-like")
        digest = hashlib.sha256(item).digest()
        h1 = int.from_bytes(digest[:16], "little")
        h2 = int.from_bytes(digest[16:], "little") or 1  # ensure non-zero
        m = len(self.bits) * 8
        for i in range(self.hash_count):
            yield (h1 + i * h2) % m

    def _set_bit(self, idx: int) -> None:
        self.bits[idx >> 3] |= 1 << (idx & 7)

    def _test_bit(self, idx: int) -> bool:
        return bool(self.bits[idx >> 3] & (1 << (idx & 7)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, item: bytes) -> None:
        """Insert *item* into the filter."""
        for idx in self._hashes(item):
            self._set_bit(idx)

    def __contains__(self, item: bytes) -> bool:  # noqa: Dunder/magic-name
        """Return ``True`` if *item* **may** be present; ``False`` if *definitely not*."""
        return all(self._test_bit(idx) for idx in self._hashes(item))
