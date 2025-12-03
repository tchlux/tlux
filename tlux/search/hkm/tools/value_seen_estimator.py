"""
ValueObserver - Bit-array value observer via hash bit mask (Bloom filter) implementation.

The `ValueObserver` class provides methods to add items and check for
possible membership, with a controlled false-positive rate. This
implementation is pure-Python and relies only on the standard library.
The hash family is produced via double hashing on the SHA-256 digest,
which offers reproducible cross-platform behaviour while avoiding
extra dependencies. A convenience constructor `ValueObserver.create`
picks a bit-array length and number of hash functions from a desired
capacity / target false-positive rate, matching the analytical
optimum k = m/n ln 2.

Example
  observer = ValueObserver.create(capacity=1000, fp_rate=0.01)
  observer.add(b"hello")
  observer.add(b"world")
  print(b"hello" in observer)  # True
  print(b"foo" in observer)    # False (probably)
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Iterable


# ValueObserver
#
# A Bloom filter that uses a bytearray as the underlying bit-array and double hashing
# for generating hash functions.
#
# Attributes
#   bits (bytearray): The bit-array where each bit represents a slot in the filter.
#   hash_count (int): The number of hash functions used to map items to bit positions.
#
@dataclass
class ValueObserver:
    bits: bytearray
    hash_count: int

    # Create a new ValueObserver sized for the given capacity and false-positive rate.
    # The bit-array length m is calculated using the formula m = -(n ln p) / (ln 2)^2
    # and rounded up to the nearest multiple of align bits (default 8).
    # The number of hash functions k is calculated as k = m/n ln 2.
    #
    # Parameters
    #   capacity (int): The expected number of items to be added.
    #   fp_rate (float, optional): The desired false-positive rate (default 0.01).
    #   align (int, optional): The alignment in bits for the bit-array length (default 8).
    #
    # Returns
    #   (ValueObserver): A new ValueObserver instance.
    #
    @classmethod
    def create(
        cls, capacity: int, fp_rate: float = 0.01, align: int = 8
    ) -> "ValueObserver":
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (0 < fp_rate < 1):
            raise ValueError("fp_rate must be in (0,1)")

        # Calculate optimal m = -(n ln p) / (ln 2)^2
        m = int(-capacity * math.log(fp_rate) / (math.log(2) ** 2))
        m = max(1, m)
        # Round up to nearest multiple of align bits
        m = (m + align - 1) // align * align
        # Calculate optimal k = m/n ln 2
        k = max(1, int(round((m / capacity) * math.log(2))))
        return cls(bits=bytearray(m // 8), hash_count=k)

    def _hashes(self, item: bytes) -> Iterable[int]:
        # Ensure item is bytes-like
        if not isinstance(item, (bytes, bytearray, memoryview)):
            raise TypeError("item for ValueObserver must be bytes-like")
        # Compute SHA-256 digest for consistent hashing
        digest = hashlib.sha256(item).digest()
        # Split digest into two 128-bit integers for double hashing
        h1 = int.from_bytes(digest[:16], "little")
        h2 = int.from_bytes(digest[16:], "little") or 1  # ensure h2 is non-zero
        m = len(self.bits) * 8
        for i in range(self.hash_count):
            # Generate hash indices using double hashing formula
            yield (h1 + i * h2) % m

    def _set_bit(self, idx: int) -> None:
        # Set the bit at index idx
        self.bits[idx >> 3] |= 1 << (idx & 7)

    def _test_bit(self, idx: int) -> bool:
        # Test if the bit at index idx is set
        return bool(self.bits[idx >> 3] & (1 << (idx & 7)))

    # Add an item to the filter. This method sets the bits in the
    # bit-array corresponding to the hash values of the item.
    #
    # Parameters
    #   item (bytes): The item to add.
    #
    def add(self, item: bytes) -> None:
        for idx in self._hashes(item):
            self._set_bit(idx)

    # Check if an item may be present in the filter.
    #
    # This method checks if all bits corresponding to the hash values of the item are set.
    # It may return false positives but never false negatives.
    #
    # Parameters
    #   item (bytes): The item to check.
    #
    # Returns
    #   (bool): True if the item may be present, False if it is definitely not present.
    #
    def __contains__(self, item: bytes) -> bool:
        return all(self._test_bit(idx) for idx in self._hashes(item))

    # Merge another ValueObserver into this one in-place.
    #
    # Both observers must have the same bit-array length and hash_count.
    # After merging, this filter will match any item present in either filter.
    #
    # Parameters
    #   other (ValueObserver): Another ValueObserver to merge.
    #
    # Raises
    #   ValueError: If bit-array lengths or hash_count differ.
    #
    def merge(self, other: "ValueObserver") -> None:
        if self.hash_count != other.hash_count:
            raise ValueError("hash_count mismatch: cannot merge")
        if len(self.bits) != len(other.bits):
            raise ValueError("bit-array size mismatch: cannot merge")
        # Merge by bitwise-OR each byte.
        for i in range(len(self.bits)):
            self.bits[i] |= other.bits[i]

    # Serialize the filter to a compact binary representation.
    #
    # The binary format consists of the hash_count as a 2-byte little-endian unsigned integer
    # followed by the bit-array bytes.
    #
    # Returns
    #   (bytes): The binary representation of the filter.
    #
    def to_bytes(self) -> bytes:
        return struct.pack("<H", self.hash_count) + bytes(self.bits)

    # Reconstruct a ValueObserver from its binary representation.
    #
    # Parameters
    #   data (bytes): The binary data produced by to_bytes.
    #
    # Returns
    #   (ValueObserver): The reconstructed ValueObserver instance.
    #
    # Raises
    #   ValueError
    #     If the data is too short to contain the hash_count.
    #
    @classmethod
    def from_bytes(cls, data: bytes) -> "ValueObserver":
        if len(data) < 2:
            raise ValueError("data too short for ValueObserver")
        k = struct.unpack("<H", data[:2])[0]
        return cls(bits=bytearray(data[2:]), hash_count=k)


if __name__ == "__main__":
    print()
    print("-"*100)
    print()
    # Demonstration of ValueObserver usage
    observer = ValueObserver.create(capacity=10, fp_rate=0.1)
    items = [b"apple", b"banana", b"cherry"]
    for item in items:
        observer.add(item)
    for item in items:
        assert item in observer
    non_present = b"date"
    if non_present in observer:
        print(f"False positive: {non_present} reported as present")
    else:
        print(f"{non_present} correctly reported as not present")
    print()
    print("-"*100)
    print()

    fp = 0.001
    print("Checking capacity byte sizes with {100*fp:.2f}% false positive rate..")
    for capacity in (2**10, 2**16, 2**20, 2**30):
        observer = ValueObserver.create(capacity=capacity, fp_rate=fp)
        size = len(observer.to_bytes())
        print()
        print(f" With capacity {capacity} at {100*fp:.1f}% false positives,")
        if size > 2**10:
            print(f"  is {size / 2**10:.1f} KB")
        elif size > 2**20:
            print(f"  is {size / 2**20:.1f} MB")
        else:
            print(f"  is {size} bytes")

    # Quick false positive rate demo
    cap = 2**12
    print()
    print("-"*100)
    print()
    print(f"Checking actual false positive rate with {cap} insertions..")
    observer = ValueObserver.create(capacity=cap, fp_rate=fp)
    # Insert odd numbers
    for x in range(1, 2 * cap, 2):
        observer.add(str(x).encode("ascii"))
    # Query even numbers
    queries = 100_000_0
    false_pos = 0
    for x in range(0, queries, 2):
        if str(x).encode("ascii") in observer:
            false_pos += 1
    print()
    print(f"Empirical false positive rate: {false_pos / queries:.6f}")
    print()
    print("-"*100)
    print()

