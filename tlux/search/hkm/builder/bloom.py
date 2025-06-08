"""Bit-array bloom filter implementation."""

from dataclasses import dataclass
from typing import Iterable


@dataclass
class BloomFilter:
    """Simple bloom filter placeholder."""

    bits: bytearray
    hash_count: int

    def add(self, item: bytes) -> None:
        raise NotImplementedError

    def __contains__(self, item: bytes) -> bool:
        raise NotImplementedError
