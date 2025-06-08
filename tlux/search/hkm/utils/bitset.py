"""Very small fixed-size bitset implementation."""

from dataclasses import dataclass


@dataclass
class BitSet:
    """A fixed-size bitset supporting up to 100 bits."""

    bits: int = 0

    def set(self, idx: int) -> None:
        self.bits |= 1 << idx

    def test(self, idx: int) -> bool:
        return bool(self.bits & (1 << idx))
