"""
unique_count_estimator.py

Compact cardinality estimator
=============================

A generalized counting utility that stores a fixed number of "registers"
and uses a hash function plus the count of leading zeros (plus one) to
estimate how many unique byte sequences have been inserted into the counter.
By default, precision = 13, which implies 2**13 = 8192 bytes of "registers".

Key points
----------
- Accuracy: RSE approximately 1.04 / sqrt(M). Ten-fold more registers reduces error by sqrt(10).
- Memory: One byte per register (could be 6 bits).
- Speed: Hashing dominates; per-item update is constant-time.
- Merge-ability: Two sketches with equal p can be OR-merged.

Example
-------
```python
from unique_count_estimator import UniqueCounter
uc = UniqueCounter(precision=13)  # 8,192 registers, relative standard error of ~1.1%
bytes_generator = (
    i.to_bytes(4)                 # Always uses a 4-byte reprsentation.
    for i in range(1000000)       # Convert integers in the range [0, 1M) to bytes
)
uc.add_batch(bytes_generator)     # Using bulk addition, same as for loop with '.add'
print(uc.estimate())              # Returns (point_estimate, lower_bound, upper_bound)
```
"""

from __future__ import annotations
import hashlib
import math
import struct
from typing import Callable, Iterable, Tuple

__all__ = ["UniqueCounter"]

# Fixed memory usage cardinality estimator class.
# This class provides a way to estimate the number of unique items in a data stream.
# It uses a "HyperLogLog sketch", which is memory-efficient and allows merging counters.
# You can choose the 'precision' by adjusting the number of registers is 2^p.
class UniqueCounter:
    # Static method to compute the correction factor for the estimate.
    # The correction factor depends on the number of registers and adjusts the raw estimate.
    @staticmethod
    def calculate_correction_factor(number_of_registers: int) -> float:
        if number_of_registers == 8:
            return 0.635
        if number_of_registers == 16:
            return 0.673
        if number_of_registers == 32:
            return 0.697
        if number_of_registers == 64:
            return 0.709
        if number_of_registers >= 128:
            return 0.7213 / (1 + 1.079 / number_of_registers)
        raise ValueError("Number of registers must be a power of two and at least 8.")

    # Static method to find the position of the first 1-bit in a number
    # This counts from the left (most significant bit), starting at position 1.
    # If the number is 0, it returns the maximum possible position plus 1.
    @staticmethod
    def find_first_one_position(number: int, max_bit_length: int) -> int:
        # For example, if max_bit_length=4:
        # number=0b1000 -> returns 1 (first 1 at position 1 from left)
        # number=0b0100 -> returns 2
        # number=0b0000 -> returns 5 (max_bit_length + 1)
        return max_bit_length - number.bit_length() + 1

    # Static method to compute the z-score for a confidence level
    # The z-score is used to calculate the width of the confidence interval.
    @staticmethod
    def compute_z_score(confidence_level: float) -> float:
        # Predefined z-scores for common confidence levels
        z_score_table = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.98: 2.326, 0.99: 2.576}
        if confidence_level in z_score_table:
            return z_score_table[confidence_level]
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("Confidence level must be between 0 and 1, exclusive.")
        try:  # Use precise calculation if available (Python 3.8+)
            return math.sqrt(2) * math.erfinv(confidence_level)
        except AttributeError:  # Fall back to approximation for older Python versions
            temp = math.sqrt(-2 * math.log(1 - confidence_level))
            return temp  # Good enough for confidence levels between 0.8 and 0.99

    # Initialize a new unique count estimator
    # - precision: Number of bits to use for register indexing (3 to 18), default 13 -> 2**13 = 8KB.
    # - hash_function: A function that hashes bytes (e.g., hashlib.sha256)
    def __init__(self, precision: int = 13, hash_function: Callable[[bytes], "hashlib._Hash"] = hashlib.sha256):
        # Check that precision is within the allowed range
        if not (3 <= precision <= 18):
            raise ValueError("Precision must be between 3 and 18 (8 to 262,144 registers).")
        self.precision: int = precision
        self.number_of_registers: int = 1 << precision  # 2^precision registers
        self.correction_factor: float = UniqueCounter.calculate_correction_factor(self.number_of_registers)
        self.registers: list[int] = [0] * self.number_of_registers  # Start all registers at 0
        self._hash_function = hash_function
        # Verify the hash function provides enough bits
        sample_hash = hash_function()
        digest_size_in_bytes = sample_hash.digest_size
        self.hash_bit_length: int = 8 * digest_size_in_bytes
        minimum_required_bits = self.precision + 64
        if self.hash_bit_length < minimum_required_bits:
            raise ValueError(f"Hash function must provide at least {minimum_required_bits} bits, got {self.hash_bit_length}.")

    # Add a single piece of data to the estimator
    # The data is hashed, and the hash updates one of the registers.
    def add(self, data: bytes) -> None:
        # Step 1: Hash the data into a fixed-size byte string
        hash_object = self._hash_function(data)
        hash_bytes = hash_object.digest()
        # Step 2: Convert the hash bytes into a large integer
        hash_value = int.from_bytes(hash_bytes, "big")
        # Step 3: Use the top 'precision' bits to pick a register
        register_index = hash_value >> (self.hash_bit_length - self.precision)
        # Step 4: Use the remaining bits to find the first 1-bit position
        remaining_bits_mask = (1 << (self.hash_bit_length - self.precision)) - 1
        remaining_bits = hash_value & remaining_bits_mask
        first_one_position = UniqueCounter.find_first_one_position(
            remaining_bits, self.hash_bit_length - self.precision
        )
        # Step 5: Update the register if this position is greater than the current value
        if first_one_position > self.registers[register_index]:
            self.registers[register_index] = first_one_position

    # Add multiple data items at once
    # This loops through the items and calls add() for each one.
    def add_batch(self, items: Iterable[bytes]) -> None:
        for item in items:
            self.add(item)

    # Estimate the number of unique items with a confidence interval
    # - confidence_level: A value between 0 and 1 (e.g., 0.95 for 95% confidence)
    # Returns a tuple: (estimate, lower_bound, upper_bound)
    def estimate(self, confidence_level: float = 0.95) -> Tuple[float, float, float]:
        # If no confidence interval is wanted, return the estimate three times
        if confidence_level <= 0.0:
            raw_estimate = self._compute_raw_estimate()
            return raw_estimate, raw_estimate, raw_estimate
        # Get the base estimate
        raw_estimate = self._compute_raw_estimate()
        # Calculate the relative standard error and confidence interval
        relative_standard_error = 1.04 / math.sqrt(self.number_of_registers)
        z_score = UniqueCounter.compute_z_score(confidence_level)
        margin_of_error = z_score * raw_estimate * relative_standard_error
        lower_bound = max(0.0, raw_estimate - margin_of_error)
        upper_bound = raw_estimate + margin_of_error
        # Return the estimate, lower, and upper bounds.
        return raw_estimate, lower_bound, upper_bound

    # Compute the raw estimate of unique items without confidence intervals
    # This applies the "HyperLogLog" formula and a correction for small counts.
    def _compute_raw_estimate(self) -> float:
        # Sum the inverse powers of 2 based on register values
        sum_of_inverse_powers = sum(2.0 ** (-register_value) for register_value in self.registers)
        # Calculate the initial estimate using the correction factor
        estimate = self.correction_factor * self.number_of_registers * self.number_of_registers / sum_of_inverse_powers
        # Correction for small counts: use linear counting if many registers are zero
        number_of_zero_registers = self.registers.count(0)
        if number_of_zero_registers > 0 and estimate <= 2.5 * self.number_of_registers:
            estimate = self.number_of_registers * math.log(self.number_of_registers / number_of_zero_registers)
        # Return the raw estimate.
        return estimate

    # Merge another unique count estimator into this one
    # Both must have the same precision value.
    def merge(self, other: "UniqueCounter") -> None:
        # Check compatibility
        if self.precision != other.precision:
            raise ValueError("Cannot merge estimators with different precision values.")
        # Update each register to the maximum of the two values
        self.registers = [max(self_val, other_val) for self_val, other_val in zip(self.registers, other.registers)]

    # Serialize the precision and registers into a byte string.
    # The format is:
    # - 4 bytes: precision (big-endian unsigned integer)
    # - followed by number_of_registers bytes: each byte represents a register value
    # Note: Register values are guaranteed to be <= 255 by the class design,
    # since max value = hash_bit_length - precision + 1, and hash_bit_length >= precision + 64.
    def dumps(self) -> bytes:
        precision_bytes = struct.pack(">I", self.precision)
        registers_bytes = bytes(self.registers)
        return precision_bytes + registers_bytes

    # Deserialize the byte string to create a new UniqueCounter instance.
    # The serialized data must follow the format from dumps():
    # - First 4 bytes: precision (big-endian unsigned integer)
    # - Next number_of_registers bytes: register values
    # The hash_function must be provided by the user and should match the original
    # to ensure consistent behavior, though its digest size is checked during initialization.
    @classmethod
    def loads(cls, serialized: bytes, hash_function: Callable[[bytes], "hashlib._Hash"] = hashlib.sha256) -> "UniqueCounter":
        if len(serialized) < 4:
            raise ValueError("Serialized data is too short.")
        precision = struct.unpack(">I", serialized[:4])[0]
        if not (3 <= precision <= 18):
            raise ValueError("Invalid precision value.")
        number_of_registers = 1 << precision
        expected_length = 4 + number_of_registers
        if len(serialized) != expected_length:
            raise ValueError(f"Serialized data has incorrect length. Expected {expected_length}, got {len(serialized)}.")
        registers_bytes = serialized[4:4 + number_of_registers]
        uc = cls(precision=precision, hash_function=hash_function)
        uc.registers = list(registers_bytes)
        return uc

    # Support in-place merging with the += operator
    def __iadd__(self, other: "UniqueCounter") -> "UniqueCounter":
        self.merge(other)
        return self

    # Provide a string representation for debugging
    def __repr__(self) -> str:
        relative_standard_error = 1.04 / math.sqrt(self.number_of_registers)
        return (
            f"UniqueCounter(precision={self.precision}, RSE approx. {relative_standard_error:.3%}, "
            f"registers={self.registers[:8]}...)"
        )


if __name__ == "__main__":
    import time
    from unique_count_estimator import UniqueCounter
    # 2**13 = 8,192 registers. Relative standard error of ~1.1%
    uc = UniqueCounter(precision=13)
    n = 1000000
    # Convert integers in the range [0, 2n) to bytes
    bytes_generator = (
        # Always uses a 4-byte reprsentation.
        (i//2).to_bytes(4, "big")
        for i in range(2*n)
    )
    print()
    print(f"Counting through {2*n} elements..", flush=True)
    start = time.time()
    # Using bulk addition, same as for loop with '.add'
    uc.add_batch(bytes_generator)
    end = time.time()
    # Returns (estimate, lower_bound, upper_bound)
    estimate, lower, upper = uc.estimate()
    print()
    print(f"  {estimate:,.0f} unique. Something between [{lower:,.0f}, {upper:,.0f}] with 95% confidence")
    print()
    print(f"Processed at {2*n / (end-start):,.0f} per second.")
    print()
    print(uc)
