"""
ValueObserver FP-rate test - Empirical false positive rate validation for capacity 2**10.

This script constructs a Bloom filter via ValueObserver.create with a target
false-positive rate, inserts exactly n=2**10 distinct items, and then issues a
large number of non-member queries drawn from a disjoint key range. It reports
the observed false positive rate, compares it to the theoretical rate
p_theory = (1 - exp(-k*n/m))**k derived from the realized (m,k), and asserts
that the observation is within a statistically reasonable margin (5 sigma)
around p_theory (binomial normal approximation).

Usage
  python test_value_observer_fp.py
"""

from __future__ import annotations

import math
import random
from typing import Dict, Tuple

# Local import of the implementation under test.
# Assumes this file sits next to value_seen_estimator.py.
from tlux.search.hkm.tools.value_seen_estimator import ValueObserver


# make_item
#
# Deterministic encoding from an integer to a short ASCII bytes payload.
#
# Parameters:
#   x (int): Non-negative integer identifier.
#   tag (str): ASCII prefix to ensure disjoint keyspaces via different tags.
#
# Returns:
#   (bytes): Encoded key.
#
def make_item(x: int, tag: str) -> bytes:
    if x < 0:
        raise ValueError("x must be non-negative")
    if not isinstance(tag, str) or any(ord(c) > 127 for c in tag):
        raise TypeError("tag must be ASCII str")
    # Use a simple, explicit ASCII encoding; zero-pad to keep lengths regular.
    return (tag + ":" + str(x).zfill(12)).encode("ascii")


# run_experiment
#
# Build a ValueObserver with the given capacity and fp_rate, insert n items,
# and probe with q disjoint non-members to estimate the false-positive rate.
#
# Example:
#   stats = run_experiment(capacity=2**10, fp_rate=0.01, n=2**10, q=200_000, seed=12345)
#   print(stats["p_obs"], stats["p_theory"])
#
# Parameters:
#   capacity (int): Filter capacity used to size m and k.
#   fp_rate (float): Target false positive rate for sizing.
#   n (int): Number of inserted items (should be <= capacity for the sizing guarantee).
#   q (int): Number of non-member queries (large for tight CI; e.g., 2e5).
#   seed (int): RNG seed for reproducible item selection/order.
#
# Returns:
#   (Dict[str, float|Tuple[int,int]]): Summary statistics.
#
def run_experiment(
    capacity: int, fp_rate: float, n: int, q: int, seed: int
) -> Dict[str, float]:
    if n <= 0 or q <= 0:
        raise ValueError("n and q must be positive")
    if n > 10_000_000 or q > 10_000_000:
        raise ValueError("n or q unreasonably large for this test")
    rng = random.Random(seed)

    obs = ValueObserver.create(capacity=capacity, fp_rate=fp_rate)
    m_bits = len(obs.bits) * 8
    k = obs.hash_count

    # Insert exactly n members from the "A" namespace.
    # Use a shuffled small range to avoid pathological clustering in the hash inputs.
    inserts = list(range(n))
    rng.shuffle(inserts)
    for x in inserts:
        obs.add(make_item(x, "A"))

    # Probe q items from a disjoint namespace "B" to estimate false positives.
    # Draw uniformly from a wide range far from the insert range.
    # Disjointness is guaranteed by the differing namespace tag.
    fp = 0
    for i in range(q):
        x = rng.randrange(1 << 61)  # large domain
        key = make_item(x, "B")
        if key in obs:
            fp += 1

    p_obs = fp / float(q)

    # Theoretical FP rate for the realized (m,k) with n inserts.
    # p_theory = (1 - exp(-k*n/m))**k
    lam = (k * n) / float(m_bits)
    p_theory = (1.0 - math.exp(-lam)) ** k

    # Binomial normal approximation: sigma = sqrt(p*(1-p)/q).
    # Use p_theory for sigma to define the acceptance band.
    sigma = math.sqrt(max(p_theory * (1.0 - p_theory), 1e-18) / float(q))
    # Require observation to be within 5 sigma of theory. This is strict yet fair for q large.
    delta = abs(p_obs - p_theory)
    if delta > 5.0 * sigma:
        raise AssertionError(
            "Observed FP rate deviates from theory: "
            + f"p_obs={p_obs:.6g}, p_theory={p_theory:.6g}, "
            + f"delta={delta:.6g}, 5sigma={5.0*sigma:.6g}, "
            + f"(m_bits={m_bits}, k={k}, n={n}, q={q})"
        )

    return {
        "p_obs": p_obs,
        "p_theory": p_theory,
        "sigma": sigma,
        "m_bits": float(m_bits),
        "k": float(k),
        "n": float(n),
        "q": float(q),
    }


if __name__ == "__main__":
    # Deterministic, capacity-matched experiment with a large probe set.
    cap = 2**12
    target_fp = 0.001
    stats = run_experiment(
        capacity=cap, fp_rate=target_fp, n=cap, q=200_000, seed=12345
    )
    print()
    print(f"Capacity:  {cap}")
    print(f"Target FP: {target_fp*100:.1f}%")
    print()
    print("Observed FP rate:   ", f"{stats['p_obs']:.6f}")
    print("Theoretical FP rate:", f"{stats['p_theory']:.6f}")
    print("Std. error (theory):", f"{stats['sigma']:.6f}")
    print("Upper FP rate:      ", f"{stats['p_theory']+3*stats['sigma']:.6f}")
    print()
    print("m_bits:", int(stats["m_bits"]), "k:", int(stats["k"]))
