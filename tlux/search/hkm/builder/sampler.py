"""Sampling utilities for HKM builder.

This module provides two independent helpers used by the index builder:

* **Reservoir sampling** for uniformly selecting *k* items from a stream
  of unknown or very large size using O(k) memory.
* **k-means++ sampling** to obtain initial centroids for Lloyd’s
  iterations.  The implementation is NumPy-only and follows the original
  Arthur & Vassilvitskii 2007 algorithm.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence, TypeVar

import numpy as np

T = TypeVar("T")  # generic item type for reservoir_sample

__all__ = ["reservoir_sample", "kmeanspp_sample"]

# ---------------------------------------------------------------------------
# Reservoir sampling
# ---------------------------------------------------------------------------

def reservoir_sample(iterable: Iterable[T], k: int) -> List[T]:
    """Return *k* items uniformly sampled from *iterable*.

    This is Vitter's Algorithm R.  The function consumes the input in a
    single pass and keeps only *k* items in memory.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    reservoir: List[T] = []
    for i, item in enumerate(iterable):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)  # inclusive
            if j < k:
                reservoir[j] = item
    if len(reservoir) < k:
        raise ValueError("iterable has fewer than k items")
    return reservoir


# ---------------------------------------------------------------------------
# k-means++ initial centroid sampling
# ---------------------------------------------------------------------------

def kmeanspp_sample(
    data: np.ndarray | Sequence[Sequence[float]], k: int, seed: int | None = None
) -> np.ndarray:
    """Return *k* initial centroids chosen with the k-means++ heuristic.

    Parameters
    ----------
    data:
        2-D array-like of shape ``(n, d)``.
    k:
        Desired number of centroids (``1 ≤ k ≤ n``).
    seed:
        Optional RNG seed for reproducible selection.

    Returns
    -------
    centroids:
        Array of shape ``(k, d)`` with *float32* centroids.
    """

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("data must be 2-D (n, d)")
    n, d = data.shape
    if not (1 <= k <= n):
        raise ValueError("k must be in [1, n]")

    rng = np.random.default_rng(seed)

    centroids = np.empty((k, d), dtype=np.float32)
    # Pick first centroid at random
    centroids[0] = data[rng.integers(0, n)]

    # Squared distances to nearest centroid so far
    dist2 = np.square(data - centroids[0]).sum(axis=1)

    for i in range(1, k):
        # Probability proportional to squared distance
        probs = dist2 / dist2.sum(dtype=np.float32)
        idx = rng.choice(n, p=probs)
        centroids[i] = data[idx]
        # Update dist2 with the newly added centroid
        new_dist2 = np.square(data - centroids[i]).sum(axis=1)
        dist2 = np.minimum(dist2, new_dist2)

    return centroids
