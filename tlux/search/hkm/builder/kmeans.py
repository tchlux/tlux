"""Lightweight k-means implementation for HKM builder.

This module provides a NumPy-only implementation of Lloyd's algorithm with
k-means++ initialisation, suitable for clustering up to 4 096 centroids as
required by the HKM specification.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def _kmeans_pp_init(
    data: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Initialise centroids with the k-means++ heuristic."""

    n, _ = data.shape
    centroids = np.empty((k, data.shape[1]), dtype=np.float32)

    # First centroid chosen uniformly at random
    centroids[0] = data[rng.integers(0, n)]

    # Squared distance of every point to its nearest centroid so far
    dist2 = np.linalg.norm(data - centroids[0], axis=1, dtype=np.float32) ** 2

    for i in range(1, k):
        probs = dist2 / dist2.sum(dtype=np.float32)
        next_idx = rng.choice(n, p=probs)
        centroids[i] = data[next_idx]
        new_dist2 = np.linalg.norm(
            data - centroids[i], axis=1, dtype=np.float32
        ) ** 2
        dist2 = np.minimum(dist2, new_dist2)

    return centroids


def kmeans(
    data: np.ndarray,
    k: int,
    max_iter: int = 20,
    tol: float = 1e-4,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Lloyd's k-means clustering on *data*.

    Parameters
    ----------
    data:
        Array of shape ``(n, d)`` containing *n* ``float32`` embeddings.
    k:
        Number of clusters (``1 ≤ k ≤ 4096``).
    max_iter:
        Maximum number of Lloyd iterations.
    tol:
        Early-stopping threshold on the maximum centroid displacement.
    seed:
        Optional RNG seed for deterministic behaviour.

    Returns
    -------
    centroids:
        Array of shape ``(k, d)`` with cluster centres.
    labels:
        Array of shape ``(n,)`` with integer cluster assignments.
    """

    if data.ndim != 2:
        raise ValueError("data must be 2-D (n, d)")
    n_samples, dim = data.shape
    if not n_samples:
        raise ValueError("data is empty")
    if not (1 <= k <= min(4096, n_samples)):
        raise ValueError(f"k must be between 1 and {min(4096, n_samples)}")

    # Ensure float32 for consistency and memory efficiency
    data = data.astype(np.float32, copy=False)

    rng = np.random.default_rng(seed)
    centroids = _kmeans_pp_init(data, k, rng)

    labels = np.empty(n_samples, dtype=np.int32)

    for _ in range(max_iter):
        # ------- Assignment step -----------------------------------------
        # Compute squared L2 distances (broadcasted, (n, k))
        sq_dists = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = sq_dists.argmin(axis=1)

        # ------- Update step ---------------------------------------------
        moved = 0.0
        for i in range(k):
            members = data[new_labels == i]
            if members.size:  # normal case
                new_centroid = members.mean(axis=0, dtype=np.float32)
            else:  # re-initialise empty cluster to a random point
                new_centroid = data[rng.integers(0, n_samples)]
            moved = max(moved, float(np.linalg.norm(centroids[i] - new_centroid)))
            centroids[i] = new_centroid

        labels = new_labels  # commit assignment

        if moved < tol:
            break

    return centroids, labels


__all__ = ["kmeans"]
