"""Ranking utilities using NumPy distance computations.

The public helper :func:`rank` returns the *indices* and *scores* of the
``top_k`` nearest embeddings (smallest distances) for each query vector.
The implementation is entirely NumPy-based and uses ``argpartition`` for
O(*n*) selection followed by an ordered argsort to obtain a fully sorted
result for each query.

This module does **not** assume any particular metric beyond *squared L2*.
If cosine / inner-product measures are required in future, they can be
built as thin wrappers converting inputs before delegating to
:func:`_pairwise_sq_l2` and :func:`rank`.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

__all__ = ["rank"]


def _pairwise_sq_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return matrix ``D[i, j] = ||a[i] - b[j]||²``.

    The computation is performed in *float32* regardless of the original
    dtype to minimise memory bandwidth while retaining adequate
    precision for ANN search.  Broadcasting formula:

        ||x − y||² = ||x||² + ||y||² − 2 x·y.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    # Pre-compute squared norms once for each side
    a_norm = (a ** 2).sum(axis=1, keepdims=True)
    b_norm = (b ** 2).sum(axis=1, keepdims=True).T  # row → col vector

    # D = ||a||² + ||b||²ᵀ − 2 a·bᵀ
    return a_norm + b_norm - 2.0 * np.matmul(a, b.T, dtype=np.float32)


def rank(
    embeddings: np.ndarray, queries: np.ndarray, top_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return *top_k* nearest indices for each query.

    Parameters
    ----------
    embeddings:
        ``(n, d)`` array of float32 candidate vectors.
    queries:
        ``(q, d)`` array of float32 query vectors.
    top_k:
        Number of nearest neighbours to return (``1 ≤ top_k ≤ n``).

    Returns
    -------
    indices:
        ``(q, top_k)`` int32 array of embedding indices, sorted by
        ascending distance per query (closest first).
    distances:
        ``(q, top_k)`` float32 array of squared L2 distances matching
        *indices*.
    """

    if embeddings.ndim != 2 or queries.ndim != 2:
        raise ValueError("embeddings and queries must be 2-D arrays")
    if embeddings.shape[1] != queries.shape[1]:
        raise ValueError("dimensionality mismatch between embeddings and queries")

    n = embeddings.shape[0]
    if not (1 <= top_k <= n):
        raise ValueError("top_k must be in [1, n]")

    # Pairwise squared distances (q, n)
    dist = _pairwise_sq_l2(queries, embeddings)

    # argpartition to get *unsorted* top-k (O(n) per row)
    part_idx = np.argpartition(dist, top_k - 1, axis=1)[:, :top_k]

    # Gather corresponding distances then sort each row (O(k log k))
    part_dist = np.take_along_axis(dist, part_idx, axis=1)
    order = np.argsort(part_dist, axis=1)

    indices = np.take_along_axis(part_idx, order, axis=1).astype(np.int32, copy=False)
    distances = np.take_along_axis(part_dist, order, axis=1)

    return indices, distances
