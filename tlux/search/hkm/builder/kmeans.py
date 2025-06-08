"""Lightweight k-means implementation."""

import numpy as np


def kmeans(data: np.ndarray, k: int, max_iter: int = 20):
    """Run Lloyd's algorithm on ``data``.

    Parameters
    ----------
    data:
        ``(n, d)`` array of float32 embeddings.
    k:
        Number of clusters, ``k`` ≤ 4096.
    max_iter:
        Maximum number of iterations.
    """
    raise NotImplementedError
