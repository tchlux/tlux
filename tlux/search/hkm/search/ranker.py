"""Ranking utilities using NumPy distance computations."""

import numpy as np


def rank(embeddings: np.ndarray, queries: np.ndarray, top_k: int):
    """Return ``top_k`` closest embedding indices per query."""
    raise NotImplementedError
