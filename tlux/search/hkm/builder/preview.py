"""Preview chunk selection utilities.

During index build, each HKM node stores a **preview set** of *128* chunk
IDs – half *representative* (uniform random) and half *diverse* (greedy
farthest-point).  The helpers below implement those two sampling
strategies with an optional RNG seed for reproducibility.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence

import numpy as np

__all__ = ["select_random", "select_diverse"]

# ---------------------------------------------------------------------------
# Uniform random sampling
# ---------------------------------------------------------------------------

def select_random(
    chunks: Sequence[int] | np.ndarray | Iterable[int],
    k: int,
    seed: int | None = None,
) -> List[int]:
    """Return *k* random chunk identifiers from *chunks* without replacement."""

    chunks = list(chunks)  # ensure indexable sequence
    if k > len(chunks):
        raise ValueError("k cannot exceed number of chunks")

    rng = random.Random(seed)
    return rng.sample(chunks, k)


# ---------------------------------------------------------------------------
# Greedy farthest-point sampling for diversity
# ---------------------------------------------------------------------------

def select_diverse(
    embeddings: np.ndarray,
    k: int,
    seed: int | None = None,
) -> List[int]:
    """Return indices of *k* diverse points via farthest-point heuristic.

    Parameters
    ----------
    embeddings:
        ``(n, d)`` array of *float32* chunk embeddings.
    k:
        Number of points to select (``1 ≤ k ≤ n``).
    seed:
        Optional RNG seed controlling the first anchor selection.

    Returns
    -------
    indices:
        List of *k* integer indices into *embeddings*, ordered by the
        sequence they were selected.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2-D (n, d)")
    n = embeddings.shape[0]
    if not (1 <= k <= n):
        raise ValueError("k must be in [1, n]")

    emb = embeddings.astype(np.float32, copy=False)

    rng = np.random.default_rng(seed)
    first = rng.integers(0, n)
    selected = [first]

    # Squared L2 distance of every point to the nearest selected point
    dist2 = np.square(emb - emb[first]).sum(axis=1, dtype=np.float32)

    for _ in range(1, k):
        next_idx = int(dist2.argmax())
        selected.append(next_idx)
        # Update dist2 with the newly selected centroid
        new_dist2 = np.square(emb - emb[next_idx]).sum(axis=1, dtype=np.float32)
        dist2 = np.minimum(dist2, new_dist2)

    return selected
