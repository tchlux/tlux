"""HKM tree traversal and preview streaming.

This implementation is **generic** – it walks any on-disk HKM directory
layout that matches the README specification *without* requiring an
in-memory tree.  It yields leaf node directories (those containing
``chunk_meta.npy``) that satisfy the *metadata upper-bound* filters in
a provided :class:`~hkm.schema.QuerySpec`.

The traversal strategy is breadth-first limited to *DESCEND_K* (8) best
children per query embedding, measured by centroid distance.  If a
node lacks centroids (e.g. root built by minimal demo), the algorithm
falls back to scanning all sub-directories.
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Generator, Iterable, List

import numpy as np

from ..schema import DESCEND_K, QuerySpec
from .filters import apply_filters
from .ranker import _pairwise_sq_l2  # internal helper for efficiency

__all__ = ["descend"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_centroids(node_dir: Path) -> np.ndarray | None:
    cpath = node_dir / "centroids.npy"
    if cpath.exists():
        return np.load(cpath, mmap_mode="r")
    return None


def _load_stats(node_dir: Path) -> dict | None:
    spath = node_dir / "stats.json"
    if spath.exists():
        with spath.open() as f:
            return json.load(f)
    return None


def _node_passes_upper_bound(stats: dict | None, query: QuerySpec) -> bool:
    """Return *True* if *stats* cannot rule out matching docs."""

    if stats is None:  # No stats – conservatively accept
        return True

    # --- Label pruning ---------------------------------------------------
    if query.label_include:
        ok = False
        # stats carries flattened keys like "lang:en" → count
        for key, allowed in query.label_include.items():
            for val in allowed:
                if f"{key}:{val}" in stats.get("labels_count", {}):
                    ok = True
                    break
            if not ok:
                return False

    # --- Numeric pruning -------------------------------------------------
    if query.numeric_range and stats.get("numeric_min") and stats.get("numeric_max"):
        mins = stats["numeric_min"]
        maxs = stats["numeric_max"]
        for i, (feat, bounds) in enumerate(query.numeric_range.items()):
            if i >= len(mins):
                return False  # query refers to feature beyond stored list
            if bounds[1] < mins[i] or bounds[0] > maxs[i]:
                return False  # no overlap

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def descend(
    tree_root: str | os.PathLike,
    query: QuerySpec | dict,
    *,
    fs=None,
) -> Generator[Path, None, None]:
    """Yield HKM *leaf* directories relevant to *query*.

    This routine requires only a file-system rooted at *tree_root*.  It
    performs on-the-fly pruning using per-node ``stats.json`` and selects
    up to :data:`~hkm.schema.DESCEND_K` closest children at each level.

    Parameters
    ----------
    tree_root:
        Path to the root node directory (string or ``Path``).
    query:
        Either a :class:`QuerySpec` or a mapping convertible to one.
    fs:
        Optional *FileSystem* instance; if ``None`` the local OS fs is
        used.  (Included for forward compatibility.)
    """

    if isinstance(query, dict):
        query = QuerySpec(**query)

    if fs is not None:
        join = fs.join  # type: ignore[attr-defined]
        listdir = fs.listdir  # type: ignore[attr-defined]
        exists = fs.exists  # type: ignore[attr-defined]
    else:  # fallback to os / pathlib
        join = os.path.join
        listdir = os.listdir
        exists = os.path.exists

    root = Path(tree_root)
    if not root.exists():
        return  # nothing to traverse

    # ------------------------------------------------------------------
    # BFS queue: (node_path, centroids_selected?)
    # ------------------------------------------------------------------
    q = deque([root])

    # Convert query.embeddings list into NumPy array once for reuse
    if query.embeddings:
        q_emb = np.asarray(query.embeddings, dtype=np.float32)
    else:
        q_emb = None

    while q:
        node = q.popleft()
        node_stats = _load_stats(node)
        if not _node_passes_upper_bound(node_stats, query):
            continue  # prune

        # Leaf test: presence of chunk_meta.npy marks a leaf
        if (node / "chunk_meta.npy").exists():
            yield node
            continue

        # Decide which children to enqueue --------------------------------
        children: List[Path] = [node / d for d in listdir(node) if (node / d).is_dir()]

        if not children:
            continue  # malformed tree, skip

        if q_emb is None:
            q.extend(children)  # no embedding filter, visit all children
            continue

        # If centroids present in parent, we can rank children quickly
        centroids = _load_centroids(node)
        if centroids is None or centroids.shape[0] != len(children):
            # Fallback: naive enqueue maintaining original order
            q.extend(children)
        else:
            # Compute distance of each child centroid to *all* query vectors
            dists = _pairwise_sq_l2(q_emb, centroids).min(axis=0)
            ranked = np.argsort(dists)[:DESCEND_K]
            for idx in ranked:
                q.append(children[int(idx)])
