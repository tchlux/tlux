"""Partition embeddings into child clusters.

Reads root embeddings and centroids, assigns each embedding to the nearest
centroid, and writes per-cluster shard arrays plus lightweight stats.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np

try:
    from ..fs import FileSystem
    from ..schema import LEAF_MAX_CHUNKS
    from .chunk_io import ChunkReader
except ImportError:  # pragma: no cover
    from tlux.search.hkm.fs import FileSystem
    from tlux.search.hkm.schema import LEAF_MAX_CHUNKS
    from tlux.search.hkm.builder.chunk_io import ChunkReader


def route_embeddings(docs_dir: str, hkm_dir: str, centroids_path: str) -> None:
    fs = FileSystem()
    centroids = np.load(centroids_path)
    cluster_count = centroids.shape[0]

    # Load embeddings from embed_root.npy
    embed_root = np.load(os.path.join(docs_dir, "embed_root.npy"))
    # Assign each embedding to nearest centroid (squared L2)
    dists = np.linalg.norm(embed_root[:, None, :] - centroids[None, :, :], axis=2)
    assign = np.argmin(dists, axis=1)

    # Split embeddings into per-cluster shards
    for cid in range(cluster_count):
        mask = assign == cid
        if not np.any(mask):
            continue
        cluster_dir = fs.join(hkm_dir, f"cluster_{cid:04d}")
        os.makedirs(cluster_dir, exist_ok=True)
        emb_c = embed_root[mask]
        np.save(os.path.join(cluster_dir, "embed_root.npy"), emb_c)
        # simple stats
        stats = {
            "doc_count": int(mask.sum()),
        }
        with open(os.path.join(cluster_dir, "stats.json"), "w", encoding="ascii") as f:
            json.dump(stats, f)
