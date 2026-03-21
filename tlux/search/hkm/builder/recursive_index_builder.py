"""Recursive HKM tree construction through the shared job manager."""


import argparse
import json
import os
from pathlib import Path
from typing import List, Any, Optional

import numpy as np

from ..jobs import run_job
from ..tools.kmeans import kmeans
from .chunk_io import ChunkReader
from .sampler import sample_embeddings


MAX_CLUSTER_COUNT: int = 1024
RANDOM_SEED: int = 42


def _count_documents(docs_dir: str) -> int:
    total = 0
    for chunk_path in Path(docs_dir).rglob("*.hkmchunk"):
        reader = ChunkReader(str(chunk_path), metadata_schema=[])
        total += reader.document_count
    return total


# Builds the HKM index for the specified cluster.
# 
# Parameters
# ----------
# index_root_directory : str
#     The root directory containing the index structure.
# 
# Raises
# ------
# TypeError
#     If index_root_directory is not a string.
# ValueError
#     If the specified directory does not exist.
# 
# Notes
# -----
# This function uses memory-mapped NumPy arrays to efficiently handle large embedding files.
# 
def build_cluster_index(
    index_root_directory: str,
    max_cluster_count: int = MAX_CLUSTER_COUNT,
    leaf_doc_limit: int = 2,
    seed: int = RANDOM_SEED,
    fs_root: Optional[str] = None,
    max_depth: int = 3,
    depth: int = 0,
) -> None:
    # Input validation
    if not isinstance(index_root_directory, str):
        raise TypeError("index_root_directory must be a string")
    root_path: Path = Path(index_root_directory)
    if not root_path.exists():
        raise ValueError(f"Directory does not exist: {index_root_directory}")

    docs_dir: str = os.path.join(index_root_directory, "docs")
    data_dir_alt = os.path.join(index_root_directory, "data")
    if os.path.isdir(data_dir_alt):
        docs_dir = data_dir_alt
    if os.path.basename(index_root_directory).startswith("cluster_"):
        hkm_dir = index_root_directory
    else:
        hkm_dir = os.path.join(index_root_directory, "hkm")
    os.makedirs(hkm_dir, exist_ok=True)

    # Sample embeddings for clustering
    sample, rnd_idx, div_idx, preview_rnd, preview_div = sample_embeddings(
        docs_dir, target=256_000, seed=seed
    )
    if sample.size == 0:
        return
    document_count = _count_documents(docs_dir)

    np.save(os.path.join(hkm_dir, "sample.npy"), sample)
    if preview_rnd.size:
        np.save(os.path.join(hkm_dir, "preview_random.npy"), preview_rnd)
    if preview_div.size:
        np.save(os.path.join(hkm_dir, "preview_diverse.npy"), preview_div)

    cluster_limit: int = min(sample.shape[0], max_cluster_count)
    stats = {"doc_count": int(document_count), "leaf": False, "depth": depth}
    if cluster_limit <= 1 or document_count <= leaf_doc_limit or depth >= max_depth:
        stats["leaf"] = True
        with open(os.path.join(hkm_dir, "chunk_meta.json"), "w", encoding="ascii") as f_meta:
            json.dump({"leaf": True, "doc_count": int(document_count)}, f_meta, separators=(",", ":"))
        with open(os.path.join(hkm_dir, "stats.json"), "w", encoding="ascii") as f_stats:
            json.dump(stats, f_stats, separators=(",", ":"))
        return

    if cluster_limit >= 2 and np.allclose(sample, sample[0]):
        eps = 1e-3
        base = sample[0]
        jitter = np.stack([base + eps, base - eps], axis=0).astype(np.float32)
        cluster_centers = jitter[:cluster_limit]
    else:
        cluster_centers, _ = kmeans(sample, cluster_limit, seed=seed)
    np.save(os.path.join(hkm_dir, "centroids.npy"), cluster_centers)
    with open(os.path.join(hkm_dir, "stats.json"), "w", encoding="ascii") as f_stats:
        json.dump(stats, f_stats, separators=(",", ":"))

    # Spawn partitioning job for this level
    assignment_jobs: List[Any] = []
    job = run_job(
        "tlux.search.hkm.builder.partitioner.route_embeddings",
        docs_dir,
        hkm_dir,
        os.path.join(hkm_dir, "centroids.npy"),
        seed=seed,
        fs_root=fs_root,
    )
    assignment_jobs.append(job)

    for cluster_id in range(cluster_centers.shape[0]):
        sub_hkm_dir: str = os.path.join(hkm_dir, f"cluster_{cluster_id:04d}")
        run_job(
            "tlux.search.hkm.builder.recursive_index_builder.build_cluster_index",
            sub_hkm_dir,
            max_cluster_count,
            leaf_doc_limit,
            seed + cluster_id + 1,
            fs_root,
            max_depth,
            depth + 1,
            dependencies=assignment_jobs,
        )



if __name__ == "__main__":
    # Command-line interface for building an HKM index.
    parser = argparse.ArgumentParser(description="Build an HKM index tree for a cluster.")
    parser.add_argument(
        "index_root_directory",
        type=str,
        help="Root directory for the index structure"
    )
    args = parser.parse_args()
    build_cluster_index(args.index_root_directory)
