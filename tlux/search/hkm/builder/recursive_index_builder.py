# Builds a hierarchical k-means (HKM) index for a specific cluster.
# 
# This script recursively constructs an HKM index tree by processing document embeddings.
# It performs the following steps:
#  1. Initiates a background job to merge metadata from all documents in the cluster.
#  2. Collects all document embeddings from files using memory mapping.
#  3. Selects small random and diverse sets of embedding indices for quick previews.
#  4. Executes k-means clustering to determine cluster centers (centroids).
#  5. Spawns jobs to partition documents into clusters based on these centroids.
#  6. Triggers new index-building jobs for each sub-cluster, continuing the recursion.
# 
# Notes:
# - Assumes the number of embeddings per cluster fits in memory.
# - For large clusters, sampling could be considered to reduce memory usage.
# - Recursion halts when an external condition deems clusters sufficiently small.
# 
# Example usage:
#     $ python launcher.py /path/to/index_root


import argparse
from pathlib import Path
from typing import List, Any

import numpy as np

from ..fs import FileSystem
from ..jobs import spawn_job
from ..tools.kmeans import kmeans
from ..tools.preview import select_random, select_diverse
from .metadata_merger import _run_merge_metadata


# Constants
MAX_PREVIEW_SIZE: int = 512
MAX_CLUSTER_COUNT: int = 1024
RANDOM_SEED: int = 42


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
def build_cluster_index(index_root_directory: str) -> None:
    # Input validation
    if not isinstance(index_root_directory, str):
        raise TypeError("index_root_directory must be a string")
    root_path: Path = Path(index_root_directory)
    if not root_path.exists():
        raise ValueError(f"Directory does not exist: {index_root_directory}")

    # Initialize filesystem and define directory paths
    filesystem: FileSystem = FileSystem()
    docs_dir: str = filesystem.join(index_root_directory, "docs")
    hkm_dir: str = filesystem.join(index_root_directory, "hkm")
    filesystem.mkdir(hkm_dir, exist_ok=True)

    # Collect embedding file paths
    embedding_paths: List[Path] = list(Path(docs_dir).glob("embed_root.npy"))
    if not embedding_paths:
        embedding_paths = list(Path(docs_dir).rglob("embed_*.npy"))
    if not embedding_paths:
        raise ValueError(f"No embedding files found in {docs_dir}")

    # Load embeddings with memory mapping for efficiency
    embedding_arrays: List[np.ndarray] = [
        np.load(str(path), mmap_mode="r") for path in embedding_paths
    ]
    combined_embeddings: np.ndarray = np.vstack(embedding_arrays) if embedding_arrays else np.empty((0, embedding_arrays[0].shape[1]), dtype=np.float32)

    # Calculate preview sample size based on data size
    preview_size: int = min(combined_embeddings.shape[0], 512)

    # Generate preview indices with deterministic randomness
    random_indices: np.ndarray = select_random(
        range(combined_embeddings.shape[0]), preview_size, seed=RANDOM_SEED
    )
    diverse_indices: np.ndarray = select_diverse(
        combined_embeddings, preview_size, seed=RANDOM_SEED
    )

    # Save preview indices to disk
    np.save(filesystem.join(hkm_dir, "preview_random.npy"), random_indices)
    np.save(filesystem.join(hkm_dir, "preview_diverse.npy"), diverse_indices)

    # Determine maximum cluster count
    cluster_limit: int = min(combined_embeddings.shape[0], MAX_CLUSTER_COUNT)
    if cluster_limit <= 1:
        return

    # Perform k-means clustering with seeded RNG
    cluster_centers: np.ndarray
    cluster_centers, _ = kmeans(combined_embeddings, cluster_limit, seed=RANDOM_SEED)
    np.save(filesystem.join(hkm_dir, "centroids.npy"), cluster_centers)

    # Spawn partitioning job for this level
    assignment_jobs: List[Any] = []
    job = spawn_job(
        "hkm.builder.partitioner.route_embeddings",
        docs_dir,
        hkm_dir,
        centroids_path=filesystem.join(hkm_dir, "centroids.npy"),
    )
    assignment_jobs.append(job)

    # Launch recursive index-building jobs for children
    for cluster_id in range(cluster_centers.shape[0]):
        sub_hkm_dir: str = filesystem.join(hkm_dir, f"cluster_{cluster_id:04d}")
        spawn_job(
            "hkm.builder.recursive_index_builder.build_cluster_index",
            sub_hkm_dir,
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
