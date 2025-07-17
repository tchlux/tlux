
import argparse
from pathlib import Path

import numpy as np

from ..fs import FileSystem
from .kmeans import kmeans
from .preview import select_random, select_diverse
from .worker_metadata_merger import _run_merge_metadata
from .jobs import spawn_job

# This script constructs a hierarchical k-means (HKM) index for a specific cluster.
# It works recursively to build the entire index tree by processing document embeddings.
#
# The main steps are:
# 1. Start a background job to combine metadata from all documents in this cluster.
# 2. Gather all document embeddings from files in the cluster using memory mapping.
# 3. Pick small random and diverse sets of embedding indices and save them for quick access.
# 4. Run k-means clustering on the embeddings to find cluster centers (centroids).
# 5. Launch jobs to group documents into clusters based on these centroids.
# 6. Trigger new index-building jobs for each sub-cluster, continuing the recursion.
#
# Notes:
# - The script assumes that the number of embeddings per cluster fits in memory.
# - For clusters with many embeddings, sampling could save memory but might affect results.
# - Recursion stops when an external condition (not in this script) deems clusters small enough.
# 
def build_cluster_index(index_root_directory: str):
    # Set up the file system and prepare directories for documents and the HKM index
    filesystem = FileSystem()
    documents_directory = filesystem.join(index_root_directory, "docs")
    hkm_directory = filesystem.join(index_root_directory, "hkm")
    filesystem.mkdir(hkm_directory, exist_ok=True)

    # Locate all embedding files (named embed_*.npy) in the documents directory
    embedding_file_paths = list(Path(documents_directory).rglob("embed_*.npy"))

    # Load embeddings from files using memory mapping to save memory
    embedding_arrays = [np.load(str(path), mmap_mode="r") for path in embedding_file_paths]

    # Combine all embeddings into one array; use an empty array if none exist
    if embedding_arrays:
        combined_embeddings = np.vstack(embedding_arrays)
    else:
        combined_embeddings = np.empty((0, 0), dtype=np.float32)

    # Set how many embeddings to sample for previews (max 512)
    preview_sample_size = min(combined_embeddings.shape[0], 512)

    # Choose indices for random and diverse preview samples, using a fixed seed for consistency
    random_preview_indices = select_random(range(combined_embeddings.shape[0]), preview_sample_size, seed=42)
    diverse_preview_indices = select_diverse(combined_embeddings, preview_sample_size, seed=42)

    # Save these indices to files for fast access later
    np.save(filesystem.join(hkm_directory, "preview_random.npy"), random_preview_indices)
    np.save(filesystem.join(hkm_directory, "preview_diverse.npy"), diverse_preview_indices)

    # Decide the maximum number of clusters (max 4096, or fewer if there are fewer embeddings)
    max_cluster_count = min(combined_embeddings.shape[0], 4096)

    # Run k-means to find cluster centers (centroids)
    cluster_centers, _ = kmeans(combined_embeddings, max_cluster_count, seed=42)

    # Save the cluster centers to a file
    np.save(filesystem.join(hkm_directory, "centroids.npy"), cluster_centers)

    # Start jobs to assign documents to their nearest cluster
    assignment_jobs = []
    for cluster_number in range(cluster_centers.shape[0]):
        job = spawn_job(
            "hkm.builder.worker_partitioner",  # Module that assigns documents to clusters
            documents_directory,
            hkm_directory,
            cluster_id=cluster_number,
            num_clusters=cluster_centers.shape[0],
        )
        assignment_jobs.append(job)

    # Launch recursive index-building jobs for each sub-cluster
    for job in assignment_jobs:
        cluster_number = job.args['cluster_id']
        subcluster_hkm_directory = filesystem.join(hkm_directory, f"cluster_{cluster_number:04d}")
        spawn_job(
            "hkm.builder.worker_index_builder",  # This script, run for the sub-cluster
            subcluster_hkm_directory,
            dependencies=[job],  # Wait for the assignment job to finish
        )

    # Start a job to merge metadata for this cluster
    # TODO: Could improve by having assignment jobs collect stats and merge them instead
    spawn_job(
        "hkm.builder.worker_metadata_merger",
        documents_directory,
        hkm_directory,
    )

if __name__ == "__main__":
    # Handle command-line arguments and start the index-building process
    parser = argparse.ArgumentParser(description="Build an HKM index tree at this cluster.")
    parser.add_argument("index_root_directory", help="Root directory for the index")
    arguments = parser.parse_args()
    build_cluster_index(arguments.index_root_directory)
