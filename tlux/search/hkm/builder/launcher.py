# Default driver orchestrator for HKM index construction.
# 
# This module spawns separate processes for:
#   1. Tokenizing and embedding raw documents in parallel workers.
#   2. Building the hierarchical K-Means (HKM) tree once all shards are ready.
# 
# Workers are launched via `jobs.spawn_job`, which invokes a Python function in
# a new process with the provided arguments and optional dependencies.
# 
# Example usage:
#   python launcher.py /path/to/index /path/to/docs --workers 4


import os
import argparse

try:                from ..fs import FileSystem
except ImportError: from tlux.search.hkm.fs import FileSystem
try:                from ..jobs import spawn_job
except ImportError: from tlux.search.hkm.jobs import spawn_job
try:                from .consolidate import consolidate
except ImportError: from tlux.search.hkm.builder.consolidate import consolidate


# Function to build the search index by orchestrating worker processes.
#
# Prepares the directory structure, spawns worker processes to tokenize and
# embed documents, and then spawns a job to build the HKM tree once all
# workers complete.
#
# Parameters:
#   fs (FileSystem): The file system object for directory operations.
#   docs_dir (str): Path to the directory containing raw text documents.
#   index_root (str): Root directory where the HKM index will be created.
#   num_workers (int): Number of parallel worker processes.
#
# Returns:
#   None
#
# Raises:
#   ValueError: If docs_dir does not exist or num_workers is not a positive integer.
# 
def build_search_index(
    fs: FileSystem,
    docs_dir: str,
    index_root: str,
    num_workers: int,
    tokenizer_main: str = "hkm.builder.tokenize_and_embed.default_worker"
) -> None:
    # Validate input parameters
    if not fs.exists(docs_dir):
        raise ValueError(f"docs_dir '{docs_dir}' does not exist")
    if not isinstance(num_workers, int) or num_workers <= 0:
        raise ValueError("num_workers must be a positive integer")

    # Create subdirectories for document shards and HKM tree
    docs_root_out = fs.join(index_root, "docs")
    fs.mkdir(docs_root_out, exist_ok=True)
    hkm_root = fs.join(index_root, "hkm")
    fs.mkdir(hkm_root, exist_ok=True)

    # Launch worker processes to tokenize and embed documents
    worker_jobs = []
    for worker_id in range(num_workers):
        work_dir = fs.join(docs_root_out, f"worker_{worker_id:04d}")
        job = spawn_job(
            tokenizer_main,
            document_directory=docs_dir,
            output_directory=work_dir,
            worker_index=worker_id,
            total_workers=num_workers,
        )
        worker_jobs.append(job)

    # Consolidate worker chunks before HKM build
    consolidate_job = spawn_job(
        "hkm.builder.consolidate.consolidate",
        index_root,
        dependencies=worker_jobs,
    )
    # Launch the HKM tree builder once consolidation is complete
    spawn_job(
        "hkm.builder.recursive_index_builder.build_cluster_index",
        index_root,
        dependencies=[consolidate_job],
    )


# Entry point for the HKM driver
#
# Description:
#   Parses command-line arguments, initializes the file system, and dispatches
#   worker and tree-building jobs.
#
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrate tokenization, embedding, and HKM build"
    )
    parser.add_argument(
        "index_root",
        help="Root directory where the HKM index will be created",
    )
    parser.add_argument(
        "docs_dir",
        help="Path to the directory containing raw text documents",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel tokenization/embedding workers",
    )
    args = parser.parse_args()

    fs = FileSystem()
    build_search_index(fs, args.docs_dir, args.index_root, args.workers)


if __name__ == "__main__":  # pragma: no cover
    main()
