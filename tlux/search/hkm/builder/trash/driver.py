
"""
Default driver orchestrator for HKM index construction.

This module spawns separate processes for:
  1. Tokenizing and embedding raw documents in parallel workers.
  2. Building the hierarchical K-Means (HKM) tree once all shards are ready.

Workers are launched via `jobs.spawn_job`, which invokes a Python function in
a new process with the provided arguments and optional dependencies.
"""

import os
import argparse

from ..fs import FileSystem
from .jobs import spawn_job


def build_hkm_tree(
    fs: FileSystem,
    docs_dir: str,
    index_root: str,
    num_workers: int,
):
    # -- Prepare index subdirectories ---------------------------------------
    docs_root_out = fs.join(index_root, "docs")
    fs.mkdir(docs_root_out, exist_ok=True)
    hkm_root = fs.join(index_root, "hkm")
    fs.mkdir(hkm_root, exist_ok=True)

    # -- Spawn tokenize+embed workers --------------------------------------
    worker_jobs = []
    for worker_id in range(num_workers):
        work_dir = fs.join(docs_root_out, f"worker_{worker_id:04d}")
        job = spawn_job(
            "hkm.builder.worker_tokenize_embed.main",
            fs.root,
            docs_dir,
            work_dir,
            worker_id=worker_id,
            num_workers=num_workers,
        )
        worker_jobs.append(job)

    # -- Once all shards are ready, kick off the HKM index build ------------
    spawn_job(
        "hkm.builder.worker_index_builder.main",
        fs.root,
        index_root,
        dependencies=worker_jobs,
    )


def main():
    """
    Entry point for the HKM driver.

    Parses command-line arguments, prepares directory structure, and
    dispatches worker and tree-building jobs.
    """
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
    build_hkm_tree(fs, args.docs_dir, args.index_root, args.workers)


if __name__ == "__main__":  # pragma: no cover
    main()
