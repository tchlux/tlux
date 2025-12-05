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
import json
import argparse
from pathlib import Path
from typing import List

try:                from ..fs import FileSystem
except ImportError: from tlux.search.hkm.fs import FileSystem
try:                from ..jobs import spawn_job
except ImportError: from tlux.search.hkm.jobs import spawn_job


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
def _bin_pack(paths: List[Path], target_bins: int) -> List[List[Path]]:
    bins: List[List[Path]] = [[] for _ in range(target_bins)]
    bin_sizes = [0] * target_bins
    for p in sorted(paths, key=lambda x: x.stat().st_size, reverse=True):
        idx = min(range(target_bins), key=lambda i: bin_sizes[i])
        bins[idx].append(p)
        bin_sizes[idx] += p.stat().st_size
    return bins


def _should_skip(path: Path, skip_list: List[Path]) -> bool:
    for s in skip_list:
        try:
            if path == s or path.is_relative_to(s):
                return True
        except Exception:
            if str(path).startswith(str(s)):
                return True
    return False


def build_search_index(
    fs: FileSystem,
    docs_dir: str,
    index_root: str,
    num_workers: int,
    tokenizer_main: str = "tlux.search.hkm.builder.tokenize_and_embed.default_worker",
    max_k: int = 2,
    leaf_doc_limit: int = 2,
    seed: int = 42,
    fs_root: str | None = None,
    skip_paths: List[str] | None = None,
) -> None:
    # Validate input parameters
    if not os.path.exists(docs_dir):
        raise ValueError(f"docs_dir '{docs_dir}' does not exist")
    if not isinstance(num_workers, int) or num_workers <= 0:
        raise ValueError("num_workers must be a positive integer")
    if fs_root is None:
        fs_root = fs.root

    docs_dir_path = Path(docs_dir)
    skip_list = [Path(p).resolve() for p in (skip_paths or [])]
    all_files = [p for p in docs_dir_path.rglob("*") if p.is_file() and not _should_skip(p, skip_list)]
    if not all_files:
        raise ValueError("No documents found to index.")

    docs_root_out = os.path.join(index_root, "docs")
    os.makedirs(docs_root_out, exist_ok=True)
    hkm_root = os.path.join(index_root, "hkm")
    os.makedirs(hkm_root, exist_ok=True)

    # bin-pack files by size across workers
    bins = _bin_pack(all_files, num_workers)
    manifest_dir = os.path.join(index_root, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    worker_jobs = []
    for worker_id, files in enumerate(bins):
        if not files:
            continue
        manifest_path = Path(manifest_dir) / f"worker_{worker_id:04d}.json"
        manifest_path.write_text(json.dumps([str(p) for p in files]), encoding="utf-8")
        work_dir = fs.join(docs_root_out, f"worker_{worker_id:04d}")
        job = spawn_job(
            tokenizer_main,
            document_directory=str(docs_dir_path),
            output_directory=work_dir,
            worker_index=worker_id,
            total_workers=num_workers,
            manifest_path=str(manifest_path),
            fs_root=fs_root,
        )
        worker_jobs.append(job)

    consolidate_job = spawn_job(
        "tlux.search.hkm.builder.consolidate.run_consolidate",
        index_root,
        fs_root=fs_root,
        dependencies=worker_jobs,
    )
    spawn_job(
        "tlux.search.hkm.builder.recursive_index_builder.build_cluster_index",
        index_root,
        max_cluster_count=max_k,
        leaf_doc_limit=leaf_doc_limit,
        seed=seed,
        fs_root=fs_root,
        run_inline=True,
        max_depth=3,
        depth=0,
        dependencies=[consolidate_job],
    )


def build_search_index_inline(
    docs_dir: str,
    index_root: str,
    num_workers: int,
    tokenizer_main: str = "tlux.search.hkm.builder.tokenize_and_embed.default_worker",
    metadata_schema: str = "[['name','str'],['num_bytes','int']]",
    max_k: int = 2,
    leaf_doc_limit: int = 2,
    seed: int = 42,
    max_docs: int = 200,
    fs_root: str | None = None,
    skip_paths: List[str] | None = None,
) -> None:
    if fs_root is None:
        fs_root = index_root
    """Single-process helper for tests and small runs."""
    docs_dir_path = Path(docs_dir)
    skip_list = [Path(p).resolve() for p in (skip_paths or [])]
    all_files = [p for p in docs_dir_path.rglob("*") if p.is_file() and not _should_skip(p, skip_list)]
    all_files = sorted(all_files, key=lambda p: p.stat().st_size)[:max_docs]
    if not all_files:
        raise ValueError("No documents found to index.")

    docs_root_out = os.path.join(index_root, "docs")
    os.makedirs(docs_root_out, exist_ok=True)
    hkm_root = os.path.join(index_root, "hkm")
    os.makedirs(hkm_root, exist_ok=True)

    bins = _bin_pack(all_files, num_workers)
    manifest_dir = os.path.join(index_root, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    # Import tokenizer entry point dynamically
    module_path, func_name = tokenizer_main.rsplit(".", 1)
    tokenizer_mod = __import__(module_path, fromlist=[func_name])
    tokenizer_fn = getattr(tokenizer_mod, func_name)

    for worker_id, files in enumerate(bins):
        if not files:
            continue
        manifest_path = Path(manifest_dir) / f"worker_{worker_id:04d}.json"
        manifest_path.write_text(json.dumps([str(p) for p in files]), encoding="utf-8")
        work_dir = os.path.join(docs_root_out, f"worker_{worker_id:04d}")
        tokenizer_fn(
            document_directory=str(docs_dir_path),
            output_directory=work_dir,
            worker_index=worker_id,
            total_workers=num_workers,
            manifest_path=str(manifest_path),
            fs_root=fs_root,
            metadata_schema=metadata_schema,
        )

    # Consolidate
    spawn_job(
        "tlux.search.hkm.builder.consolidate.run_consolidate",
        index_root,
        fs_root=fs_root,
        inline=True,
    )

    # Build HKM tree inline
    spawn_job(
        "tlux.search.hkm.builder.recursive_index_builder.build_cluster_index",
        index_root,
        max_cluster_count=max_k,
        leaf_doc_limit=leaf_doc_limit,
        seed=seed,
        fs_root=fs_root,
        run_inline=True,
        max_depth=3,
        depth=0,
        inline=True,
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
    build_search_index(fs, args.docs_dir, args.index_root, args.workers, fs_root=fs.root)


if __name__ == "__main__":  # pragma: no cover
    main()
