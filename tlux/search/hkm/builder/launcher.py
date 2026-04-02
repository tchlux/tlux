"""Job-managed HKM build orchestration."""


import os
import json
import argparse
from pathlib import Path
from typing import List

from ..embedder import get_backend

try:
    from ..jobs import Job, run_job, set_jobs_root
except ImportError:
    from tlux.search.hkm.jobs import Job, run_job, set_jobs_root


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


#
# Enqueue the full HKM build pipeline on the shared job manager.
#
# Arguments:
#   docs_dir (str): Directory of input documents.
#   index_root (str): Root directory that will hold docs, hkm, and jobs.
#   num_workers (int): Number of worker shards to enqueue.
#
# Returns:
#   (Job): Root HKM build job.
#
def build_search_index(
    docs_dir: str,
    index_root: str,
    num_workers: int,
    tokenizer_main: str = "tlux.search.hkm.builder.tokenize_and_embed.default_worker",
    metadata_schema: str = "[['source_path','bytes'],['file_kind','str'],['num_bytes','float'],['tags','list'],['attrs','dict']]",
    max_k: int = 8,
    leaf_doc_limit: int = 1024,
    max_n_gram: int = 3,
    n_gram_fp_rate: float = 0.01,
    seed: int = 42,
    fs_root: str | None = None,
    jobs_root: str | None = None,
    skip_paths: List[str] | None = None,
) -> Job:
    # Validate input parameters
    if not os.path.exists(docs_dir):
        raise ValueError(f"docs_dir '{docs_dir}' does not exist")
    if not isinstance(num_workers, int) or num_workers <= 0:
        raise ValueError("num_workers must be a positive integer")
    if fs_root is None:
        try:
            fs_root = os.path.commonpath([os.path.abspath(docs_dir), os.path.abspath(index_root)])
        except Exception:
            fs_root = os.path.abspath(index_root)
        if fs_root in ("", os.sep):
            fs_root = os.path.abspath(index_root)
    if jobs_root is None:
        jobs_root = os.path.join(index_root, ".hkm_jobs")
    set_jobs_root(jobs_root)
    try:
        metadata_schema_value = json.loads(metadata_schema)
    except Exception:
        import ast
        metadata_schema_value = ast.literal_eval(metadata_schema)

    docs_dir_path = Path(docs_dir)
    skip_list = [Path(p).resolve() for p in (skip_paths or [])]
    all_files = [p for p in docs_dir_path.rglob("*") if p.is_file() and not _should_skip(p, skip_list)]
    if not all_files:
        raise ValueError("No documents found to index.")

    docs_root_out = os.path.join(index_root, "docs")
    os.makedirs(docs_root_out, exist_ok=True)
    hkm_root = os.path.join(index_root, "hkm")
    os.makedirs(hkm_root, exist_ok=True)
    Path(index_root, "index.json").write_text(json.dumps({
        "version": 1,
        "source_root": os.path.abspath(docs_dir),
        "jobs_root": os.path.abspath(jobs_root),
        "embedder_backend": get_backend().name,
        "metadata_schema": metadata_schema_value,
        "build_config": {
            "num_workers": num_workers,
            "max_cluster_count": max_k,
            "leaf_doc_limit": leaf_doc_limit,
            "max_n_gram": max_n_gram,
            "n_gram_fp_rate": n_gram_fp_rate,
            "seed": seed,
        },
        "max_n_gram": max_n_gram,
        "n_gram_fp_rate": n_gram_fp_rate,
        "docs_path": "docs",
        "hkm_path": "hkm",
    }, indent=2), encoding="utf-8")

    # bin-pack files by size across workers
    bins = _bin_pack(all_files, num_workers)
    manifest_dir = os.path.join(index_root, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    worker_jobs = []
    doc_id_base = 0
    for worker_id, files in enumerate(bins):
        if not files:
            continue
        manifest_path = Path(manifest_dir) / f"worker_{worker_id:04d}.json"
        manifest_path.write_text(json.dumps([str(p) for p in files]), encoding="utf-8")
        work_dir = os.path.join(docs_root_out, f"worker_{worker_id:04d}")
        job = run_job(
            tokenizer_main,
            document_directory=str(docs_dir_path),
            output_directory=work_dir,
            worker_index=worker_id,
            total_workers=num_workers,
            manifest_path=str(manifest_path),
            fs_root=fs_root,
            metadata_schema=metadata_schema,
            n_gram=max_n_gram,
            doc_id_base=doc_id_base,
        )
        worker_jobs.append(job)
        doc_id_base += len(files)

    consolidate_job = run_job(
        "tlux.search.hkm.builder.consolidate.run_consolidate",
        index_root,
        fs_root=fs_root,
        dependencies=worker_jobs,
    )
    return run_job(
        "tlux.search.hkm.builder.recursive_index_builder.build_cluster_index",
        index_root,
        max_cluster_count=max_k,
        leaf_doc_limit=leaf_doc_limit,
        max_n_gram=max_n_gram,
        n_gram_fp_rate=n_gram_fp_rate,
        seed=seed,
        fs_root=fs_root,
        max_depth=3,
        depth=0,
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

    build_search_index(args.docs_dir, args.index_root, args.workers)


if __name__ == "__main__":  # pragma: no cover
    main()
