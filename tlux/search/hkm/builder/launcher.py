"""Job-managed HKM build orchestration."""

from __future__ import annotations

import os
import json
import argparse
import fnmatch
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from ..embedder import get_backend
from ..schema import DEFAULT_METADATA_SCHEMA

try:
    from ..jobs import Job, run_job, set_jobs_root
except ImportError:
    from tlux.search.hkm.jobs import Job, run_job, set_jobs_root


DEFAULT_SKIP_PARTS = {
    ".git",
    ".env",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    "build",
    "dist",
    ".hkm_jobs",
}
DEFAULT_SKIP_NAMES = [
    "tmp_index*",
    "tmp_hkm*",
    "tokenizer*.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]
DEFAULT_SKIP_SUFFIXES = {
    ".pyc",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".npy",
    ".npz",
    ".zip",
    ".tar",
    ".gz",
    ".xz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".sqlite",
    ".db",
    ".safetensors",
    ".pt",
    ".pth",
    ".onnx",
    ".gguf",
}
DEFAULT_METADATA_SCHEMA_TEXT = json.dumps(DEFAULT_METADATA_SCHEMA)


# Return the current UTC timestamp as a compact ISO string.
#
# Arguments:
#   None.
#
# Returns:
#   (str): Timestamp ending in Z.
#
def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# Resolve the source metadata manifest used for document enrichment.
#
# Arguments:
#   docs_dir (Path): Source document directory.
#   source_manifest (str | None): Explicit manifest path.
#
# Returns:
#   (str | None): Manifest path when available.
#
def _source_manifest_path(docs_dir: Path, source_manifest: str | None) -> str | None:
    if source_manifest:
        return str(Path(source_manifest).resolve())
    for candidate in (docs_dir / "manifest.jsonl", docs_dir.parent / "manifest.jsonl"):
        if candidate.exists():
            return str(candidate.resolve())
    return None


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


# Match relative path globs while accepting basename-only patterns.
#
# Arguments:
#   rel_path (str): POSIX-style path relative to the document root.
#   patterns (List[str]): Glob patterns.
#
# Returns:
#   (bool): True when any pattern matches.
#
def _glob_match(rel_path: str, patterns: List[str]) -> bool:
    return any(
        fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(Path(rel_path).name, pattern)
        for pattern in patterns
    )


# Return the built-in skip reason for generated, model, cache, or binary paths.
#
# Arguments:
#   path (Path): Absolute or source-root-relative file path.
#   rel_path (str): POSIX-style path relative to the document root.
#
# Returns:
#   (str | None): Skip reason, or None if defaults allow the path.
#
def _default_skip_reason(path: Path, rel_path: str) -> str | None:
    if any(part in DEFAULT_SKIP_PARTS or part.endswith(".hkmchunk") for part in Path(rel_path).parts):
        return "default_path"
    if _glob_match(rel_path, DEFAULT_SKIP_NAMES):
        return "default_name"
    if path.suffix.lower() in DEFAULT_SKIP_SUFFIXES:
        return "default_suffix"
    return None


# Select files for indexing and collect pre-worker skip records.
#
# Arguments:
#   docs_dir (Path): Source document directory.
#   skip_paths (List[str] | None): Explicit paths to skip.
#   include_globs (List[str] | None): Relative include globs.
#   exclude_globs (List[str] | None): Relative exclude globs.
#   default_skips (bool): Apply built-in skip rules.
#   max_file_bytes (int | None): Maximum source bytes per file.
#
# Returns:
#   (tuple[List[Path], List[Dict[str, object]], Dict[str, int]]): Planned files,
#   skipped file records, and skip counts by reason.
#
def _ingest_plan(
    docs_dir: Path,
    skip_paths: List[str] | None,
    include_globs: List[str] | None,
    exclude_globs: List[str] | None,
    default_skips: bool,
    max_file_bytes: int | None,
) -> tuple[List[Path], List[Dict[str, object]], Dict[str, int]]:
    skip_list = [Path(p).resolve() for p in (skip_paths or [])]
    includes = include_globs or []
    excludes = exclude_globs or []
    files: List[Path] = []
    skipped: List[Dict[str, object]] = []
    reasons: Dict[str, int] = {}

    for path in sorted(p for p in docs_dir.rglob("*") if p.is_file()):
        rel_path = path.relative_to(docs_dir).as_posix()
        reason = None
        if _should_skip(path.resolve(), skip_list):
            reason = "skip_path"
        elif includes and not _glob_match(rel_path, includes):
            reason = "include_glob"
        elif excludes and _glob_match(rel_path, excludes):
            reason = "exclude_glob"
        elif default_skips:
            reason = _default_skip_reason(path, rel_path)
        if reason is None and max_file_bytes is not None and path.stat().st_size > max_file_bytes:
            reason = "max_file_bytes"
        if reason is None:
            files.append(path)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
            skipped.append({"path": str(path), "reason": reason})
    return files, skipped, reasons


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
    metadata_schema: str = DEFAULT_METADATA_SCHEMA_TEXT,
    max_k: int = 8,
    leaf_embedding_limit: int = 1024,
    leaf_doc_limit: int = 1024,
    max_n_gram: int = 3,
    n_gram_fp_rate: float = 0.01,
    seed: int = 42,
    fs_root: str | None = None,
    jobs_root: str | None = None,
    skip_paths: List[str] | None = None,
    include_globs: List[str] | None = None,
    exclude_globs: List[str] | None = None,
    default_skips: bool = True,
    max_file_bytes: int | None = 8 * 2**20,
    max_tokens: int | None = 200_000,
    source_manifest: str | None = None,
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
    source_manifest_path = _source_manifest_path(docs_dir_path, source_manifest)
    build_id = _utc_now()
    all_files, skipped_files, skip_reasons = _ingest_plan(
        docs_dir_path,
        skip_paths,
        include_globs,
        exclude_globs,
        default_skips,
        max_file_bytes,
    )
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
        "source_manifest": source_manifest_path,
        "build_config": {
            "num_workers": num_workers,
            "max_cluster_count": max_k,
            "leaf_embedding_limit": leaf_embedding_limit,
            "leaf_doc_limit": leaf_doc_limit,
            "max_n_gram": max_n_gram,
            "n_gram_fp_rate": n_gram_fp_rate,
            "seed": seed,
            "build_id": build_id,
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
    Path(manifest_dir, "ingest_summary.json").write_text(json.dumps({
        "scanned": len(all_files) + len(skipped_files),
        "planned": len(all_files),
        "indexed": 0,
        "skipped": len(skipped_files),
        "failed": 0,
        "skip_reasons": skip_reasons,
        "skipped_files": skipped_files,
        "failed_files": [],
    }, indent=2), encoding="utf-8")

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
            max_tokens=max_tokens,
            source_manifest=source_manifest_path,
            build_id=build_id,
            ingested_at=build_id,
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
        leaf_embedding_limit=leaf_embedding_limit,
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
    parser.add_argument("--max-k", type=int, default=8, help="Max clusters per level")
    parser.add_argument("--leaf-embedding-limit", type=int, default=1024, help="Embeddings per leaf")
    parser.add_argument("--leaf-doc-limit", type=int, default=1024, help="Docs per leaf")
    parser.add_argument("--skip", action="append", default=[], help="Path to exclude; repeatable")
    parser.add_argument("--include", action="append", default=[], help="Relative glob to include; repeatable")
    parser.add_argument("--exclude", action="append", default=[], help="Relative glob to exclude; repeatable")
    parser.add_argument("--max-file-bytes", type=int, default=8 * 2**20, help="Maximum source file bytes")
    parser.add_argument("--max-tokens", type=int, default=200_000, help="Maximum tokens per document")
    parser.add_argument("--source-manifest", default=None, help="Optional JSONL source metadata manifest")
    parser.add_argument("--no-default-skips", action="store_true", help="Disable built-in cache/model/binary skips")
    args = parser.parse_args()

    print(build_search_index(
        docs_dir=args.docs_dir,
        index_root=args.index_root,
        num_workers=args.workers,
        max_k=args.max_k,
        leaf_embedding_limit=args.leaf_embedding_limit,
        leaf_doc_limit=args.leaf_doc_limit,
        skip_paths=args.skip,
        include_globs=args.include,
        exclude_globs=args.exclude,
        default_skips=not args.no_default_skips,
        max_file_bytes=args.max_file_bytes,
        max_tokens=args.max_tokens,
        source_manifest=args.source_manifest,
    ).id)


if __name__ == "__main__":  # pragma: no cover
    main()
