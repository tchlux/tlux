"""Job-managed HKM build orchestration."""

from __future__ import annotations

import os
import json
import argparse
import fnmatch
import hashlib
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ..embedder import get_backend
from ..fs import FileSystem
from ..schema import DEFAULT_METADATA_SCHEMA
from .consolidate import consolidate
from .tokenize_and_embed import DocumentValue, process_documents, split_text_passages

try:
    from ..jobs import Job, drain_jobs, run_job, set_jobs_root
except ImportError:
    from tlux.search.hkm.fs import FileSystem
    from tlux.search.hkm.jobs import Job, drain_jobs, run_job, set_jobs_root


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
    ".hkm_cache",
    ".hkm_builds",
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


# Return the currently published generation for a public index root.
#
# Arguments:
#   public_root (Path): User-facing index path.
#
# Returns:
#   (Path): Active generation directory, or the legacy root.
#
def _active_generation_root(public_root: Path) -> Path:
    manifest_path = public_root / "index.json"
    if not manifest_path.exists():
        return public_root
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = str(data.get("generation_path", "") or "")
    if not relative:
        return public_root
    generation = (public_root / relative).resolve()
    if not generation.is_relative_to(public_root.resolve()):
        raise ValueError(f"generation_path escapes index root: {relative}")
    return generation


# Create a unique staging generation beneath the public root.
#
# Arguments:
#   public_root (Path): User-facing index path.
#   build_id (str): Build identifier.
#
# Returns:
#   (Path): Empty staging generation directory.
#
def _new_generation_root(public_root: Path, build_id: str) -> Path:
    name = build_id.replace(":", "").replace("-", "") + "-" + uuid.uuid4().hex[:8]
    generation = public_root / ".hkm_builds" / name
    generation.mkdir(parents=True, exist_ok=False)
    return generation


# Clone an active generation before an append-only update.
#
# Arguments:
#   source (Path): Existing active generation.
#   destination (Path): New staging generation.
#
# Returns:
#   (tuple[int, float]): Copied bytes and elapsed seconds.
#
def _clone_generation(source: Path, destination: Path) -> tuple[int, float]:
    started = time.perf_counter()
    copied_bytes = sum(
        path.stat().st_size
        for path in source.rglob("*")
        if path.is_file() and path.name != "index.json"
    )
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("index.json", ".hkm_builds", ".hkm_jobs", ".hkm_cache"),
    )
    return copied_bytes, time.perf_counter() - started


# Replace one public compatibility alias with a generation symlink.
#
# Arguments:
#   public_root (Path): User-facing index path.
#   generation (Path): New generation directory.
#   name (str): Alias name such as docs, hkm, or manifests.
#
# Returns:
#   (bool): True when the alias was replaced.
#
def _replace_generation_alias(public_root: Path, generation: Path, name: str) -> bool:
    alias = public_root / name
    if alias.exists() and not alias.is_symlink():
        generated = {
            "docs": (alias / "doc_index.npy").exists() or any(alias.rglob("*.hkmchunk")),
            "hkm": (alias / "node.json").exists(),
            "manifests": (alias / "ingest_summary.json").exists() or any(alias.glob("worker_*.json")),
        }[name]
        if not generated:
            return False
        backup = public_root / ".hkm_builds" / ".legacy" / uuid.uuid4().hex[:8] / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        alias.rename(backup)
    elif alias.is_symlink():
        alias.unlink()
    temporary = public_root / f".{name}.next-{uuid.uuid4().hex[:8]}"
    temporary.symlink_to(os.path.relpath(generation / name, public_root), target_is_directory=True)
    os.replace(temporary, alias)
    return True


# Atomically publish an audited generation while retaining a stable public root.
#
# Arguments:
#   generation_root (Path): Audited immutable build directory.
#   public_root (Path): User-facing index path.
#
# Returns:
#   (None): Replaces the public manifest in one filesystem operation.
#
def publish_generation(generation_root: str, public_root: str) -> None:
    generation = Path(generation_root).resolve()
    public = Path(public_root).resolve()
    public.parent.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    previous = _active_generation_root(public)
    from ..search.searcher import audit_index

    audit_index(str(generation))
    stage = json.loads((generation / "index.json").read_text(encoding="utf-8"))
    stage["generation_path"] = Path(os.path.relpath(generation, public)).as_posix()
    stage["jobs_root"] = stage.get("jobs_root", str(public / ".hkm_jobs"))
    temporary = public / f".index.next-{uuid.uuid4().hex[:8]}"
    temporary.write_text(json.dumps(stage, indent=2), encoding="utf-8")
    os.replace(temporary, public / "index.json")
    for name in ("docs", "hkm", "manifests"):
        _replace_generation_alias(public, generation, name)
    builds_root = (public / ".hkm_builds").resolve()
    if previous != public and previous != generation and previous.is_relative_to(builds_root):
        shutil.rmtree(previous, ignore_errors=True)


# Parse a serialized or Python metadata schema into JSON and runtime forms.
#
# Arguments:
#   metadata_schema (str | list): Schema as JSON text or Python list.
#
# Returns:
#   (tuple[list, list[tuple[str, type]]]): JSON-safe schema and parsed schema.
#
def _parse_metadata_schema_value(metadata_schema: str | List[List[str]] | List[Tuple[str, type]]) -> tuple[list, list[tuple[str, type]]]:
    if isinstance(metadata_schema, str):
        try:
            schema_value = json.loads(metadata_schema)
        except Exception:
            import ast
            schema_value = ast.literal_eval(metadata_schema)
    else:
        schema_value = [
            [name, typ if isinstance(typ, str) else typ.__name__]
            for name, typ in metadata_schema
        ]
    type_map = {"str": str, "float": float, "int": int, "json": dict, "bytes": bytes, "list": list, "dict": dict}
    return schema_value, [(name, type_map.get(kind, str)) for name, kind in schema_value]


# Coerce user metadata into the storage type declared by the schema.
#
# Arguments:
#   field_type (type): Runtime metadata field type.
#   value (object): User metadata value.
#
# Returns:
#   (DocumentValue): Value accepted by process_documents().
#
def _coerce_document_metadata(field_type: type, value: object) -> DocumentValue:
    if value is None:
        return None
    if field_type is bytes:
        return value if isinstance(value, bytes) else str(value).encode("utf-8")
    if field_type is int:
        return int(value)
    if field_type is float:
        return float(value)
    return value


# Convert user document dictionaries into process_documents batches.
#
# Arguments:
#   documents (Iterable[dict]): Records with text and optional metadata.
#   metadata_schema (list[tuple[str, type]]): Parsed metadata schema.
#   build_id (str): Current build identifier.
#
# Returns:
#   (Iterable[tuple[list[str], list[list[DocumentValue]]]]): Single-document batches.
#
def _document_batches(
    documents: Iterable[Dict[str, Any]],
    metadata_schema: List[Tuple[str, type]],
    build_id: str,
) -> Iterable[Tuple[List[str], List[List[DocumentValue]]]]:
    for idx, record in enumerate(documents):
        if not isinstance(record, dict) or "text" not in record:
            raise ValueError("documents must yield dictionaries with a text field")
        text = str(record["text"])
        metadata = dict(record.get("metadata", {}) or {})
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_path = str(metadata.get("source_path", f"document_{idx:08d}"))
        defaults = {
            "source_path": source_path,
            "source_type": "iterator",
            "file_kind": str(metadata.get("file_kind", "none")),
            "title": str(metadata.get("title", source_path)),
            "section_path": str(metadata.get("section_path", "")),
            "byte_start": int(metadata.get("byte_start", 0) or 0),
            "byte_end": int(metadata.get("byte_end", len(text.encode("utf-8"))) or 0),
            "token_start": int(metadata.get("token_start", 0) or 0),
            "token_end": int(metadata.get("token_end", 0) or 0),
            "content_hash": content_hash,
            "build_id": build_id,
            "ingested_at": build_id,
            "source_id": str(metadata.get("source_id", "")),
            "source_url": str(metadata.get("source_url", "")),
            "source_date": str(metadata.get("source_date", "")),
            "source_token_count": int(metadata.get("source_token_count", 0) or 0),
            "num_bytes": len(text.encode("utf-8")),
            "document_preview": text[:512],
        }
        row = [
            _coerce_document_metadata(field_type, metadata.get(field_name, defaults.get(field_name)))
            for field_name, field_type in metadata_schema
        ]
        yield [text], [row]


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


# Return the SHA256 hash of a file.
#
# Arguments:
#   path (Path): File to hash.
#
# Returns:
#   (str): Hex digest.
#
def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Estimate how many passage records a source file will publish.
#
# Arguments:
#   path (Path): UTF-8 source file.
#
# Returns:
#   (int): Passage count, or one record for unreadable files.
#
def _passage_count(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return 1
    return max(1, len(split_text_passages(text, path.suffix.lower())))


# Load the current source snapshot, if it is compatible enough for reuse.
#
# Arguments:
#   index_root (str): Index root.
#   docs_dir (str): Source root for the requested build.
#   metadata_schema_value (list): Parsed metadata schema.
#   backend_name (str): Active embedder backend.
#
# Returns:
#   (dict | None): Snapshot payload when compatible.
#
def _load_snapshot(
    index_root: str,
    docs_dir: str,
    metadata_schema_value: list,
    backend_name: str,
) -> dict | None:
    path = Path(index_root) / "manifests" / "source_snapshot.json"
    if not path.exists():
        return None
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if snapshot.get("source_root") != os.path.abspath(docs_dir):
        return None
    if snapshot.get("embedder_backend") != backend_name:
        return None
    if snapshot.get("metadata_schema") != metadata_schema_value:
        return None
    if not (Path(index_root) / "docs" / "doc_index.npy").exists():
        return None
    if not (Path(index_root) / "hkm" / "node.json").exists():
        return None
    return snapshot


# Split planned files into reused and work-needed groups.
#
# Arguments:
#   docs_dir (Path): Source document root.
#   files (list[Path]): Planned source files.
#   snapshot (dict): Existing source snapshot.
#
# Returns:
#   (tuple[list[dict], list[Path], int, int]): Reused docs, changed/new files,
#   changed count, and deleted count.
#
def _classify_incremental(
    docs_dir: Path,
    files: List[Path],
    snapshot: dict,
) -> tuple[List[dict], List[Path], int, int]:
    old: Dict[str, List[dict]] = {}
    for item in snapshot.get("documents", []):
        old.setdefault(item["source_path"], []).append(item)
    current_paths = set()
    reused: List[dict] = []
    work: List[Path] = []
    changed = 0
    for path in files:
        rel_path = path.relative_to(docs_dir).as_posix()
        current_paths.add(rel_path)
        digest = _file_hash(path)
        previous = old.get(rel_path, [])
        if previous and all(item.get("content_hash") == digest for item in previous):
            reused.extend(previous)
        else:
            changed += 1 if previous else 0
            work.append(path)
    deleted = len(set(old) - current_paths)
    return reused, work, changed, deleted


# Return the next available canonical worker id.
#
# Arguments:
#   index_root (str): Index root.
#
# Returns:
#   (int): Worker id.
#
def _next_worker_id(index_root: str) -> int:
    workers = sorted((Path(index_root) / "docs").glob("worker_*"))
    if not workers:
        return 0
    return max(int(path.name.split("_")[-1]) for path in workers) + 1


# Return the largest active doc id.
#
# Arguments:
#   snapshot (dict | None): Existing source snapshot.
#
# Returns:
#   (int): Maximum doc id or 0.
#
def _max_doc_id(snapshot: dict | None) -> int:
    if snapshot is None:
        return 0
    return max([int(item.get("doc_id", 0)) for item in snapshot.get("documents", [])] or [0])


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
    incremental: bool = True,
) -> Job:
    docs_dir_path = Path(docs_dir).resolve()
    public_index_root = Path(index_root).expanduser().absolute()
    public_index_root.mkdir(parents=True, exist_ok=True)
    active_index_root = _active_generation_root(public_index_root)
    docs_dir = str(docs_dir_path)
    index_root = str(active_index_root)

    # Validate input parameters.
    if not docs_dir_path.exists():
        raise ValueError(f"docs_dir '{docs_dir}' does not exist")
    if not isinstance(num_workers, int) or num_workers <= 0:
        raise ValueError("num_workers must be a positive integer")
    if max_file_bytes is not None and (not isinstance(max_file_bytes, int) or max_file_bytes <= 0):
        raise ValueError("max_file_bytes must be positive or None")
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
        raise ValueError("max_tokens must be positive or None")
    if fs_root is None:
        try:
            fs_root = os.path.commonpath([docs_dir, str(public_index_root)])
        except Exception:
            fs_root = str(public_index_root)
        if fs_root in ("", os.sep):
            fs_root = str(public_index_root)
    else:
        fs_root = str(Path(fs_root).resolve())
    if jobs_root is None:
        jobs_root = str(public_index_root / ".hkm_jobs")
    else:
        jobs_root = str(Path(jobs_root).resolve())
    try:
        metadata_schema_value = json.loads(metadata_schema)
    except Exception:
        import ast
        metadata_schema_value = ast.literal_eval(metadata_schema)

    source_manifest_path = _source_manifest_path(docs_dir_path, source_manifest)
    build_id = _utc_now()
    backend_name = get_backend().name
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

    snapshot = _load_snapshot(str(active_index_root), docs_dir, metadata_schema_value, backend_name) if incremental else None
    reused_docs: List[dict] = []
    work_files = all_files
    changed_count = 0
    deleted_count = 0
    if snapshot is not None:
        reused_docs, work_files, changed_count, deleted_count = _classify_incremental(docs_dir_path, all_files, snapshot)
        new_count = len(work_files) - changed_count
        summary = {
            "scanned": len(all_files) + len(skipped_files),
            "planned": len(all_files),
            "indexed": len(reused_docs),
            "skipped": len(skipped_files),
            "failed": 0,
            "reused": len(reused_docs),
            "new": new_count,
            "changed": changed_count,
            "deleted": deleted_count,
            "cache_hits": len(reused_docs),
            "cache_misses": 0,
            "skip_reasons": skip_reasons,
            "skipped_files": skipped_files,
            "failed_files": [],
        }
        active_summary = active_index_root / "manifests" / "ingest_summary.json"
        active_summary.parent.mkdir(parents=True, exist_ok=True)
        active_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if not work_files and not deleted_count:
            return run_job("tlux.search.hkm.builder.incremental.noop")

    generation_root = _new_generation_root(public_index_root, build_id)
    staging_copy_bytes = 0
    staging_copy_seconds = 0.0
    if snapshot is not None:
        staging_copy_bytes, staging_copy_seconds = _clone_generation(active_index_root, generation_root)
    index_root_path = generation_root
    index_root = str(generation_root.resolve())
    manifest_dir = os.path.join(index_root, "manifests")
    cache_dir = os.path.join(str(public_index_root), ".hkm_cache", "embeddings")
    if snapshot is None:
        for name in ("docs", "hkm", "manifests"):
            path = generation_root / name
            if path.exists():
                shutil.rmtree(path)
        for name in ("docs", "hkm", "manifests"):
            (generation_root / name).mkdir(parents=True, exist_ok=True)
        for name in ("docs", "hkm", "manifests"):
            _replace_generation_alias(public_index_root, generation_root, name)
    else:
        (generation_root / "docs").mkdir(parents=True, exist_ok=True)
        (generation_root / "hkm").mkdir(parents=True, exist_ok=True)
        (generation_root / "manifests").mkdir(parents=True, exist_ok=True)
    set_jobs_root(jobs_root)

    docs_root_out = os.path.join(index_root, "docs")
    hkm_root = os.path.join(index_root, "hkm")
    os.makedirs(docs_root_out, exist_ok=True)
    os.makedirs(hkm_root, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    if snapshot is not None:
        new_count = len(work_files) - changed_count
        Path(manifest_dir, "ingest_summary.json").write_text(json.dumps({
            "scanned": len(all_files) + len(skipped_files),
            "planned": len(all_files),
            "indexed": len(reused_docs),
            "skipped": len(skipped_files),
            "failed": 0,
            "reused": len(reused_docs),
            "new": new_count,
            "changed": changed_count,
            "deleted": deleted_count,
            "cache_hits": len(reused_docs),
            "cache_misses": 0,
            "skip_reasons": skip_reasons,
            "skipped_files": skipped_files,
            "failed_files": [],
        }, indent=2), encoding="utf-8")
    Path(index_root, "index.json").write_text(json.dumps({
        "version": 1,
        "source_root": docs_dir,
        "jobs_root": jobs_root,
        "embedder_backend": backend_name,
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
            "staging_copy_bytes": staging_copy_bytes,
            "staging_copy_seconds": staging_copy_seconds,
        },
        "max_n_gram": max_n_gram,
        "n_gram_fp_rate": n_gram_fp_rate,
        "docs_path": "docs",
        "hkm_path": "hkm",
        "append_only": snapshot is not None,
    }, indent=2), encoding="utf-8")

    # bin-pack files by size across workers
    bins = _bin_pack(work_files, num_workers)
    if snapshot is None:
        Path(manifest_dir, "ingest_summary.json").write_text(json.dumps({
            "scanned": len(all_files) + len(skipped_files),
            "planned": len(all_files),
            "indexed": 0,
            "skipped": len(skipped_files),
            "failed": 0,
            "reused": 0,
            "new": len(all_files),
            "changed": 0,
            "deleted": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "skip_reasons": skip_reasons,
            "skipped_files": skipped_files,
            "failed_files": [],
        }, indent=2), encoding="utf-8")

    worker_jobs = []
    doc_id_base = 0 if snapshot is None else _max_doc_id(snapshot)
    worker_id_base = 0 if snapshot is None else _next_worker_id(index_root)
    worker_ids = []
    for worker_id, files in enumerate(bins):
        if not files:
            continue
        actual_worker_id = worker_id_base + worker_id
        worker_ids.append(actual_worker_id)
        manifest_path = Path(manifest_dir) / f"worker_{worker_id:04d}.json"
        manifest_path.write_text(json.dumps([str(p) for p in files]), encoding="utf-8")
        work_dir = os.path.join(docs_root_out, f"worker_{actual_worker_id:04d}")
        job = run_job(
            tokenizer_main,
            document_directory=str(docs_dir_path),
            output_directory=work_dir,
            worker_index=actual_worker_id,
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
            embedding_cache_dir=cache_dir,
        )
        worker_jobs.append(job)
        doc_id_base += sum(_passage_count(path) for path in files)

    if snapshot is not None:
        plan_path = Path(manifest_dir) / "incremental_plan.json"
        plan_path.write_text(json.dumps({
            "build_id": build_id,
            "reused_documents": reused_docs,
            "worker_ids": worker_ids,
        }, indent=2), encoding="utf-8")
        return run_job(
            "tlux.search.hkm.builder.incremental.finalize_incremental",
            index_root,
            str(plan_path),
            publish_root=str(public_index_root),
            max_cluster_count=max_k,
            leaf_embedding_limit=leaf_embedding_limit,
            leaf_doc_limit=leaf_doc_limit,
            max_n_gram=max_n_gram,
            n_gram_fp_rate=n_gram_fp_rate,
            seed=seed,
            dependencies=worker_jobs,
        )

    consolidate_job = run_job(
        "tlux.search.hkm.builder.consolidate.run_consolidate",
        index_root,
        fs_root=fs_root,
        dependencies=worker_jobs,
    )
    build_job = run_job(
        "tlux.search.hkm.builder.recursive_index_builder.build_cluster_index",
        index_root,
        max_cluster_count=max_k,
        leaf_embedding_limit=leaf_embedding_limit,
        leaf_doc_limit=leaf_doc_limit,
        max_n_gram=max_n_gram,
        n_gram_fp_rate=n_gram_fp_rate,
        seed=seed,
        fs_root=fs_root,
        max_depth=0,
        depth=0,
        dependencies=[consolidate_job],
    )
    return run_job(
        "tlux.search.hkm.builder.incremental.write_source_snapshot",
        index_root,
        publish_root=str(public_index_root),
        dependencies=[build_job],
    )


# Build a queryable HKM index from Python document dictionaries.
#
# Arguments:
#   index_root (str): Root directory where the HKM index will be created.
#   documents (Iterable[dict]): Records with text and optional metadata.
#   metadata_schema (str | list): Metadata schema for stored fields.
#   num_workers (int): Local workers used for HKM tree jobs.
#
# Returns:
#   (Job): Final HKM tree build job, already drained to completion.
#
def build_search_index_from_documents(
    index_root: str,
    documents: Iterable[Dict[str, Any]],
    metadata_schema: str | List[List[str]] | List[Tuple[str, type]] = DEFAULT_METADATA_SCHEMA_TEXT,
    num_workers: int = 1,
    max_k: int = 8,
    leaf_embedding_limit: int = 1024,
    leaf_doc_limit: int = 1024,
    max_n_gram: int = 3,
    n_gram_fp_rate: float = 0.01,
    seed: int = 42,
    fs_root: str | None = None,
    jobs_root: str | None = None,
    chunk_size_limit: int = 8 * 2**20,
    max_tokens: int | None = 200_000,
) -> Job:
    public_index_root = Path(index_root).expanduser().absolute()
    public_index_root.mkdir(parents=True, exist_ok=True)
    build_id = _utc_now()
    index_root_path = _new_generation_root(public_index_root, build_id)
    index_root = str(index_root_path.resolve())
    fs_root = str(Path(fs_root).resolve()) if fs_root is not None else str(public_index_root)
    jobs_root = str(Path(jobs_root).resolve()) if jobs_root is not None else str(public_index_root / ".hkm_jobs")
    if not isinstance(num_workers, int) or num_workers <= 0:
        raise ValueError("num_workers must be a positive integer")
    schema_value, parsed_schema = _parse_metadata_schema_value(metadata_schema)
    backend_name = get_backend().name

    (index_root_path / "docs" / "worker_0000").mkdir(parents=True, exist_ok=True)
    (index_root_path / "hkm").mkdir(parents=True, exist_ok=True)
    (index_root_path / "manifests").mkdir(parents=True, exist_ok=True)
    set_jobs_root(jobs_root)

    Path(index_root, "index.json").write_text(json.dumps({
        "version": 1,
        "source_root": str(public_index_root),
        "jobs_root": jobs_root,
        "embedder_backend": backend_name,
        "metadata_schema": schema_value,
        "source_manifest": None,
        "build_config": {
            "num_workers": num_workers,
            "max_cluster_count": max_k,
            "leaf_embedding_limit": leaf_embedding_limit,
            "leaf_doc_limit": leaf_doc_limit,
            "max_n_gram": max_n_gram,
            "n_gram_fp_rate": n_gram_fp_rate,
            "seed": seed,
            "build_id": build_id,
            "staging_copy_bytes": 0,
            "staging_copy_seconds": 0.0,
        },
        "max_n_gram": max_n_gram,
        "n_gram_fp_rate": n_gram_fp_rate,
        "docs_path": "docs",
        "hkm_path": "hkm",
        "append_only": False,
    }, indent=2), encoding="utf-8")

    summary_path = index_root_path / "manifests" / "ingest_summary.json"
    summary_path.write_text(json.dumps({
        "scanned": 0,
        "planned": 0,
        "indexed": 0,
        "skipped": 0,
        "failed": 0,
        "reused": 0,
        "new": 0,
        "changed": 0,
        "deleted": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "skip_reasons": {},
        "skipped_files": [],
        "failed_files": [],
    }, indent=2), encoding="utf-8")
    worker_dir = index_root_path / "docs" / "worker_0000"
    process_documents(
        str(worker_dir),
        str(worker_dir),
        _document_batches(documents, parsed_schema, build_id),
        parsed_schema,
        chunk_size_limit=chunk_size_limit,
        n_gram=max_n_gram,
        fs_root=fs_root,
        max_tokens=max_tokens,
        ingest_report_path=str(worker_dir / "ingest_report.json"),
        embedding_cache_dir=None,
    )
    report = json.loads((worker_dir / "ingest_report.json").read_text(encoding="utf-8"))
    if int(report.get("indexed", 0)) <= 0:
        raise ValueError("No documents found to index.")
    consolidate(FileSystem(root=fs_root), index_root)
    root_job = run_job(
        "tlux.search.hkm.builder.recursive_index_builder.build_cluster_index",
        index_root,
        max_cluster_count=max_k,
        leaf_embedding_limit=leaf_embedding_limit,
        leaf_doc_limit=leaf_doc_limit,
        max_n_gram=max_n_gram,
        n_gram_fp_rate=n_gram_fp_rate,
        seed=seed,
        fs_root=fs_root,
        max_depth=0,
        depth=0,
    )
    drain_jobs(FileSystem(root=jobs_root), max_workers=num_workers)
    root_job.reload()
    if root_job.status != "SUCCEEDED":
        raise RuntimeError(root_job.status_reason or root_job.stderr or "HKM build failed")
    from .incremental import write_source_snapshot

    write_source_snapshot(index_root, publish_root=str(public_index_root))
    return root_job


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
    parser.add_argument("--full-rebuild", action="store_true", help="Ignore incremental snapshot and rebuild the index")
    args = parser.parse_args()

    old_disable = os.environ.get("HKM_DISABLE_WATCHER_LAUNCH")
    os.environ["HKM_DISABLE_WATCHER_LAUNCH"] = "1"
    try:
        root_job = build_search_index(
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
            incremental=not args.full_rebuild,
        )
        print(root_job.id, flush=True)
        jobs_root = Path(args.index_root).resolve() / ".hkm_jobs"
        drain_jobs(FileSystem(root=str(jobs_root)), max_workers=args.workers)
        root_job.reload()
    finally:
        if old_disable is None:
            os.environ.pop("HKM_DISABLE_WATCHER_LAUNCH", None)
        else:
            os.environ["HKM_DISABLE_WATCHER_LAUNCH"] = old_disable

    if root_job.status == "SUCCEEDED":
        return
    if root_job.status_reason:
        print(root_job.status_reason, file=sys.stderr)
    if root_job.stderr:
        print(root_job.stderr[-4000:], file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
