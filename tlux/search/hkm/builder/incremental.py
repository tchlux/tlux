"""Incremental build helpers for active snapshots and leaf appends.

This module keeps incremental state small: source snapshots define active
documents, appended chunks reuse the existing HKM tree, and full rebuilds remain
the path for global rebalancing.

Example:
  write_source_snapshot("idx")
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .chunk_io import ChunkReader, ChunkWriter
from .recursive_index_builder import build_cluster_index
from ..fs import make_filesystem


# Drain descendants created by recursive tree jobs while leaving this barrier job running.
#
# Arguments:
#   jobs_root (str): Shared jobs directory.
#   max_workers (int): Configured build worker count.
#
# Returns:
#   (None): Waits until no queued or waiting descendant remains.
#
def _drain_descendant_jobs(jobs_root: str, max_workers: int) -> None:
    from ..jobs import watcher

    fs = make_filesystem(jobs_root)
    deadline = time.time() + 300.0
    while time.time() < deadline:
        watcher(fs=fs, max_workers=max(2, int(max_workers)))
        waiting = len(fs.listdir("waiting"))
        queued = len(fs.listdir("queued"))
        running = len(fs.listdir("running"))
        if waiting == 0 and queued == 0 and running <= 1:
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for recursive HKM jobs under {jobs_root}")
from ..schema import DOC_INDEX_DTYPE
from ..tools.unique_count_estimator import UniqueCounter
from ..tools.value_seen_estimator import ValueObserver


# Convert a JSON metadata schema into runtime Python types.
#
# Arguments:
#   schema (list[list[str]]): Serialized metadata schema.
#
# Returns:
#   (list[tuple[str, type]]): Parsed schema.
#
def _parse_schema(schema: List[List[str]]) -> List[Tuple[str, type]]:
    type_map = {"str": str, "float": float, "int": int, "json": dict, "bytes": bytes, "list": list, "dict": dict}
    return [(name, type_map.get(kind, str)) for name, kind in schema]


# Decode a metadata scalar into a string.
#
# Arguments:
#   value (object): Metadata value.
#
# Returns:
#   (str): Decoded string or empty string.
#
def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if value is None:
        return ""
    return str(value)


# Return the canonical chunk path for a doc-index row.
#
# Arguments:
#   docs_root (Path): Canonical docs directory.
#   row (np.void): Row from doc_index.npy.
#
# Returns:
#   (Path): Chunk directory.
#
def _row_chunk_path(docs_root: Path, row: np.void) -> Path:
    return docs_root / f"worker_{int(row['worker']):04d}" / f"shard_{int(row['shard']):08d}.hkmchunk"


# Load active doc-index rows keyed by doc id.
#
# Arguments:
#   index_root (Path): Index root.
#
# Returns:
#   (dict[int, np.void]): Active doc rows.
#
def _doc_rows(index_root: Path) -> Dict[int, np.void]:
    path = index_root / "docs" / "doc_index.npy"
    if not path.exists():
        return {}
    rows = np.load(path)
    return {int(row["doc_id"]): row for row in rows}


# Yield all leaf node directories in an HKM tree.
#
# Arguments:
#   hkm_root (Path): HKM root directory.
#
# Returns:
#   (Iterable[Path]): Leaf node directories.
#
def _leaf_dirs(hkm_root: Path) -> Iterable[Path]:
    for node_path in hkm_root.rglob("node.json"):
        node = json.loads(node_path.read_text(encoding="utf-8"))
        if node.get("is_leaf", False):
            yield node_path.parent


# Map doc ids to the leaf directory where search can find them.
#
# Arguments:
#   hkm_root (Path): HKM tree root.
#
# Returns:
#   (dict[int, str]): Doc id to leaf path relative to hkm root.
#
def _leaf_map(hkm_root: Path) -> Dict[int, str]:
    output: Dict[int, str] = {}
    if not hkm_root.exists():
        return output
    for leaf_dir in _leaf_dirs(hkm_root):
        node = json.loads((leaf_dir / "node.json").read_text(encoding="utf-8"))
        rel_leaf = "." if leaf_dir == hkm_root else leaf_dir.relative_to(hkm_root).as_posix()
        for chunk_root in node.get("chunk_roots", []):
            root = leaf_dir / chunk_root
            for chunk_path in root.rglob("*.hkmchunk") if root.exists() else []:
                reader = ChunkReader(str(chunk_path), metadata_schema=[])
                base = int(reader.chunk_metadata().get("min_document_id", 0) or 0)
                for i in range(reader.document_count):
                    output.setdefault(base + i, rel_leaf)
    return output


# Return decoded metadata for a doc-index row.
#
# Arguments:
#   docs_root (Path): Canonical docs directory.
#   row (np.void): Row from doc_index.npy.
#   schema (list[tuple[str, type]]): Parsed metadata schema.
#
# Returns:
#   (dict[str, object]): Metadata values keyed by field name.
#
def _row_metadata(docs_root: Path, row: np.void, schema: List[Tuple[str, type]]) -> Dict[str, object]:
    reader = ChunkReader(str(_row_chunk_path(docs_root, row)), metadata_schema=schema)
    _, _, _, values = reader[int(row["idx"])]
    return {name: value for (name, _typ), value in zip(schema, values)}


# Fail a build that planned files but published no active documents.
#
# Arguments:
#   index_root (Path): Index root.
#   active_count (int): Count of active document rows.
#
# Returns:
#   (None): Raises when the build would publish an empty active index.
#
def _assert_non_empty_build(index_root: Path, active_count: int) -> None:
    summary_path = index_root / "manifests" / "ingest_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    planned = int(summary.get("planned", 0))
    if planned <= 0 or active_count > 0:
        return
    skipped = int(summary.get("skipped", 0))
    failed = int(summary.get("failed", 0))
    reasons = dict(summary.get("skip_reasons", {}))
    raise RuntimeError(
        f"Build indexed zero documents: planned={planned} skipped={skipped} "
        f"failed={failed} reasons={reasons}"
    )


# Write the active source snapshot from the current doc_index.
#
# Arguments:
#   index_root (str): Index root.
#
#   publish_root (str | None): Public path to swap after audit.
#
# Returns:
#   (None): Writes manifests/source_snapshot.json and optionally publishes it.
#
def write_source_snapshot(index_root: str, publish_root: str | None = None) -> None:
    root = Path(index_root)
    manifest = json.loads((root / "index.json").read_text(encoding="utf-8"))
    _drain_descendant_jobs(manifest.get("jobs_root", str(root / ".hkm_jobs")), int(manifest.get("build_config", {}).get("num_workers", 1)))
    schema = _parse_schema(manifest.get("metadata_schema", []))
    docs_root = root / manifest.get("docs_path", "docs")
    doc_rows = _doc_rows(root)
    _assert_non_empty_build(root, len(doc_rows))
    from ..search.searcher import audit_index

    audit_index(index_root)
    leaf_paths = _leaf_map(root / manifest.get("hkm_path", "hkm"))
    entries = []
    for doc_id, row in sorted(doc_rows.items()):
        meta = _row_metadata(docs_root, row, schema)
        entries.append({
            "source_path": _text(meta.get("source_path")),
            "content_hash": _text(meta.get("content_hash")),
            "num_bytes": int(meta.get("num_bytes") or 0),
            "doc_id": doc_id,
            "worker": int(row["worker"]),
            "shard": int(row["shard"]),
            "idx": int(row["idx"]),
            "leaf_path": leaf_paths.get(doc_id, ""),
            "embedding_count": int(_row_chunk_reader(docs_root, row).embed_index["document_id"].tolist().count(doc_id)),
        })
    snapshot = {
        "version": 1,
        "source_root": manifest.get("source_root", ""),
        "embedder_backend": manifest.get("embedder_backend", ""),
        "metadata_schema": manifest.get("metadata_schema", []),
        "build_config": manifest.get("build_config", {}),
        "documents": entries,
    }
    out = root / "manifests" / "source_snapshot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    if publish_root:
        from .launcher import publish_generation

        publish_generation(str(root), publish_root)


# Return the ChunkReader for a doc-index row.
#
# Arguments:
#   docs_root (Path): Canonical docs directory.
#   row (np.void): Row from doc_index.npy.
#
# Returns:
#   (ChunkReader): Reader with no metadata decoding.
#
def _row_chunk_reader(docs_root: Path, row: np.void) -> ChunkReader:
    return ChunkReader(str(_row_chunk_path(docs_root, row)), metadata_schema=[])


# Return all n-gram byte strings for a token sequence.
#
# Arguments:
#   tokens (list[int]): Token sequence.
#   max_n_gram (int): Maximum n-gram length.
#
# Returns:
#   (list[bytes]): Encoded n-grams.
#
def _ngrams(tokens: List[int], max_n_gram: int) -> List[bytes]:
    return [
        b"".join(int(tok & 0xFFFFFFFF).to_bytes(4, "little") for tok in tokens[i : i + n])
        for n in range(1, max_n_gram + 1)
        for i in range(len(tokens) - n + 1)
    ]


# Add n-grams and count deltas to one node manifest.
#
# Arguments:
#   node_dir (Path): HKM node directory.
#   ngrams (list[bytes]): Encoded n-grams to add.
#   doc_delta (int): Document count increment.
#   emb_delta (int): Embedding count increment.
#
# Returns:
#   (None): Updates node files in place.
#
def _update_node(node_dir: Path, ngrams: List[bytes], doc_delta: int, emb_delta: int) -> None:
    counter_path = node_dir / "n_gram_counter.bytes"
    counter = UniqueCounter.from_bytes(counter_path.read_bytes()) if counter_path.exists() else UniqueCounter()
    for ngram in ngrams:
        counter.add(ngram)
    counter_path.write_bytes(counter.to_bytes())
    exists_path = node_dir / "n_gram_exists.bytes"
    if exists_path.exists():
        observer = ValueObserver.from_bytes(exists_path.read_bytes())
    else:
        _, _, upper = counter.estimate()
        observer = ValueObserver.create(capacity=max(1, int(math.ceil(upper))))
    for ngram in ngrams:
        observer.add(ngram)
    exists_path.write_bytes(observer.to_bytes())
    node_path = node_dir / "node.json"
    if node_path.exists():
        node = json.loads(node_path.read_text(encoding="utf-8"))
        node["doc_count"] = int(node.get("doc_count", 0)) + doc_delta
        node["embedding_count"] = int(node.get("embedding_count", 0)) + emb_delta
        node["n_gram_counter_path"] = "n_gram_counter.bytes"
        node["n_gram_exists_path"] = "n_gram_exists.bytes"
        node["estimated_unique_ngrams"] = int(math.ceil(counter.estimate(0.0)[0]))
        node_path.write_text(json.dumps(node, separators=(",", ":")), encoding="utf-8")
    stats_path = node_dir / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="ascii"))
        stats["doc_count"] = int(stats.get("doc_count", 0)) + doc_delta
        stats["emb_count"] = int(stats.get("emb_count", 0)) + emb_delta
        stats_path.write_text(json.dumps(stats, separators=(",", ":")), encoding="ascii")


# Find the existing leaf for a document embedding.
#
# Arguments:
#   hkm_root (Path): HKM root directory.
#   embedding (np.ndarray): Document embedding rows.
#
# Returns:
#   (tuple[Path, list[Path]]): Leaf directory and path from root to leaf.
#
def _route_leaf(hkm_root: Path, embedding: np.ndarray) -> Tuple[Path, List[Path]]:
    node_dir = hkm_root
    path = [node_dir]
    doc_emb = embedding.mean(axis=0) if embedding.size else np.zeros((1,), dtype=np.float32)
    while True:
        node_path = node_dir / "node.json"
        if not node_path.exists():
            return node_dir, path
        node = json.loads(node_path.read_text(encoding="utf-8"))
        children = node.get("children", [])
        centroids_path = node_dir / "centroids.npy"
        if node.get("is_leaf", False) or not children or not centroids_path.exists():
            return node_dir, path
        centroids = np.load(centroids_path)
        cid = int(np.argmin(np.linalg.norm(centroids - doc_emb[None, :], axis=1)))
        node_dir = node_dir / children[cid]
        path.append(node_dir)


# Ensure a leaf manifest will scan appended data chunks.
#
# Arguments:
#   leaf_dir (Path): HKM leaf directory.
#
# Returns:
#   (None): Adds data to chunk_roots when needed.
#
def _ensure_leaf_data_root(leaf_dir: Path) -> None:
    node_path = leaf_dir / "node.json"
    if not node_path.exists():
        return
    node = json.loads(node_path.read_text(encoding="utf-8"))
    roots = list(node.get("chunk_roots", []))
    if "data" not in roots:
        roots.append("data")
        node["chunk_roots"] = roots
        node["has_data"] = True
        node_path.write_text(json.dumps(node, separators=(",", ":")), encoding="utf-8")


# Append canonical new chunks to the existing HKM leaves.
#
# Arguments:
#   index_root (Path): Index root.
#   rows (list[np.ndarray]): New doc-index rows.
#   max_n_gram (int): Maximum n-gram length.
#
# Returns:
#   (None): Writes leaf append chunks and updates node summaries.
#
def _append_rows_to_tree(index_root: Path, rows: List[np.ndarray], max_n_gram: int) -> set[Path]:
    fs = make_filesystem(str(index_root))
    docs_root = index_root / "docs"
    hkm_root = index_root / "hkm"
    writers: Dict[Path, ChunkWriter] = {}
    touched: set[Path] = set()
    try:
        for row in rows:
            reader = _row_chunk_reader(docs_root, row)
            tokens, embeddings, embed_meta, _ = reader[int(row["idx"])]
            doc_id = int(row["doc_id"])
            leaf_dir, node_path = _route_leaf(hkm_root, embeddings)
            touched.add(leaf_dir)
            _ensure_leaf_data_root(leaf_dir)
            out_dir = leaf_dir / "data" / f"worker_{int(row['worker']):04d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            if leaf_dir not in writers:
                writers[leaf_dir] = ChunkWriter(fs, str(out_dir), 8 * 2**20, metadata_schema=[])
            windows = [(int(item["token_start"]), int(item["token_end"]), int(item["window_size"])) for item in embed_meta]
            token_list = tokens.tolist()
            writers[leaf_dir].add_document(doc_id, token_list, embeddings, windows, [])
            grams = _ngrams(token_list, max_n_gram)
            for node_dir in node_path:
                _update_node(node_dir, grams, 1, int(embeddings.shape[0]))
    finally:
        for writer in writers.values():
            writer.save_chunk()
    return touched


# Split touched leaves that exceed configured limits and own their data root.
#
# Arguments:
#   leaves (set[Path]): Leaf directories touched by the incremental append.
#   max_cluster_count (int): Maximum child cluster count.
#   leaf_embedding_limit (int): Embedding threshold.
#   leaf_doc_limit (int): Document threshold.
#   max_n_gram (int): Maximum n-gram length.
#   n_gram_fp_rate (float): Bloom filter false-positive rate.
#   seed (int): Random seed.
#
# Returns:
#   (None): Rebuilds oversized leaf subtrees in place.
#
def _split_oversized_leaves(
    leaves: set[Path],
    max_cluster_count: int,
    leaf_embedding_limit: int,
    leaf_doc_limit: int,
    max_n_gram: int,
    n_gram_fp_rate: float,
    seed: int,
) -> None:
    for leaf_dir in sorted(leaves):
        node_path = leaf_dir / "node.json"
        if not node_path.exists() or not (leaf_dir / "data").exists():
            continue
        node = json.loads(node_path.read_text(encoding="utf-8"))
        if "data" not in node.get("chunk_roots", []):
            continue
        too_many_docs = 0 < leaf_doc_limit < int(node.get("doc_count", 0))
        too_many_embeddings = 0 < leaf_embedding_limit < int(node.get("embedding_count", 0))
        if not (too_many_docs or too_many_embeddings):
            continue
        for child in node.get("children", []):
            shutil.rmtree(leaf_dir / child, ignore_errors=True)
        build_cluster_index(
            str(leaf_dir),
            max_cluster_count=max_cluster_count,
            leaf_embedding_limit=leaf_embedding_limit,
            leaf_doc_limit=leaf_doc_limit,
            max_n_gram=max_n_gram,
            n_gram_fp_rate=n_gram_fp_rate,
            seed=seed,
            max_depth=0,
            depth=int(node.get("depth", 0)),
        )


# Rename new worker chunks and return their doc-index rows.
#
# Arguments:
#   docs_root (Path): Canonical docs root.
#   worker_ids (list[int]): New worker ids.
#
# Returns:
#   (list[np.ndarray]): Doc-index rows for new documents.
#
def _new_worker_rows(docs_root: Path, worker_ids: List[int]) -> List[np.ndarray]:
    rows: List[np.ndarray] = []
    for worker_id in worker_ids:
        worker_path = docs_root / f"worker_{worker_id:04d}"
        if not worker_path.exists():
            continue
        for shard_id, chunk_dir in enumerate(sorted(worker_path.glob("chunk_*.hkmchunk"))):
            target = worker_path / f"shard_{shard_id:08d}.hkmchunk"
            if chunk_dir != target:
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                os.rename(chunk_dir, target)
            reader = ChunkReader(str(target), metadata_schema=[])
            base = int(reader.chunk_metadata().get("min_document_id", 0) or 0)
            for local_idx in range(reader.document_count):
                rows.append(np.array((base + local_idx, worker_id, shard_id, local_idx), dtype=DOC_INDEX_DTYPE))
    return rows


# Merge worker reports into the incremental ingest summary.
#
# Arguments:
#   index_root (Path): Index root.
#   worker_ids (list[int]): New worker ids.
#   active_count (int): Active document count after finalization.
#
# Returns:
#   (None): Updates ingest_summary.json.
#
def _finish_summary(index_root: Path, worker_ids: List[int], active_count: int) -> None:
    summary_path = index_root / "manifests" / "ingest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    skipped_files = list(summary.get("skipped_files", []))
    failed_files = list(summary.get("failed_files", []))
    reasons = dict(summary.get("skip_reasons", {}))
    cache_hits = int(summary.get("cache_hits", 0))
    cache_misses = int(summary.get("cache_misses", 0))
    for worker_id in worker_ids:
        report_path = index_root / "docs" / f"worker_{worker_id:04d}" / "ingest_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cache_hits += int(report.get("cache_hits", 0))
        cache_misses += int(report.get("cache_misses", 0))
        for item in report.get("skipped_files", []):
            reason = item.get("reason", "worker_skip")
            reasons[reason] = reasons.get(reason, 0) + 1
            skipped_files.append(item)
        failed_files.extend(report.get("failed_files", []))
    summary["indexed"] = active_count
    summary["skipped"] = len(skipped_files)
    summary["failed"] = len(failed_files)
    summary["cache_hits"] = cache_hits
    summary["cache_misses"] = cache_misses
    summary["skip_reasons"] = reasons
    summary["skipped_files"] = skipped_files
    summary["failed_files"] = failed_files
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


# Finalize an incremental build after new document workers finish.
#
# Arguments:
#   index_root (str): Index root.
#   plan_path (str): Incremental plan JSON path.
#   max_n_gram (int): Maximum n-gram length.
#
# Returns:
#   (None): Publishes active doc_index and source snapshot.
#
def finalize_incremental(
    index_root: str,
    plan_path: str,
    publish_root: str | None = None,
    max_cluster_count: int = 8,
    leaf_embedding_limit: int = 1024,
    leaf_doc_limit: int = 1024,
    max_n_gram: int = 3,
    n_gram_fp_rate: float = 0.01,
    seed: int = 42,
) -> None:
    root = Path(index_root)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    docs_root = root / "docs"
    old_rows = _doc_rows(root)
    active_rows = [old_rows[int(item["doc_id"])] for item in plan.get("reused_documents", []) if int(item["doc_id"]) in old_rows]
    new_rows = _new_worker_rows(docs_root, [int(worker_id) for worker_id in plan.get("worker_ids", [])])
    touched_leaves = _append_rows_to_tree(root, new_rows, max_n_gram)
    _split_oversized_leaves(
        touched_leaves,
        max_cluster_count,
        leaf_embedding_limit,
        leaf_doc_limit,
        max_n_gram,
        n_gram_fp_rate,
        seed,
    )
    rows = active_rows + new_rows
    if rows:
        doc_index = np.stack(rows).astype(DOC_INDEX_DTYPE, copy=False)
        doc_index.sort(order="doc_id")
        np.save(docs_root / "doc_index.npy", doc_index)
    _finish_summary(root, [int(worker_id) for worker_id in plan.get("worker_ids", [])], len(rows))
    write_source_snapshot(index_root, publish_root=publish_root)


# No-op job entrypoint used for unchanged incremental builds.
#
# Arguments:
#   None.
#
# Returns:
#   (None): Job succeeds.
#
def noop() -> None:
    return None
