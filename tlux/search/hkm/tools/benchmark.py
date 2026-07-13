"""Reproducible HKM quality, cost, and scaling measurements."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..builder.chunk_io import ChunkReader
from ..search.searcher import Searcher, audit_index


# Iterate every active embedding window in every published leaf.
#
# Arguments:
#   searcher (Searcher): Open HKM index.
#   max_windows (int | None): Optional safety limit for local experiments.
#
# Returns:
#   (Iterable[tuple[np.ndarray, tuple[int, int, int], int]): Embedding, key, and window size.
#
def _iter_windows(
    searcher: Searcher,
    max_windows: int | None = None,
) -> Iterable[tuple[np.ndarray, tuple[int, int, int], int]]:
    active = searcher._active_doc_ids()
    emitted = 0
    root = Path(searcher.generation_hkm_root or searcher.hkm_root)
    for node_path in sorted(root.rglob("node.json")):
        node = json.loads(node_path.read_text(encoding="utf-8"))
        if not node.get("is_leaf", False):
            continue
        for chunk_root in node.get("chunk_roots", []):
            chunk_dir = node_path.parent / str(chunk_root)
            for chunk_path in sorted(chunk_dir.rglob("*.hkmchunk")):
                reader = ChunkReader(str(chunk_path), metadata_schema=[])
                for embedding, meta in zip(reader.embeddings, reader.embed_index):
                    doc_id = int(meta["document_id"])
                    if doc_id not in active:
                        continue
                    key = (doc_id, int(meta["token_start"]), int(meta["token_end"]))
                    yield embedding.astype(np.float32, copy=False), key, int(meta["window_size"])
                    emitted += 1
                    if max_windows is not None and emitted >= max_windows:
                        return


# Return the byte size of an index component while excluding transient state.
#
# Arguments:
#   root (Path): Directory to measure.
#   excluded (set[str]): Child directory names to omit.
#
# Returns:
#   (int): Total regular-file bytes.
#
def _tree_bytes(root: Path, excluded: set[str]) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file() and not excluded.intersection(path.relative_to(root).parts):
            total += path.stat().st_size
    return total


# Break canonical index storage into coarse artifacts for optimization decisions.
#
# Arguments:
#   root (Path): Index root.
#
# Returns:
#   (dict[str, int]): Bytes grouped by artifact family.
#
def _storage_components(root: Path) -> dict[str, int]:
    components: dict[str, int] = {}
    for path in root.rglob("*"):
        if not path.is_file() or {".hkm_jobs", ".hkm_cache"}.intersection(path.relative_to(root).parts):
            continue
        name = path.name
        if name.startswith("preview_"):
            family = "previews"
        elif name in {"embeddings.npy", "embed_index.npy"}:
            family = "embeddings"
        elif name in {"tokens.bin", "tokens_index.npy"}:
            family = "tokens"
        elif name.startswith("n_gram_") or name.startswith("observer."):
            family = "token_sketches"
        elif name == "centroids.npy":
            family = "centroids"
        else:
            family = "metadata_and_manifests"
        components[family] = components.get(family, 0) + path.stat().st_size
    return components


# Summarize persisted job stages from one completed build.
#
# Arguments:
#   root (Path): Index root containing `.hkm_jobs`.
#
# Returns:
#   (dict[str, Any]): Stage counts, durations, and ingest summary.
#
def _build_evidence(root: Path) -> dict[str, Any]:
    stages: dict[str, dict[str, float | int]] = {}
    ids_root = root / ".hkm_jobs" / "ids"
    for config_path in sorted(ids_root.glob("*/job_config")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            function = (config_path.parent / "exec_function").read_text(encoding="utf-8").strip()
            stage = function.rsplit(".", 1)[-1]
            start = float(config.get("start_ts", 0.0) or 0.0)
            end = float(config.get("end_ts", 0.0) or 0.0)
            duration = max(0.0, end - start) if end and start else 0.0
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        entry = stages.setdefault(stage, {"count": 0, "duration_seconds": 0.0, "max_seconds": 0.0})
        entry["count"] += 1
        entry["duration_seconds"] += duration
        entry["max_seconds"] = max(float(entry["max_seconds"]), duration)
    failures = []
    for job_dir in sorted((root / ".hkm_jobs" / "failed").iterdir()) if (root / ".hkm_jobs" / "failed").exists() else []:
        stderr_path = job_dir / "stderr"
        text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
        failures.append({"job_id": job_dir.name, "stderr": text[-4000:]})
    manifest_path = root / "index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    summary_path = root / "manifests" / "ingest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "stages": stages,
        "ingest_summary": summary,
        "staging": {
            "bytes": int(manifest.get("build_config", {}).get("staging_copy_bytes", 0)),
            "seconds": float(manifest.get("build_config", {}).get("staging_copy_seconds", 0.0)),
        },
        "failures": failures,
    }


# Return a stable percentile summary for repeated measurements.
#
# Arguments:
#   values (list[float]): Measured values.
#
# Returns:
#   (dict[str, float]): Minimum, median, p95, p99, and maximum values.
#
def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "min": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]),
        "p99": float(ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]),
        "max": float(ordered[-1]),
    }


# Rank every embedding exactly for one query.
#
# Arguments:
#   embeddings (np.ndarray): Matrix with shape (n, d).
#   query_embedding (np.ndarray): Query vector with shape (d,).
#   keys (list[tuple[int, int, int]]): Window identities.
#   top_k (int): Number of distinct documents to retain.
#
# Returns:
#   (tuple[list[tuple[int, int, int]], int]): Exact keys and distance count.
#
def _exact_rank(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    keys: list[tuple[int, int, int]],
    top_k: int,
) -> tuple[list[tuple[int, int, int]], int]:
    distances = np.linalg.norm(embeddings - query_embedding[None, :], axis=1)
    order = np.argsort(distances, kind="stable")
    output = []
    seen_docs = set()
    for index in order:
        key = keys[int(index)]
        if key[0] in seen_docs:
            continue
        output.append(key)
        seen_docs.add(key[0])
        if len(output) >= top_k:
            break
    return output, int(embeddings.shape[0])


# Rank with a temporary quantized copy of the stored vectors.
#
# Arguments:
#   embeddings (np.ndarray): Float32 stored vectors.
#   query_embedding (np.ndarray): Float32 query vector.
#   keys (list[tuple[int, int, int]]): Window identities.
#   top_k (int): Number of distinct documents to retain.
#   dtype (str): Either `float16` or symmetric per-dimension `int8`.
#
# Returns:
#   (list[tuple[int, int, int]]): Quantized ranking keys.
#
def _quantized_rank(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    keys: list[tuple[int, int, int]],
    top_k: int,
    dtype: str,
) -> list[tuple[int, int, int]]:
    if dtype == "float16":
        restored = embeddings.astype(np.float16).astype(np.float32)
    elif dtype == "int8":
        scale = np.maximum(np.max(np.abs(embeddings), axis=0), 1e-8) / 127.0
        restored = np.round(embeddings / scale).clip(-127, 127).astype(np.int8).astype(np.float32) * scale
    else:
        raise ValueError(f"unsupported quantization dtype: {dtype}")
    ranked, _ = _exact_rank(restored, query_embedding, keys, top_k)
    return ranked


# Return a semantic recall value for two ranked key lists.
#
# Arguments:
#   expected (list[tuple[int, int, int]]): Exact oracle keys.
#   actual (list[tuple[int, int, int]]): HKM keys.
#
# Returns:
#   (float): Fraction of expected keys recovered.
#
def _recall(expected: list[tuple[int, int, int]], actual: list[tuple[int, int, int]]) -> float:
    if not expected:
        return 1.0
    return len(set(expected).intersection(actual)) / float(len(expected))


# Measure deterministic exhaustive distance work at small scale points.
#
# Arguments:
#   dimension (int): Synthetic vector dimension.
#   seed (int): Reproducible random seed.
#
# Returns:
#   (list[dict[str, Any]]): Measured points, with larger estimates kept separate.
#
def _synthetic_scale(dimension: int = 8, seed: int = 42) -> list[dict[str, Any]]:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    rng = np.random.default_rng(seed)
    query = rng.standard_normal(dimension).astype(np.float32)
    points = []
    for count in (1_000, 10_000, 100_000):
        vectors = rng.standard_normal((count, dimension)).astype(np.float32)
        start = time.perf_counter()
        np.linalg.norm(vectors - query[None, :], axis=1)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        points.append({
            "vectors": count,
            "dimension": dimension,
            "bytes": int(vectors.nbytes),
            "exact_scan_ms": elapsed_ms,
            "measured": True,
        })
    return points


# Return the independently verified documents containing an exact token phrase.
#
# Arguments:
#   searcher (Searcher): Open HKM index.
#   text (str): Phrase to tokenize and verify.
#
# Returns:
#   (set[int]): Matching active document ids.
#
def _exact_token_docs(searcher: Searcher, text: str) -> set[int]:
    token_ids = searcher._backend().tokenize([text])[0]
    if not token_ids:
        return set()
    target = np.asarray(token_ids, dtype=np.uint32)
    matches = set()
    for doc_id in searcher._active_doc_ids():
        tokens, _ = searcher._doc_context(doc_id)
        for start in range(max(0, len(tokens) - len(target) + 1)):
            if np.array_equal(tokens[start : start + len(target)], target):
                matches.add(doc_id)
                break
    return matches


# Measure one index against exhaustive semantic retrieval.
#
# Arguments:
#   index_root (str): Index directory.
#   queries (list[str]): Semantic query texts.
#   top_k (int): Number of oracle results to compare.
#   repeats (int): Warm query repetitions.
#   max_windows (int | None): Optional local safety limit.
#   probe_count (int): Children to probe per non-leaf; zero is exhaustive.
#
# Returns:
#   (dict[str, Any]): JSON-compatible evidence report.
#
def measure_index(
    index_root: str,
    queries: list[str],
    top_k: int = 10,
    repeats: int = 3,
    max_windows: int | None = None,
    probe_count: int = 2,
    exact_queries: list[str] | None = None,
    where: dict[str, Any] | None = None,
    synthetic: bool = True,
) -> dict[str, Any]:
    if top_k < 1 or repeats < 1:
        raise ValueError("top_k and repeats must be positive")
    root = audit_index(index_root)
    searcher = Searcher.from_index_root(str(root))
    generation_root = Path(searcher.generation_root or root)
    rows = list(_iter_windows(searcher, max_windows=max_windows))
    if not rows:
        raise ValueError(f"Index contains no active embedding windows: {root}")
    embeddings = np.stack([row[0] for row in rows]).astype(np.float32, copy=False)
    keys = [row[1] for row in rows]
    window_sizes = np.asarray([row[2] for row in rows], dtype=np.int32)
    backend = searcher._backend()
    query_reports = []
    for text in queries:
        query_ids = backend.tokenize([text])
        query_embedding = backend.embed(query_ids, role="query")[0]
        exact_keys, exact_work = _exact_rank(embeddings, query_embedding, keys, top_k)
        quantized_recall = {
            dtype: _recall(exact_keys, _quantized_rank(embeddings, query_embedding, keys, top_k, dtype))
            for dtype in ("float16", "int8")
        }
        window_ablation = {}
        overlap_ablation = {}
        for window_size in sorted(set(int(value) for value in window_sizes)):
            mask = window_sizes != window_size
            reduced_keys, _ = _exact_rank(embeddings[mask], query_embedding, [key for key, keep in zip(keys, mask) if keep], top_k)
            window_ablation[str(window_size)] = {
                "recall_without_size": _recall(exact_keys, reduced_keys),
                "windows_removed": int((~mask).sum()),
            }
            ordinals: dict[int, int] = {}
            overlap_mask = np.ones(len(keys), dtype=bool)
            for index, (key, size) in enumerate(zip(keys, window_sizes)):
                if int(size) != window_size:
                    continue
                ordinal = ordinals.get(key[0], 0)
                ordinals[key[0]] = ordinal + 1
                overlap_mask[index] = ordinal % 2 == 0
            overlap_keys = [key for key, keep in zip(keys, overlap_mask) if keep]
            reduced_keys, _ = _exact_rank(embeddings[overlap_mask], query_embedding, overlap_keys, top_k)
            overlap_ablation[str(window_size)] = {
                "recall_after_dropping_alternate_windows": _recall(exact_keys, reduced_keys),
                "windows_removed": int((~overlap_mask).sum()),
                "method": "drop alternate windows within each document and size",
            }
        hkm_ranked: list[tuple[float, int, tuple[int, int], int]] = []
        start = time.perf_counter()
        searcher._search_node(Path(searcher.generation_hkm_root or searcher.hkm_root), query_embedding, hkm_ranked, probe_count)
        traversal_ms = (time.perf_counter() - start) * 1000.0
        hkm_hits = searcher._search_embeddings(query_embedding, top_k, text, probe_count)
        hkm_keys = [(hit.doc_id, hit.span[0], hit.span[1]) for hit in hkm_hits]
        query_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            searcher._search_embeddings(query_embedding, top_k, text, probe_count)
            query_times.append((time.perf_counter() - start) * 1000.0)
        query_reports.append({
            "query": text,
            "recall_at_k": _recall(exact_keys, hkm_keys),
            "window_recall_at_k": _recall(exact_keys, hkm_keys),
            "quantized_recall_at_k": quantized_recall,
            "window_ablation": window_ablation,
            "overlap_ablation": overlap_ablation,
            "exact_work_windows": exact_work,
            "hkm_work_windows": len(hkm_ranked),
            "hkm_work_fraction": len(hkm_ranked) / float(exact_work),
            "first_traversal_ms": traversal_ms,
            "warm_traversal_ms": _summary(query_times),
            "exact_keys": [list(key) for key in exact_keys],
            "hkm_keys": [list(key) for key in hkm_keys],
        })
    exact_checks = []
    for text in exact_queries or []:
        expected = _exact_token_docs(searcher, text)
        actual = {
            hit.doc_id
            for hit in searcher.search({
                "text_ast": {"phrase": text},
                "top_k": max(1, searcher._doc_count()),
            }).docs
        }
        exact_checks.append({
            "query": text,
            "expected_documents": sorted(expected),
            "actual_documents": sorted(actual),
            "exact": expected == actual,
        })
    filter_check = None
    if where is not None:
        actual = {
            hit.doc_id
            for hit in searcher.search({
                "mode": "semantic",
                "text": queries[0],
                "top_k": max(1, searcher._doc_count()),
                "where": where,
            }).docs
        }
        expected = set()
        for doc_id in searcher._active_doc_ids():
            hit = searcher._hit(doc_id, 0.0, (0, 0), "", "")
            if searcher._filter_hit(hit, where):
                expected.add(doc_id)
        filter_check = {
            "where": where,
            "expected_documents": sorted(expected),
            "actual_documents": sorted(actual),
            "exact": expected == actual,
        }
    canonical_bytes = _tree_bytes(generation_root, {".hkm_jobs", ".hkm_cache", ".hkm_builds"})
    embedding_bytes = sum(
        path.stat().st_size
        for path in generation_root.rglob("embeddings.npy")
        if not {".hkm_jobs", ".hkm_cache", ".hkm_builds"}.intersection(path.relative_to(generation_root).parts)
    )
    cache_bytes = _tree_bytes(root / ".hkm_cache", set())
    source_root = Path(searcher.source_root)
    source_bytes = _tree_bytes(source_root, {".hkm_jobs", ".hkm_cache", "docs", "hkm"}) if source_root != root else 0
    scale_targets = [1_000, 10_000, 100_000, 1_000_000, 1_000_000_000]
    bytes_per_window = canonical_bytes / float(len(rows))
    mean_work_fraction = statistics.mean(row["hkm_work_fraction"] for row in query_reports)
    scale = [
        {
            "windows": target,
            "estimated_index_bytes": int(bytes_per_window * target),
            "estimated_index_gib": bytes_per_window * target / 2**30,
            "estimated_candidate_windows": int(target if probe_count == 0 else target * mean_work_fraction),
            "estimated_tree_levels": int(max(1, np.ceil(np.log(max(1.0, target / 1024.0)) / np.log(8.0)))),
            "basis": "linear observed bytes/window extrapolation",
        }
        for target in scale_targets
    ]
    return {
        "index_root": str(root),
        "backend": searcher.backend_name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "documents": searcher._doc_count(),
        "embedding_windows": len(rows),
        "embedding_dimension": int(embeddings.shape[1]),
        "source_bytes": source_bytes,
        "canonical_index_bytes": canonical_bytes,
        "storage_components": _storage_components(generation_root),
        "storage_ablations": {
            dtype: int(canonical_bytes - embedding_bytes + embedding_bytes * factor)
            for dtype, factor in (("float16", 0.5), ("int8", 0.25))
        },
        "embedding_cache_bytes": cache_bytes,
        "bytes_per_window": bytes_per_window,
        "bytes_per_source_byte": canonical_bytes / float(source_bytes) if source_bytes else None,
        "top_k": top_k,
        "probe_count": probe_count,
        "mean_work_fraction": mean_work_fraction,
        "queries": query_reports,
        "exact_checks": exact_checks,
        "filter_check": filter_check,
        "build": _build_evidence(root),
        "synthetic_scale": _synthetic_scale() if synthetic else [],
        "scaling": scale,
        "scaling_note": "Scale rows are extrapolations from observed bytes/window; they are not 1B-document measurements.",
    }


# Render a compact report that keeps measured values distinct from estimates.
#
# Arguments:
#   report (dict[str, Any]): Measurement report.
#
# Returns:
#   (str): Markdown report.
#
def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# HKM Benchmark Report",
        "",
        f"- Backend: `{report['backend']}`",
        f"- Python/platform: `{report.get('python', '')}` / `{report.get('platform', '')}`",
        f"- Documents/windows: {report['documents']} / {report['embedding_windows']}",
        f"- Source bytes: {report.get('source_bytes', 0):,}",
        f"- Canonical bytes: {report['canonical_index_bytes']:,}",
        f"- Cache bytes: {report['embedding_cache_bytes']:,}",
        f"- Bytes/window: {report['bytes_per_window']:.2f}",
        f"- Probe count: {report['probe_count']} (0 means exhaustive)",
        "",
        "## Query evidence",
        "",
        "| Query | Recall@k | HKM windows | Exact windows | Work fraction | Warm p50/p95/p99 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["queries"]:
        timing = row.get("warm_traversal_ms", {})
        lines.append(
            f"| {row['query']} | {row['recall_at_k']:.3f} | {row['hkm_work_windows']} | "
            f"{row['exact_work_windows']} | {row['hkm_work_fraction']:.3f} | "
            f"{float(timing.get('median', timing.get('p95', 0.0))):.2f}/"
            f"{float(timing.get('p95', 0.0)):.2f}/{float(timing.get('p99', timing.get('p95', 0.0))):.2f} |"
        )
        quantized = row.get("quantized_recall_at_k")
        if quantized:
            lines.append(
                f"  Quantized recall: float16={quantized['float16']:.3f}, "
                f"int8={quantized['int8']:.3f}."
            )
        if row.get("window_ablation"):
            ablation = ", ".join(
                f"drop {size}: {values['recall_without_size']:.3f}"
                for size, values in sorted(row["window_ablation"].items())
            )
            lines.append(f"  Window ablation recall: {ablation}.")
        if row.get("overlap_ablation"):
            ablation = ", ".join(
                f"{size}: {values['recall_after_dropping_alternate_windows']:.3f}"
                for size, values in sorted(row["overlap_ablation"].items())
            )
            lines.append(f"  Alternate-overlap removal recall: {ablation}.")
    lines.extend(["", "## Exact checks", ""])
    for row in report.get("exact_checks", []):
        lines.append(f"- `{row['query']}`: `{'PASS' if row['exact'] else 'FAIL'}`")
    if report.get("filter_check") is not None:
        row = report["filter_check"]
        lines.append(f"- metadata filter `{row['where']}`: `{'PASS' if row['exact'] else 'FAIL'}`")
    lines.extend(["", "## Build evidence", "", "| Stage | Jobs | Total seconds | Max seconds |", "|---|---:|---:|---:|"])
    for name, row in sorted(report.get("build", {}).get("stages", {}).items()):
        lines.append(
            f"| {name} | {int(row['count'])} | {float(row['duration_seconds']):.3f} | "
            f"{float(row['max_seconds']):.3f} |"
        )
    ingest = report.get("build", {}).get("ingest_summary", {})
    if ingest:
        lines.append(
            f"\nIngest: planned={ingest.get('planned', 0)}, indexed={ingest.get('indexed', 0)}, "
            f"reused={ingest.get('reused', 0)}, changed={ingest.get('changed', 0)}, "
            f"cache_hits={ingest.get('cache_hits', 0)}, cache_misses={ingest.get('cache_misses', 0)}."
        )
    staging = report.get("build", {}).get("staging", {})
    if staging and (staging.get("bytes") or staging.get("seconds")):
        lines.append(
            f"Staging copy: {int(staging.get('bytes', 0)):,} bytes in "
            f"{float(staging.get('seconds', 0.0)):.3f} seconds."
        )
    lines.extend(["", "## Canonical storage components", "", "| Component | Bytes |", "|---|---:|"])
    for name, size in sorted(report.get("storage_components", {}).items()):
        lines.append(f"| {name} | {size:,} |")
    if report.get("storage_ablations"):
        lines.extend(["", "Estimated canonical bytes after embedding-only quantization:"])
        for name, size in sorted(report["storage_ablations"].items()):
            lines.append(f"- `{name}`: {size:,}")
    lines.extend([
        "",
        "## Scaling estimate",
        "",
        "These rows extrapolate observed storage per embedding window. They do not claim that a laptop has built a billion-document index.",
        "",
        "| Windows | Estimated index GiB | Modeled candidate windows | Tree levels |",
        "|---:|---:|---:|---:|",
    ])
    for row in report["scaling"]:
        lines.append(
            f"| {row['windows']:,} | {row['estimated_index_gib']:.2f} | "
            f"{row['estimated_candidate_windows']:,} | {row['estimated_tree_levels']} |"
        )
    lines.extend([
        "",
        "```mermaid",
        "xychart-beta",
        "    title \"Observed-storage extrapolation\"",
        "    x-axis [1K, 10K, 100K, 1M, 1B]",
        "    y-axis \"GiB\" 0 --> auto",
        "    line [" + ", ".join(f"{row['estimated_index_gib']:.2f}" for row in report["scaling"]) + "]",
        "```",
        "",
        "At positive probe counts, candidate work is modeled as the measured "
        "candidate fraction held constant while tree depth grows logarithmically "
        "with the corpus. This supports a scaling hypothesis, not a claim that "
        "the laptop measured a billion-document build.",
        "",
        "## Deterministic exhaustive-scan scaling",
        "",
        "These points are measured NumPy scans on this machine; they are not HKM query results.",
        "",
        "| Vectors | Dimension | Vector bytes | Exact scan ms |",
        "|---:|---:|---:|---:|",
    ])
    for row in report.get("synthetic_scale", []):
        lines.append(f"| {row['vectors']:,} | {row['dimension']} | {row['bytes']:,} | {row['exact_scan_ms']:.3f} |")
    lines.extend([
        "",
    ])
    return "\n".join(lines)


# Parse CLI arguments and write the evidence report.
#
# Arguments:
#   None
#
# Returns:
#   (None): Writes requested outputs and prints Markdown.
#
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure HKM quality, work, storage, and scaling.")
    parser.add_argument("index_root")
    parser.add_argument("--query", action="append", default=["0 1 2 3"], help="Semantic query; repeatable")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--probes", type=int, default=2, help="Children to probe per level; zero is exhaustive")
    parser.add_argument("--exact-query", action="append", default=[], help="Exact token phrase check; repeatable")
    parser.add_argument("--where-json", default=None, help="Metadata filter object to verify")
    parser.add_argument("--no-synthetic", action="store_true", help="Skip deterministic NumPy scale points")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--report-output", default=None)
    args = parser.parse_args()
    where = json.loads(args.where_json) if args.where_json else None
    report = measure_index(
        args.index_root,
        args.query,
        args.top_k,
        args.repeat,
        args.max_windows,
        args.probes,
        args.exact_query,
        where,
        not args.no_synthetic,
    )
    markdown = render_report(report)
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.report_output:
        Path(args.report_output).write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
