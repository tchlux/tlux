"""Evaluate HKM and BM25 on a standard BEIR corpus/query/qrels layout.

The dataset is read from a local directory and never copied into tracked
fixtures. The optional HKM lane builds a separate index and reports measured
retrieval metrics beside the external qrels oracle.

Example:
    bin/hkm-beir /private/tmp/scifact/scifact --limit 25
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..builder.launcher import build_search_index_from_documents
from ..search.searcher import Searcher
from .quality_benchmark import bm25_search_batch, evaluate_run


@dataclass(frozen=True)
class BeirDataset:
    name: str
    split: str
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, float]]


# Read a JSONL file keyed by its BEIR identifier.
#
# Arguments:
#   path (Path): JSONL path.
#   text_fields (tuple[str, ...]): Fields joined into searchable text.
#
# Returns:
#   (dict[str, str]): Stable identifier-to-text mapping.
def _read_jsonl(path: Path, text_fields: tuple[str, ...]) -> dict[str, str]:
    records: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(record, dict) or not isinstance(record.get("_id"), str):
            raise ValueError(f"{path}:{line_number}: record requires string _id")
        text = "\n".join(str(record.get(field, "")).strip() for field in text_fields).strip()
        if not text:
            raise ValueError(f"{path}:{line_number}: {_id(record)} has empty text")
        identifier = record["_id"]
        if identifier in records:
            raise ValueError(f"{path}:{line_number}: duplicate _id {identifier!r}")
        records[identifier] = text
    if not records:
        raise ValueError(f"{path}: no records")
    return records


# Return an identifier for a malformed-record error without masking its cause.
def _id(record: dict[str, Any]) -> str:
    return repr(record.get("_id", "<missing>"))


# Read a BEIR qrels TSV and validate every referenced query and document.
#
# Arguments:
#   path (Path): Qrels TSV path.
#   queries (dict[str, str]): Loaded query IDs.
#   corpus (dict[str, str]): Loaded corpus IDs.
#
# Returns:
#   (dict[str, dict[str, float]]): Query-to-graded-document judgments.
def _read_qrels(path: Path, queries: dict[str, str], corpus: dict[str, str]) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].lower() == "query-id":
            continue
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: qrels rows require query ID, corpus ID, score")
        query_id, document_id, raw_score = fields
        if query_id not in queries or document_id not in corpus:
            raise ValueError(f"{path}:{line_number}: qrels references an unknown ID")
        if document_id in qrels.get(query_id, {}):
            raise ValueError(f"{path}:{line_number}: duplicate judgment")
        try:
            score = float(raw_score)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: score must be numeric") from exc
        if score < 0:
            raise ValueError(f"{path}:{line_number}: score must be non-negative")
        qrels.setdefault(query_id, {})[document_id] = score
    if not qrels:
        raise ValueError(f"{path}: no qrels")
    return qrels


# Load a BEIR dataset directory and its requested split.
#
# Arguments:
#   root (str | Path): Dataset root or directory containing corpus.jsonl.
#   split (str): Qrels split name, normally test.
#
# Returns:
#   (BeirDataset): Validated corpus, queries, and qrels.
def load_beir_dataset(root: str | Path, split: str = "test") -> BeirDataset:
    path = Path(root).expanduser().absolute()
    if not (path / "corpus.jsonl").exists() and (path / path.name / "corpus.jsonl").exists():
        path = path / path.name
    corpus = _read_jsonl(path / "corpus.jsonl", ("title", "text"))
    queries = _read_jsonl(path / "queries.jsonl", ("text",))
    qrels = _read_qrels(path / "qrels" / f"{split}.tsv", queries, corpus)
    return BeirDataset(path.name, split, corpus, queries, qrels)


# Select a deterministic qrels-ordered query subset.
def _query_ids(dataset: BeirDataset, limit: int | None) -> list[str]:
    query_ids = sorted(dataset.qrels)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        query_ids = query_ids[:limit]
    return query_ids


# Evaluate a ranking run and attach external dataset accounting.
def _report(dataset: BeirDataset, system: str, run: dict[str, list[str]], k: int, latencies: list[float] | None = None) -> dict[str, Any]:
    qrels = {query_id: dataset.qrels[query_id] for query_id in run}
    evaluation = evaluate_run(run, qrels, k, policy="nonrelevant")
    report: dict[str, Any] = {
        "dataset": dataset.name,
        "split": dataset.split,
        "system": system,
        "corpus_documents": len(dataset.corpus),
        "queries_available": len(dataset.queries),
        "queries_judged": len(dataset.qrels),
        "queries_evaluated": len(run),
        "qrels": sum(len(qrels[query_id]) for query_id in run),
        "k": k,
        "metrics": evaluation,
    }
    if latencies is not None:
        values = sorted(latencies)
        report["latency_ms"] = {
            "median": statistics.median(values) if values else 0.0,
            "p95": values[min(len(values) - 1, max(0, int(0.95 * len(values))))] if values else 0.0,
        }
    return report


# Run the standard in-memory BM25 baseline over external qrels.
#
# Arguments:
#   dataset (BeirDataset): Loaded external dataset.
#   k (int): Rank cutoff.
#   limit (int | None): Optional deterministic query limit.
#
# Returns:
#   (dict[str, Any]): BM25 metrics and dataset accounting.
def evaluate_bm25(dataset: BeirDataset, k: int = 10, limit: int | None = None) -> dict[str, Any]:
    query_ids = _query_ids(dataset, limit)
    rankings = bm25_search_batch([dataset.queries[query_id] for query_id in query_ids], dataset.corpus, k)
    run = dict(zip(query_ids, rankings))
    return _report(dataset, "bm25", run, k)


# Build an HKM index whose source paths preserve BEIR document IDs.
#
# Arguments:
#   dataset (BeirDataset): Loaded external dataset.
#   index_root (str | Path): Destination index root.
#   workers (int): Build workers.
#   max_k (int): Maximum cluster count.
#
# Returns:
#   (None): Builds and drains the index.
def build_hkm_index(dataset: BeirDataset, index_root: str | Path, workers: int = 1, max_k: int = 8) -> None:
    if workers < 1 or max_k < 1:
        raise ValueError("workers and max_k must be positive")
    documents = [
        {"text": text, "metadata": {"source_path": document_id}}
        for document_id, text in dataset.corpus.items()
    ]
    build_search_index_from_documents(str(index_root), documents, num_workers=workers, max_k=max_k)


# Evaluate a published HKM index against external qrels.
#
# Arguments:
#   dataset (BeirDataset): Loaded external dataset.
#   index_root (str | Path): Published HKM index root.
#   mode (str): HKM query mode.
#   k (int): Rank cutoff.
#   limit (int | None): Optional deterministic query limit.
#
# Returns:
#   (dict[str, Any]): HKM metrics and latency accounting.
def evaluate_hkm(
    dataset: BeirDataset,
    index_root: str | Path,
    mode: str = "token",
    k: int = 10,
    limit: int | None = None,
) -> dict[str, Any]:
    if mode not in {"token", "semantic", "hybrid"}:
        raise ValueError("mode must be token, semantic, or hybrid")
    searcher = Searcher.from_index_root(str(index_root))
    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    for query_id in _query_ids(dataset, limit):
        started = time.perf_counter()
        result = searcher.search({"text": dataset.queries[query_id], "mode": mode, "top_k": k})
        latencies.append((time.perf_counter() - started) * 1000.0)
        run[query_id] = [hit.source_path for hit in result.docs]
    return _report(dataset, f"hkm-{mode}", run, k, latencies)


# Run BM25 and optionally build/evaluate HKM on one external dataset.
def run_benchmark(
    dataset_root: str | Path,
    index_root: str | Path | None = None,
    build: bool = False,
    mode: str = "token",
    k: int = 10,
    limit: int | None = None,
    workers: int = 1,
    max_k: int = 8,
) -> dict[str, Any]:
    dataset = load_beir_dataset(dataset_root)
    systems = [evaluate_bm25(dataset, k, limit)]
    if build:
        if index_root is None:
            raise ValueError("index_root is required with build=True")
        build_hkm_index(dataset, index_root, workers, max_k)
    if index_root is not None:
        systems.append(evaluate_hkm(dataset, index_root, mode, k, limit))
    return {
        "dataset": dataset.name,
        "split": dataset.split,
        "systems": systems,
        "external_qrels": True,
        "query_selection": "sorted qrels IDs, optionally truncated by --limit",
    }


# Parse CLI arguments and print one external benchmark report.
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HKM and BM25 on a BEIR dataset.")
    parser.add_argument("dataset_root")
    parser.add_argument("--index-root")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--mode", choices=["token", "semantic", "hybrid"], default="token")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    logs = io.StringIO()
    with contextlib.redirect_stdout(logs):
        report = run_benchmark(
            args.dataset_root,
            args.index_root,
            args.build_index,
            args.mode,
            args.top_k,
            args.limit,
            args.workers,
            args.max_k,
        )
    if logs.getvalue():
        print(logs.getvalue(), end="", file=sys.stderr)
    payload = json.dumps(report, indent=2)
    if args.json_output:
        Path(args.json_output).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
