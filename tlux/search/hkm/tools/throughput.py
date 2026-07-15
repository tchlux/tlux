"""Measure concurrent search throughput with one searcher per worker process."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import threading
import time
from pathlib import Path
from typing import Any

from ..search.searcher import Searcher, audit_index
from .benchmark import _summary


_WORKER_SEARCHER: Searcher | None = None
_WORKER_MODE = "hybrid"
_WORKER_TOP_K = 10
_THREAD_LOCAL = threading.local()


# Load one immutable index view in each worker process.
#
# Arguments:
#   index_root (str): Index directory.
#   mode (str): Search mode.
#   top_k (int): Number of results to request.
#
# Returns:
#   (None): Stores process-local worker configuration.
def _init_worker(index_root: str, mode: str, top_k: int) -> None:
    global _WORKER_SEARCHER, _WORKER_MODE, _WORKER_TOP_K
    _WORKER_SEARCHER = Searcher.from_index_root(index_root)
    _WORKER_MODE = mode
    _WORKER_TOP_K = top_k


# Load one independent index view in each fallback thread.
#
# Arguments:
#   index_root (str): Index directory.
#   mode (str): Search mode.
#   top_k (int): Number of results to request.
#
# Returns:
#   (None): Stores thread-local worker configuration.
def _init_thread(index_root: str, mode: str, top_k: int) -> None:
    _THREAD_LOCAL.searcher = Searcher.from_index_root(index_root)
    _THREAD_LOCAL.mode = mode
    _THREAD_LOCAL.top_k = top_k


# Execute one timed search inside a worker process.
#
# Arguments:
#   text (str): Query text.
#
# Returns:
#   (dict[str, Any]): Query result count and elapsed milliseconds.
def _search_one(text: str) -> dict[str, Any]:
    searcher = _WORKER_SEARCHER or getattr(_THREAD_LOCAL, "searcher", None)
    if searcher is None:
        raise RuntimeError("throughput worker was not initialized")
    mode = _WORKER_MODE if _WORKER_SEARCHER is not None else _THREAD_LOCAL.mode
    top_k = _WORKER_TOP_K if _WORKER_SEARCHER is not None else _THREAD_LOCAL.top_k
    started = time.perf_counter()
    result = searcher.search({"text": text, "mode": mode, "top_k": top_k})
    return {
        "query": text,
        "results": result.count,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
    }


# Measure concurrent search throughput using isolated worker processes.
#
# Arguments:
#   index_root (str): Index directory.
#   queries (list[str]): Queries assigned round-robin to requests.
#   requests (int): Number of timed requests.
#   concurrency (int): Number of worker processes.
#   top_k (int): Number of results to request.
#   mode (str): Search mode.
#
# Returns:
#   (dict[str, Any]): Throughput, latency, and error evidence.
def measure_throughput(
    index_root: str,
    queries: list[str],
    requests: int = 32,
    concurrency: int = 1,
    top_k: int = 10,
    mode: str = "hybrid",
) -> dict[str, Any]:
    if not queries:
        raise ValueError("queries must not be empty")
    if requests < 1 or concurrency < 1 or top_k < 1:
        raise ValueError("requests, concurrency, and top_k must be positive")
    if mode not in {"token", "semantic", "hybrid"}:
        raise ValueError("mode must be token, semantic, or hybrid")
    root = audit_index(index_root)
    request_queries = [queries[index % len(queries)] for index in range(requests)]
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=concurrency,
            mp_context=context,
            initializer=_init_worker,
            initargs=(str(root), mode, top_k),
        )
        executor = "process"
    except (NotImplementedError, OSError, PermissionError):
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency,
            initializer=_init_thread,
            initargs=(str(root), mode, top_k),
        )
        executor = "thread"
    with pool:
        futures = [pool.submit(_search_one, query) for query in request_queries]
        for query, future in zip(request_queries, futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append({"query": query, "error": str(exc)})
    wall_ms = (time.perf_counter() - started) * 1000.0
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "index_root": str(root),
        "requests": requests,
        "completed": len(rows),
        "errors": errors,
        "concurrency": concurrency,
        "executor": executor,
        "mode": mode,
        "top_k": top_k,
        "wall_ms": wall_ms,
        "throughput_qps": len(rows) / (wall_ms / 1000.0) if wall_ms else 0.0,
        "latency_ms": _summary(latencies),
        "rows": rows,
    }


# Parse CLI arguments and print one JSON throughput report.
#
# Arguments:
#   None.
#
# Returns:
#   (None): Prints the report.
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure concurrent HKM search throughput.")
    parser.add_argument("index_root")
    parser.add_argument("--query", action="append", default=["0 1 2 3"])
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", choices=["token", "semantic", "hybrid"], default="hybrid")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    report = measure_throughput(
        args.index_root,
        args.query,
        args.requests,
        args.concurrency,
        args.top_k,
        args.mode,
    )
    payload = json.dumps(report, indent=2)
    if args.json_output:
        Path(args.json_output).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
