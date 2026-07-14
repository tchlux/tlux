"""Evaluate language-query search against the checked-in concept-group benchmark.

The evaluator keeps the relevance oracle in ``plan/benchmark_language_queries.md``
and only expects the persistent agent JSON response contract.  It therefore
works across rebuilt indexes whose document IDs are different.

Example:
    bin/hkm-language-benchmark data/fourth_wing_hkm_index --warmup
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .agent_benchmark import LMSTUDIO_TIMEOUT
from .local_agent import LocalSearchAgent


_CASE_ROW = re.compile(r"^\|\s*(LQ-\d+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$")


# Normalize text for alias matching while preserving word boundaries.
def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# Collapse the small set of suffixes used by the hand-reviewed aliases.
def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


# Return true when an alias appears as a whole word or normalized phrase.
def _contains_alias(text: str, alias: str) -> bool:
    haystack = [_stem(token) for token in _tokens(text)]
    needle = [_stem(token) for token in _tokens(alias)]
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(
        haystack[index:index + width] == needle
        for index in range(len(haystack) - width + 1)
    )


# Parse comma-separated aliases from one markdown table cell.
def _aliases(cell: str) -> List[str]:
    cleaned = cell.replace("`", "").strip()
    if not cleaned or cleaned.lower() == "none" or cleaned.lower().startswith("do not require"):
        return []
    return [part.strip() for part in cleaned.split(",") if part.strip()]


# Parse the benchmark table into ID, query, required groups, and antipatterns.
def parse_cases(markdown: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for line in markdown.splitlines():
        match = _CASE_ROW.match(line)
        if match is None or match.group(1) == "ID":
            continue
        case_id, query, required, antipatterns = match.groups()
        groups = [_aliases(group) for group in required.split("+")]
        cases.append({
            "id": case_id,
            "query": query,
            "groups": [group for group in groups if group],
            "antipatterns": _aliases(antipatterns),
        })
    if not cases:
        raise ValueError("benchmark contains no LQ table rows")
    return cases


# Combine the fields exposed by the JSONL agent into searchable evidence text.
def _hit_text(hit: Dict[str, Any]) -> str:
    document = hit.get("document") or {}
    return " ".join(str(value) for value in (
        hit.get("preview_text", ""),
        hit.get("source_path", ""),
        document.get("document_preview", "") if isinstance(document, dict) else "",
    ))


# Return aliases represented by one hit.
def _hit_groups(hit: Dict[str, Any], groups: Sequence[Sequence[str]]) -> List[int]:
    text = _hit_text(hit)
    return [
        index for index, aliases in enumerate(groups)
        if any(_contains_alias(text, alias) for alias in aliases)
    ]


# Return the nearest-rank p95 value used by the other HKM benchmarks.
def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


# Evaluate one agent response against one concept-group case.
def evaluate_case(case: Dict[str, Any], response: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    hits = response.get("docs", [])[:top_k]
    groups = case["groups"]
    hit_groups = [_hit_groups(hit, groups) for hit in hits]
    covered = {group for matched in hit_groups for group in matched}
    coherent = bool(hit_groups) and (
        len(groups) <= 1 or any(len(matched) >= 2 for matched in hit_groups)
    )
    anti_rank = None
    for rank, hit in enumerate(hits, 1):
        if any(_contains_alias(_hit_text(hit), alias) for alias in case["antipatterns"]):
            anti_rank = rank
            break
    return {
        "id": case["id"],
        "query": case["query"],
        "group_coverage_at_5": len(covered) / len(groups) if groups else 0.0,
        "coherent_hit_at_5": coherent,
        "top_hit_groups": len(hit_groups[0]) if hit_groups else 0,
        "anti_rank": anti_rank,
        "grounded": bool(response.get("grounded")) and bool(hits),
        "doc_count": len(hits),
        "agent_ms": float(response.get("agent_ms", 0.0)),
        "search_ms": float(response.get("search_ms", 0.0)),
        "queries": response.get("queries", []),
        "antipatterns": response.get("antipatterns", []),
        "rounds": int(response.get("rounds", 0)),
        "hit_groups": hit_groups,
        "docs": hits,
    }


# Summarize per-case results and apply the benchmark's all-case gate.
def summarize(results: Sequence[Dict[str, Any]], wall_ms: Sequence[float]) -> Dict[str, Any]:
    coverage = [float(result["group_coverage_at_5"]) for result in results]
    coherent = sum(bool(result["coherent_hit_at_5"]) for result in results)
    wall = list(wall_ms)
    return {
        "cases": len(results),
        "passed": sum(
            result["group_coverage_at_5"] == 1.0 and result["coherent_hit_at_5"]
            for result in results
        ),
        "gate_passed": all(
            result["group_coverage_at_5"] == 1.0 and result["coherent_hit_at_5"]
            for result in results
        ),
        "mean_group_coverage_at_5": statistics.fmean(coverage) if coverage else 0.0,
        "min_group_coverage_at_5": min(coverage) if coverage else 0.0,
        "coherent_hits": coherent,
        "wall_ms": {
            "median": statistics.median(wall) if wall else 0.0,
            "p95": _p95(wall),
        },
        "agent_ms": {
            "median": statistics.median(
                result["agent_ms"] for result in results
            ) if results else 0.0,
            "p95": _p95([result["agent_ms"] for result in results]),
        },
        "search_ms": {
            "median": statistics.median(
                result["search_ms"] for result in results
            ) if results else 0.0,
            "p95": _p95([result["search_ms"] for result in results]),
        },
    }


# Run benchmark cases through one persistent agent and return an auditable report.
def run_benchmark(
    agent: LocalSearchAgent,
    cases: Iterable[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    wall_ms: List[float] = []
    for case in cases:
        started = time.perf_counter()
        response = agent.run(case["query"], top_k)
        elapsed = (time.perf_counter() - started) * 1000.0
        wall_ms.append(elapsed)
        result = evaluate_case(case, response, top_k)
        result["wall_ms"] = elapsed
        results.append(result)
    return {"summary": summarize(results, wall_ms), "results": results}


# Parse CLI arguments, run the evaluator, and emit one JSON report.
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate language-query concept groups.")
    parser.add_argument("index_root")
    parser.add_argument("--benchmark", default="plan/benchmark_language_queries.md")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=LMSTUDIO_TIMEOUT)
    parser.add_argument("--tool-mode", choices=["hybrid", "token", "semantic"], default="semantic")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-first", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--jsonl", help="write one raw case result per line")
    parser.add_argument(
        "--require-gate",
        action="store_true",
        help="exit nonzero when the all-case gate fails",
    )
    args = parser.parse_args()
    cases = parse_cases(Path(args.benchmark).read_text(encoding="utf-8"))
    agent = LocalSearchAgent(
        args.index_root,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        mode=args.tool_mode,
        deterministic_first=not args.model_first,
        language_query=True,
    )
    if args.warmup:
        agent.warmup()
    report = run_benchmark(agent, cases, args.top_k)
    if args.jsonl:
        with Path(args.jsonl).open("w", encoding="utf-8") as handle:
            for result in report["results"]:
                handle.write(json.dumps(result, ensure_ascii=True) + "\n")
    json.dump(report, sys.stdout, ensure_ascii=True, indent=2)
    sys.stdout.write("\n")
    if args.require_gate and not report["summary"]["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
