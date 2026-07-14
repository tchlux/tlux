"""Benchmark language-like memory queries generated from random indexed passages.

The benchmark samples raw source evidence, turns each sample into vague,
specific, conditional, and missing-entity requests, and evaluates an HKM
language-search agent against both the exact document and raw evidence.

Example:
    bin/hkm-random-language-benchmark data/fourth_wing_hkm_index --samples 10
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .agent_benchmark import (
    LANGUAGE_QUERY_MAX_WORDS,
    LMSTUDIO_TIMEOUT,
    LMStudioQueryGenerator,
    _evidence_rank,
    _target_rank,
    sample_passages,
)
from .local_agent import LocalSearchAgent


_QUERY_WORDS = {
    "about", "after", "again", "also", "because", "before", "being",
    "could", "every", "first", "from", "have", "into", "just", "more",
    "other", "over", "said", "some", "than", "that", "their", "there",
    "these", "they", "this", "through", "under", "what", "when", "where",
    "which", "while", "with", "would", "a", "an", "and", "as", "at", "by",
    "for", "in", "is", "it", "of", "on", "or", "the", "to", "was", "were",
    "will", "you", "your", "us", "even", "find", "passage", "scene", "thing",
}
_CONTENT_STOP_WORDS = _QUERY_WORDS | {
    "another", "answer", "around", "been", "beneath", "bond", "called",
    "come", "doesn", "differ", "details", "even", "general", "get",
    "going", "happens", "head", "keep", "know", "like", "looking", "maybe",
    "mean", "much", "near", "need", "part", "please", "point", "present",
    "right", "scene", "separate", "side", "something", "specifically", "start",
    "started", "steady", "still", "than", "then", "think", "thoughts", "those",
    "though", "topic", "toward", "trying", "under", "want", "without", "wording",
    "wrong", "yourself", "slowly", "cracks", "yawn",
}
_STYLE_TEMPLATES = {
    "vague": (
        "A scene involving {a}, {b}, and {c}, with {d}, without relying on exact wording",
        "Something happens around {a} while {b}, {c}, and {d} are present",
    ),
    "specific": (
        "Find the passage where {a}, {b}, and {c} appear near {d}",
        "A passage specifically about {a}, {b}, {c}, and {d}",
    ),
    "conditional": (
        "If {a} appears while {b} is happening near {c}, and {d} is present, find it even if "
        "the surrounding details differ",
        "Find where {a} occurs after {b}, unless {c} is a separate scene and {d} is absent",
    ),
    "missing_entity": (
        "I remember {b} near {c}, {d}, and {e}, but not who or what was involved",
        "I cannot recall the person or object; search for the part with {b}, {c}, {d}, and {e}",
    ),
}
_STYLES = tuple(_STYLE_TEMPLATES)


# One generated language request and its sampled raw-evidence target.
@dataclass(frozen=True)
class LanguageCase:
    sample_id: int
    doc_id: int
    source_path: str
    excerpt: str
    style: str
    query: str
    query_origin: str = "deterministic"


# Keep generated requests within the language-agent query budget.
def _bounded_query(query: str) -> str:
    return " ".join(query.split()[:LANGUAGE_QUERY_MAX_WORDS])


# Return distinctive source words in original order for a memory query.
def _content_terms(text: str, limit: int = 4) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.replace("\u2019", "'"))
    terms: List[tuple[str, int, int]] = []
    seen = set()
    for index, word in enumerate(words):
        normalized = word.lower()
        if len(normalized) < 4 or normalized in _CONTENT_STOP_WORDS:
            continue
        if normalized not in seen:
            specificity = len(normalized)
            if word[:1].isupper() and index:
                specificity += 3
            if "-" in normalized:
                specificity += 2
            terms.append((word, index, specificity))
            seen.add(normalized)
    if terms:
        if len(terms) <= limit:
            return [word for word, _, _ in terms]
        early = terms[: min(3, limit)]
        later = sorted(terms[len(early):], key=lambda item: (-item[2], item[1]))
        selected = early + later[: limit - len(early)]
        return [word for word, _, _ in selected]
    fallback = [word for word in words if word.lower() not in _QUERY_WORDS]
    return fallback[:limit] or words[:limit] or ["event"]


# Generate one deterministic natural-language query from raw evidence.
def deterministic_query(excerpt: str, style: str, variant: int = 0) -> str:
    if style not in _STYLE_TEMPLATES:
        raise ValueError(f"unknown query style: {style}")
    limit = 5 if style == "missing_entity" else 4
    terms = _content_terms(excerpt, limit)
    while len(terms) < limit:
        terms.append(terms[-1])
    template = _STYLE_TEMPLATES[style][variant % len(_STYLE_TEMPLATES[style])]
    return _bounded_query(template.format(
        a=terms[0], b=terms[1], c=terms[2], d=terms[3], e=terms[4] if len(terms) > 4 else terms[-1],
    ))


# Parse a bounded JSON query returned by a compatible language model.
def _parse_model_query(response: str) -> str:
    text = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        value = json.loads(match.group(0)) if match else {}
    if not isinstance(value, dict):
        raise ValueError("memory-query planner returned a non-object")
    query = " ".join(str(value.get("query", "")).split())
    if not query:
        raise ValueError("memory-query planner returned no query")
    return _bounded_query(query)


# Reject a model query that does not retain any supplied evidence term.
def _query_evidence_overlap(query: str, excerpt: str) -> int:
    query_terms = {
        word.lower() for word in re.findall(r"[A-Za-z0-9]+", query)
        if word.lower() not in _QUERY_WORDS
    }
    evidence_terms = {
        word.lower() for word in re.findall(r"[A-Za-z0-9]+", excerpt)
        if word.lower() not in _QUERY_WORDS
    }
    return len(query_terms.intersection(evidence_terms))


# Return true only when a query retains enough supplied evidence to be audited.
def _query_overlaps_evidence(query: str, excerpt: str, minimum: int = 2) -> bool:
    return _query_evidence_overlap(query, excerpt) >= minimum


# Return true when a missing-entity request omits its first distinctive clue.
def _missing_entity_query_is_valid(query: str, excerpt: str) -> bool:
    terms = _content_terms(excerpt, 4)
    if len(terms) < 3:
        return False
    query_terms = {
        word.lower() for word in re.findall(r"[A-Za-z0-9]+", query)
    }
    return terms[0].lower() not in query_terms and _query_overlaps_evidence(query, excerpt)


# Ask LM Studio to turn raw evidence into a grounded memory-style request.
def model_query(
    generator: LMStudioQueryGenerator,
    excerpt: str,
    style: str,
) -> str:
    memory_query = getattr(generator, "generate_memory_query", None)
    if not callable(memory_query):
        raise TypeError("query generator must implement generate_memory_query")
    value = memory_query(excerpt, style)
    if isinstance(value, dict):
        value = value.get("query", "")
    query = " ".join(str(value).split())
    if not _query_overlaps_evidence(query, excerpt):
        raise ValueError("memory-query planner did not quote supplied evidence")
    if style == "missing_entity" and not _missing_entity_query_is_valid(query, excerpt):
        raise ValueError("missing-entity planner restored the omitted clue")
    return _bounded_query(query)


# Build reproducible cases from a random sample and selected query styles.
def build_cases(samples: Sequence[Any], styles: Sequence[str]) -> List[LanguageCase]:
    if not styles:
        raise ValueError("at least one query style is required")
    unknown = [style for style in styles if style not in _STYLE_TEMPLATES]
    if unknown:
        raise ValueError(f"unknown query styles: {', '.join(unknown)}")
    cases: List[LanguageCase] = []
    for sample in samples:
        for index, style in enumerate(styles):
            cases.append(LanguageCase(
                sample_id=int(sample.sample_id),
                doc_id=int(sample.doc_id),
                source_path=str(sample.source_path),
                excerpt=str(sample.excerpt),
                style=style,
                query=deterministic_query(
                    str(sample.excerpt),
                    style,
                    int(sample.sample_id) + index,
                ),
            ))
    return cases


# Return summary statistics for ranks and request timings.
def _metrics(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    ranks = [row.get(key) for row in rows]
    reciprocal = [1.0 / rank if rank else 0.0 for rank in ranks]
    return {
        "recall_at_k": sum(rank is not None for rank in ranks) / len(ranks) if ranks else 0.0,
        "precision_at_1": sum(rank == 1 for rank in ranks) / len(ranks) if ranks else 0.0,
        "mrr": sum(reciprocal) / len(reciprocal) if reciprocal else 0.0,
    }


# Return median and p95 values without hiding slow tail requests.
def _timings(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    values = sorted(float(row.get(key, 0.0)) for row in rows)
    if not values:
        return {"median_ms": 0.0, "p95_ms": 0.0}
    return {
        "median_ms": statistics.median(values),
        "p95_ms": values[min(len(values) - 1, max(0, math.ceil(0.95 * len(values)) - 1))],
    }


# Return true when every case has rank-one evidence and no runner errors.
def _evidence_gate_failed(report: Dict[str, Any]) -> bool:
    metrics = report.get("evidence", {})
    return bool(report.get("errors")) or any(
        float(metrics.get(name, 0.0)) < 1.0
        for name in ("recall_at_k", "precision_at_1", "mrr")
    )


# Run generated cases through one persistent language agent.
def run_benchmark(
    agent: LocalSearchAgent,
    cases: Iterable[LanguageCase],
    top_k: int,
) -> Dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        try:
            # Call the runner directly so evidence ranking sees full HKM hits.
            run = agent.runner.run(case.query, agent.searcher, top_k)
            result = run["result"]
            row = {
                "sample_id": case.sample_id,
                "doc_id": case.doc_id,
                "source_path": case.source_path,
                "style": case.style,
                "excerpt": case.excerpt,
                "query": case.query,
                "query_origin": case.query_origin,
                "target_rank": _target_rank(result, case.doc_id),
                "evidence_rank": _evidence_rank(result, case.excerpt, agent.searcher),
                "agent_ms": float(run.get("agent_ms", 0.0)),
                "search_ms": float(run.get("search_ms", 0.0)),
                "queries": run.get("queries", []),
                "antipatterns": run.get("antipatterns", []),
                "rounds": int(run.get("rounds", 0)),
                "top_docs": [int(hit.doc_id) for hit in result.docs[:top_k]],
                "top_previews": [str(hit.preview_text)[:240] for hit in result.docs[:top_k]],
                "wall_ms": (time.perf_counter() - started) * 1000.0,
            }
            rows.append(row)
        except Exception as exc:
            errors.append({"sample_id": case.sample_id, "style": case.style, "error": str(exc)})
    evidence_metrics = _metrics(rows, "evidence_rank")
    return {
        "cases": len(rows),
        "errors": errors,
        "target": _metrics(rows, "target_rank"),
        "evidence": evidence_metrics,
        "evidence_gate_passed": not errors and all(
            evidence_metrics[name] == 1.0
            for name in ("recall_at_k", "precision_at_1", "mrr")
        ),
        "by_style": {
            style: _metrics([row for row in rows if row["style"] == style], "evidence_rank")
            for style in _STYLES
            if any(row["style"] == style for row in rows)
        },
        "by_query_origin": {
            origin: _metrics(
                [row for row in rows if row["query_origin"] == origin],
                "evidence_rank",
            )
            for origin in ("lm", "deterministic", "deterministic_fallback")
            if any(row["query_origin"] == origin for row in rows)
        },
        "latency_ms": {
            "agent": _timings(rows, "agent_ms"),
            "search": _timings(rows, "search_ms"),
            "wall": _timings(rows, "wall_ms"),
        },
        "rows": rows,
    }


# Parse CLI options, sample the corpus, and emit JSON for auditability.
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark random language-like HKM queries.")
    parser.add_argument("index_root")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-tokens", type=int, default=36)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--styles", default=",".join(_STYLES))
    parser.add_argument("--query-source", choices=["deterministic", "lm"], default="deterministic")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=LMSTUDIO_TIMEOUT)
    parser.add_argument("--model-first", action="store_true")
    parser.add_argument("--always-refine", action="store_true", help="force one LM inspection/refinement round per query")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--require-evidence", action="store_true", help="fail unless all evidence metrics are 1.0")
    parser.add_argument("--json-output")
    args = parser.parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    styles = [style.strip() for style in args.styles.split(",") if style.strip()]
    from ..search.searcher import Searcher
    searcher = Searcher.from_index_root(args.index_root)
    samples = sample_passages(searcher, args.samples, args.seed, args.max_tokens)
    cases = build_cases(samples, styles)
    planner_errors: List[Dict[str, Any]] = []
    if args.query_source == "lm":
        generator = LMStudioQueryGenerator(args.base_url, args.model, args.timeout)
        replaced: List[LanguageCase] = []
        for case in cases:
            origin = "lm"
            try:
                query = model_query(generator, case.excerpt, case.style)
            except Exception as exc:
                planner_errors.append({"sample_id": case.sample_id, "style": case.style, "error": str(exc)})
                query = case.query
                origin = "deterministic_fallback"
            replaced.append(LanguageCase(
                case.sample_id,
                case.doc_id,
                case.source_path,
                case.excerpt,
                case.style,
                query,
                origin,
            ))
        cases = replaced
    agent = LocalSearchAgent(
        args.index_root,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        mode="semantic",
        deterministic_first=not args.model_first,
        language_query=True,
        always_refine=args.always_refine,
    )
    if args.warmup:
        agent.warmup()
    report = run_benchmark(agent, cases, args.top_k)
    report.update({
        "index_root": str(Path(args.index_root).expanduser().absolute()),
        "seed": args.seed,
        "samples_requested": args.samples,
        "sample_count": len(samples),
        "styles": styles,
        "query_source": args.query_source,
        "planner_errors": planner_errors,
        "planner_error_count": len(planner_errors),
        "model_query_successes": sum(row["query_origin"] == "lm" for row in report["rows"]),
        "model_query_fallbacks": sum(row["query_origin"] == "deterministic_fallback" for row in report["rows"]),
        "model_query_success_rate": (
            sum(row["query_origin"] == "lm" for row in report["rows"]) / len(report["rows"])
            if report["rows"] else 0.0
        ),
        "model_first": args.model_first,
    })
    encoded = json.dumps(report, ensure_ascii=True, indent=2)
    if args.json_output:
        Path(args.json_output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if args.require_evidence and _evidence_gate_failed(report):
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
