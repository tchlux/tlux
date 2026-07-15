"""Load strict source-blind query and graded-qrels manifests.

Each non-empty line is one JSON object. The loader rejects incomplete
provenance labels, duplicate IDs, invalid grades, and answerability conflicts.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .quality_benchmark import DocumentId, JudgmentPolicy, QueryCase, evaluate_retriever


# Return a required non-empty string field from one manifest record.
def _string(record: dict[str, Any], name: str, line_number: int) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {name} must be a non-empty string")
    return value.strip()


# Return a required boolean field from one manifest record.
def _boolean(record: dict[str, Any], name: str, line_number: int) -> bool:
    value = record.get(name)
    if type(value) is not bool:
        raise ValueError(f"line {line_number}: {name} must be a boolean")
    return value


# Parse one JSON object into a validated query case.
def _case(record: dict[str, Any], line_number: int) -> QueryCase:
    query_id = _string(record, "query_id", line_number)
    text = _string(record, "text", line_number)
    category = _string(record, "category", line_number)
    natural = _boolean(record, "natural", line_number)
    source_blind = _boolean(record, "source_blind", line_number)
    answerable = _boolean(record, "answerable", line_number)
    facets = record.get("required_facets", [])
    relevance = record.get("relevance", {})
    if not isinstance(facets, list) or not all(isinstance(item, str) and item.strip() for item in facets):
        raise ValueError(f"line {line_number}: required_facets must contain non-empty strings")
    if not isinstance(relevance, dict):
        raise ValueError(f"line {line_number}: relevance must be an object")
    cleaned_relevance: dict[DocumentId, float] = {}
    for document_id, grade in relevance.items():
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"line {line_number}: relevance IDs must be non-empty strings")
        if isinstance(grade, bool) or not isinstance(grade, (int, float)) or not math.isfinite(float(grade)):
            raise ValueError(f"line {line_number}: relevance grades must be finite numbers")
        if float(grade) < 0:
            raise ValueError(f"line {line_number}: relevance grades must be non-negative")
        cleaned_relevance[document_id] = float(grade)
    positive = any(grade > 0 for grade in cleaned_relevance.values())
    if answerable != positive:
        raise ValueError(
            f"line {line_number}: answerable must match whether relevance has a positive grade"
        )
    return QueryCase(
        query_id=query_id,
        text=text,
        natural=natural,
        source_blind=source_blind,
        category=category,
        required_facets=frozenset(item.strip() for item in facets),
        relevance=cleaned_relevance,
        answerable=answerable,
    )


# Load one JSONL query/qrels manifest with strict case validation.
#
# Arguments:
#   path (str | Path): JSONL manifest path.
#
# Returns:
#   (tuple[QueryCase, ...]): Ordered, validated query cases.
def load_query_manifest(path: str | Path) -> tuple[QueryCase, ...]:
    manifest_path = Path(path)
    cases: list[QueryCase] = []
    seen: set[str] = set()
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read query manifest {manifest_path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number}: record must be a JSON object")
        case = _case(record, line_number)
        if case.query_id in seen:
            raise ValueError(f"line {line_number}: duplicate query_id {case.query_id!r}")
        seen.add(case.query_id)
        cases.append(case)
    if not cases:
        raise ValueError("query manifest contains no records")
    return tuple(cases)


# Summarize provenance, answerability, category, and qrels coverage.
#
# Arguments:
#   cases (Iterable[QueryCase]): Validated query cases.
#
# Returns:
#   (dict[str, Any]): Manifest composition and judgment coverage.
def query_manifest_report(cases: Iterable[QueryCase]) -> dict[str, Any]:
    rows = tuple(cases)
    categories = Counter(case.category for case in rows)
    positive = sum(any(float(grade) > 0 for grade in case.relevance.values()) for case in rows)
    return {
        "queries": len(rows),
        "natural": sum(case.natural for case in rows),
        "source_blind": sum(case.source_blind for case in rows),
        "answerable": sum(case.answerable for case in rows),
        "unanswerable": sum(not case.answerable for case in rows),
        "positive_qrels": positive,
        "categories": dict(sorted(categories.items())),
    }


# Evaluate a retriever against a validated external query manifest.
#
# Arguments:
#   retriever (Callable[[str], Sequence[DocumentId]]): Query-to-ranking function.
#   path (str | Path): JSONL manifest path.
#   k (int): Rank cutoff.
#   policy (JudgmentPolicy): Unjudged-result policy.
#
# Returns:
#   (dict[str, Any]): Manifest composition and retrieval metrics.
def evaluate_query_manifest(
    retriever: Callable[[str], Sequence[DocumentId]],
    path: str | Path,
    k: int = 10,
    policy: JudgmentPolicy = "nonrelevant",
) -> dict[str, Any]:
    cases = load_query_manifest(path)
    report = evaluate_retriever(retriever, cases, k, policy)
    return {"manifest": query_manifest_report(cases), "evaluation": report}


__all__ = [
    "evaluate_query_manifest",
    "load_query_manifest",
    "query_manifest_report",
]
