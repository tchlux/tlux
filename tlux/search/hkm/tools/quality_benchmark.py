"""Compact deterministic retrieval-quality metrics, diagnostics, and fixtures.

The module uses only the standard library and NumPy and is useful for small
offline regressions as well as repeatable comparisons of retrieval pipelines.

Example:
    fixture = make_fixture()
    runs = run_baselines(fixture.queries[0].text, fixture.document_map, 5)
    report = evaluate_run(runs["bm25"], fixture.qrels, 5)
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import re
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Callable, Hashable, Iterable, Literal, Mapping, Sequence, TypeAlias

import numpy as np


DocumentId: TypeAlias = Hashable
JudgmentPolicy: TypeAlias = Literal["ignore", "nonrelevant", "error"]
Alternative: TypeAlias = Literal["two-sided", "greater", "less"]


# A query and its metadata, including graded judgments and required evidence facets.
@dataclass(frozen=True)
class QueryCase:
    query_id: str
    text: str
    natural: bool
    source_blind: bool
    category: str
    required_facets: frozenset[str] = frozenset()
    relevance: Mapping[DocumentId, float] = field(default_factory=dict)
    answerable: bool = True

    # Return metadata in a format suitable for JSON reports.
    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "natural": self.natural,
            "source_blind": self.source_blind,
            "category": self.category,
            "required_facets": sorted(self.required_facets),
            "answerable": self.answerable,
        }


# A compact document record used by the heterogeneous fixture and baselines.
@dataclass(frozen=True)
class FixtureDocument:
    doc_id: str
    text: str
    category: str
    facets: frozenset[str]
    language: str = "en"
    format: str = "text"
    metadata: Mapping[str, str] = field(default_factory=dict)


# A deterministic collection of documents and annotated queries.
@dataclass(frozen=True)
class Fixture:
    documents: tuple[FixtureDocument, ...]
    queries: tuple[QueryCase, ...]

    # Return documents keyed by stable identifier.
    @property
    def document_map(self) -> dict[str, str]:
        return {document.doc_id: document.text for document in self.documents}

    # Return document facets keyed by stable identifier.
    @property
    def doc_facets(self) -> dict[str, frozenset[str]]:
        return {document.doc_id: document.facets for document in self.documents}

    # Return query judgments keyed by stable identifier.
    @property
    def qrels(self) -> dict[str, Mapping[DocumentId, float]]:
        return {query.query_id: query.relevance for query in self.queries}


# Validate a cutoff and return a safe integer cutoff.
#
# Parameters:
#   k (int): Requested rank cutoff.
#   size (int): Number of available results.
#
# Returns:
#   (int): Validated cutoff.
#
def _cutoff(k: int | None, size: int) -> int:
    if k is None:
        return size
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    return k


# Validate non-negative finite graded relevance values.
#
# Parameters:
#   relevance (Mapping[DocumentId, float]): Graded document judgments.
#
# Returns:
#   (None): Raises ValueError for invalid judgments.
#
def _validate_relevance(relevance: Mapping[DocumentId, float]) -> None:
    for document_id, grade in relevance.items():
        if not np.isfinite(float(grade)) or float(grade) < 0:
            raise ValueError(f"relevance grade must be finite and non-negative: {document_id!r}")


# Prepare ranked grades while making the unjudged policy explicit.
#
# Parameters:
#   retrieved (Sequence[DocumentId]): Ranked result identifiers.
#   relevance (Mapping[DocumentId, float]): Graded judgments.
#   policy (JudgmentPolicy): Ignore, treat as nonrelevant, or reject unjudged IDs.
#   k (int | None): Optional cutoff.
#
# Returns:
#   (tuple[list[float], int, int, int]): Grades, judged count, unjudged count,
#       and unique result count.
#
def _ranked_grades(
    retrieved: Sequence[DocumentId],
    relevance: Mapping[DocumentId, float],
    policy: JudgmentPolicy,
    k: int | None,
) -> tuple[list[float], int, int, int]:
    if policy not in {"ignore", "nonrelevant", "error"}:
        raise ValueError("policy must be ignore, nonrelevant, or error")
    _validate_relevance(relevance)
    limit = _cutoff(k, len(retrieved))
    grades: list[float] = []
    judged = 0
    unjudged = 0
    seen: set[DocumentId] = set()
    seen_judged: set[DocumentId] = set()
    for document_id in retrieved[:limit]:
        is_judged = document_id in relevance
        judged += int(is_judged)
        unjudged += int(not is_judged)
        if not is_judged and policy == "error":
            raise ValueError(f"unjudged result: {document_id!r}")
        if not is_judged and policy == "ignore":
            seen.add(document_id)
            continue
        # Repeated IDs consume a rank but cannot earn relevance twice.
        grades.append(float(relevance.get(document_id, 0.0)) if document_id not in seen_judged else 0.0)
        seen.add(document_id)
        if is_judged:
            seen_judged.add(document_id)
    return grades, judged, unjudged, len(seen)


# Return accounting for judged, unjudged, and duplicate results.
#
# Parameters:
#   retrieved (Sequence[DocumentId]): Ranked result identifiers.
#   relevance (Mapping[DocumentId, float]): Graded judgments.
#   k (int | None): Optional cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#
# Returns:
#   (dict[str, float | int | str]): Counts and rates for the evaluated prefix.
#
def judgment_accounting(
    retrieved: Sequence[DocumentId],
    relevance: Mapping[DocumentId, float],
    k: int | None = None,
    policy: JudgmentPolicy = "nonrelevant",
) -> dict[str, float | int | str]:
    _, judged, unjudged, unique = _ranked_grades(retrieved, relevance, policy, k)
    total = len(retrieved[:_cutoff(k, len(retrieved))])
    return {
        "policy": policy,
        "retrieved": total,
        "judged": judged,
        "unjudged": unjudged,
        "judged_fraction": judged / total if total else 0.0,
        "unjudged_fraction": unjudged / total if total else 0.0,
        "unique_retrieved": unique,
        "duplicates": total - unique,
    }


# Compute binary precision at a rank cutoff.
#
# Parameters:
#   retrieved (Sequence[DocumentId]): Ranked result identifiers.
#   relevance (Mapping[DocumentId, float]): Graded judgments; positive is relevant.
#   k (int): Rank cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#
# Returns:
#   (float): Precision at k.
#
def precision_at_k(
    retrieved: Sequence[DocumentId],
    relevance: Mapping[DocumentId, float],
    k: int,
    policy: JudgmentPolicy = "nonrelevant",
) -> float:
    grades, _, _, _ = _ranked_grades(retrieved, relevance, policy, k)
    denominator = len(grades) if policy == "ignore" else k
    return sum(grade > 0 for grade in grades) / denominator if denominator else 0.0


# Compute binary recall at a rank cutoff.
#
# Parameters:
#   retrieved (Sequence[DocumentId]): Ranked result identifiers.
#   relevance (Mapping[DocumentId, float]): Graded judgments; positive is relevant.
#   k (int): Rank cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#
# Returns:
#   (float): Recall at k.
#
def recall_at_k(
    retrieved: Sequence[DocumentId],
    relevance: Mapping[DocumentId, float],
    k: int,
    policy: JudgmentPolicy = "nonrelevant",
) -> float:
    grades, _, _, _ = _ranked_grades(retrieved, relevance, policy, k)
    total = sum(float(grade) > 0 for grade in relevance.values())
    return sum(grade > 0 for grade in grades) / total if total else 0.0


# Compute normalized graded precision using clipped gain in [0, 1].
#
# Parameters:
#   retrieved (Sequence[DocumentId]): Ranked result identifiers.
#   relevance (Mapping[DocumentId, float]): Graded judgments.
#   k (int): Rank cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#   max_grade (float | None): Maximum grade used to normalize gain.
#
# Returns:
#   (float): Graded precision at k.
#
def graded_precision_at_k(
    retrieved: Sequence[DocumentId],
    relevance: Mapping[DocumentId, float],
    k: int,
    policy: JudgmentPolicy = "nonrelevant",
    max_grade: float | None = None,
) -> float:
    grades, _, _, _ = _ranked_grades(retrieved, relevance, policy, k)
    if max_grade is not None and max_grade <= 0:
        raise ValueError("max_grade must be positive")
    ceiling = max_grade if max_grade is not None else max(
        [float(value) for value in relevance.values()] or [0.0]
    )
    if ceiling <= 0:
        return 0.0
    denominator = len(grades) if policy == "ignore" else k
    gain = sum(min(1.0, grade / ceiling) for grade in grades)
    return gain / denominator if denominator else 0.0


# Compute graded recall as retrieved gain divided by all judged relevant gain.
#
# Parameters:
#   retrieved (Sequence[DocumentId]): Ranked result identifiers.
#   relevance (Mapping[DocumentId, float]): Graded judgments.
#   k (int): Rank cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#   max_grade (float | None): Maximum grade used to normalize gain.
#
# Returns:
#   (float): Graded recall at k.
#
def graded_recall_at_k(
    retrieved: Sequence[DocumentId],
    relevance: Mapping[DocumentId, float],
    k: int,
    policy: JudgmentPolicy = "nonrelevant",
    max_grade: float | None = None,
) -> float:
    grades, _, _, _ = _ranked_grades(retrieved, relevance, policy, k)
    if max_grade is not None and max_grade <= 0:
        raise ValueError("max_grade must be positive")
    ceiling = max_grade if max_grade is not None else max(
        [float(value) for value in relevance.values()] or [0.0]
    )
    if ceiling <= 0:
        return 0.0
    retrieved_gain = sum(min(1.0, grade / ceiling) for grade in grades)
    total_gain = sum(min(1.0, float(value) / ceiling) for value in relevance.values())
    return retrieved_gain / total_gain if total_gain else 0.0


# Compute average precision for one ranked result list.
#
# Parameters:
#   retrieved (Sequence[DocumentId]): Ranked result identifiers.
#   relevance (Mapping[DocumentId, float]): Graded judgments; positive is relevant.
#   k (int | None): Optional rank cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#
# Returns:
#   (float): Average precision.
#
def average_precision(
    retrieved: Sequence[DocumentId],
    relevance: Mapping[DocumentId, float],
    k: int | None = None,
    policy: JudgmentPolicy = "nonrelevant",
) -> float:
    grades, _, _, _ = _ranked_grades(retrieved, relevance, policy, k)
    total = sum(float(value) > 0 for value in relevance.values())
    if not total:
        return 0.0
    hits = 0
    score = 0.0
    for rank, grade in enumerate(grades, 1):
        if grade > 0:
            hits += 1
            score += hits / rank
    return score / total


# Compute mean average precision across query IDs.
#
# Parameters:
#   run (Mapping[str, Sequence[DocumentId]]): Ranked results by query ID.
#   qrels (Mapping[str, Mapping[DocumentId, float]]): Judgments by query ID.
#   k (int | None): Optional rank cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#
# Returns:
#   (float): Mean average precision.
#
def mean_average_precision(
    run: Mapping[str, Sequence[DocumentId]],
    qrels: Mapping[str, Mapping[DocumentId, float]],
    k: int | None = None,
    policy: JudgmentPolicy = "nonrelevant",
) -> float:
    if not qrels:
        return 0.0
    return float(np.mean([average_precision(run.get(query_id, ()), judgments, k, policy)
                          for query_id, judgments in qrels.items()]))


# Compute reciprocal rank for one ranked result list.
#
# Parameters:
#   retrieved (Sequence[DocumentId]): Ranked result identifiers.
#   relevance (Mapping[DocumentId, float]): Graded judgments; positive is relevant.
#   k (int | None): Optional rank cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#
# Returns:
#   (float): Reciprocal rank.
#
def reciprocal_rank(
    retrieved: Sequence[DocumentId],
    relevance: Mapping[DocumentId, float],
    k: int | None = None,
    policy: JudgmentPolicy = "nonrelevant",
) -> float:
    grades, _, _, _ = _ranked_grades(retrieved, relevance, policy, k)
    for rank, grade in enumerate(grades, 1):
        if grade > 0:
            return 1.0 / rank
    return 0.0


# Compute mean reciprocal rank across query IDs.
#
# Parameters:
#   run (Mapping[str, Sequence[DocumentId]]): Ranked results by query ID.
#   qrels (Mapping[str, Mapping[DocumentId, float]]): Judgments by query ID.
#   k (int | None): Optional rank cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#
# Returns:
#   (float): Mean reciprocal rank.
#
def mean_reciprocal_rank(
    run: Mapping[str, Sequence[DocumentId]],
    qrels: Mapping[str, Mapping[DocumentId, float]],
    k: int | None = None,
    policy: JudgmentPolicy = "nonrelevant",
) -> float:
    if not qrels:
        return 0.0
    values = [reciprocal_rank(run.get(query_id, ()), judgments, k, policy)
              for query_id, judgments in qrels.items()]
    return float(np.mean(values))


# Compute normalized discounted cumulative gain for one ranked result list.
#
# Parameters:
#   retrieved (Sequence[DocumentId]): Ranked result identifiers.
#   relevance (Mapping[DocumentId, float]): Graded judgments.
#   k (int): Rank cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#
# Returns:
#   (float): nDCG at k using gain 2**grade - 1.
#
def ndcg_at_k(
    retrieved: Sequence[DocumentId],
    relevance: Mapping[DocumentId, float],
    k: int,
    policy: JudgmentPolicy = "nonrelevant",
) -> float:
    grades, _, _, _ = _ranked_grades(retrieved, relevance, policy, k)
    discounts = [1.0 / math.log2(rank + 1) for rank in range(1, len(grades) + 1)]
    dcg = sum((2.0**grade - 1.0) * discount for grade, discount in zip(grades, discounts))
    ideal = sorted((float(value) for value in relevance.values()), reverse=True)[:k]
    ideal_discounts = [1.0 / math.log2(rank + 1) for rank in range(1, len(ideal) + 1)]
    ideal_dcg = sum((2.0**grade - 1.0) * discount
                    for grade, discount in zip(ideal, ideal_discounts))
    return dcg / ideal_dcg if ideal_dcg else 0.0


# Compute the common metric bundle and preserve judgment accounting beside it.
#
# Parameters:
#   retrieved (Sequence[DocumentId]): Ranked result identifiers.
#   relevance (Mapping[DocumentId, float]): Graded judgments.
#   k (int): Rank cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#
# Returns:
#   (dict[str, float | int | str]): Retrieval metrics and accounting.
#
def retrieval_metrics(
    retrieved: Sequence[DocumentId],
    relevance: Mapping[DocumentId, float],
    k: int,
    policy: JudgmentPolicy = "nonrelevant",
) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {
        "precision_at_k": precision_at_k(retrieved, relevance, k, policy),
        "graded_precision_at_k": graded_precision_at_k(retrieved, relevance, k, policy),
        "recall_at_k": recall_at_k(retrieved, relevance, k, policy),
        "graded_recall_at_k": graded_recall_at_k(retrieved, relevance, k, policy),
        "average_precision": average_precision(retrieved, relevance, k, policy),
        "reciprocal_rank": reciprocal_rank(retrieved, relevance, k, policy),
        "ndcg_at_k": ndcg_at_k(retrieved, relevance, k, policy),
    }
    metrics.update(judgment_accounting(retrieved, relevance, k, policy))
    return metrics


# Evaluate a complete ranked run and report per-query and macro-average metrics.
#
# Parameters:
#   run (Mapping[str, Sequence[DocumentId]]): Ranked results by query ID.
#   qrels (Mapping[str, Mapping[DocumentId, float]]): Judgments by query ID.
#   k (int): Rank cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#
# Returns:
#   (dict[str, Any]): Per-query results, macro averages, and totals.
#
def evaluate_run(
    run: Mapping[str, Sequence[DocumentId]],
    qrels: Mapping[str, Mapping[DocumentId, float]],
    k: int,
    policy: JudgmentPolicy = "nonrelevant",
) -> dict[str, Any]:
    per_query = {
        query_id: retrieval_metrics(run.get(query_id, ()), judgments, k, policy)
        for query_id, judgments in qrels.items()
    }
    metric_names = (
        "precision_at_k", "graded_precision_at_k", "recall_at_k",
        "graded_recall_at_k", "average_precision", "reciprocal_rank", "ndcg_at_k",
    )
    mean = {
        name: float(np.mean([float(metrics[name]) for metrics in per_query.values()]))
        if per_query else 0.0
        for name in metric_names
    }
    totals = {
        name: sum(int(metrics[name]) for metrics in per_query.values())
        for name in ("retrieved", "judged", "unjudged", "unique_retrieved", "duplicates")
    }
    total_results = totals["retrieved"]
    totals["judged_fraction"] = totals["judged"] / total_results if total_results else 0.0
    totals["unjudged_fraction"] = totals["unjudged"] / total_results if total_results else 0.0
    return {"k": k, "policy": policy, "per_query": per_query, "mean": mean, "totals": totals}


# Return the fraction of required facets covered by the ranked evidence.
#
# Parameters:
#   required_facets (Iterable[str]): Facets needed to answer the query.
#   retrieved (Sequence[DocumentId]): Ranked document identifiers.
#   doc_facets (Mapping[DocumentId, Iterable[str]]): Facets supplied by documents.
#   k (int | None): Optional rank cutoff.
#
# Returns:
#   (float): Fraction of required facets covered, with empty requirements scoring 1.
#
def facet_coverage(
    required_facets: Iterable[str],
    retrieved: Sequence[DocumentId],
    doc_facets: Mapping[DocumentId, Iterable[str]],
    k: int | None = None,
) -> float:
    required = set(required_facets)
    if not required:
        return 1.0
    covered: set[str] = set()
    for document_id in retrieved[:_cutoff(k, len(retrieved))]:
        covered.update(doc_facets.get(document_id, ()))
    return len(required.intersection(covered)) / len(required)


# Return covered and missing facets for complementary-evidence inspection.
#
# Parameters:
#   required_facets (Iterable[str]): Facets needed to answer the query.
#   retrieved (Sequence[DocumentId]): Ranked document identifiers.
#   doc_facets (Mapping[DocumentId, Iterable[str]]): Facets supplied by documents.
#   k (int | None): Optional rank cutoff.
#
# Returns:
#   (dict[str, Any]): Coverage score, covered facets, and missing facets.
#
def facet_coverage_report(
    required_facets: Iterable[str],
    retrieved: Sequence[DocumentId],
    doc_facets: Mapping[DocumentId, Iterable[str]],
    k: int | None = None,
) -> dict[str, Any]:
    required = set(required_facets)
    covered = set()
    for document_id in retrieved[:_cutoff(k, len(retrieved))]:
        covered.update(doc_facets.get(document_id, ()))
    return {
        "required": tuple(sorted(required)),
        "covered": tuple(sorted(required.intersection(covered))),
        "missing": tuple(sorted(required - covered)),
        "coverage": len(required.intersection(covered)) / len(required) if required else 1.0,
        "complete": required.issubset(covered),
    }


# Measure average pairwise Jaccard distance of result facets.
#
# Parameters:
#   retrieved (Sequence[DocumentId]): Ranked document identifiers.
#   doc_facets (Mapping[DocumentId, Iterable[str]]): Facets supplied by documents.
#   k (int | None): Optional rank cutoff.
#
# Returns:
#   (float): Mean pairwise diversity in [0, 1].
#
def diversity_at_k(
    retrieved: Sequence[DocumentId],
    doc_facets: Mapping[DocumentId, Iterable[str]],
    k: int | None = None,
) -> float:
    items = list(retrieved[:_cutoff(k, len(retrieved))])
    if len(items) < 2:
        return 0.0
    values = []
    for left, right in itertools.combinations(items, 2):
        first = set(doc_facets.get(left, ()))
        second = set(doc_facets.get(right, ()))
        union = first | second
        values.append(1.0 - len(first & second) / len(union) if union else 0.0)
    return float(np.mean(values))


# Measure average pairwise Jaccard similarity as result redundancy.
#
# Parameters:
#   retrieved (Sequence[DocumentId]): Ranked document identifiers.
#   doc_facets (Mapping[DocumentId, Iterable[str]]): Facets supplied by documents.
#   k (int | None): Optional rank cutoff.
#
# Returns:
#   (float): Mean pairwise redundancy in [0, 1].
#
def redundancy_at_k(
    retrieved: Sequence[DocumentId],
    doc_facets: Mapping[DocumentId, Iterable[str]],
    k: int | None = None,
) -> float:
    return 1.0 - diversity_at_k(retrieved, doc_facets, k)


# Return diversity, redundancy, and duplicate-rate diagnostics together.
#
# Parameters:
#   retrieved (Sequence[DocumentId]): Ranked document identifiers.
#   doc_facets (Mapping[DocumentId, Iterable[str]]): Facets supplied by documents.
#   k (int | None): Optional rank cutoff.
#
# Returns:
#   (dict[str, float | int]): Diversity and redundancy diagnostics.
#
def diversity_report(
    retrieved: Sequence[DocumentId],
    doc_facets: Mapping[DocumentId, Iterable[str]],
    k: int | None = None,
) -> dict[str, float | int]:
    items = list(retrieved[:_cutoff(k, len(retrieved))])
    unique = len(set(items))
    return {
        "retrieved": len(items),
        "unique": unique,
        "duplicate_rate": (len(items) - unique) / len(items) if items else 0.0,
        "diversity": diversity_at_k(items, doc_facets),
        "redundancy": redundancy_at_k(items, doc_facets),
    }


# Compute Brier score, expected calibration error, and reliability bins.
#
# Parameters:
#   confidences (Sequence[float]): Predicted probabilities in [0, 1].
#   outcomes (Sequence[bool | int]): Observed correctness labels.
#   bins (int): Number of equal-width confidence bins.
#
# Returns:
#   (dict[str, Any]): Brier score, ECE, MCE, and bin summaries.
#
def calibration_metrics(
    confidences: Sequence[float],
    outcomes: Sequence[bool | int],
    bins: int = 10,
) -> dict[str, Any]:
    probabilities = np.asarray(confidences, dtype=float).reshape(-1)
    labels = np.asarray(outcomes, dtype=float).reshape(-1)
    if probabilities.size != labels.size:
        raise ValueError("confidences and outcomes must have the same length")
    if bins < 1 or np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("bins must be positive and confidences must be in [0, 1]")
    if probabilities.size == 0:
        return {"brier": 0.0, "ece": 0.0, "mce": 0.0, "bins": []}
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("outcomes must be binary")
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, bins - 1)
    summaries: list[dict[str, float | int]] = []
    gaps: list[float] = []
    for index in range(bins):
        selected = indices == index
        count = int(np.sum(selected))
        confidence = float(np.mean(probabilities[selected])) if count else 0.0
        accuracy = float(np.mean(labels[selected])) if count else 0.0
        gap = abs(confidence - accuracy) if count else 0.0
        gaps.append(gap)
        summaries.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": count,
            "confidence": confidence,
            "accuracy": accuracy,
            "gap": gap,
        })
    return {
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece": float(sum(item["count"] * item["gap"] for item in summaries) / probabilities.size),
        "mce": max(gaps, default=0.0),
        "bins": summaries,
    }


# Evaluate selective answering and no-answer behavior at a confidence threshold.
#
# Parameters:
#   confidences (Sequence[float]): Probability that a query is answerable.
#   correct (Sequence[bool | int]): Whether an emitted answer is correct.
#   answerable (Sequence[bool | int]): Ground-truth answerability labels.
#   threshold (float): Answer when confidence is at least this value.
#
# Returns:
#   (dict[str, float | int]): Coverage, risk, and answerability metrics.
#
def abstention_metrics(
    confidences: Sequence[float],
    correct: Sequence[bool | int],
    answerable: Sequence[bool | int],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    probability = np.asarray(confidences, dtype=float).reshape(-1)
    is_correct = np.asarray(correct, dtype=bool).reshape(-1)
    can_answer = np.asarray(answerable, dtype=bool).reshape(-1)
    if not (probability.size == is_correct.size == can_answer.size):
        raise ValueError("confidences, correct, and answerable must have the same length")
    if not 0.0 <= threshold <= 1.0 or np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("threshold and confidences must be in [0, 1]")
    answered = probability >= threshold
    answer_count = int(np.sum(answered))
    positive_count = int(np.sum(can_answer))
    true_answered = int(np.sum(answered & can_answer))
    correct_answered = int(np.sum(answered & is_correct))
    true_abstained = int(np.sum(~answered & ~can_answer))
    negative_count = len(can_answer) - positive_count
    answerability_precision = true_answered / answer_count if answer_count else 0.0
    answerability_recall = true_answered / positive_count if positive_count else 0.0
    no_answer_precision = true_abstained / (len(can_answer) - answer_count) if len(can_answer) - answer_count else 0.0
    no_answer_recall = true_abstained / negative_count if negative_count else 0.0
    return {
        "answered": answer_count,
        "abstained": len(can_answer) - answer_count,
        "coverage": answer_count / len(can_answer) if can_answer.size else 0.0,
        "selective_accuracy": correct_answered / answer_count if answer_count else 0.0,
        "selective_risk": 1.0 - correct_answered / answer_count if answer_count else 0.0,
        "answerability_precision": answerability_precision,
        "answerability_recall": answerability_recall,
        "answerability_f1": (2.0 * answerability_precision * answerability_recall /
                              (answerability_precision + answerability_recall)
                              if answerability_precision + answerability_recall else 0.0),
        "no_answer_precision": no_answer_precision,
        "no_answer_recall": no_answer_recall,
        "no_answer_accuracy": no_answer_recall,
        "abstention_accuracy": no_answer_precision,
    }


# A deterministic percentile confidence interval returned by bootstrap_ci.
@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float
    resamples: int


# Estimate a statistic and a deterministic percentile bootstrap interval.
#
# Parameters:
#   values (Sequence[float]): Independent observations.
#   statistic (Callable[[np.ndarray], float]): Statistic applied to samples.
#   confidence (float): Central interval mass in (0, 1).
#   resamples (int): Number of bootstrap samples.
#   seed (int): NumPy random seed.
#
# Returns:
#   (ConfidenceInterval): Point estimate and percentile bounds.
#
def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> ConfidenceInterval:
    data = np.asarray(values, dtype=float).reshape(-1)
    if data.size == 0:
        raise ValueError("values must not be empty")
    if not 0.0 < confidence < 1.0 or resamples < 1:
        raise ValueError("confidence must be in (0, 1) and resamples must be positive")
    estimate = float(statistic(data))
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, data.size, size=(resamples, data.size))
    bootstrapped = np.asarray([statistic(data[index]) for index in samples], dtype=float)
    tail = (1.0 - confidence) * 50.0
    low, high = np.percentile(bootstrapped, [tail, 100.0 - tail])
    return ConfidenceInterval(estimate, float(low), float(high), resamples)


# Test a paired metric difference with exact or seeded sign randomization.
#
# Parameters:
#   first (Sequence[float]): Per-query scores for the first system.
#   second (Sequence[float]): Per-query scores for the second system.
#   resamples (int): Maximum random sign assignments for large samples.
#   seed (int): NumPy random seed.
#   alternative (Alternative): Direction of the alternative hypothesis.
#
# Returns:
#   (dict[str, Any]): Observed difference, p-value, and test metadata.
#
def paired_randomization_test(
    first: Sequence[float],
    second: Sequence[float],
    resamples: int = 10000,
    seed: int = 0,
    alternative: Alternative = "two-sided",
) -> dict[str, Any]:
    left = np.asarray(first, dtype=float).reshape(-1)
    right = np.asarray(second, dtype=float).reshape(-1)
    if left.size == 0 or left.size != right.size:
        raise ValueError("paired samples must be non-empty and have equal length")
    if resamples < 1 or alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("resamples must be positive and alternative is invalid")
    differences = left - right
    observed = float(np.mean(differences))
    exact = differences.size <= 16
    if exact:
        statistics = np.asarray([
            np.mean(differences * np.asarray(signs, dtype=float))
            for signs in itertools.product((-1.0, 1.0), repeat=differences.size)
        ])
    else:
        rng = np.random.default_rng(seed)
        statistics = np.asarray([
            np.mean(differences * rng.choice((-1.0, 1.0), size=differences.size))
            for _ in range(resamples)
        ])
    if alternative == "two-sided":
        extreme = np.abs(statistics) >= abs(observed) - 1e-15
    elif alternative == "greater":
        extreme = statistics >= observed - 1e-15
    else:
        extreme = statistics <= observed + 1e-15
    p_value = float(np.mean(extreme)) if exact else float((np.sum(extreme) + 1) / (len(statistics) + 1))
    return {
        "difference": observed,
        "p_value": p_value,
        "n": int(differences.size),
        "samples": int(len(statistics)),
        "exact": exact,
        "alternative": alternative,
    }


# Estimate sample size and normal-approximation power for a standardized effect.
#
# Parameters:
#   effect_size (float): Absolute standardized paired effect; must be non-zero.
#   alpha (float): Type-I error rate.
#   target_power (float): Desired power.
#   alternative (Alternative): One- or two-sided test.
#
# Returns:
#   (dict[str, float | int | str]): Required n and estimated achieved power.
#
def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    target_power: float = 0.8,
    alternative: Alternative = "two-sided",
) -> dict[str, float | int | str]:
    effect = abs(float(effect_size))
    if effect == 0.0 or not 0.0 < alpha < 1.0 or not 0.0 < target_power < 1.0:
        raise ValueError("effect_size must be non-zero; alpha and target_power must be in (0, 1)")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative is invalid")
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - alpha / (2.0 if alternative == "two-sided" else 1.0))
    target_z = normal.inv_cdf(target_power)
    required = max(2, int(math.ceil(((critical + target_z) / effect) ** 2)))
    delta = effect * math.sqrt(required)
    if alternative == "two-sided":
        achieved = normal.cdf(-critical - delta) + 1.0 - normal.cdf(critical - delta)
    else:
        achieved = normal.cdf(delta - critical)
    return {
        "required_n": required,
        "effect_size": effect,
        "alpha": alpha,
        "target_power": target_power,
        "estimated_power": achieved,
        "alternative": alternative,
    }


# Tokenize text with stable ASCII-friendly lexical units.
#
# Parameters:
#   text (str): Input text.
#
# Returns:
#   (tuple[str, ...]): Lowercase word, number, or underscore tokens.
#
def tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


# Normalize text for duplicate and leakage checks.
#
# Parameters:
#   text (str): Input text.
#
# Returns:
#   (str): Whitespace-normalized lowercase text.
#
def _normalize(text: str) -> str:
    return " ".join(tokenize(text))


# Convert common document inputs to an ID-to-text mapping.
#
# Parameters:
#   documents (Mapping | Sequence): Mapping, fixture documents, ID-text pairs,
#       or plain text strings.
#
# Returns:
#   (dict[DocumentId, str]): Stable document text mapping.
#
def _document_texts(documents: Any) -> dict[DocumentId, str]:
    if isinstance(documents, Mapping):
        return {document_id: str(text) for document_id, text in documents.items()}
    output: dict[DocumentId, str] = {}
    for index, item in enumerate(documents):
        if isinstance(item, FixtureDocument):
            output[item.doc_id] = item.text
        elif isinstance(item, tuple) and len(item) >= 2:
            output[item[0]] = str(item[1])
        else:
            output[str(index)] = str(item)
    return output


# Return deterministic exact and near-duplicate diagnostics for a corpus.
#
# Parameters:
#   documents (Mapping | Sequence): Documents accepted by _document_texts.
#   near_threshold (float): Jaccard threshold for non-identical near duplicates.
#
# Returns:
#   (dict[str, Any]): Exact groups, near pairs, and duplicate rates.
#
def duplicate_checks(documents: Any, near_threshold: float = 0.8) -> dict[str, Any]:
    if not 0.0 <= near_threshold <= 1.0:
        raise ValueError("near_threshold must be in [0, 1]")
    texts = _document_texts(documents)
    normalized = {document_id: _normalize(text) for document_id, text in texts.items()}
    exact_map: dict[str, list[DocumentId]] = {}
    for document_id, text in normalized.items():
        exact_map.setdefault(text, []).append(document_id)
    exact_groups = tuple(tuple(group) for group in exact_map.values() if len(group) > 1)
    near_pairs: list[tuple[DocumentId, DocumentId, float]] = []
    ids = list(texts)
    for index, left in enumerate(ids):
        first = set(normalized[left].split())
        for right in ids[index + 1:]:
            if normalized[left] == normalized[right]:
                continue
            second = set(normalized[right].split())
            union = first | second
            score = len(first & second) / len(union) if union else 1.0
            if score >= near_threshold:
                near_pairs.append((left, right, float(score)))
    duplicate_count = sum(len(group) - 1 for group in exact_groups)
    return {
        "documents": len(ids),
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_pairs": tuple(near_pairs),
        "duplicate_rate": duplicate_count / len(ids) if ids else 0.0,
    }


# Check query leakage, source-derived query mix, and cross-split duplicates.
#
# Parameters:
#   documents (Mapping | Sequence): Corpus documents.
#   queries (Sequence[str | QueryCase]): Query strings or annotated queries.
#   split (Mapping[DocumentId, str] | None): Optional train/test document split.
#   overlap_threshold (float): Query-token recall that flags likely leakage.
#
# Returns:
#   (dict[str, Any]): Leakage records, source-blind rate, and duplicate findings.
#
def leakage_checks(
    documents: Any,
    queries: Sequence[str | QueryCase],
    split: Mapping[DocumentId, str] | None = None,
    overlap_threshold: float = 0.8,
) -> dict[str, Any]:
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be in [0, 1]")
    texts = _document_texts(documents)
    records: list[dict[str, Any]] = []
    source_blind = 0
    for index, query in enumerate(queries):
        if isinstance(query, QueryCase):
            query_id = query.query_id
            text = query.text
            is_source_blind = query.source_blind
            targets = [doc for doc, grade in query.relevance.items() if grade > 0]
        else:
            query_id = str(index)
            text = query
            is_source_blind = False
            targets = list(texts)
        source_blind += int(is_source_blind)
        terms = set(tokenize(text))
        targets = [doc for doc in targets if doc in texts] or list(texts)
        best_doc: DocumentId | None = None
        best_overlap = 0.0
        exact_phrase = False
        normalized_query = _normalize(text)
        for document_id in targets:
            document = _normalize(texts[document_id])
            document_terms = set(document.split())
            overlap = len(terms & document_terms) / len(terms) if terms else 0.0
            if normalized_query and normalized_query in document:
                exact_phrase = True
            if overlap > best_overlap:
                best_overlap = overlap
                best_doc = document_id
        records.append({
            "query_id": query_id,
            "source_blind": is_source_blind,
            "source_derived": not is_source_blind,
            "best_document": best_doc,
            "token_overlap": best_overlap,
            "exact_phrase_match": exact_phrase,
            "leaked": exact_phrase or best_overlap >= overlap_threshold,
        })
    duplicates = duplicate_checks(texts)
    cross_split: list[tuple[DocumentId, DocumentId, float]] = []
    if split is not None:
        for left, right, score in duplicates["near_duplicate_pairs"]:
            if {split.get(left), split.get(right)} == {"train", "test"}:
                cross_split.append((left, right, score))
        for group in duplicates["exact_duplicate_groups"]:
            for left, right in itertools.combinations(group, 2):
                if {split.get(left), split.get(right)} == {"train", "test"}:
                    cross_split.append((left, right, 1.0))
    leaked = sum(bool(record["leaked"]) for record in records)
    return {
        "queries": len(records),
        "source_blind_fraction": source_blind / len(records) if records else 0.0,
        "leaked_queries": leaked,
        "leakage_rate": leaked / len(records) if records else 0.0,
        "records": tuple(records),
        "duplicates": duplicates,
        "cross_split_duplicates": tuple(cross_split),
    }


# Summarize corpus size, token volume, format diversity, and duplicate rate.
#
# Parameters:
#   documents (Mapping | Sequence): Corpus documents.
#
# Returns:
#   (dict[str, Any]): Corpus counts and UTF-8 storage statistics.
#
def corpus_report(documents: Any) -> dict[str, Any]:
    texts = _document_texts(documents)
    duplicate = duplicate_checks(texts)
    return {
        "documents": len(texts),
        "unique_documents": duplicate["documents"] - sum(len(group) - 1 for group in duplicate["exact_duplicate_groups"]),
        "utf8_bytes": sum(len(text.encode("utf-8")) for text in texts.values()),
        "characters": sum(len(text) for text in texts.values()),
        "tokens": sum(len(tokenize(text)) for text in texts.values()),
        "duplicate_rate": duplicate["duplicate_rate"],
    }


# Return index storage amplification relative to source corpus bytes.
#
# Parameters:
#   index_bytes (int): Persisted index size.
#   corpus_bytes (int): UTF-8 source corpus size.
#
# Returns:
#   (float): Index bytes divided by corpus bytes.
#
def storage_amplification(index_bytes: int, corpus_bytes: int) -> float:
    if index_bytes < 0 or corpus_bytes < 0:
        raise ValueError("byte counts must be non-negative")
    if corpus_bytes == 0:
        return 0.0 if index_bytes == 0 else float("inf")
    return index_bytes / corpus_bytes


# Read process peak RSS when the platform exposes it.
#
# Returns:
#   (int): Peak resident bytes, or zero when unavailable.
#
def _peak_rss_bytes() -> int:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (AttributeError, OSError):
        return 0


# Time one index-building callable and record process peak memory.
#
# Parameters:
#   build (Callable[[], Any]): Callable that constructs an index or artifact.
#
# Returns:
#   (dict[str, Any]): Build result, elapsed seconds, and peak RSS bytes.
#
def benchmark_build(build: Callable[[], Any]) -> dict[str, Any]:
    before = _peak_rss_bytes()
    started = time.perf_counter()
    result = build()
    elapsed = time.perf_counter() - started
    return {"result": result, "seconds": elapsed, "peak_rss_bytes": max(_peak_rss_bytes(), before)}


# Return percentile latency and throughput for a callable under concurrency.
#
# Parameters:
#   search (Callable[[str], Any]): Search callable accepting one query string.
#   queries (Sequence[str]): Queries to execute.
#   concurrency (int): Maximum simultaneous calls.
#   repeats (int): Number of passes over queries.
#
# Returns:
#   (dict[str, Any]): Latency percentiles, samples, wall time, and QPS.
#
def benchmark_latency(
    search: Callable[[str], Any],
    queries: Sequence[str],
    concurrency: int = 1,
    repeats: int = 1,
) -> dict[str, Any]:
    if concurrency < 1 or repeats < 1:
        raise ValueError("concurrency and repeats must be positive")
    work = tuple(query for _ in range(repeats) for query in queries)

    # Measure each call in its worker and wall-clock throughput around the batch.
    def timed(query: str) -> float:
        started = time.perf_counter()
        search(query)
        return time.perf_counter() - started

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        samples = tuple(executor.map(timed, work))
    wall = max(time.perf_counter() - started, np.finfo(float).tiny)
    if not samples:
        percentiles = {name: 0.0 for name in ("min", "p50", "p95", "p99", "max")}
    else:
        values = np.asarray(samples, dtype=float)
        percentiles = {
            "min": float(np.min(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }
    return {
        **percentiles,
        "median": percentiles["p50"],
        "samples": samples,
        "count": len(samples),
        "concurrency": concurrency,
        "wall_seconds": wall,
        "throughput_qps": len(samples) / wall,
    }


# Benchmark several concurrency levels with one stable report shape.
#
# Parameters:
#   search (Callable[[str], Any]): Search callable accepting one query string.
#   queries (Sequence[str]): Queries to execute.
#   levels (Sequence[int]): Concurrency levels to measure.
#   repeats (int): Number of passes over queries per level.
#
# Returns:
#   (dict[int, dict[str, Any]]): Report keyed by concurrency.
#
def benchmark_concurrency(
    search: Callable[[str], Any],
    queries: Sequence[str],
    levels: Sequence[int] = (1, 2, 4),
    repeats: int = 1,
) -> dict[int, dict[str, Any]]:
    return {level: benchmark_latency(search, queries, level, repeats) for level in levels}


# Score claim correctness, support, contradictions, and citation coverage.
#
# Parameters:
#   reference_claims (Iterable[str]): Claims expected in a correct answer.
#   answer_claims (Iterable[str]): Claims emitted by the answer.
#   supported_claims (Iterable[str]): Emitted claims supported by retrieved evidence.
#   cited_evidence (Iterable[str]): Evidence IDs cited by the answer.
#   required_evidence (Iterable[str]): Evidence IDs required by the reference.
#   contradictions (Iterable[str]): Emitted claims contradicting the reference.
#
# Returns:
#   (dict[str, float | bool]): Claim and citation faithfulness metrics.
#
def answer_quality_metrics(
    reference_claims: Iterable[str],
    answer_claims: Iterable[str],
    supported_claims: Iterable[str],
    cited_evidence: Iterable[str],
    required_evidence: Iterable[str],
    contradictions: Iterable[str] = (),
) -> dict[str, float | bool]:
    reference = set(reference_claims)
    answer = set(answer_claims)
    supported = set(supported_claims)
    cited = set(cited_evidence)
    required = set(required_evidence)
    correct = reference.intersection(answer)
    claim_precision = len(correct) / len(answer) if answer else (1.0 if not reference else 0.0)
    claim_recall = len(correct) / len(reference) if reference else (1.0 if not answer else 0.0)
    f1 = 2.0 * claim_precision * claim_recall / (claim_precision + claim_recall) if claim_precision + claim_recall else 0.0
    citation_precision = len(cited & required) / len(cited) if cited else (1.0 if not required else 0.0)
    citation_recall = len(cited & required) / len(required) if required else (1.0 if not cited else 0.0)
    return {
        "claim_precision": claim_precision,
        "claim_recall": claim_recall,
        "claim_f1": f1,
        "claim_exact": answer == reference,
        "unsupported_claim_rate": len(answer - supported) / len(answer) if answer else 0.0,
        "contradiction_rate": len(set(contradictions)) / len(answer) if answer else 0.0,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
    }


# Validate citation/provenance fields on serialized search hits.
#
# Parameters:
#   results (Sequence[Mapping]): JSON-compatible result records.
#
# Returns:
#   (dict[str, float | int]): Completeness, validity, and duplicate counts.
#
def provenance_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    valid = 0
    citation_keys: list[tuple[str, str, str]] = []
    for result in results:
        document = result.get("document", {})
        span = result.get("span", ())
        if not isinstance(document, Mapping) or not isinstance(span, (tuple, list)) or len(span) != 2:
            continue
        source = str(document.get("source_id") or result.get("source_path", ""))
        content_hash = str(document.get("content_hash", ""))
        build_id = str(document.get("build_id", ""))
        if not source or not content_hash or not build_id:
            continue
        try:
            if int(span[0]) < 0 or int(span[1]) < int(span[0]):
                continue
        except (TypeError, ValueError):
            continue
        valid += 1
        citation_keys.append((source, content_hash, build_id))
    return {
        "results": len(results),
        "valid_citations": valid,
        "citation_completeness": valid / len(results) if results else 0.0,
        "duplicate_citations": len(citation_keys) - len(set(citation_keys)),
    }


# Hash text into a deterministic normalized dense vector.
#
# Parameters:
#   text (str): Input text.
#   dimension (int): Vector dimension.
#
# Returns:
#   (np.ndarray): Float32 unit vector.
#
def hashed_embedding(text: str, dimension: int = 64) -> np.ndarray:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    vector = np.zeros(dimension, dtype=np.float32)
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


# Rank score pairs stably, retaining insertion order for exact ties.
#
# Parameters:
#   scores (Mapping[DocumentId, float]): Document scores.
#   k (int): Number of IDs to return.
#
# Returns:
#   (list[DocumentId]): Stable top-k IDs.
#
def _top_scores(scores: Mapping[DocumentId, float], k: int) -> list[DocumentId]:
    _cutoff(k, len(scores))
    return [document_id for _, document_id in sorted(
        enumerate(scores), key=lambda item: (-float(scores[item[1]]), item[0])
    )[:k]]


# Score lexical term overlap for one query and corpus.
#
# Parameters:
#   query (str): Query text.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#
# Returns:
#   (dict[DocumentId, float]): Stable lexical scores.
#
def _lexical_scores(query: str, documents: Mapping[DocumentId, str]) -> dict[DocumentId, float]:
    terms = set(tokenize(query))
    return {document_id: float(sum(token in terms for token in tokenize(text)))
            for document_id, text in documents.items()}


# Retrieve by simple lexical term overlap.
#
# Parameters:
#   query (str): Query text.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#   k (int): Number of IDs to return.
#
# Returns:
#   (list[DocumentId]): Stable top-k IDs.
#
def lexical_search(query: str, documents: Mapping[DocumentId, str], k: int = 10) -> list[DocumentId]:
    return _top_scores(_lexical_scores(query, documents), k)


# Compute BM25 scores with deterministic corpus statistics.
#
# Parameters:
#   query (str): Query text.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#   k1 (float): BM25 term-frequency saturation.
#   b (float): BM25 length normalization.
#
# Returns:
#   (dict[DocumentId, float]): BM25 scores.
#
def _bm25_scores(
    query: str,
    documents: Mapping[DocumentId, str],
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[DocumentId, float]:
    if k1 < 0.0 or not 0.0 <= b <= 1.0:
        raise ValueError("k1 must be non-negative and b must be in [0, 1]")
    token_lists = {document_id: tokenize(text) for document_id, text in documents.items()}
    document_sets = {document_id: set(tokens) for document_id, tokens in token_lists.items()}
    document_frequency = Counter(
        term for tokens in document_sets.values() for term in tokens
    )
    return _bm25_scores_from_statistics(
        query, token_lists, document_frequency, k1, b
    )


# Score BM25 using token statistics prepared once for a corpus.
def _bm25_scores_from_statistics(
    query: str,
    token_lists: Mapping[DocumentId, Sequence[str]],
    document_frequency: Mapping[str, int],
    k1: float,
    b: float,
) -> dict[DocumentId, float]:
    lengths = np.asarray([len(tokens) for tokens in token_lists.values()], dtype=float)
    average_length = float(np.mean(lengths)) if lengths.size else 0.0
    query_terms = set(tokenize(query))
    count = len(token_lists)
    scores: dict[DocumentId, float] = {}
    for document_id, tokens in token_lists.items():
        frequencies = {term: tokens.count(term) for term in query_terms}
        score = 0.0
        for term, frequency in frequencies.items():
            if not frequency:
                continue
            idf = math.log(1.0 + (count - document_frequency[term] + 0.5) /
                           (document_frequency[term] + 0.5))
            length_factor = b * (len(tokens) / average_length - 1.0) if average_length else 0.0
            score += idf * frequency * (k1 + 1.0) / (frequency + k1 * (1.0 + length_factor))
        scores[document_id] = score
    return scores


# Retrieve by Okapi BM25.
#
# Parameters:
#   query (str): Query text.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#   k (int): Number of IDs to return.
#   k1 (float): BM25 term-frequency saturation.
#   b (float): BM25 length normalization.
#
# Returns:
#   (list[DocumentId]): Stable top-k IDs.
#
def bm25_search(
    query: str,
    documents: Mapping[DocumentId, str],
    k: int = 10,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[DocumentId]:
    return _top_scores(_bm25_scores(query, documents, k1, b), k)


# Run BM25 for several queries while tokenizing the corpus only once.
#
# Parameters:
#   queries (Sequence[str]): Query texts.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#   k (int): Number of IDs to return for each query.
#   k1 (float): BM25 term-frequency saturation.
#   b (float): BM25 length normalization.
#
# Returns:
#   (list[list[DocumentId]]): One stable ranking per query.
def bm25_search_batch(
    queries: Sequence[str],
    documents: Mapping[DocumentId, str],
    k: int = 10,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[list[DocumentId]]:
    if k1 < 0.0 or not 0.0 <= b <= 1.0:
        raise ValueError("k1 must be non-negative and b must be in [0, 1]")
    token_lists = {document_id: tokenize(text) for document_id, text in documents.items()}
    document_sets = {document_id: set(tokens) for document_id, tokens in token_lists.items()}
    document_frequency = Counter(
        term for tokens in document_sets.values() for term in tokens
    )
    return [
        _top_scores(
            _bm25_scores_from_statistics(query, token_lists, document_frequency, k1, b),
            k,
        )
        for query in queries
    ]


# Retrieve by inverse-document-frequency weighted sparse overlap.
#
# Parameters:
#   query (str): Query text.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#   k (int): Number of IDs to return.
#
# Returns:
#   (list[DocumentId]): Stable top-k IDs.
#
def sparse_search(query: str, documents: Mapping[DocumentId, str], k: int = 10) -> list[DocumentId]:
    token_lists = {document_id: tokenize(text) for document_id, text in documents.items()}
    terms = set(tokenize(query))
    count = len(token_lists)
    scores = {
        document_id: sum(
            tokens.count(term) * math.log((count + 1.0) / (1.0 + sum(term in set(other) for other in token_lists.values())))
            for term in terms
        )
        for document_id, tokens in token_lists.items()
    }
    return _top_scores(scores, k)


# Retrieve by deterministic hashed-vector cosine similarity.
#
# Parameters:
#   query (str): Query text.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#   k (int): Number of IDs to return.
#   dimension (int): Hash-vector dimension.
#
# Returns:
#   (list[DocumentId]): Stable top-k IDs.
#
def dense_search(
    query: str,
    documents: Mapping[DocumentId, str],
    k: int = 10,
    dimension: int = 64,
) -> list[DocumentId]:
    query_vector = hashed_embedding(query, dimension)
    scores = {document_id: float(np.dot(query_vector, hashed_embedding(text, dimension)))
              for document_id, text in documents.items()}
    return _top_scores(scores, k)


# Retrieve with a small late-interaction token matching score.
#
# Parameters:
#   query (str): Query text.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#   k (int): Number of IDs to return.
#   dimension (int): Hash-vector dimension for token interactions.
#
# Returns:
#   (list[DocumentId]): Stable top-k IDs.
#
def late_interaction_search(
    query: str,
    documents: Mapping[DocumentId, str],
    k: int = 10,
    dimension: int = 64,
) -> list[DocumentId]:
    query_tokens = tuple(dict.fromkeys(tokenize(query)))
    query_vectors = [hashed_embedding(token, dimension) for token in query_tokens]
    scores: dict[DocumentId, float] = {}
    for document_id, text in documents.items():
        document_vectors = [hashed_embedding(token, dimension) for token in dict.fromkeys(tokenize(text))]
        scores[document_id] = float(np.mean([
            max((float(np.dot(query_vector, document_vector)) for document_vector in document_vectors), default=0.0)
            for query_vector in query_vectors
        ])) if query_vectors else 0.0
    return _top_scores(scores, k)


# Retrieve with a normalized lexical and dense score fusion.
#
# Parameters:
#   query (str): Query text.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#   k (int): Number of IDs to return.
#   dimension (int): Hash-vector dimension.
#
# Returns:
#   (list[DocumentId]): Stable top-k IDs.
#
def hybrid_search(
    query: str,
    documents: Mapping[DocumentId, str],
    k: int = 10,
    dimension: int = 64,
) -> list[DocumentId]:
    lexical = _lexical_scores(query, documents)
    dense = {document_id: float(np.dot(hashed_embedding(query, dimension), hashed_embedding(text, dimension)))
             for document_id, text in documents.items()}
    lexical_max = max(lexical.values(), default=0.0)
    dense_max = max(dense.values(), default=0.0)
    scores = {
        document_id: 0.5 * (lexical[document_id] / lexical_max if lexical_max else 0.0)
        + 0.5 * (dense[document_id] / dense_max if dense_max else 0.0)
        for document_id in documents
    }
    return _top_scores(scores, k)


# Rerank candidates with exact phrase and query-term coverage signals.
#
# Parameters:
#   query (str): Query text.
#   candidates (Sequence[DocumentId]): Candidate IDs in rank order.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#   k (int): Number of IDs to return.
#
# Returns:
#   (list[DocumentId]): Stable reranked IDs.
#
def rerank_results(
    query: str,
    candidates: Sequence[DocumentId],
    documents: Mapping[DocumentId, str],
    k: int = 10,
) -> list[DocumentId]:
    terms = set(tokenize(query))
    normalized_query = _normalize(query)
    scores = {}
    for rank, document_id in enumerate(candidates):
        if document_id not in documents:
            continue
        text = _normalize(documents[document_id])
        coverage = len(terms.intersection(text.split())) / len(terms) if terms else 0.0
        scores[document_id] = 2.0 * float(normalized_query in text) + coverage - rank * 1e-9
    return _top_scores(scores, k)


# Retrieve through deterministic multi-query fusion as an agentic baseline.
#
# Parameters:
#   query (str): Query text.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#   k (int): Number of IDs to return.
#
# Returns:
#   (list[DocumentId]): Stable fused IDs.
#
def agentic_search(query: str, documents: Mapping[DocumentId, str], k: int = 10) -> list[DocumentId]:
    variants = [query] + [part for part in re.split(r"\b(?:and|or|but|if|when|with|without)\b", query, flags=re.I)
                          if len(tokenize(part)) > 1]
    scores: dict[DocumentId, float] = {document_id: 0.0 for document_id in documents}
    for variant in dict.fromkeys(variants):
        for rank, document_id in enumerate(hybrid_search(variant, documents, len(documents)), 1):
            scores[document_id] += 1.0 / rank
    return _top_scores(scores, k)


# Retrieve from a deterministic subset to expose approximate-search tradeoffs.
#
# Parameters:
#   query (str): Query text.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#   k (int): Number of IDs to return.
#   fraction (float): Fraction of documents eligible for exact dense scoring.
#   seed (int): Stable candidate-selection seed.
#
# Returns:
#   (list[DocumentId]): Approximate dense ranking.
#
def approximate_dense_search(
    query: str,
    documents: Mapping[DocumentId, str],
    k: int = 10,
    fraction: float = 0.5,
    seed: int = 0,
) -> list[DocumentId]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    candidates = [document_id for document_id in documents if
                  int.from_bytes(hashlib.blake2b(f"{seed}:{document_id}".encode("utf-8"), digest_size=4).digest(), "little")
                  / 2**32 < fraction]
    if documents and not candidates:
        candidates = [next(iter(documents))]
    return dense_search(query, {document_id: documents[document_id] for document_id in candidates}, k)


# Run all compact baseline implementations side by side.
#
# Parameters:
#   query (str): Query text.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#   k (int): Number of IDs returned by each baseline.
#
# Returns:
#   (dict[str, list[DocumentId]]): Rankings for lexical, sparse, dense, late,
#       hybrid, reranker, and agentic baselines.
#
def run_baselines(
    query: str,
    documents: Mapping[DocumentId, str],
    k: int = 10,
) -> dict[str, list[DocumentId]]:
    lexical = lexical_search(query, documents, k)
    bm25 = bm25_search(query, documents, k)
    return {
        "lexical": lexical,
        "bm25": bm25,
        "sparse": sparse_search(query, documents, k),
        "dense": dense_search(query, documents, k),
        "late_interaction": late_interaction_search(query, documents, k),
        "hybrid": hybrid_search(query, documents, k),
        "reranker": rerank_results(query, bm25, documents, k),
        "agentic": agentic_search(query, documents, k),
    }


# Compare all compact baselines over annotated fixture queries.
#
# Parameters:
#   fixture (Fixture): Heterogeneous corpus and query judgments.
#   k (int): Rank cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#
# Returns:
#   (dict[str, dict[str, Any]]): Evaluation report by baseline name.
#
def compare_baselines(
    fixture: Fixture,
    k: int = 10,
    policy: JudgmentPolicy = "nonrelevant",
) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, list[DocumentId]]] = {}
    for query in fixture.queries:
        for name, ranking in run_baselines(query.text, fixture.document_map, k).items():
            runs.setdefault(name, {})[query.query_id] = ranking
    return {name: evaluate_run(run, fixture.qrels, k, policy) for name, run in runs.items()}


# Audit retrieved records for access, tenant, citation, and provenance failures.
#
# Parameters:
#   results (Sequence[Mapping[str, Any]]): Result records with a doc_id or id.
#   known_documents (Mapping[DocumentId, Any]): Documents visible to the evaluator.
#   allowed_ids (Iterable[DocumentId] | None): Optional permission allow-list.
#   tenant (str | None): Optional tenant required by every result.
#
# Returns:
#   (dict[str, Any]): Failure counts and per-result audit records.
#
def result_audit(
    results: Sequence[Mapping[str, Any]],
    known_documents: Mapping[DocumentId, Any],
    allowed_ids: Iterable[DocumentId] | None = None,
    tenant: str | None = None,
) -> dict[str, Any]:
    allowed = set(allowed_ids) if allowed_ids is not None else set(known_documents)
    records: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        document_id = result.get("doc_id", result.get("id"))
        issues: list[str] = []
        if document_id not in known_documents:
            issues.append("unknown_document")
        if document_id not in allowed:
            issues.append("forbidden_document")
        if not any(result.get(key) for key in ("citation", "source", "source_path")):
            issues.append("missing_citation")
        if not any(result.get(key) for key in ("provenance", "source", "source_path")):
            issues.append("missing_provenance")
        if tenant is not None and result.get("tenant") != tenant:
            issues.append("tenant_mismatch")
        records.append({"rank": index + 1, "doc_id": document_id, "issues": tuple(issues)})
    counts = {issue: sum(issue in record["issues"] for record in records)
              for issue in ("unknown_document", "forbidden_document", "missing_citation",
                            "missing_provenance", "tenant_mismatch")}
    return {"valid": not any(counts.values()), "records": tuple(records), **counts}


# Explain lexical support for each ranked result without hiding its provenance.
#
# Parameters:
#   query (str): Query text.
#   ranked (Sequence[DocumentId]): Ranked document identifiers.
#   documents (Mapping[DocumentId, str]): Corpus text by ID.
#
# Returns:
#   (tuple[dict[str, Any], ...]): Matched terms and phrase indicators by rank.
#
def explain_ranking(
    query: str,
    ranked: Sequence[DocumentId],
    documents: Mapping[DocumentId, str],
) -> tuple[dict[str, Any], ...]:
    terms = set(tokenize(query))
    normalized_query = _normalize(query)
    explanations = []
    for rank, document_id in enumerate(ranked, 1):
        text = _normalize(documents[document_id]) if document_id in documents else ""
        matched = tuple(sorted(terms.intersection(text.split())))
        explanations.append({
            "rank": rank,
            "doc_id": document_id,
            "matched_terms": matched,
            "term_coverage": len(matched) / len(terms) if terms else 0.0,
            "phrase_match": bool(normalized_query and normalized_query in text),
        })
    return tuple(explanations)


# Measure lexical answer support and citation validity from retrieved evidence.
#
# Parameters:
#   answer (str): Downstream answer text.
#   evidence (Sequence[str]): Evidence passages supplied to the answerer.
#   citations (Sequence[DocumentId]): IDs cited by the answer.
#   retrieved (Sequence[DocumentId]): IDs available as evidence.
#
# Returns:
#   (dict[str, float | int]): Lexical support and citation coverage metrics.
#
def grounding_metrics(
    answer: str,
    evidence: Sequence[str],
    citations: Sequence[DocumentId] = (),
    retrieved: Sequence[DocumentId] = (),
) -> dict[str, float | int]:
    answer_terms = set(tokenize(answer))
    evidence_terms = set(token for text in evidence for token in tokenize(text))
    cited = set(citations)
    available = set(retrieved)
    supported = len(answer_terms & evidence_terms)
    cited_valid = len(cited & available)
    return {
        "answer_terms": len(answer_terms),
        "supported_terms": supported,
        "answer_term_support": supported / len(answer_terms) if answer_terms else 0.0,
        "evidence_terms": len(evidence_terms),
        "citation_count": len(cited),
        "valid_citations": cited_valid,
        "citation_precision": cited_valid / len(cited) if cited else 0.0,
        "citation_recall": cited_valid / len(available) if available else 0.0,
    }


# Generate a deterministic heterogeneous corpus and annotated query set.
#
# Parameters:
#   seed (int): Seed for optional distractor generation.
#   distractors (int): Number of deterministic topic-sharing distractors.
#
# Returns:
#   (Fixture): Documents and query metadata with graded relevance judgments.
#
def make_fixture(seed: int = 0, distractors: int = 2) -> Fixture:
    if distractors < 0:
        raise ValueError("distractors must be non-negative")
    documents = [
        FixtureDocument("d_solar", "Solar panels use sunlight to lower emissions but need storage at night.",
                         "science", frozenset({"mechanism", "benefit", "tradeoff"}), metadata={"tenant": "a"}),
        FixtureDocument("d_wind", "Wind turbines lower emissions and operate without sunlight, with variable output.",
                         "science", frozenset({"mechanism", "benefit", "tradeoff"}), metadata={"tenant": "a"}),
        FixtureDocument("d_code", "def normalize_temperature(value_c): return (value_c + 273.15) / 100.0",
                         "code", frozenset({"function", "conversion"}), format="python", metadata={"tenant": "a"}),
        FixtureDocument("d_table", "year panels output\n2023 12 40\n2024 18 63\n2025 21 77",
                         "table", frozenset({"year", "count", "output"}), format="table", metadata={"tenant": "b"}),
        FixtureDocument("d_metadata", '{"owner":"ops","tags":["alpine","night"],"status":"active"}',
                         "metadata", frozenset({"owner", "tag", "status"}), format="json", metadata={"tenant": "b"}),
        FixtureDocument("d_ocr", "WARN1NG: north gate closed after ra1n; report to the guard.",
                         "ocr", frozenset({"warning", "location", "condition"}), format="ocr", metadata={"tenant": "b"}),
        FixtureDocument("d_fr", "Le renard traverse la riviere avant la nuit.", "multilingual",
                         frozenset({"animal", "movement", "time"}), language="fr", metadata={"tenant": "c"}),
        FixtureDocument("d_es", "La energia solar reduce emisiones y requiere almacenamiento nocturno.",
                         "multilingual", frozenset({"mechanism", "benefit", "tradeoff"}), language="es", metadata={"tenant": "c"}),
        FixtureDocument("d_conflict", "A small report claims wind output is always constant, contrary to field data.",
                         "conflict", frozenset({"claim", "contradiction"}), metadata={"tenant": "a"}),
        FixtureDocument("d_long", "Solar energy is useful. " * 20 + "Storage balances the night cycle.",
                         "long", frozenset({"mechanism", "tradeoff"}), format="long", metadata={"tenant": "a"}),
        FixtureDocument("d_solar_copy", "Solar panels use sunlight to lower emissions but need storage at night.",
                         "duplicate", frozenset({"mechanism", "benefit", "tradeoff"}), metadata={"tenant": "a"}),
    ]
    for index in range(distractors):
        topic = "solar" if (seed + index) % 2 == 0 else "wind"
        documents.append(FixtureDocument(
            f"d_distractor_{index}",
            f"A {topic} policy note discusses output, planning, and local permits.",
            "distractor", frozenset({"policy", "planning"}), metadata={"tenant": "d"},
        ))
    queries = [
        QueryCase("q_solar", "Which energy source uses sunlight and lowers emissions?", True, True,
                  "conceptual", frozenset({"mechanism", "benefit"}), {"d_solar": 3, "d_solar_copy": 2, "d_wind": 1}),
        QueryCase("q_code", "What does normalize_temperature do?", True, True, "code",
                  frozenset({"function", "conversion"}), {"d_code": 3}),
        QueryCase("q_table", "How many panels were installed in 2024?", True, True, "table",
                  frozenset({"year", "count"}), {"d_table": 3}),
        QueryCase("q_metadata", "Find records tagged alpine with owner ops.", True, True, "metadata",
                  frozenset({"tag", "owner"}), {"d_metadata": 3}),
        QueryCase("q_ocr", "What warning appears near the north gate?", True, True, "ocr",
                  frozenset({"warning", "location"}), {"d_ocr": 3}),
        QueryCase("q_french", "Quel animal traverse la riviere?", True, True, "multilingual",
                  frozenset({"animal", "movement"}), {"d_fr": 3}),
        QueryCase("q_composite", "Which source uses sunlight and requires storage at night?", True, True,
                  "compositional", frozenset({"mechanism", "tradeoff"}), {"d_solar": 3, "d_es": 2}),
        QueryCase("q_ambiguous", "What happened at the gate?", True, True, "ambiguous",
                  frozenset({"location"}), {"d_ocr": 1}),
        QueryCase("q_misspelled", "Which report mentiones solar storaje?", True, True, "misspelled",
                  frozenset({"mechanism"}), {"d_solar": 3, "d_solar_copy": 2}),
        QueryCase("q_conversational", "I am trying to remember that code thing that changes celsius.", True, True,
                  "conversational", frozenset({"function", "conversion"}), {"d_code": 3}),
        QueryCase("q_negated", "Find energy records that are not about wind.", True, True, "negated",
                  frozenset({"mechanism"}), {"d_solar": 3, "d_es": 2, "d_wind": 0}),
        QueryCase("q_temporal", "What changed between the 2023 and 2024 rows?", True, True, "temporal",
                  frozenset({"year", "count"}), {"d_table": 3}),
        QueryCase("q_conditional", "If sunlight disappears at night, which source discusses storage?", True, True,
                  "conditional", frozenset({"mechanism", "tradeoff"}), {"d_solar": 3, "d_es": 2}),
        QueryCase("q_contradictory", "Which report says wind is constant but admits field disagreement?", True, True,
                  "contradictory", frozenset({"claim", "contradiction"}), {"d_conflict": 3}),
        QueryCase("q_vague", "That thing by the gate.", True, True, "vague",
                  frozenset({"location"}), {"d_ocr": 1}),
        QueryCase("q_cross_language", "Which Spanish source describes energy storage after dark?", True, True,
                  "cross_lingual", frozenset({"mechanism", "tradeoff"}), {"d_es": 3}),
        QueryCase("q_source", "normalize_temperature value_c 273.15", False, False, "source-derived",
                  frozenset({"function"}), {"d_code": 3}),
        QueryCase("q_none", "Which document explains lunar battery gardening?", True, True, "no-answer",
                  frozenset(), {}, answerable=False),
    ]
    return Fixture(tuple(documents), tuple(queries))


# Evaluate an arbitrary retriever over annotated query cases.
#
# Parameters:
#   retriever (Callable[[str], Sequence[DocumentId]]): Query-to-ranking callable.
#   queries (Sequence[QueryCase]): Annotated queries.
#   k (int): Rank cutoff.
#   policy (JudgmentPolicy): Explicit handling for unjudged results.
#
# Returns:
#   (dict[str, Any]): Standard evaluate_run report.
#
def evaluate_retriever(
    retriever: Callable[[str], Sequence[DocumentId]],
    queries: Sequence[QueryCase],
    k: int = 10,
    policy: JudgmentPolicy = "nonrelevant",
) -> dict[str, Any]:
    run = {query.query_id: retriever(query.text) for query in queries}
    qrels = {query.query_id: query.relevance for query in queries}
    return evaluate_run(run, qrels, k, policy)


# Run the deterministic fixture baselines and print one JSON report.
#
# Parameters:
#   None: Arguments are read from the command line.
#
# Returns:
#   (None): Prints the benchmark report.
#
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate compact HKM retrieval baselines.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distractors", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(compare_baselines(make_fixture(args.seed, args.distractors), args.k), indent=2))


# Public short aliases commonly used in retrieval reports.
map_score = mean_average_precision
mrr = mean_reciprocal_rank
ndcg = ndcg_at_k
complementary_evidence_coverage = facet_coverage
bootstrap_confidence_interval = bootstrap_ci
randomization_significance = paired_randomization_test


__all__ = [
    "ConfidenceInterval", "DocumentId", "Fixture", "FixtureDocument", "QueryCase",
    "abstention_metrics", "agentic_search", "answer_quality_metrics", "approximate_dense_search", "average_precision",
    "benchmark_build", "benchmark_concurrency", "benchmark_latency", "bm25_search",
    "bootstrap_ci", "bootstrap_confidence_interval", "calibration_metrics", "compare_baselines",
    "complementary_evidence_coverage", "corpus_report", "dense_search", "diversity_at_k",
    "diversity_report", "duplicate_checks", "evaluate_retriever", "evaluate_run", "facet_coverage",
    "facet_coverage_report", "graded_precision_at_k", "graded_recall_at_k", "grounding_metrics",
    "explain_ranking", "hashed_embedding",
    "hybrid_search", "judgment_accounting", "late_interaction_search", "leakage_checks", "lexical_search",
    "make_fixture", "map_score", "mean_average_precision", "mean_reciprocal_rank", "mrr", "ndcg",
    "ndcg_at_k", "paired_randomization_test", "power_analysis", "precision_at_k", "recall_at_k",
    "redundancy_at_k", "randomization_significance", "rerank_results", "result_audit", "retrieval_metrics",
    "provenance_metrics", "run_baselines", "sparse_search", "storage_amplification", "tokenize", "bm25_search_batch", "main",
]


if __name__ == "__main__":
    main()
