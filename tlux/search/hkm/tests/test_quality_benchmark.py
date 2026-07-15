from __future__ import annotations

import pytest

from tlux.search.hkm.tools.quality_benchmark import (
    abstention_metrics,
    answer_quality_metrics,
    average_precision,
    bm25_search,
    bootstrap_ci,
    calibration_metrics,
    compare_baselines,
    diversity_at_k,
    evaluate_run,
    facet_coverage_report,
    leakage_checks,
    make_fixture,
    ndcg_at_k,
    paired_randomization_test,
    power_analysis,
    precision_at_k,
    provenance_metrics,
    reciprocal_rank,
)


def test_standard_metrics_use_graded_qrels() -> None:
    ranking = ["a", "b", "c", "d"]
    qrels = {"a": 3, "b": 0, "c": 2, "d": 1}
    assert precision_at_k(ranking, qrels, 3) == 2 / 3
    assert evaluate_run({"q": ranking}, {"q": qrels}, 3)["mean"]["recall_at_k"] == 2 / 3
    assert average_precision(ranking, qrels) == (1 + 2 / 3 + 3 / 4) / 3
    assert reciprocal_rank(ranking, qrels) == 1.0
    assert ndcg_at_k(["a", "c", "d", "b"], qrels, 4) == 1.0


def test_unjudged_policy_is_explicit() -> None:
    ranking = ["a", "unknown"]
    qrels = {"a": 1}
    assert precision_at_k(ranking, qrels, 2, "nonrelevant") == 0.5
    assert precision_at_k(ranking, qrels, 2, "ignore") == 1.0
    report = evaluate_run({"q": ranking}, {"q": qrels}, 2)
    assert report["totals"]["unjudged"] == 1
    assert report["totals"]["judged_fraction"] == 0.5


def test_complementary_evidence_and_diversity_are_separate() -> None:
    doc_facets = {"a": {"first"}, "b": {"first"}, "c": {"second"}}
    report = facet_coverage_report({"first", "second"}, ["a", "c"], doc_facets, 2)
    assert report["complete"] is True
    assert diversity_at_k(["a", "b", "c"], doc_facets, 3) > 0.0


def test_calibration_and_abstention_report_no_answer_behavior() -> None:
    calibration = calibration_metrics([0.9, 0.1], [1, 0], bins=2)
    assert calibration["brier"] == pytest.approx(0.01)
    assert calibration["ece"] == pytest.approx(0.1)
    abstention = abstention_metrics([0.9, 0.2, 0.1], [1, 0, 0], [1, 0, 0], threshold=0.5)
    assert abstention["coverage"] == 1 / 3
    assert abstention["no_answer_accuracy"] == 1.0


def test_statistics_are_seeded_and_power_is_visible() -> None:
    interval_a = bootstrap_ci([1.0, 1.0, 1.0], resamples=50, seed=7)
    interval_b = bootstrap_ci([1.0, 1.0, 1.0], resamples=50, seed=7)
    assert interval_a == interval_b
    assert interval_a.low == interval_a.high == 1.0
    result = paired_randomization_test([1, 1, 1, 1], [0, 0, 0, 0], resamples=100, seed=7)
    assert result["difference"] == 1.0
    assert 0.0 < result["p_value"] <= 1.0
    assert power_analysis(0.5)["required_n"] > 1


def test_leakage_report_detects_duplicates_and_source_derived_queries() -> None:
    report = leakage_checks(
        {"a": "same text", "b": "same text", "c": "other words"},
        ["same text"],
    )
    assert report["duplicates"]["exact_duplicate_groups"]
    assert report["leaked_queries"] == 1
    fixture = make_fixture()
    fixture_report = leakage_checks(fixture.documents, fixture.queries)
    assert fixture_report["source_blind_fraction"] < 1.0


def test_baselines_run_side_by_side_on_heterogeneous_fixture() -> None:
    fixture = make_fixture()
    assert bm25_search("solar storage", fixture.document_map, 1)[0] == "d_solar"
    report = compare_baselines(fixture, k=5)
    assert {"lexical", "bm25", "dense", "sparse", "late_interaction", "hybrid", "reranker", "agentic"} <= set(report)
    assert all("ndcg_at_k" in value["mean"] for value in report.values())
    formats = {document.format for document in fixture.documents}
    assert {"python", "table", "ocr"} <= formats
    assert {
        "conceptual", "compositional", "conditional", "contradictory", "conversational",
        "cross_lingual", "misspelled", "negated", "no-answer", "temporal", "vague",
    } <= {query.category for query in fixture.queries}
    assert any(not query.natural and not query.source_blind for query in fixture.queries)
    assert any(not query.answerable for query in fixture.queries)


def test_answer_faithfulness_and_provenance_are_scored_separately() -> None:
    answer = answer_quality_metrics(
        ["retains seven years"],
        ["retains seven years", "adds unsupported detail"],
        ["retains seven years"],
        ["policy-a", "wrong-source"],
        ["policy-a"],
        ["adds unsupported detail"],
    )
    assert answer["claim_recall"] == 1.0
    assert answer["unsupported_claim_rate"] == 0.5
    assert answer["citation_precision"] == 0.5
    results = [
        {"source_path": "policy.txt", "span": [0, 10], "document": {
            "source_id": "policy", "content_hash": "hash", "build_id": "build",
        }},
        {"source_path": "missing.txt", "span": [0, 10], "document": {}},
    ]
    assert provenance_metrics(results)["valid_citations"] == 1
