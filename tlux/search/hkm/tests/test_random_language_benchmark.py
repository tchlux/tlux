from types import SimpleNamespace

import pytest

from tlux.search.hkm.tools.random_language_benchmark import (
    _metrics,
    _evidence_gate_failed,
    _parse_model_query,
    _query_overlaps_evidence,
    build_cases,
    deterministic_query,
    model_query,
)


def test_deterministic_queries_keep_evidence_terms_and_vary_style() -> None:
    excerpt = "The archivist crosses the basalt bridge beneath thunder."
    queries = [deterministic_query(excerpt, style) for style in (
        "vague", "specific", "conditional", "missing_entity",
    )]
    assert all("archivist" in query and "crosses" in query for query in queries[:3])
    assert all("basalt" in query for query in queries)
    assert "archivist" not in queries[-1]
    assert len(set(queries)) == 4
    assert "who or what" in queries[-1]
    assert "If " in queries[2]
    assert max(len(query.split()) for query in queries) <= 24


def test_build_cases_is_reproducible_and_rejects_unknown_style() -> None:
    samples = [SimpleNamespace(
        sample_id=3,
        doc_id=17,
        source_path="source.md",
        excerpt="A silver bridge crosses the ravine.",
    )]
    cases = build_cases(samples, ["vague", "conditional"])
    assert [(case.sample_id, case.style) for case in cases] == [
        (3, "vague"), (3, "conditional"),
    ]
    assert all(case.doc_id == 17 for case in cases)
    with pytest.raises(ValueError, match="unknown query styles"):
        build_cases(samples, ["made-up"])


def test_parse_model_query_accepts_bounded_json() -> None:
    assert _parse_model_query('```json\n{"query":"find the hidden bridge"}\n```') == (
        "find the hidden bridge"
    )
    with pytest.raises(ValueError, match="no query"):
        _parse_model_query('{"query":""}')


def test_model_query_requires_an_evidence_term() -> None:
    assert _query_overlaps_evidence("find the basalt bridge", "A basalt bridge")
    assert not _query_overlaps_evidence("find the missing event", "A basalt bridge")


def test_model_query_uses_optional_generator_method() -> None:
    class Generator:
        def generate_memory_query(self, excerpt: str, style: str) -> str:
            return "find the basalt bridge"

    assert model_query(Generator(), "A basalt bridge", "specific") == "find the basalt bridge"


def test_metrics_report_recall_precision_and_mrr() -> None:
    rows = [{"target_rank": 1}, {"target_rank": 3}, {"target_rank": None}]
    metrics = _metrics(rows, "target_rank")
    assert metrics["recall_at_k"] == pytest.approx(2 / 3)
    assert metrics["precision_at_1"] == pytest.approx(1 / 3)
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3) / 3)


def test_evidence_gate_requires_perfect_metrics() -> None:
    passing = {"errors": [], "evidence": {"recall_at_k": 1.0, "precision_at_1": 1.0, "mrr": 1.0}}
    failing = {"errors": [], "evidence": {"recall_at_k": 1.0, "precision_at_1": 0.5, "mrr": 1.0}}
    assert _evidence_gate_failed(passing) is False
    assert _evidence_gate_failed(failing) is True
