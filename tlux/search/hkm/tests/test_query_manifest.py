import json

import pytest

from tlux.search.hkm.tools.query_manifest import (
    evaluate_query_manifest,
    load_query_manifest,
    query_manifest_report,
)


def _write_manifest(path, records) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_query_manifest_preserves_provenance_and_graded_qrels(tmp_path) -> None:
    path = tmp_path / "queries.jsonl"
    _write_manifest(path, [
        {
            "query_id": "q1",
            "text": "find alpha",
            "natural": True,
            "source_blind": True,
            "category": "conceptual",
            "required_facets": ["mechanism"],
            "answerable": True,
            "relevance": {"a": 3, "b": 1},
        },
        {
            "query_id": "q2",
            "text": "derive beta",
            "natural": False,
            "source_blind": False,
            "category": "source-derived",
            "required_facets": [],
            "answerable": True,
            "relevance": {"b": 2},
        },
        {
            "query_id": "q3",
            "text": "find missing",
            "natural": True,
            "source_blind": True,
            "category": "no-answer",
            "required_facets": [],
            "answerable": False,
            "relevance": {"a": 0},
        },
    ])
    cases = load_query_manifest(path)
    summary = query_manifest_report(cases)
    assert len(cases) == 3
    assert cases[0].relevance["a"] == 3.0
    assert summary["source_blind"] == 2
    assert summary["unanswerable"] == 1
    assert summary["positive_qrels"] == 2
    assert summary["categories"]["source-derived"] == 1

    report = evaluate_query_manifest(
        lambda text: ["a"] if text == "find alpha" else ["b"], path, k=1
    )
    assert report["evaluation"]["mean"]["recall_at_k"] == pytest.approx(0.5)


@pytest.mark.parametrize("mutation", [
    {"duplicate": True},
    {"missing_source_blind": True},
    {"answerability_mismatch": True},
    {"negative_grade": True},
])
def test_query_manifest_rejects_incomplete_or_inconsistent_records(tmp_path, mutation) -> None:
    record = {
        "query_id": "q1",
        "text": "find alpha",
        "natural": True,
        "source_blind": True,
        "category": "conceptual",
        "required_facets": [],
        "answerable": True,
        "relevance": {"a": 1},
    }
    if mutation.get("duplicate"):
        records = [record, dict(record)]
    else:
        records = [dict(record)]
        if mutation.get("missing_source_blind"):
            records[0].pop("source_blind")
        if mutation.get("answerability_mismatch"):
            records[0]["answerable"] = False
        if mutation.get("negative_grade"):
            records[0]["relevance"] = {"a": -1}
    path = tmp_path / "invalid.jsonl"
    _write_manifest(path, records)
    with pytest.raises(ValueError):
        load_query_manifest(path)
