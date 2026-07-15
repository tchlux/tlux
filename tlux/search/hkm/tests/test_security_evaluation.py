from tlux.search.hkm.tools.quality_benchmark import (
    explain_ranking,
    grounding_metrics,
    provenance_metrics,
    result_audit,
)


def test_result_audit_reports_permission_tenant_and_provenance_failures() -> None:
    results = [
        {"doc_id": "allowed", "source_path": "allowed.txt", "tenant": "a"},
        {"doc_id": "forbidden", "source_path": "forbidden.txt", "tenant": "b"},
        {"doc_id": "missing", "tenant": "b"},
    ]
    report = result_audit(
        results,
        {"allowed": {}, "forbidden": {}},
        allowed_ids={"allowed"},
        tenant="a",
    )
    assert report["valid"] is False
    assert report["unknown_document"] == 1
    assert report["forbidden_document"] == 2
    assert report["missing_citation"] == 1
    assert report["missing_provenance"] == 1
    assert report["tenant_mismatch"] == 2


def test_provenance_explanations_and_grounding_keep_failures_visible() -> None:
    valid = {
        "source_path": "a.txt",
        "span": [0, 4],
        "document": {"source_id": "a", "content_hash": "hash", "build_id": "build"},
    }
    invalid = {"source_path": "a.txt", "span": [-1, 4], "document": valid["document"]}
    provenance = provenance_metrics([valid, valid, invalid])
    assert provenance["valid_citations"] == 2
    assert provenance["duplicate_citations"] == 1

    explanations = explain_ranking(
        "solar storage", ["solar", "unknown"], {"solar": "Solar panels need storage at night."}
    )
    assert explanations[0]["matched_terms"] == ("solar", "storage")
    assert explanations[1]["term_coverage"] == 0.0

    grounding = grounding_metrics(
        "solar storage claim", ["solar panels need storage"], ["solar", "unknown"], ["solar"]
    )
    assert grounding["answer_term_support"] == 2 / 3
    assert grounding["citation_precision"] == 0.5
    assert grounding["citation_recall"] == 1.0
