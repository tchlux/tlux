import json

import pytest

from tlux.search.hkm.tools.beir_benchmark import evaluate_bm25, load_beir_dataset
from tlux.search.hkm.tools.quality_benchmark import bm25_search, bm25_search_batch


def _jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _make_dataset(root) -> None:
    (root / "qrels").mkdir(parents=True)
    _jsonl(root / "corpus.jsonl", [
        {"_id": "d1", "title": "Alpha", "text": "alpha evidence supports the claim"},
        {"_id": "d2", "title": "Beta", "text": "beta evidence supports another claim"},
    ])
    _jsonl(root / "queries.jsonl", [
        {"_id": "q1", "text": "alpha evidence"},
        {"_id": "q2", "text": "beta evidence"},
    ])
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq2\td2\t1\n",
        encoding="utf-8",
    )


def test_bm25_batch_matches_single_query_search() -> None:
    documents = {"a": "alpha evidence", "b": "beta evidence"}
    queries = ["alpha", "beta"]
    assert bm25_search_batch(queries, documents, 1) == [
        bm25_search(query, documents, 1) for query in queries
    ]


def test_beir_loader_and_bm25_report_use_external_qrels(tmp_path) -> None:
    _make_dataset(tmp_path)
    dataset = load_beir_dataset(tmp_path)
    report = evaluate_bm25(dataset, k=1)

    assert report["corpus_documents"] == 2
    assert report["queries_judged"] == 2
    assert report["queries_evaluated"] == 2
    assert report["metrics"]["mean"]["recall_at_k"] == 1.0


def test_beir_loader_rejects_duplicate_qrels(tmp_path) -> None:
    _make_dataset(tmp_path)
    qrels = tmp_path / "qrels" / "test.tsv"
    qrels.write_text(qrels.read_text(encoding="utf-8") + "q1\td1\t1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate judgment"):
        load_beir_dataset(tmp_path)
