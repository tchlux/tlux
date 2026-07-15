import gzip
import json

import pytest

from tlux.search.hkm.tools.recognized_benchmarks import (
    load_beir,
    load_miracl,
    load_trec,
    run_benchmark,
)


def _jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_miracl(path, rows) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_beir(root):
    base = root / "scifact"
    (base / "qrels").mkdir(parents=True)
    _jsonl(base / "corpus.jsonl", [
        {"_id": "d1", "title": "Alpha", "text": "alpha evidence document supports the claim with details and context"},
        {"_id": "d2", "title": "Beta", "text": "beta distractor document discusses unrelated material and background context"},
    ])
    _jsonl(base / "queries.jsonl", [{"_id": "q1", "text": "alpha evidence"}])
    (base / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\n", encoding="utf-8"
    )


def _make_miracl(root):
    corpus = root / "miracl-corpus-v1.0-en"
    topics = root / "miracl-v1.0-en" / "topics"
    qrels = root / "miracl-v1.0-en" / "qrels"
    corpus.mkdir(parents=True)
    topics.mkdir(parents=True)
    qrels.mkdir(parents=True)
    _write_miracl(corpus / "docs-10.jsonl.gz", [{"docid": "d10", "title": "later", "text": "later"}])
    _write_miracl(corpus / "docs-2.jsonl.gz", [{"docid": "d2", "title": "alpha", "text": "alpha evidence document supports the claim with details and context"}])
    (topics / "topics.miracl-v1.0-en-dev.tsv").write_text("q1\talpha evidence\n", encoding="utf-8")
    (qrels / "qrels.miracl-v1.0-en-dev.tsv").write_text("q1\tQ0\td2\t1\n", encoding="utf-8")


def _make_trec(root):
    root.mkdir(parents=True)
    (root / "collection.tsv").write_text(
        "d1\talpha evidence document supports the claim with details and context\n"
        "d2\tbeta distractor document discusses unrelated material and background context\n", encoding="utf-8"
    )
    (root / "queries.tsv").write_text("qid\tquery\nq1\talpha evidence\n", encoding="utf-8")
    (root / "qrels.test").write_text("q1 0 d1 1\n", encoding="utf-8")


def test_loaders_preserve_standard_ids_and_grades(tmp_path) -> None:
    beir_root = tmp_path / "beir"
    miracl_root = tmp_path / "miracl"
    trec_root = tmp_path / "trec"
    _make_beir(beir_root)
    _make_miracl(miracl_root)
    _make_trec(trec_root)

    beir = load_beir(beir_root, "scifact")
    miracl = load_miracl(miracl_root, "en", "dev")
    trec = load_trec(trec_root)

    assert beir.qrels["q1"]["d1"] == 2.0
    assert [row["metadata"]["source_id"] for row in beir.documents()] == ["d1", "d2"]
    assert [row["metadata"]["source_id"] for row in miracl.documents()] == ["d2", "d10"]
    assert trec.qrels["q1"]["d1"] == 1.0
    assert [row["metadata"]["source_id"] for row in trec.documents()] == ["d1", "d2"]


@pytest.mark.parametrize("kind", ["beir", "miracl", "trec"])
def test_runner_builds_and_scores_each_supported_benchmark(tmp_path, monkeypatch, kind) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    root = tmp_path / kind
    if kind == "beir":
        _make_beir(root)
        data = load_beir(root, "scifact")
    elif kind == "miracl":
        _make_miracl(root)
        data = load_miracl(root, "en", "dev")
    else:
        _make_trec(root)
        data = load_trec(root)

    report = run_benchmark(data, tmp_path / f"{kind}-index", mode="hybrid", k=1)

    assert report["queries"] == 1
    assert report["evaluation"]["mean"]["recall_at_k"] == 1.0
