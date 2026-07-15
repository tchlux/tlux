from tlux.search.hkm import build_search_index_from_documents
from tlux.search.hkm.tools.benchmark import measure_index


def test_approximate_report_compares_probe_to_exact_oracle(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    report_root = tmp_path / "index"
    build_search_index_from_documents(
        str(report_root),
        [
            {"text": "1 2 3 4 5 6 7 8", "metadata": {"source_path": "a.txt"}},
            {"text": "10 11 12 13 14 15 16 17", "metadata": {"source_path": "b.txt"}},
            {"text": "20 21 22 23 24 25 26 27", "metadata": {"source_path": "c.txt"}},
            {"text": "30 31 32 33 34 35 36 37", "metadata": {"source_path": "d.txt"}},
        ],
        num_workers=1,
        max_k=2,
    )
    report = measure_index(
        str(report_root),
        ["1 2"],
        top_k=1,
        repeats=1,
        probe_count=1,
        exact_queries=["1 2"],
    )
    query = report["queries"][0]
    assert report["probe_count"] == 1
    assert 0.0 <= query["recall_at_k"] <= 1.0
    assert query["quantized_recall_at_k"].keys() == {"float16", "int8"}
    assert query["window_ablation"]
    assert report["exact_checks"][0]["exact"] is True
