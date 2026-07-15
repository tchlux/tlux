from tlux.search.hkm import build_search_index_from_documents
from tlux.search.hkm.tools.throughput import measure_throughput


def test_throughput_uses_isolated_workers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    root = tmp_path / "idx"
    build_search_index_from_documents(
        str(root),
        [
            {"text": "alpha one", "metadata": {"source_path": "a.txt"}},
            {"text": "beta two", "metadata": {"source_path": "b.txt"}},
        ],
        num_workers=1,
        max_k=1,
    )
    report = measure_throughput(
        str(root),
        ["alpha", "beta"],
        requests=4,
        concurrency=2,
        top_k=1,
        mode="token",
    )
    assert report["completed"] == 4
    assert report["errors"] == []
    assert report["executor"] in {"process", "thread"}
    assert report["throughput_qps"] > 0.0
    assert report["latency_ms"]["p95"] >= 0.0
