from tlux.search.hkm import build_search_index_from_documents
from tlux.search.hkm.tools.inspect import inspect_index


def test_inspect_reports_audited_storage_and_search(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    root = tmp_path / "idx"
    build_search_index_from_documents(
        str(root),
        [{"text": "one two three", "metadata": {"source_path": "one.txt"}}],
        num_workers=1,
        max_k=1,
    )
    report = inspect_index(str(root), ["one"], repeats=1)
    assert report["audit"] == "PASS"
    assert report["documents"] == 1
    assert report["storage_components"]["embeddings"] > 0
    assert report["search"][0]["timing_ms"]["p99"] >= 0.0
    assert report["build"]["resources"]["samples"] >= 1
    assert report["build"]["resources"]["peak_rss_bytes"] >= 0
