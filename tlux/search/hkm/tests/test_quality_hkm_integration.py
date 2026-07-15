from tlux.search.hkm import build_search_index, drain_jobs
from tlux.search.hkm.fs import FileSystem
from tlux.search.hkm.search.searcher import Searcher
from tlux.search.hkm.tools.quality_benchmark import (
    QueryCase,
    benchmark_concurrency,
    evaluate_retriever,
    storage_amplification,
)


def test_quality_metrics_evaluate_a_real_hkm_index(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    index_root = tmp_path / "index"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "alpha.txt").write_text("100 101 102 103 104 105 106 107 108", encoding="utf-8")
    (docs / "beta.txt").write_text("200 201 202 203 204 205 206 207 208", encoding="utf-8")
    job = build_search_index(
        str(docs),
        str(index_root),
        num_workers=1,
        max_k=2,
    )
    drain_jobs(FileSystem(root=str(index_root / ".hkm_jobs")), max_workers=1)
    job.reload()
    assert job.status == "SUCCEEDED", job.stderr
    searcher = Searcher.from_index_root(str(index_root))
    queries = [
        QueryCase("alpha", "100", True, True, "natural", relevance={"alpha.txt": 3}),
        QueryCase("beta", "200", True, True, "natural", relevance={"beta.txt": 3}),
    ]

    def retrieve(text: str) -> list[str]:
        return [
            hit.source_path
            for hit in searcher.search({"text": text, "mode": "token", "top_k": 1}).docs
        ]

    report = evaluate_retriever(retrieve, queries, k=1)
    assert report["mean"]["recall_at_k"] == 1.0
    assert report["mean"]["ndcg_at_k"] == 1.0


def test_quality_operational_helpers_report_latency_and_storage() -> None:
    report = benchmark_concurrency(lambda query: query, ["a", "b"], levels=(1, 2), repeats=2)
    assert set(report) == {1, 2}
    assert all(row["p95"] >= row["p50"] >= row["min"] for row in report.values())
    assert all(row["throughput_qps"] > 0.0 for row in report.values())
    assert storage_amplification(20, 10) == 2.0
