import json

import numpy as np

from tlux.search.hkm.tools.benchmark import _exact_rank, _recall, _resource_evidence, render_report


def test_exact_rank_returns_one_best_window_per_document() -> None:
    embeddings = np.asarray([[0.0], [0.01], [2.0]], dtype=np.float32)
    keys = [(1, 0, 4), (1, 4, 8), (2, 0, 4)]
    ranked, work = _exact_rank(embeddings, np.asarray([0.0], dtype=np.float32), keys, 2)
    assert ranked == [(1, 0, 4), (2, 0, 4)]
    assert work == 3


def test_recall_and_report_are_machine_and_human_readable() -> None:
    expected = [(1, 0, 4), (2, 0, 4)]
    assert _recall(expected, [(2, 0, 4), (1, 0, 4)]) == 1.0
    report = {
        "backend": "fake",
        "documents": 2,
        "embedding_windows": 3,
        "canonical_index_bytes": 30,
        "embedding_cache_bytes": 0,
        "bytes_per_window": 10.0,
        "probe_count": 2,
        "queries": [{
            "query": "1 2",
            "recall_at_k": 1.0,
            "hkm_work_windows": 2,
            "exact_work_windows": 3,
            "hkm_work_fraction": 2 / 3,
            "warm_traversal_ms": {"p95": 1.0},
        }],
        "exact_checks": [],
        "filter_check": None,
        "scaling": [{
            "windows": 1000,
            "estimated_index_gib": 0.01,
            "estimated_candidate_windows": 500,
            "estimated_tree_levels": 1,
        }],
    }
    output = render_report(report)
    assert "Recall@k" in output
    assert "1B" in output


def test_resource_evidence_aggregates_valid_samples(tmp_path) -> None:
    resource_path = tmp_path / ".hkm_jobs" / "ids" / "1" / "resources"
    resource_path.parent.mkdir(parents=True)
    resource_path.write_text(
        "\n".join([
            json.dumps({"rss": 4, "cpu_percent": 1.5, "gpu_percent": None}),
            json.dumps({"rss": 8, "cpu_percent": 3.0, "gpu_percent": 12.0}),
            "malformed",
        ]),
        encoding="utf-8",
    )
    report = _resource_evidence(tmp_path)
    assert report["jobs"] == 1
    assert report["samples"] == 2
    assert report["peak_rss_bytes"] == 8
    assert report["peak_cpu_percent"] == 3.0
    assert report["peak_gpu_percent"] == 12.0
