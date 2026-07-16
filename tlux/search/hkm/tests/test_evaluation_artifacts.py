from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tlux.search.hkm.tools.active_search import WindowCandidate, WindowPool, write_exhaustive_map
from tlux.search.hkm.tools.evaluation_artifacts import (
    STATUSES,
    benchmark_svc_configurations,
    load_failure_registry,
    load_hard_queries,
    load_small_lm_map,
    replay_active_labels,
    validate_failure_registry,
)


def _pool() -> WindowPool:
    windows = [
        WindowCandidate(index, 0, 128, 128, np.asarray([1.0, float(index)]), f"text {index}", f"c{index % 2}")
        for index in range(8)
    ]
    return WindowPool(windows)


def test_registry_contains_hard_cases_and_separate_artifact_statuses() -> None:
    records = load_failure_registry()
    assert {record.query_id for record in records}.issuperset({
        "q303", "q437", "q560", "q691", "q887", "q914", "q1213", "q1226",
    })
    assert {record.status for record in records} == STATUSES
    hard = load_hard_queries()
    assert {row["id"] for row in hard}.issuperset({
        "q303", "q437", "q560", "q691", "q887", "q914", "q1213", "q1226",
        "LQ-01", "LQ-02", "LQ-03", "LQ-05", "LQ-12", "LQ-16", "LQ-17",
    })


def test_registry_validation_rejects_duplicate_assessments() -> None:
    row = {
        "query_id": "q",
        "qrels_targets": ["d"],
        "observed_behavior": "missed",
        "assessment": "search failure",
        "status": "confirmed_search_failure",
    }
    with pytest.raises(ValueError, match="unique"):
        validate_failure_registry([row, row])


def test_exhaustive_map_accepts_only_small_lm_oracles(tmp_path: Path) -> None:
    path = tmp_path / "oracle.jsonl"
    pool = _pool()
    counts = write_exhaustive_map(
        path, "hard-query", "request", pool,
        lambda query, window: ("relevant", "match") if window.document_id in {1, 6} else ("not_relevant", "no"),
    )
    labels = load_small_lm_map(path)
    assert counts["relevant"] == 2
    assert {key for key, label in labels["hard-query"].items() if label == "relevant"} == {
        pool.windows[1].key, pool.windows[6].key,
    }
    calls = []
    write_exhaustive_map(path, "hard-query", "request", pool, lambda query, window: calls.append(window.key) or ("relevant", ""))
    assert calls == []
    row = json.loads(path.read_text(encoding="ascii").splitlines()[0])
    row["judge"] = "large_model"
    path.write_text(json.dumps(row) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="not labeled by the small LM"):
        load_small_lm_map(path)


def test_replay_compares_baseline_and_global_svc_choices() -> None:
    pool = _pool()
    labels = {window.key: "relevant" if window.document_id in {0, 7} else "not_relevant" for window in pool.windows}
    replay = replay_active_labels(pool, labels, list(range(8)), 2, "linear")
    assert replay["calls_to_final_positive"] <= 8
    assert replay["baseline_calls_to_final_positive"] == 8
    report = benchmark_svc_configurations([(pool, labels, list(range(8)))], (1, 2), ("linear", "rbf"))
    assert report["selected"]["kernel"] in {"linear", "rbf"}
    assert report["selected"]["temporary_negatives"] in {1, 2}


def test_runtime_import_does_not_load_evaluation_labels() -> None:
    package_root = Path(__file__).resolve().parents[4]
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(package_root)!r}); "
        "import tlux.search.hkm.tools.active_search; "
        "print('tlux.search.hkm.tools.evaluation_artifacts' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
