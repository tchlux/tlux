from tlux.search.hkm.tools.capacity import measure_capacity, measure_capacity_sweep


def test_capacity_report_records_measured_build_and_storage(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    report = measure_capacity(
        str(tmp_path / "one"),
        document_count=2,
        tokens_per_document=8,
        max_k=2,
    )
    assert report["measurement"] == {
        "status": "measured",
        "corpus": "deterministic_numeric_documents",
        "extrapolated": False,
    }
    assert report["build"]["documents_indexed"] == 2
    assert report["build"]["seconds"] > 0.0
    assert report["storage"]["source_bytes"] > 0
    assert report["storage"]["canonical_index_bytes"] > 0
    assert report["storage"]["bytes_per_source_byte"] > 0.0
    assert report["hardware"]["cpu_count"] >= 1


def test_capacity_sweep_keeps_each_size_measured(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    report = measure_capacity_sweep(
        str(tmp_path / "sweep"),
        document_counts=(2, 3),
        tokens_per_document=4,
        max_k=2,
    )
    assert report["document_counts"] == [2, 3]
    assert len(report["rows"]) == 2
    assert all(row["measurement"]["status"] == "measured" for row in report["rows"])
    assert all(row["build"]["documents_indexed"] == count for row, count in zip(report["rows"], (2, 3)))
    assert report["scaling_note"].startswith("Every row is measured")
