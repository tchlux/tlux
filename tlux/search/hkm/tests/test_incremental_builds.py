"""Incremental build tests for embedding cache and active-document updates."""

import json
from pathlib import Path

from tlux.search.hkm import Searcher, build_search_index, drain_jobs
from tlux.search.hkm.fs import FileSystem


# Build an index and drain local jobs.
#
# Arguments:
#   docs (Path): Source corpus directory.
#   index_root (Path): Index root.
#   kwargs (dict): Build overrides.
#
# Returns:
#   (dict): Ingest summary.
#
def _build(docs: Path, index_root: Path, **kwargs) -> dict:
    job = build_search_index(
        str(docs),
        str(index_root),
        1,
        fs_root=str(index_root),
        max_k=2,
        leaf_doc_limit=1,
        leaf_embedding_limit=1,
        seed=0,
        **kwargs,
    )
    drain_jobs(FileSystem(root=str(index_root / ".hkm_jobs")), max_workers=1)
    job.reload()
    assert job.status == "SUCCEEDED", job.stderr
    return json.loads((index_root / "manifests" / "ingest_summary.json").read_text(encoding="utf-8"))


# Return source paths from a token search.
#
# Arguments:
#   index_root (Path): Index root.
#   text (str): Token query text.
#
# Returns:
#   (list[str]): Hit source paths.
#
def _token_paths(index_root: Path, text: str) -> list[str]:
    return [hit.source_path for hit in Searcher.from_index_root(str(index_root)).search({"mode": "token", "text": text, "top_k": 10}).docs]


# Count canonical index bytes while excluding transient jobs and embedding cache.
def _canonical_bytes(index_root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in index_root.rglob("*")
        if path.is_file() and not {".hkm_jobs", ".hkm_cache"}.intersection(path.relative_to(index_root).parts)
    )


def test_incremental_reuses_adds_changes_and_deletes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "keep.txt").write_text("1 2 3", encoding="utf-8")
    (docs / "change.txt").write_text("10 11 12", encoding="utf-8")
    (docs / "delete.txt").write_text("20 21 22", encoding="utf-8")
    index_root = tmp_path / "idx"

    first = _build(docs, index_root)
    assert first["cache_misses"] == 3
    assert (index_root / "manifests" / "source_snapshot.json").exists()

    unchanged = _build(docs, index_root)
    assert unchanged["reused"] == 3
    assert unchanged["cache_hits"] == 3
    assert unchanged["cache_misses"] == 0

    (docs / "change.txt").write_text("10 11 99", encoding="utf-8")
    (docs / "delete.txt").unlink()
    (docs / "new.txt").write_text("30 31 32", encoding="utf-8")
    updated = _build(docs, index_root)
    assert updated["reused"] == 1
    assert updated["changed"] == 1
    assert updated["new"] == 1
    assert updated["deleted"] == 1
    manifest = json.loads((index_root / "index.json").read_text(encoding="utf-8"))
    assert manifest["build_config"]["staging_copy_bytes"] > 0
    assert manifest["build_config"]["staging_copy_seconds"] >= 0

    assert _token_paths(index_root, "3") == ["keep.txt"]
    assert _token_paths(index_root, "99") == ["change.txt"]
    assert _token_paths(index_root, "32") == ["new.txt"]
    assert _token_paths(index_root, "12") == []
    assert _token_paths(index_root, "22") == []
    incremental_bytes = _canonical_bytes(index_root)

    searcher = Searcher.from_index_root(str(index_root))
    leaf_paths = [path.parent for path in (index_root / "hkm").rglob("node.json") if json.loads(path.read_text()).get("is_leaf")]
    browsed = [hit.source_path for leaf in leaf_paths for hit in searcher.leaf_docs(leaf)]
    assert "delete.txt" not in browsed

    compacted = _build(docs, index_root, incremental=False)
    assert compacted["cache_hits"] >= 3
    assert _canonical_bytes(index_root) <= incremental_bytes


def test_full_rebuild_uses_embedding_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("1 2 3", encoding="utf-8")
    (docs / "b.txt").write_text("4 5 6", encoding="utf-8")
    index_root = tmp_path / "idx"

    assert _build(docs, index_root)["cache_misses"] == 2
    rebuilt = _build(docs, index_root, incremental=False)
    assert rebuilt["reused"] == 0
    assert rebuilt["cache_hits"] == 2
    assert rebuilt["cache_misses"] == 0


def test_incremental_split_only_touched_oversized_leaf(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "low.txt").write_text("1 2 3", encoding="utf-8")
    (docs / "high.txt").write_text("100 101 102", encoding="utf-8")
    index_root = tmp_path / "idx"
    _build(docs, index_root)

    (docs / "low_new.txt").write_text("2 3 4", encoding="utf-8")
    _build(docs, index_root)

    child_nodes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (index_root / "hkm").glob("cluster_*/node.json")
    ]
    assert any(node.get("children") for node in child_nodes)
    assert _token_paths(index_root, "4") == ["low_new.txt"]
