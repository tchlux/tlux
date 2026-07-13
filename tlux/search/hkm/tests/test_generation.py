from pathlib import Path

import json

from tlux.search.hkm import Searcher, build_search_index, build_search_index_from_documents, drain_jobs
from tlux.search.hkm.builder.launcher import publish_generation
from tlux.search.hkm.fs import FileSystem


def test_publish_generation_swaps_public_pointer_and_keeps_old(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    public = tmp_path / "index"
    public.mkdir()
    (public / "old.txt").write_text("old", encoding="utf-8")
    generation = tmp_path / "generation"
    build_search_index_from_documents(
        str(generation),
        [{"text": "new", "metadata": {"source_path": "new.txt"}}],
        max_k=1,
    )
    active = generation / json.loads((generation / "index.json").read_text(encoding="utf-8"))["generation_path"]

    publish_generation(str(active), str(public))

    assert not public.is_symlink()
    manifest = json.loads((public / "index.json").read_text(encoding="utf-8"))
    assert (public / manifest["generation_path"]).is_dir()
    assert (public / "old.txt").read_text(encoding="utf-8") == "old"


def test_failed_generation_keeps_previous_searchable_index(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "one.txt"
    source.write_text("1 2 3", encoding="utf-8")
    index = tmp_path / "index"

    first = build_search_index(str(docs), str(index), 1, fs_root=str(index), max_k=1)
    drain_jobs(FileSystem(root=str(index / ".hkm_jobs")), max_workers=1)
    first.reload()
    assert first.status == "SUCCEEDED", first.stderr
    previous = json.loads((index / "index.json").read_text(encoding="utf-8"))["generation_path"]

    source.write_text("1 2 3", encoding="utf-8")
    failed = build_search_index(
        str(docs),
        str(index),
        1,
        fs_root=str(index),
        max_k=1,
        max_tokens=1,
        incremental=False,
    )
    drain_jobs(FileSystem(root=str(index / ".hkm_jobs")), max_workers=1)
    failed.reload()

    assert failed.status == "FAILED"
    assert json.loads((index / "index.json").read_text(encoding="utf-8"))["generation_path"] == previous
    assert (index / previous).is_dir()
    assert Searcher.from_index_root(str(index)).search({"mode": "token", "text": "1 2"}).docs
