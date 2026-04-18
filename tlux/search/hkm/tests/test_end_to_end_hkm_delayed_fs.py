from pathlib import Path

from tlux.search.hkm import Searcher, build_search_index, jobs
from tlux.search.hkm.fs import make_filesystem
from tlux.search.hkm.tests.support import enable_delayed_fs, run_watchers, sync_fs


def _write_corpus(root: Path) -> Path:
    docs = root / "corpus"
    docs.mkdir()
    for name, text in {
        "a.txt": "0 1 2 3 4",
        "b.txt": "10 11 12 13 14 99",
        "c.txt": "20 21 22 23 24",
        "d.txt": "40 41 42 43 44 45 99 777",
    }.items():
        (docs / name).write_text(text, encoding="utf-8")
    return docs


def test_hkm_build_on_delayed_fs_stays_queryable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    enable_delayed_fs(monkeypatch, listdir_delay=0.01)
    docs = _write_corpus(tmp_path)
    root_job = build_search_index(
        docs_dir=str(docs),
        index_root=str(tmp_path),
        num_workers=2,
        max_k=2,
        leaf_embedding_limit=1,
        leaf_doc_limit=1,
        seed=0,
    )
    jobs_fs = make_filesystem(str(tmp_path / ".hkm_jobs"))
    run_watchers(jobs_fs, watcher_count=2, max_workers=2, timeout=20.0)
    sync_fs(jobs_fs, settle=0.05)
    sync_fs(make_filesystem(str(tmp_path)), settle=0.05)
    root_job.reload()
    assert root_job.status == "SUCCEEDED", root_job.stderr
    searcher = Searcher.from_index_root(str(tmp_path))
    token_hits = searcher.search({"mode": "token", "text": "99", "top_k": 5})
    semantic_hits = searcher.search({"mode": "semantic", "text": "40 41 42 43 44 45 99 777", "top_k": 2})
    assert token_hits.docs
    assert semantic_hits.docs
    assert semantic_hits.docs[0].source_path == "d.txt"
    for node_path in (tmp_path / "hkm").rglob("node.json"):
        node_dir = node_path.parent
        assert (node_dir / "n_gram_counter.bytes").exists()
        if node_dir != (tmp_path / "hkm"):
            assert node_path.exists()


def test_hkm_build_recovery_after_watcher_kill_reaches_terminal_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    enable_delayed_fs(monkeypatch, listdir_delay=0.01)
    docs = _write_corpus(tmp_path)
    root_job = build_search_index(
        docs_dir=str(docs),
        index_root=str(tmp_path),
        num_workers=2,
        max_k=2,
        leaf_embedding_limit=1,
        leaf_doc_limit=1,
        seed=0,
    )
    jobs_fs = make_filesystem(str(tmp_path / ".hkm_jobs"))
    run_watchers(jobs_fs, watcher_count=1, max_workers=1, kill_one=True, timeout=20.0)
    sync_fs(jobs_fs, settle=0.05)
    sync_fs(make_filesystem(str(tmp_path)), settle=0.05)
    root_job.reload()
    assert root_job.status in {"SUCCEEDED", "FAILED"}
    assert not jobs_fs.listdir("waiting")
    assert not jobs_fs.listdir("queued")
    assert not jobs_fs.listdir("running")
    if root_job.status == "SUCCEEDED":
        hits = Searcher.from_index_root(str(tmp_path)).search({"mode": "token", "text": "777", "top_k": 1})
        assert hits.docs and hits.docs[0].source_path == "d.txt"
    else:
        assert jobs_fs.listdir("failed")
