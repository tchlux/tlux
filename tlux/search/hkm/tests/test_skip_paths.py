import json
import os
import subprocess
from pathlib import Path

from tlux.search.hkm import Searcher, build_search_index, drain_jobs
from tlux.search.hkm.fs import FileSystem


def test_skip_paths_excludes_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs = tmp_path / "docs"
    docs.mkdir()
    keep = docs / "keep.txt"
    skip_dir = docs / "skipme"
    skip_dir.mkdir()
    skip_file = skip_dir / "skip.txt"
    keep.write_text("keep me", encoding="utf-8")
    skip_file.write_text("skip me", encoding="utf-8")

    index_root = tmp_path / "idx"
    index_root.mkdir()

    build_search_index(
        docs_dir=str(docs),
        index_root=str(index_root),
        num_workers=1,
        fs_root=str(index_root),
        skip_paths=[str(skip_dir)],
    )
    drain_jobs(FileSystem(root=str(index_root / ".hkm_jobs")), max_workers=1)

    manifest = Path(index_root / "manifests" / "worker_0000.json")
    assert manifest.exists()
    files = json.loads(manifest.read_text(encoding="utf-8"))
    assert str(keep) in files
    assert str(skip_file) not in files


def _manifest_files(index_root: Path) -> list[str]:
    return json.loads((index_root / "manifests" / "worker_0000.json").read_text(encoding="utf-8"))


def _summary(index_root: Path) -> dict:
    return json.loads((index_root / "manifests" / "ingest_summary.json").read_text(encoding="utf-8"))


def test_hkm_index_cli_accepts_repeatable_skip(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    keep = docs / "keep.txt"
    skip_dir = docs / "skipme"
    skip_dir.mkdir()
    keep.write_text("keep me", encoding="utf-8")
    (skip_dir / "skip.txt").write_text("skip me", encoding="utf-8")

    index_root = tmp_path / "idx"
    env = dict(os.environ, HKM_FAKE_EMBEDDER="1")
    completed = subprocess.run(
        [
            str(Path(__file__).parents[1] / "bin" / "hkm-index"),
            str(index_root),
            str(docs),
            "--workers",
            "1",
            "--skip",
            str(skip_dir),
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert _manifest_files(index_root) == [str(keep)]
    root_job_id = completed.stdout.strip().splitlines()[0]
    root_config_path = index_root / ".hkm_jobs" / "ids" / root_job_id / "job_config"
    root_config = json.loads(root_config_path.read_text(encoding="utf-8"))
    assert root_config["status"] == "SUCCEEDED"
    assert (index_root / "hkm" / "node.json").exists()
    for bucket in ("waiting", "queued", "running"):
        assert not list((index_root / ".hkm_jobs" / bucket).iterdir())


def test_default_skips_exclude_generated_and_binary_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs = tmp_path / "docs"
    docs.mkdir()
    keep = docs / "keep.txt"
    keep.write_text("keep me", encoding="utf-8")
    for path in [
        docs / ".git" / "config",
        docs / ".env" / "pyvenv.cfg",
        docs / "__pycache__" / "x.pyc",
        docs / "tmp_index_old" / "old.txt",
        docs / "shard.hkmchunk" / "tokens.bin",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("skip me", encoding="utf-8")
    for path in [
        docs / "tokenizer.json",
        docs / "tokenizer_config.json",
        docs / "weights.safetensors",
        docs / "image.png",
    ]:
        path.write_text("skip me", encoding="utf-8")

    index_root = tmp_path / "idx"
    build_search_index(str(docs), str(index_root), 1, fs_root=str(index_root))

    assert _manifest_files(index_root) == [str(keep)]
    summary = _summary(index_root)
    assert summary["planned"] == 1
    assert summary["skipped"] == 9


def test_include_and_exclude_globs_use_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs = tmp_path / "docs"
    (docs / "sub").mkdir(parents=True)
    keep = docs / "keep.txt"
    keep.write_text("keep me", encoding="utf-8")
    (docs / "drop.md").write_text("drop me", encoding="utf-8")
    (docs / "sub" / "drop.txt").write_text("drop me", encoding="utf-8")

    index_root = tmp_path / "idx"
    build_search_index(
        str(docs),
        str(index_root),
        1,
        fs_root=str(index_root),
        include_globs=["*.txt"],
        exclude_globs=["sub/*"],
    )

    assert _manifest_files(index_root) == [str(keep)]
    summary = _summary(index_root)
    assert summary["skip_reasons"]["include_glob"] == 1
    assert summary["skip_reasons"]["exclude_glob"] == 1


def test_size_token_and_decode_skips_are_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs = tmp_path / "docs"
    docs.mkdir()
    keep = docs / "keep.txt"
    keep.write_text("1 2", encoding="utf-8")
    (docs / "large.txt").write_text("too large", encoding="utf-8")
    (docs / "too_many_tokens.txt").write_text("1 2 3", encoding="utf-8")
    (docs / "bad.txt").write_bytes(b"\xff\xfe\x00")

    index_root = tmp_path / "idx"
    root_job = build_search_index(
        str(docs),
        str(index_root),
        1,
        fs_root=str(index_root),
        max_file_bytes=7,
        max_tokens=2,
    )
    drain_jobs(FileSystem(root=str(index_root / ".hkm_jobs")), max_workers=1)
    root_job.reload()

    assert root_job.status == "SUCCEEDED", root_job.stderr
    summary = _summary(index_root)
    assert summary["planned"] == 3
    assert summary["indexed"] == 1
    assert summary["skipped"] == 2
    assert summary["failed"] == 1
    assert summary["skip_reasons"]["max_file_bytes"] == 1
    assert summary["skip_reasons"]["max_tokens"] == 1
    assert summary["failed_files"][0]["reason"] == "decode_error"


def test_relative_build_paths_publish_under_index_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    monkeypatch.chdir(tmp_path)
    docs = Path("data/docs")
    docs.mkdir(parents=True)
    docs.joinpath("keep.txt").write_text("1 2 3", encoding="utf-8")

    root_job = build_search_index("data/docs", "data/idx", 1)
    index_root = tmp_path / "data" / "idx"
    drain_jobs(FileSystem(root=str(index_root / ".hkm_jobs")), max_workers=1)
    root_job.reload()

    assert root_job.status == "SUCCEEDED", root_job.stderr
    assert sorted(index_root.rglob("*.hkmchunk"))
    assert not (tmp_path / "data" / "data").exists()
    hits = Searcher.from_index_root("data/idx").search({"mode": "token", "text": "1", "top_k": 10})
    assert [hit.source_path for hit in hits.docs] == ["keep.txt"]


def test_all_worker_skipped_documents_fail_root_build(tmp_path, monkeypatch):
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs = tmp_path / "docs"
    docs.mkdir()
    docs.joinpath("too_many_tokens.txt").write_text("1 2 3", encoding="utf-8")

    index_root = tmp_path / "idx"
    root_job = build_search_index(
        str(docs),
        str(index_root),
        1,
        fs_root=str(index_root),
        max_tokens=2,
    )
    drain_jobs(FileSystem(root=str(index_root / ".hkm_jobs")), max_workers=1)
    root_job.reload()

    assert root_job.status == "FAILED"
    assert "Build indexed zero documents" in root_job.stderr
    assert "max_tokens" in root_job.stderr
    summary = _summary(index_root)
    assert summary["planned"] == 1
    assert summary["indexed"] == 0
    assert summary["skip_reasons"]["max_tokens"] == 1
