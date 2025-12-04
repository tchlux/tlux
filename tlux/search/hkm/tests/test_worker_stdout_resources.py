import os
import tempfile
from pathlib import Path

from tlux.search.hkm import jobs
from tlux.search.hkm.job_runner_helper import run_default_worker_job


def _setup_jobs_root() -> str:
    root = Path(tempfile.mkdtemp()) / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    fs = jobs.FileSystem(str(root))
    for bucket in ("ids", "queued", "running", "finished", "failed", "next"):
        fs.mkdir(fs.join(bucket), exist_ok=True)
    jobs.JOBS_ROOT = str(root)
    return str(root), fs


def test_worker_logs_and_resources(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("hello world", encoding="utf-8")
    (docs_dir / "b.txt").write_text("another doc", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Fake embedder for speed but still exercise worker path + logging.
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    jobs_root, fs = _setup_jobs_root()

    job = jobs.spawn_job("tlux.search.hkm.job_runner_helper.run_default_worker_job", str(docs_dir), str(out_dir))
    job.wait_for_completion(poll_interval=0.1)

    assert job.status == "FINISHED"
    assert "[worker] start" in job.stdout, "stdout should include worker start log"
    res_path = Path(job.path) / "resources"
    assert res_path.exists(), "resources file missing"
    lines = [ln for ln in res_path.read_text().splitlines() if ln.strip()]
    assert lines, "resources file empty"
    last = lines[-1]
    assert "cpu_percent" in last and "gpu_percent" in last and "rss" in last
