import json
import tempfile
import time
from pathlib import Path

import pytest
import multiprocessing

from tlux.search.hkm import jobs


def _setup_jobs_root() -> tuple[str, jobs.FileSystem]:
    root = Path(tempfile.mkdtemp()) / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    fs = jobs.FileSystem(str(root))
    for bucket in ("ids", "waiting", "queued", "running", "succeeded", "failed", "next", "workers"):
        fs.mkdir(fs.join(bucket), exist_ok=True)
    jobs.JOBS_ROOT = str(root)
    return str(root), fs


def test_worker_executes_job_and_records_resources():
    _, fs = _setup_jobs_root()
    job = jobs.run_job("tlux.search.hkm.tests.job_runner_helper.job_cpu_burner")
    jobs.watcher(fs=fs, max_workers=1)
    job.wait_for_completion(poll_interval=0.1)
    assert job.status == "SUCCEEDED"
    res_path = Path(job.path) / "resources"
    assert res_path.exists()
    lines = [ln for ln in res_path.read_text().splitlines() if ln.strip()]
    assert lines, "resources heartbeat should be recorded"
    last = json.loads(lines[-1])
    assert "cpu_percent" in last


def test_job_kill_sets_failed_status():
    _, fs = _setup_jobs_root()
    job = jobs.run_job("tlux.search.hkm.tests.job_runner_helper.job_cpu_burner")
    jobs.watcher(fs=fs, max_workers=1)
    with pytest.raises(RuntimeError):
        job.kill(reason="test kill")
