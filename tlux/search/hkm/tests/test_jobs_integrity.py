import tempfile
import time
from pathlib import Path

import pytest

from tlux.search.hkm import jobs


@pytest.fixture()
def temp_jobs_root():
    old = jobs.JOBS_ROOT
    root = Path(tempfile.mkdtemp())
    jobs.JOBS_ROOT = str(root)
    fs = jobs.FileSystem(jobs.JOBS_ROOT)
    for bucket in ("ids", "queued", "running", "finished", "failed", "next"):
        fs.mkdir(fs.join(bucket), exist_ok=True)
    yield root
    jobs.JOBS_ROOT = old


def test_job_dir_has_config(temp_jobs_root):
    job = jobs.spawn_job("tlux.search.hkm.job_runner_helper.job_ok")
    job_dir = Path(temp_jobs_root) / "running" / job.id
    assert job_dir.exists(), "job directory should be created in running/"
    assert (job_dir / "job_config").exists(), "job_config must exist immediately after spawn"
    job.wait_for_completion(poll_interval=0.05)
    finished_dir = Path(temp_jobs_root) / "finished" / job.id
    assert finished_dir.exists(), "job should move to finished/"
    assert (finished_dir / "job_config").exists(), "job_config must persist after completion"
