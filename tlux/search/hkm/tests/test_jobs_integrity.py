import tempfile
from pathlib import Path

import pytest

from tlux.search.hkm import jobs
from tlux.search.hkm.tests.support import setup_jobs_root


@pytest.fixture()
def temp_jobs_root():
    old = jobs.JOBS_ROOT
    root = Path(tempfile.mkdtemp())
    setup_jobs_root(root)
    yield root
    jobs.JOBS_ROOT = old


def test_job_dir_has_config(temp_jobs_root):
    job = jobs.run_job("tlux.search.hkm.tests.job_runner_helper.job_ok")
    fs = setup_jobs_root(temp_jobs_root)
    jobs.watcher(fs=fs, max_workers=1)
    job_dir = Path(temp_jobs_root) / "ids" / job.id
    assert job_dir.exists(), "job id directory should be created under ids/"
    assert (job_dir / "job_config").exists(), "job_config must exist in ids directory"
    job.wait_for_completion(poll_interval=0.05)
    finished_dir = Path(temp_jobs_root) / "succeeded" / job.id
    assert finished_dir.exists(), "job should move to succeeded/"
    assert (job_dir / "job_config").exists(), "job_config must persist after completion"
