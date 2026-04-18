import os

from tlux.search.hkm import jobs
from tlux.search.hkm.tests.support import setup_jobs_root


def test_run_job_can_import_package(tmp_path) -> None:
    """Regression: worker must find tlux.search.hkm modules via injected sys.path."""
    old_root = jobs.JOBS_ROOT
    try:
        fs = setup_jobs_root(tmp_path / "jobs")

        job = jobs.run_job("tlux.search.hkm.tests.job_runner_helper.job_ok")
        jobs.watcher(fs=fs, max_workers=1)
        job.wait_for_completion(poll_interval=0.05)
        job.reload()
        assert job.exit_code == 0, f"stderr: {job.stderr}"
        assert job.status == "SUCCEEDED"
    finally:
        jobs.JOBS_ROOT = old_root
