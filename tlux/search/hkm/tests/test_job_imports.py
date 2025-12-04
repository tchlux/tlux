import os
import time

from tlux.search.hkm import jobs


def test_spawn_job_can_import_package(tmp_path) -> None:
    """Regression: worker must find tlux.search.hkm modules via injected sys.path."""
    old_root = jobs.JOBS_ROOT
    try:
        jobs_root = tmp_path / "jobs"
        jobs_root.mkdir(parents=True, exist_ok=True)
        jobs.JOBS_ROOT = str(jobs_root)
        fs = jobs.FileSystem(jobs.JOBS_ROOT)
        for bucket in ("ids", "queued", "running", "finished", "failed", "next"):
            fs.mkdir(fs.join(bucket), exist_ok=True)

        job = jobs.spawn_job("tlux.search.hkm.job_runner_helper.job_ok")
        job.wait_for_completion(poll_interval=0.05)
        assert job.exit_code == 0, f"stderr: {job.stderr}"
        assert job.status == "FINISHED"
    finally:
        jobs.JOBS_ROOT = old_root
