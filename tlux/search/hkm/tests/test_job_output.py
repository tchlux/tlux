import tempfile
import time
from pathlib import Path

from tlux.search.hkm import jobs


def test_job_stdout_visible(tmp_path) -> None:
    """Spawn a chatty job and ensure stdout is captured to file."""
    old_root = jobs.JOBS_ROOT
    try:
        jobs_root = tmp_path / "jobs"
        jobs_root.mkdir(parents=True, exist_ok=True)
        jobs.JOBS_ROOT = str(jobs_root)
        fs = jobs.FileSystem(jobs.JOBS_ROOT)
        for bucket in ("ids", "queued", "running", "finished", "failed", "next"):
            fs.mkdir(fs.join(bucket), exist_ok=True)

        job = jobs.spawn_job("tlux.search.hkm.job_runner_helper.job_with_output")
        job.wait_for_completion(poll_interval=0.05)

        stdout_text = job.stdout
        assert "[job_with_output] tick" in stdout_text, f"missing job output: {stdout_text!r}"
        assert job.status == "FINISHED"
    finally:
        jobs.JOBS_ROOT = old_root
