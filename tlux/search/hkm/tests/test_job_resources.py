from pathlib import Path

from tlux.search.hkm import jobs
from tlux.search.hkm.tests.support import setup_jobs_root


def test_resources_capture_cpu(tmp_path) -> None:
    old = jobs.JOBS_ROOT
    try:
        setup_jobs_root(tmp_path / "jobs")
        job = jobs.run_job("tlux.search.hkm.tests.job_runner_helper.job_cpu_burner")
        jobs.watcher(fs=setup_jobs_root(jobs.JOBS_ROOT), max_workers=1)
        job.wait_for_completion(poll_interval=0.05)
        assert job.status == "SUCCEEDED"
        res_path = Path(job.path) / "resources"
        assert res_path.exists()
        lines = res_path.read_text().strip().splitlines()
        assert lines, "resources file empty"
        last = lines[-1]
        assert "cpu_percent" in last, last
    finally:
        jobs.JOBS_ROOT = old
