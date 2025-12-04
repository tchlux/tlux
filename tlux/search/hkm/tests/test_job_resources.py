import tempfile
from pathlib import Path

from tlux.search.hkm import jobs


def _setup_jobs_root() -> str:
    root = Path(tempfile.mkdtemp()) / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    fs = jobs.FileSystem(str(root))
    for bucket in ("ids", "queued", "running", "finished", "failed", "next"):
        fs.mkdir(fs.join(bucket), exist_ok=True)
    jobs.JOBS_ROOT = str(root)
    return str(root)


def test_resources_capture_cpu(tmp_path) -> None:
    old = jobs.JOBS_ROOT
    try:
        _setup_jobs_root()
        job = jobs.spawn_job("tlux.search.hkm.job_runner_helper.job_cpu_burner")
        job.wait_for_completion(poll_interval=0.05)
        assert job.status == "FINISHED"
        res_path = Path(job.path) / "resources"
        assert res_path.exists()
        lines = res_path.read_text().strip().splitlines()
        assert lines, "resources file empty"
        last = lines[-1]
        assert "cpu_percent" in last, last
    finally:
        jobs.JOBS_ROOT = old
