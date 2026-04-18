import json
from pathlib import Path

import pytest

from tlux.search.hkm import jobs
from tlux.search.hkm.tests.support import setup_jobs_root


def test_worker_executes_job_and_records_resources(tmp_path):
    fs = setup_jobs_root(tmp_path / "jobs")
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


def test_job_kill_sets_failed_status(tmp_path):
    fs = setup_jobs_root(tmp_path / "jobs")
    job = jobs.run_job("tlux.search.hkm.tests.job_runner_helper.job_cpu_burner")
    jobs.watcher(fs=fs, max_workers=1)
    with pytest.raises(RuntimeError):
        job.kill(reason="test kill")
