import json
import subprocess
import sys
import time
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


def test_reaper_keeps_live_executor_when_watcher_is_dead(tmp_path, monkeypatch):
    old_root = jobs.JOBS_ROOT
    monkeypatch.setenv("HKM_DISABLE_WATCHER_LAUNCH", "1")
    dead_monitor = subprocess.Popen([sys.executable, "-c", ""])
    dead_monitor.wait(timeout=1.0)
    executor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        fs = setup_jobs_root(tmp_path / "jobs")
        job = jobs.run_job("tlux.search.hkm.tests.job_runner_helper.job_ok")
        assert fs.rename(fs.join("queued", job.id), fs.join("running", job.id))
        job.status = "RUNNING"
        job.start_ts = time.time() - (jobs.ORPHAN_GRACE_SECONDS * 2)
        job.monitor_pid = dead_monitor.pid
        job.executor_pid = executor.pid
        job._save()

        jobs.reap_running_jobs(fs)
        job.reload()

        assert job.status == "RUNNING"
        assert job.id in fs.listdir("running")
        assert not fs.listdir("failed")
        assert executor.poll() is None
    finally:
        jobs.JOBS_ROOT = old_root
        if executor.poll() is None:
            executor.terminate()
            try:
                executor.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                executor.kill()
                executor.wait(timeout=1.0)
