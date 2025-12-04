import json
import os
import signal
import socket
import tempfile
import time
import multiprocessing
from pathlib import Path
import pytest

from tlux.search.hkm import jobs
from tlux.search.hkm.jobs import Job
from tlux.search.hkm.jobs_worker import start_worker


# Utility to construct an empty jobs tree for tests.
def _setup_jobs_root() -> tuple[str, jobs.FileSystem]:
    root = Path(tempfile.mkdtemp()) / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    fs = jobs.FileSystem(str(root))
    for bucket in ("ids", "queued", "running", "finished", "failed", "next", "workers"):
        fs.mkdir(fs.join(bucket), exist_ok=True)
    jobs.JOBS_ROOT = str(root)
    return str(root), fs


def _make_job_without_spawn(fs: jobs.FileSystem, command: str) -> Job:
    # Patch the launch helper to avoid nested subprocess creation.
    original = jobs._launch_worker
    try:
        jobs._launch_worker = lambda fs, job_dir, module_path=None: Job(fs, job_dir)
        job = jobs.create_new_job(fs, command)
    finally:
        jobs._launch_worker = original
    # Ensure job is in running bucket for worker pickup.
    job_dir = fs.join("running", job.id)
    assert fs.exists(job_dir)
    return Job(fs, job_dir)


# Test functions importable via dotted path.
def _quick_job() -> int:
    print("[quick] start")
    time.sleep(0.05)
    print("[quick] end")
    return 0


def _slow_job() -> int:
    print("[slow] start")
    time.sleep(2.0)
    print("[slow] end")
    return 0
def test_worker_executes_job_and_heartbeats(monkeypatch):
    root, fs = _setup_jobs_root()
    job = _make_job_without_spawn(fs, "tests.test_jobs_worker._quick_job")
    proc = multiprocessing.Process(target=start_worker, args=(root,), kwargs={"max_workers": 1, "poll_interval": 0.1})
    proc.start()
    # Wait for job to complete.
    deadline = time.time() + 5
    while time.time() < deadline and not job.is_done():
        time.sleep(0.1)
    assert job.is_done(), "job should finish under worker daemon"
    assert job.status in {"FINISHED", "FAILED"}
    heartbeat_dir = Path(root) / "workers" / socket.gethostname()
    files = list(heartbeat_dir.glob("*.json"))
    assert files, "worker heartbeat file missing"
    payload = json.loads(files[0].read_text())
    assert payload["pid"] == proc.pid
    proc.terminate()
    proc.join(timeout=2)


def test_worker_kill_command(monkeypatch):
    root, fs = _setup_jobs_root()
    job = _make_job_without_spawn(fs, "tests.test_jobs_worker._slow_job")
    proc = multiprocessing.Process(target=start_worker, args=(root,), kwargs={"max_workers": 1, "poll_interval": 0.1})
    proc.start()
    # Wait for the worker to claim the job and record pid.
    deadline = time.time() + 5
    job_pid = None
    while time.time() < deadline and job_pid is None:
        cfg = json.loads(fs.read(fs.join(job.path, "job_config")).decode())
        job_pid = cfg.get("pid")
        time.sleep(0.05)
    assert job_pid, "worker never claimed job"
    os.kill(proc.pid, signal.SIGTERM)
    # Allow time for process to exit and be harvested.
    deadline = time.time() + 5
    while time.time() < deadline and not job.is_done():
        time.sleep(0.1)
    with pytest.raises(OSError):
        os.kill(int(job_pid), 0)
    proc.terminate()
    proc.join(timeout=2)
