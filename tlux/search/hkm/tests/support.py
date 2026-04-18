from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

from tlux.search.hkm import jobs
from tlux.search.hkm.fs import make_filesystem


# Build a fresh jobs root for tests and create all scheduler buckets.
#
# Arguments:
#   root (str | Path): Filesystem path for the jobs root
#
# Returns:
#   (object): Filesystem rooted at *root*
#
def setup_jobs_root(root: str | Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    jobs.JOBS_ROOT = str(root)
    return jobs.ensure_jobs_root(make_filesystem(str(root)))


# Enable the delayed filesystem wrapper through env for subprocess tests.
#
# Arguments:
#   monkeypatch (object): Pytest monkeypatch fixture
#   write_delay (float): Write visibility delay in seconds
#   rename_delay (float): Rename visibility delay in seconds
#   remove_delay (float): Remove visibility delay in seconds
#   listdir_delay (float): Extra directory visibility delay in seconds
#
# Returns:
#   (None): Environment configured for delayed filesystem tests
#
def enable_delayed_fs(
    monkeypatch,
    *,
    write_delay: float = 0.01,
    rename_delay: float = 0.01,
    remove_delay: float = 0.01,
    listdir_delay: float = 0.0,
) -> None:
    monkeypatch.setenv("HKM_FILESYSTEM_CLASS", "tlux.search.hkm.tests.delayed_fs.DelayedFileSystem")
    monkeypatch.setenv("HKM_FS_WRITE_DELAY", str(write_delay))
    monkeypatch.setenv("HKM_FS_RENAME_DELAY", str(rename_delay))
    monkeypatch.setenv("HKM_FS_REMOVE_DELAY", str(remove_delay))
    monkeypatch.setenv("HKM_FS_LISTDIR_DELAY", str(listdir_delay))
    monkeypatch.setenv("HKM_DISABLE_WATCHER_LAUNCH", "1")


# Trigger delayed filesystem event application after a real-time wait.
#
# Arguments:
#   fs (object): Filesystem instance
#   settle (float): Optional sleep before syncing
#
# Returns:
#   (None): Pending delayed operations applied
#
def sync_fs(fs, settle: float = 0.0) -> None:
    if settle > 0:
        time.sleep(settle)
    fs.exists(fs.root)


def _watcher_target(root: str, max_workers: int) -> None:
    jobs.watcher(fs=make_filesystem(root), max_workers=max_workers)


# Run watcher processes until the queue converges, optionally killing one mid-run.
#
# Arguments:
#   fs (object): Job-root filesystem
#   watcher_count (int): Target number of watcher processes
#   max_workers (int): Scheduler max_workers value
#   timeout (float): Maximum wall-clock wait
#   poll_interval (float): Poll cadence
#   kill_one (bool): Kill one active watcher once while jobs remain
#
# Returns:
#   (None): Scheduler reached a terminal state
#
def run_watchers(
    fs,
    *,
    watcher_count: int = 1,
    max_workers: int = 1,
    timeout: float = 10.0,
    poll_interval: float = 0.01,
    kill_one: bool = False,
) -> None:
    deadline = time.time() + timeout
    processes: list[multiprocessing.Process] = []
    killed = False
    idle_since: float | None = None
    try:
        while time.time() < deadline:
            live = []
            for process in processes:
                process.join(timeout=0)
                if process.is_alive():
                    live.append(process)
            processes = live
            pending = any(fs.listdir(bucket) for bucket in ("waiting", "queued", "running"))
            if kill_one and (not killed) and processes and fs.listdir("running"):
                processes[0].terminate()
                processes[0].join(timeout=1.0)
                killed = True
                processes = processes[1:]
            if pending:
                idle_since = None
                while len(processes) < watcher_count:
                    process = multiprocessing.Process(target=_watcher_target, args=(fs.root, max_workers))
                    process.start()
                    processes.append(process)
            elif not processes:
                idle_since = idle_since or time.time()
                sync_fs(fs, settle=poll_interval)
                if time.time() - idle_since >= max(
                    poll_interval * 2,
                    float(os.environ.get("HKM_FS_WRITE_DELAY", "0") or 0.0),
                    float(os.environ.get("HKM_FS_RENAME_DELAY", "0") or 0.0),
                    float(os.environ.get("HKM_FS_REMOVE_DELAY", "0") or 0.0),
                    float(os.environ.get("HKM_FS_LISTDIR_DELAY", "0") or 0.0),
                ) + 0.02:
                    return
            sync_fs(fs, settle=poll_interval)
        running = fs.listdir("running")
        details = []
        for job_id in running:
            job = jobs.Job(fs, fs.join("ids", job_id))
            details.append(
                {
                    "id": job_id,
                    "monitor_pid": job.monitor_pid,
                    "executor_pid": job.executor_pid,
                    "start_ts": job.start_ts,
                    "reason": job.status_reason,
                }
            )
        raise TimeoutError(
            "watchers did not converge before timeout: "
            f"waiting={fs.listdir('waiting')} queued={fs.listdir('queued')} running={details}"
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
