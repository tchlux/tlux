# File-system-backed job scheduler.
# 
# A lean, deterministic job orchestrator using only atomic directory
# operations on a shared file tree. Jobs are Python callables referenced
# by import path; their entire life-cycle and metadata live under a single
# sub-directory in `jobs/`. Multiple hosts can cooperate by pointing their
# supervisor processes at the same root.
# 
# Example
# -------
# >>> from job_system import spawn_job
# >>> job = spawn_job("my_pkg.training.run", args=("config.yaml",))
# >>> print(job.id)
# >>> print(job.is_running())
# >>> job.kill()
# >>> print(job.stdout)
#

from __future__ import annotations

import json
import os
import shutil
import subprocess
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

try:
    from .fs import FileSystem, make_filesystem
    from .monitor import proc_usage # pyright: ignore
except:
    from tlux.search.hkm.fs import FileSystem, make_filesystem
    from tlux.search.hkm.monitor import proc_usage # pyright: ignore

CODE_ROOT: str = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(CODE_ROOT)))
JOBS_ROOT: str = os.environ.get("HKM_JOBS_ROOT", os.path.join(CODE_ROOT, "jobs"))
JOB_BUCKETS: tuple[str, ...] = ("ids", "waiting", "queued", "running", "succeeded", "failed", "next", "workers")
ID_WIDTH: int = 9
STATUS_VALUES: set[str] = {"WAITING", "QUEUED", "RUNNING", "SUCCEEDED", "FAILED"}
ORPHAN_GRACE_SECONDS: float = 1.0
if __name__ == "__main__":
    MODULE_PATH: str = os.path.splitext(os.path.basename(__file__))[0]
else:
    MODULE_PATH: str = __name__

import os
import subprocess


# Wrapper for a single job stored in the FileSystem.
#
# Parameters:
#   fs (FileSystem): File-system abstraction.
#   path (str | Path | None): Path to existing job directory.
#   resource_config (dict | None): Resource configuration.
#
# Attributes:
#   id (str), status (str), exit_code (int | None), submit_ts (float),
#   start_ts (float | None), end_ts (float | None), resource_config (dict)
# 
class Job:
    status: str = ""

    # A descriptive string representation of this Job.
    def __repr__(self):
        p = os.path.relpath(self.path, os.getcwd())
        t = time.ctime(self.submit_ts)
        return f"Job({self.id}, path={repr(p)}) at {t}"

    # Initialize a Job from the given file system and job directory path.
    #
    # Parameters:
    #   fs (FileSystem): File-system abstraction.
    #   path (str | Path): Path to the job directory.
    #   active_job (subprocess.Popen | None): Optional running process handle.
    #
    def __init__(self, fs: FileSystem, path: Union[str, Path], *,
                 active_job: Optional[subprocess.Popen] = None,
                 config: Optional[Dict[str, Any]] = None) -> None:
        self._fs: FileSystem = fs
        self.job = active_job
        self.path = str(path)
        if config is None:
            self._load(Path(path))
        else:
            self._load_config(Path(path), config)
        if active_job is not None:
            self.hostname = socket.gethostname()
            self.monitor_pid = active_job.pid
            self._save()

    # Ensure that the job's path is correct by checking the current location.
    # If the job has moved due to a state change, update internal state.
    #
    # Raises:
    #   FileNotFoundError: If the job cannot be found in any status bucket.
    #
    def _ensure_path(self) -> None:
        # TODO: If the "mtime" of the job config file is newer, reload.
        # 
        # If current path exists, nothing to do.
        if self._fs.exists(self._fs.join(self.status, self.id)):
            return
        # Search all status buckets for a job with the same ID.
        for _status in STATUS_VALUES:
            candidate_path = self._fs.join(_status.lower(), self.id)
            if self._fs.exists(candidate_path):
                self.status = _status
                return
        raise FileNotFoundError(f"Job '{self.id}' not found in any status bucket.")

    # Override attribute access to trigger a path check for selected attributes.
    #
    # Parameters:
    #   name (str): Attribute name being accessed.
    #
    # Returns:
    #   Any: Attribute value.
    #
    def __getattribute__(self, name: str) -> Any:
        _refresh_on = {"WAITING", "QUEUED", "RUNNING"}
        _refresh_for = {"stdout", "stderr", "exit_code", "pid", "is_done", "is_running", "kill", "pause", "resume", "_save"}
        if (object.__getattribute__(self, "status") in _refresh_on) and (name in _refresh_for):
            self._ensure_path()
        return object.__getattribute__(self, name)

    # Return the import path of the function to execute for this job.
    #
    # Returns:
    #   str: Dotted import path as written in exec_function file.
    #
    @property
    def command(self) -> str:
        return self._fs.read(self._fs.join(self.path, "exec_function")).decode()

    # Return the positional and keyword arguments to pass to the job function.
    #
    # Returns:
    #   Tuple[Tuple[Any, ...], Dict[str, Any]]: (args, kwargs) as deserialized from exec_args.
    #
    @property
    def arguments(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        blob: bytes = self._fs.read(self._fs.join(self.path, "exec_args"))
        payload: Dict[str, Any] = json.loads(blob.decode())
        return tuple(payload.get("args", ())), payload.get("kwargs", {})

    # Return the standard output produced by the job process.
    #
    # Returns:
    #   str: Contents of the stdout file (decoded as UTF-8).
    #
    @property
    def stdout(self) -> str:
        path = self._fs.join(self.path, "stdout")
        return self._fs.read(path).decode(errors="ignore") if self._fs.exists(path) else ""

    # Return the standard error output produced by the job process.
    #
    # Returns:
    #   str: Contents of the stderr file (decoded as UTF-8).
    #
    @property
    def stderr(self) -> str:
        path = self._fs.join(self.path, "stderr")
        return self._fs.read(path).decode(errors="ignore") if self._fs.exists(path) else ""

    # Return process ID from active process or saved file.
    #
    # Returns:
    #   int: The PID of the job process.
    #
    # Raises:
    #   RuntimeError: If PID cannot be determined.
    #
    @property
    def pid(self) -> Optional[int]:
        if (self.job is not None):
            return self.job.pid
        elif (self.monitor_pid is not None):
            return self.monitor_pid
        return None

    # Check if the job process is done executing.
    #
    # Returns:
    #   bool: True if process is done executing, False otherwise.
    #
    # Raises:
    #   NotImplementedError: On unsupported platforms.
    #
    def is_done(self) -> bool:
        return self.status in {"SUCCEEDED", "FAILED"}

    # Check if the job process is running.
    #
    # Returns:
    #   bool: True if process is alive, False otherwise.
    #
    # Raises:
    #   NotImplementedError: On unsupported platforms.
    #
    def is_running(self) -> bool:
        if (self.status != "RUNNING"):
            return False
        # If this is the owning process, we can actually check directly to see if it is done.
        if self.job is not None:
            try:
                os.kill(self.job.pid, 0)
            except (OSError, RuntimeError):
                return False
            return True
        # Otherwise we have to assume the owning process is not done.
        else:
            return True

    # Send SIGKILL to the job process.
    #
    # Raises:
    #   RuntimeError: If PID is unavailable or signaling fails.
    #   NotImplementedError: On unsupported platforms.
    #
    def kill(self, reason: str = "Killed.") -> None:
        target_pid: Optional[int] = None
        if self.job is not None:
            target_pid = self.job.pid
        elif self.monitor_pid is not None:
            target_pid = self.monitor_pid
        if target_pid is None:
            raise RuntimeError("Cannot kill job without a recorded pid.")
        try:
            os.kill(target_pid, signal.SIGKILL)
        except Exception as e:
            raise RuntimeError(f"Failed to kill job process: {e}")
        # Also verify executor pid is killed.
        target_pid = getattr(self, "executor_pid", None)
        if target_pid:
            try:
                os.kill(int(target_pid), signal.SIGKILL)
            except Exception:
                pass
        # Mark as failed and move to failed bucket.
        try:
            # Set the status and save.
            self.status = "FAILED"
            self.status_reason = reason
            self.exit_code = -9
            self.end_ts = time.time()
            self._save()
            # Rename from the current status to "failed" (final atomic lock on state).
            self._fs.rename(
                self._fs.join("running", self.id),
                self._fs.join("failed", self.id)
            )
        except Exception:
            pass

    # Send SIGSTOP to the job process.
    #
    # Raises:
    #   RuntimeError: If PID is unavailable or signaling fails.
    #   NotImplementedError: On unsupported platforms.
    #
    def pause(self) -> None:
        if (self.status != "RUNNING"):
            return
        if (self.job is None):
            raise RuntimeError("Cannot pause a job that is not owned by this host process. `self.job is None`")
        try:
            os.kill(self.job.pid, signal.SIGSTOP)
        except Exception as e:
            raise RuntimeError(f"Failed to pause job process: {e}")

    # Send SIGCONT to the job process.
    #
    # Raises:
    #   RuntimeError: If PID is unavailable or signaling fails.
    #   NotImplementedError: On unsupported platforms.
    #
    def resume(self) -> None:
        if self.job is None:
            raise RuntimeError("This Job does not have a process to resume.")
        try:
            os.kill(self.job.pid, signal.SIGCONT)
        except Exception as e:
            raise RuntimeError(f"Failed to resume job process: {e}")

    # Wait until this job has finished execution.
    #
    # Parameters:
    #   poll_interval (float): Time in seconds between status checks (default 1.0).
    #
    # Returns
    #   bool: True if the job succeeded and False if it failed.
    #
    def wait_for_completion(self, poll_interval: float = 1.0) -> bool:
        # Polls the job's running status at fixed intervals until finished.
        while self.is_running():
            time.sleep(poll_interval)
        return self.status == "SUCCEEDED"

    # Write the job configuration and metadata to job_config file.
    #
    # Raises:
    #   OSError: If file write fails.
    #
    def _save(self) -> None:
        cfg = {
            "id": self.id,
            "status": self.status,
            "status_reason": self.status_reason,
            "exit_code": self.exit_code,
            "submit_ts": self.submit_ts,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "resource_config": self.resource_config,
            "hostname": self.hostname,
            "monitor_pid": self.pid,
            "executor_pid": getattr(self, "executor_pid", None),
        }
        self._fs.write(self._fs.join(self.path, "job_config"),
                       json.dumps(cfg).encode(), overwrite=True)

    # Reload this job.
    def reload(self) -> None:
        if self.path:
            return self._load(self.path)

    # Load in-memory job metadata and initialize attributes.
    #
    # Parameters:
    #   path (Path): Path to the job directory.
    #   data (Dict[str, Any]): Job metadata payload.
    #
    # Returns:
    #   (None): Attributes updated in place.
    #
    def _load_config(self, path: str | Path, data: Dict[str, Any]) -> None:
        status: str = data.get("status", "")
        if status not in STATUS_VALUES:
            raise RuntimeError(f"Corrupt status '{status}' for job '{path}'.")
        self.id = data["id"]
        self.status = status
        self.status_reason = data.get("status_reason", "")
        self.exit_code = data.get("exit_code")
        self.submit_ts = data.get("submit_ts")
        self.start_ts = data.get("start_ts")
        self.end_ts = data.get("end_ts")
        self.resource_config = data.get("resource_config", {})
        self.hostname = data.get("hostname")
        self.monitor_pid = data.get("monitor_pid", data.get("watcher_pid"))
        self.executor_pid = data.get("executor_pid")
        self.path = str(path)

    # Load job metadata from job_config file and initialize attributes.
    #
    # Parameters:
    #   path (Path): Path to the job directory.
    #
    # Raises:
    #   FileNotFoundError: If job_config is missing.
    #   RuntimeError: If status is invalid or data is corrupt.
    #
    def _load(self, path: str | Path) -> None:
        cfg_path = self._fs.join(str(path), "job_config")
        if not self._fs.exists(cfg_path):
            raise FileNotFoundError(f"job_config missing at '{cfg_path}'.")
        self._load_config(path, json.loads(self._fs.read(cfg_path).decode()))


# Ensure the filesystem contains the standard job buckets.
#
# Arguments:
#   fs (FileSystem): Job-root filesystem.
#
# Returns:
#   (FileSystem): The same filesystem after bucket creation.
#
def ensure_jobs_root(fs: FileSystem) -> FileSystem:
    for bucket in JOB_BUCKETS:
        fs.mkdir(fs.join(bucket), exist_ok=True)
    return fs


# Set the global jobs root and initialize its bucket layout.
#
# Arguments:
#   path (str): Filesystem path that will hold all job state.
#   reset (bool): Whether to delete any existing job state first.
#
# Returns:
#   (FileSystem): Filesystem rooted at the configured jobs path.
#
def set_jobs_root(path: str, reset: bool = False) -> FileSystem:
    root = os.path.abspath(path)
    if reset and os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)
    global JOBS_ROOT
    JOBS_ROOT = root
    os.environ["HKM_JOBS_ROOT"] = root
    return ensure_jobs_root(make_filesystem(root))


# Return the first visible bucket name containing *job_id*.
#
# Arguments:
#   fs (FileSystem): Job-root filesystem.
#   job_id (str): Job identifier.
#
# Returns:
#   (str | None): Matching uppercase status or None.
#
def job_status(fs: FileSystem, job_id: str) -> str | None:
    cfg_path = fs.join("ids", job_id, "job_config")
    if fs.exists(cfg_path):
        status = json.loads(fs.read(cfg_path).decode()).get("status")
        if status in STATUS_VALUES:
            return status
    for status in ("FAILED", "SUCCEEDED", "RUNNING", "QUEUED", "WAITING"):
        if fs.exists(fs.join(status.lower(), job_id)):
            return status
    return None


# Move a waiting job into failed with a persisted reason.
#
# Arguments:
#   fs (FileSystem): Job-root filesystem.
#   job_id (str): Job identifier.
#   reason (str): Failure reason.
#
# Returns:
#   (None): Waiting job failed if still claimable.
#
def fail_waiting_job(fs: FileSystem, job_id: str, reason: str) -> None:
    waiting_path = fs.join("waiting", job_id)
    failed_path = fs.join("failed", job_id)
    if not fs.rename(waiting_path, failed_path):
        return
    job = Job(fs=fs, path=fs.join("ids", job_id))
    job.status = "FAILED"
    job.status_reason = reason
    job.end_ts = time.time()
    job._save()


# Reconcile waiting jobs until they converge to queued or failed states.
#
# Arguments:
#   fs (FileSystem): Job-root filesystem.
#
# Returns:
#   (None): Waiting jobs updated in place.
#
def reconcile_waiting_jobs(fs: FileSystem) -> None:
    waiting_ids = sorted(fs.listdir("waiting")) if fs.exists("waiting") else []
    active = any(fs.listdir(bucket) for bucket in ("workers", "queued", "running"))
    now = time.time()
    for job_id in waiting_ids:
        upstream_dir = fs.join("ids", job_id, "upstream_jobs")
        upstream_ids = sorted(fs.listdir(upstream_dir)) if fs.exists(upstream_dir) else []
        orphan_path = fs.join("ids", job_id, "orphaned_at")
        failed_upstreams = [upstream_id for upstream_id in upstream_ids if job_status(fs, upstream_id) == "FAILED"]
        if failed_upstreams:
            fail_waiting_job(fs, job_id, f"Upstream job {failed_upstreams[0]} failed.")
            continue
        if not upstream_ids:
            if fs.rename(fs.join("waiting", job_id), fs.join("queued", job_id)):
                job = Job(fs=fs, path=fs.join("ids", job_id))
                job.status = "QUEUED"
                job._save()
            if fs.exists(orphan_path):
                try:
                    fs.remove(orphan_path)
                except Exception:
                    pass
            continue
        if active:
            if fs.exists(orphan_path):
                try:
                    fs.remove(orphan_path)
                except Exception:
                    pass
            continue
        if not fs.exists(orphan_path):
            fs.write(orphan_path, str(now).encode(), overwrite=True)
            continue
        if now - float(fs.read(orphan_path).decode()) >= ORPHAN_GRACE_SECONDS:
            fail_waiting_job(fs, job_id, "WAITING remained blocked with no active watchers or runnable jobs.")


# Fail running jobs whose watcher process has disappeared.
#
# Arguments:
#   fs (FileSystem): Job-root filesystem.
#
# Returns:
#   (None): Stale running jobs moved to failed.
#
def reap_running_jobs(fs: FileSystem) -> None:
    running_ids = sorted(fs.listdir("running")) if fs.exists("running") else []
    for job_id in running_ids:
        try:
            job = Job(fs=fs, path=fs.join("ids", job_id))
        except Exception:
            continue
        pid = getattr(job, "monitor_pid", None)
        if pid is None:
            if (
                getattr(job, "executor_pid", None) is None
                and (
                    ((job.start_ts is not None) and (time.time() - float(job.start_ts) >= ORPHAN_GRACE_SECONDS))
                    or ((job.start_ts is None) and (job.submit_ts is not None) and (time.time() - float(job.submit_ts) >= ORPHAN_GRACE_SECONDS))
                )
            ):
                if not fs.rename(fs.join("running", job_id), fs.join("failed", job_id)):
                    continue
                job.status = "FAILED"
                job.status_reason = "Watcher claimed the job but disappeared before launching it."
                job.exit_code = -9 if job.exit_code is None else job.exit_code
                job.end_ts = time.time()
                job._save()
            continue
        try:
            os.kill(int(pid), 0)
            continue
        except OSError:
            pass
        for target_pid in (getattr(job, "executor_pid", None),):
            if target_pid:
                try:
                    os.kill(int(target_pid), signal.SIGKILL)
                except Exception:
                    pass
        if not fs.rename(fs.join("running", job_id), fs.join("failed", job_id)):
            continue
        job.status = "FAILED"
        job.status_reason = "Watcher disappeared while the job was RUNNING."
        job.exit_code = -9 if job.exit_code is None else job.exit_code
        job.end_ts = time.time()
        job._save()


# Wait until a renamed path is visible on disk for direct file access.
#
# Arguments:
#   path (str): Absolute path expected to appear.
#   timeout (float): Maximum wait in seconds.
#
# Returns:
#   (None): Path is now visible.
#
def wait_for_path(path: str, timeout: float = 1.0) -> None:
    deadline = time.time() + timeout
    while not os.path.exists(path):
        if time.time() >= deadline:
            raise FileNotFoundError(f"Timed out waiting for path visibility: {path}")
        time.sleep(0.005)



# Create and configure a new job directory.
#
# Parameters:
#   fs (FileSystem): File-system abstraction.
#   command (str): Import path to the function.
#   dependencies (Iterable[Job]): Optional upstream jobs.
#   args (Iterable[Any] | None): Positional arguments.
#   kwargs (dict | None): Keyword arguments.
#   resource_config (dict | None): Resource configuration.
#
# Returns:
#   Job: The created job wrapper.
# 
def create_job(
    fs: FileSystem,
    command: str, *,
    dependencies: Iterable[Job] = (),
    args: Optional[Iterable[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    resource_config: Optional[Dict[str, Any]] = None
) -> Job:
    # Reserve a unique job ID via atomic directory creation.
    #
    # Parameters:
    #   fs (FileSystem): File-system abstraction.
    #
    # Returns:
    #   str: Reserved job ID.
    # 
    def _reserve_id(fs: FileSystem) -> Tuple[str, str]:
        while True:
            tick = int(time.time() * 10_000)
            candidate = f"{tick % 10 ** ID_WIDTH:0{ID_WIDTH}d}"
            try:
                candidate_dir = fs.mkdir(fs.join("ids", candidate), exist_ok=False)
                return candidate, candidate_dir
            except OSError:
                time.sleep(0.0001)

    args = tuple(args or ())
    kwargs = dict(kwargs or {})
    resource_config = dict(resource_config or {})
    job_id, job_id_dir = _reserve_id(fs)
    # Set the job directory (where this will move once populated).
    has_deps = bool(dependencies)
    bucket = "waiting" if has_deps else "queued"
    job_state_dir = fs.join(bucket, job_id)
    fs.write(fs.join(job_id_dir, "exec_function"), command.encode(), overwrite=True)
    fs.write(fs.join(job_id_dir, "exec_args"), json.dumps({"args": args, "kwargs": kwargs}).encode(), overwrite=True)
    now_ts = time.time()
    cfg: Dict[str, Any] = {
        "id": job_id,
        "status": bucket.upper(),
        "status_reason": "",
        "exit_code": None,
        "submit_ts": now_ts,
        "start_ts": None,
        "end_ts": None,
        "resource_config": resource_config,
        "hostname": None,
        "monitor_pid": None,
        "executor_pid": None,
    }
    fs.write(fs.join(job_id_dir, "job_config"), json.dumps(cfg).encode(), overwrite=True)
    if has_deps:
        up_root = fs.join(job_id_dir, "upstream_jobs")
        fs.mkdir(up_root, exist_ok=True)
        for dep in dependencies:
            fs.mkdir(fs.join(up_root, dep.id), exist_ok=True)
            next_entry = fs.join("next", dep.id, job_id)
            fs.mkdir(fs.join(next_entry), exist_ok=True)
            # Assert that the job we just gave a "next" is still RUNNING,
            #  if it is not, then do not create a "next" entry for it.
            if (
                (not fs.exists(fs.join("waiting", dep.id)))
                and (not fs.exists(fs.join("queued", dep.id)))
                and (not fs.exists(fs.join("running", dep.id)))
            ):
                fs.remove(fs.join(next_entry))
    if not fs.mkdir(job_state_dir, exist_ok=False):
        raise RuntimeError(f"Failed to assign job dir into {repr(bucket)}: {job_id_dir} -> {job_state_dir}")
    # Return the Job object.
    return Job(fs=fs, path=job_id_dir, config=cfg)


# Run a job using the default file system rooted at JOBS_ROOT.
#
# Parameters:
#   command (str): Import path to function.
#   dependencies (Iterable[Job]): Optional upstream jobs.
#   *args (Any): Positional arguments.
#   **kwargs (Any): Keyword arguments.
#
# Returns:
#   Job: Created job wrapper.
#
def run_job(
    command: str,
    *args: Any,
    dependencies: Iterable[Job] = (),
    **kwargs: Any,
) -> Job:
    default_fs = ensure_jobs_root(make_filesystem(JOBS_ROOT))
    new_job = create_job(
        default_fs,
        command,
        dependencies=dependencies,
        args=args,
        kwargs=kwargs
    )
    # Ensure a worker is running (up to the max).
    max_workers = max(1, int(os.environ.get("HKM_MAX_WORKERS", "1")))
    if os.environ.get("HKM_DISABLE_WATCHER_LAUNCH", "") != "1":
        watcher(launch=True, max_workers=max_workers)
    # return _launch_worker(fs=fs, job_dir=job_dir)
    return new_job


# Run local watchers until the queue is empty.
#
# Arguments:
#   fs (FileSystem | None): Optional filesystem rooted at the jobs directory.
#   max_workers (int): Maximum number of local watcher processes to use.
#   poll_interval (float): Delay between watcher passes.
#
# Returns:
#   None
#
def drain_jobs(
    fs: Optional[FileSystem] = None,
    max_workers: int = 1,
    poll_interval: float = 0.05,
) -> None:
    fs = ensure_jobs_root(fs or make_filesystem(JOBS_ROOT))
    while True:
        watcher(fs=fs, max_workers=max_workers)
        pending = sum(len(fs.listdir(bucket)) for bucket in ("waiting", "queued", "running"))
        active_workers = len(fs.listdir("workers"))
        if (pending == 0) and (active_workers == 0):
            return
        time.sleep(poll_interval)


# Watcher that monitors the state of the jobs directory, executes jobs, and cleans.
# 
# Steps:
#  - check active registrations for watchers
#  - register self if < max workers running (global config)
#  - verify that < max workers are running (kill latest to spawn, including self, if over)
#  - check for QUEUED jobs, execute "worker" directly on first job
#  - if none QUEUED, then look for WAITING with none RUNNING (kill those WAITING as unsatisfiable)
#  - if all empty, then exit
# 
def watcher(fs: Optional[FileSystem] = None, max_workers: int = 1, launch: bool=False) -> None:
    # Otherwise assume this is the primary process.
    if fs is None:
        fs = make_filesystem(JOBS_ROOT)
    fs = ensure_jobs_root(fs)
    if (max_workers == 1) and os.environ.get("HKM_MAX_WORKERS", "").isdigit():
        max_workers = max(1, int(os.environ["HKM_MAX_WORKERS"]))
    # If a launch is desired, create a process and return.
    if launch:
        # The process will overwrite its own STDOUT and STDERR when ready.
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), fs.root, str(max_workers)],
            env={
                **os.environ,
                "HKM_JOBS_ROOT": fs.root,
                "HKM_MAX_WORKERS": str(max_workers),
                "PYTHONPATH": CODE_ROOT + ":" + REPO_ROOT + ":" + os.environ.get("PYTHONPATH", ""),
            },
        )
        return
    # Check how many registered workers there are.
    fs.mkdir("workers", exist_ok=True)
    registered_watchers = fs.listdir("workers")
    # Drop stale worker directories whose PID is no longer alive.
    live_watchers = []
    for name in registered_watchers:
        pid = int(name) if name.isdigit() else None
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                pass
        if alive:
            live_watchers.append(name)
        else:
            try:
                fs.remove(fs.join("workers", name))
            except Exception:
                pass
    registered_watchers = live_watchers
    reap_running_jobs(fs)
    if len(registered_watchers) >= max_workers:
        # a = "are" if len(registered_watchers) > 1 else "is"
        # w = "watchers" if len(registered_watchers) > 1 else "watcher"
        # print(f"[jobs.WATCHER] Exiting since there {a} {len(registered_watchers)} {w} already.", flush=True)
        return
    # Register self as a worker.
    pid = str(os.getpid())
    wdir = fs.join("workers", pid)
    fs.mkdir(wdir)
    # Validate the number of workers (deterministically exit if too many run).
    registered_watchers = fs.listdir("workers")  # naturally ordered by start time
    if (len(registered_watchers) >= max_workers) and (pid not in registered_watchers[:max_workers]):
        # a = "are" if len(registered_watchers) > 1 else "is"
        # w = "watchers" if len(registered_watchers) > 1 else "watcher"
        # print(f"[jobs.WATCHER] Exiting and removing watcher directory since there {a} {len(registered_watchers)} {w}.", flush=True)
        fs.remove(wdir)
        return
    # Set standard output and error for this process to be owned worker directory.
    out_file = open(fs.join(wdir, "logs"), "a")
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = out_file
    sys.stderr = out_file
    # Check for queued jobs, execute "worker" on first available.
    #   - reserve a queued job by moving it from 'queued' to 'running' (if successful, it is owned).
    while ((next_jid := next(iter(sorted(fs.listdir("queued")) + [None]))) is not None):
        if os.getppid() == 1:
            break
        try:
            job = Job(fs=fs, path=fs.join("ids", next_jid))
            if not fs.rename(fs.join("queued", next_jid), fs.join("running", next_jid)):
                continue
            wait_for_path(fs.join("running", next_jid))
            worker(fs=fs, job=job)
        except: # Exception as e:
            # print(f"[jobs.WATCHER] Exception claiming job {next_jid} encountered {e}", file=sys.stderr, flush=True)
            continue
    # All jobs completed, moving on to cleanup, indicate by saying this watcher is no longer active.
    fs.remove(wdir)
    reconcile_waiting_jobs(fs)
    # Reset standard output streams.
    sys.stdout = stdout
    sys.stderr = stderr
    out_file.close()


# Run *job* in a subprocess, monitor resources, finalize state, trigger deps.
def worker(fs: FileSystem, job: Job) -> Job:
    # --- launch target ---
    args, kwargs = job.arguments
    payload_hex = json.dumps({"args": args, "kwargs": kwargs}).encode().hex()
    launcher_code = (
        "import importlib, json, sys; "
        "mod_path, blob = sys.argv[1], sys.argv[2]; "
        "mod = importlib.import_module('.'.join(mod_path.split('.')[:-1])); "
        "func = getattr(mod, mod_path.split('.')[-1]); "
        "payload = json.loads(bytes.fromhex(blob).decode()); "
        "ret = func(*payload['args'], **payload['kwargs']); "
        "sys.exit(ret if isinstance(ret, int) else 0)"
    )
    out_path = fs.join(job.path, "stdout")
    err_path = fs.join(job.path, "stderr")
    job.start_ts = time.time()
    job.status = "RUNNING"
    job.monitor_pid = os.getpid()
    job._save()
    with open(out_path, "ab") as so, open(err_path, "ab") as se:
        proc = subprocess.Popen(
            [sys.executable, "-c", launcher_code, job.command, payload_hex],
            stdout=so,
            stderr=se,
            close_fds=True,
        )
    # Update the job config.
    job.executor_pid = proc.pid
    job._save()
    # --- monitoring loop ---
    res_file = Path(fs.join(job.path, "resources"))
    res_file.touch(exist_ok=True)
    while True:
        if os.getppid() == 1:
            try:
                proc.kill()
            except Exception:
                pass
            exit_code = -9
            break
        # Wait for exit of the process.
        try:
            exit_code = proc.wait(timeout=5)
            break
        except:
            pass
        # If the process hasn't exited, do a resource utilization check as a heartbeat.
        pid_list = [proc.pid]
        try:
            out = subprocess.check_output(["ps", "-o", "pid=", "-p", str(proc.pid)], text=True)
            for line in out.strip().splitlines():
                try:
                    pid_list.append(int(line.strip()))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            # Also write a lightweight heartbeat to stdout for UIs.
            snap = proc_usage(pid_list)
            with res_file.open("a", encoding="utf-8") as rf:
                rf.write(json.dumps({"ts": time.time(), **snap}) + "\n")
        except Exception:
            pass
    # --- finalize job ---
    job.exit_code = exit_code
    job.end_ts = time.time()
    job.status = "SUCCEEDED" if exit_code == 0 else "FAILED"
    job._save()
    dest_dir = fs.join(job.status.lower(), job.id)
    # Before checking for downstreams, remove from "running".
    fs.rename(fs.join("running", job.id), dest_dir)
    # --- trigger downstreams ---
    next_dir = fs.join("next", job.id)
    if not fs.exists(next_dir):
        return job
    for did in fs.listdir(next_dir):
        njob = fs.join(next_dir, did)  # next job entry (MINE for downstream)
        wjob = fs.join("waiting", did)  # waiting job entry (downstream job)
        updir = fs.join("ids", did, "upstream_jobs")
        wentry = fs.join(updir, job.id)  # waiting-on entry (downstream for ME)
        # Remove the downstream from THIS job's waiting list.
        try:
            fs.remove(wentry)
            fs.remove(njob)
        except: # Exception as e:
            # print(f"[jobs.WORKER] Failed to remove downstream wait-lock for {job.id} blocking downstream {did}. {e}", file=sys.stderr, flush=True)
            pass
        # Fail all downstream jobs if this job failed.
        if job.status == "FAILED":
            try:
                j = Job(fs=fs, path=fs.join("ids", did))
                j.status = "FAILED"
                j.status_reason = f"Upstream job {job.id} failed."
                j._save()
                fs.rename(wjob, fs.join("failed", did))
            except:
                pass
        # If the downstream job has no more upstreams, move it to its next state.
        elif len(list(fs.listdir(updir))) == 0:
            try:
                j = Job(fs=fs, path=fs.join("ids", did))
                if (j.status != "FAILED"):
                    if not fs.rename(wjob, fs.join("queued", did)):
                        continue
                    j.status = "QUEUED"
                    j._save()
                    # Ensure a watcher exists to execute the job.
                    if os.environ.get("HKM_DISABLE_WATCHER_LAUNCH", "") != "1":
                        watcher(fs=fs, launch=True)
            except: # Exception as e:
                # print(f"[jobs.WORKER] Failed to move downstream {did} into QUEUED state: {e}", file=sys.stderr, flush=True)
                pass
    reconcile_waiting_jobs(fs)
    # TODO: Log the downstreams into the config at time of launching and delete the next directory (since it will not be watched further).
    # # Remove the "next" directory indicating it has been processed.
    # fs.remove(next_dir)
    return job


if __name__ == "__main__":
    # 
    # print(os.getpid(), "-"*100, flush=True)
    # print("sys.argv: ", sys.argv, flush=True)
    # 
    if len(sys.argv) > 1:
        # This is a watcher process.
        # Ensure CODE_ROOT and REPO_ROOT are in the sys path.
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        if CODE_ROOT not in sys.path:
            sys.path.insert(0, CODE_ROOT)
        watcher(
            fs=make_filesystem(sys.argv[1]),
            max_workers=int(sys.argv[2]) if (len(sys.argv) > 2 and sys.argv[2].isdigit()) else 1,
        )
    else:
        print("[jobs.TESTER] begin.", flush=True)
        # Example: spawn a job that computes a simple function, wait, and print result.
        #
        # The test function below is for demonstration only.
        import time

        # Test 1 (failed job)
        job = run_job(f"tests.test_jobs._example_fail", 7, 8)
        print("job: ", job, flush=True)
        print(f"Spawned job ID: {job.id}")
        # Wait for job to finish.
        for _ in range(2):
            time.sleep(1)
            if not job.is_running():
                break
            print("Waiting for job to finish...")
        # Print results.
        # TODO: Job should reload itself here automatically!
        job.reload()
        print("Job stdout:", repr(job.stdout))
        print("Job stderr:", repr(job.stderr))
        print(f"Job finished? {not job.is_running()}")
        print(f"Job exit code: {job.exit_code}")
        print()
        print(job.stderr)

        # Test 2 (successful job)
        job = run_job(f"tests.test_jobs._example_add", 7, 8)
        print("job: ", job, flush=True)
        print(f"Spawned job ID: {job.id}")
        # Wait for job to finish.
        for _ in range(2):
            time.sleep(1)
            if not job.is_running():
                break
            print("Waiting for job to finish...")
        # Update job config.
        job._load(Path(job.path))
        # Print results.
        job.reload()
        print("Job stdout:", repr(job.stdout))
        print("Job stderr:", repr(job.stderr))
        print(f"Job finished? {not job.is_running()}")
        print(f"Job exit code: {job.exit_code}")
        print()
        print(job.stderr)

        # Test 3 (job dependencies, upstream succeeds)
        job_a = run_job("tests.test_jobs._example_add", 2, 3)
        job_b = run_job("tests.test_jobs._example_dep", 5, dependencies=[job_a])
        print(f"Spawned\n job_a: {job_a.id}\n job_b: {job_b.id}")
        while job_a.is_running():
            time.sleep(0.01)
        print("done waiting on a", flush=True)
        while not job_b.is_done():
            print(f"job_b {job_b.id} is done?", job_b.is_done(), flush=True)
            time.sleep(1.0)
        print("done waiting on b", flush=True)
        print("Stdout:")
        print("", "job_a", repr(job_a.stdout))
        if job_b:
            print("", "job_b", repr(job_b.stdout))
    # 
    # print(os.getpid(), "^"*100, flush=True)
    # 
