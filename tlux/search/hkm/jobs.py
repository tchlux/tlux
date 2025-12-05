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

import json
import os
import subprocess
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

try:
    from .fs import FileSystem
    from .monitor import proc_usage
except:
    # exec(open(os.path.join(os.path.dirname(__file__), "fs.py")).read())
    from tlux.search.hkm.fs import FileSystem
    from tlux.search.hkm.monitor import proc_usage

CODE_ROOT: str = os.path.abspath(os.path.dirname(__file__))
JOBS_ROOT: str = os.path.join(CODE_ROOT, "jobs")
ID_WIDTH: int = 9
STATUS_VALUES: set[str] = {"WAITING", "QUEUED", "RUNNING", "SUCCEEDED", "FAILED"}
if __name__ == "__main__":
    MODULE_PATH: str = os.path.splitext(os.path.basename(__file__))[0]
else:
    MODULE_PATH: str = __name__

import tempfile
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

    # Initialize a Job from the given file system and job directory path.
    #
    # Parameters:
    #   fs (FileSystem): File-system abstraction.
    #   path (str | Path): Path to the job directory.
    #   active_job (subprocess.Popen | None): Optional running process handle.
    #
    def __init__(self, fs: FileSystem, path: Union[str, Path], *,
                 active_job: Optional[subprocess.Popen] = None) -> None:
        self._fs: FileSystem = fs
        self.job = active_job
        self._load(Path(path))
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
        # If current path exists, nothing to do.
        if self._fs.exists(self.path):
            return
        # Search all status buckets for a job with the same ID.
        for _status in STATUS_VALUES:
            candidate_path = self._fs.join(_status.lower(), self.id)
            if self._fs.exists(candidate_path):
                self._load(Path(candidate_path))
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
        _refresh_on = {"QUEUED", "RUNNING"}
        _refresh_for = {"stdout", "stderr", "pid", "is_done", "is_running", "kill", "pause", "resume", "_save"}
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
    def pid(self) -> int:
        if (self.job is not None):
            return self.job.pid
        elif (self.monitor_pid is not None):
            return self.monitor_pid
        raise RuntimeError("No PID available for job.")

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
            "status_reason": "",
            "exit_code": self.exit_code,
            "submit_ts": self.submit_ts,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "resource_config": self.resource_config,
            "hostname": self.hostname,
            "watcher_pid": self.pid,
            "executor_pid": getattr(self, "executor_pid", None),
        }
        self._fs.write(self._fs.join(self.path, "job_config"),
                       json.dumps(cfg).encode(), overwrite=True)

    # Load job metadata from job_config file and initialize attributes.
    #
    # Parameters:
    #   path (Path): Path to the job directory.
    #
    # Raises:
    #   FileNotFoundError: If job_config is missing.
    #   RuntimeError: If status is invalid or data is corrupt.
    #
    def _load(self, path: Path) -> None:
        cfg_path = self._fs.join(str(path), "job_config")
        if not self._fs.exists(cfg_path):
            raise FileNotFoundError(f"job_config missing at '{cfg_path}'.")
        data: Dict[str, Any] = json.loads(self._fs.read(cfg_path).decode())
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
        self.monitor_pid = data.get("monitor_pid")
        self.executor_pid = data.get("executor_pid")
        self.path = str(path)



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
            next_root = fs.join("next", dep.id)
            fs.mkdir(next_root, exist_ok=True)
            fs.mkdir(fs.join(next_root, job_id), exist_ok=True)
    if not fs.mkdir(job_state_dir, exist_ok=False):
        raise RuntimeError(f"Failed to assign job dir into {repr(bucket)}: {job_id_dir} -> {job_state_dir}")
    # Return the Job object.
    return Job(fs=fs, path=job_id_dir)


# Run a job using the default file system rooted at JOBS_ROOT.
#
# Parameters:
#   command (str): Import path to function.
#   dependencies (Iterable[Job]): Optional upstream jobs.
#   inline (bool): Execute immediately in-process when True.
#   *args (Any): Positional arguments.
#   **kwargs (Any): Keyword arguments.
#
# Returns:
#   Job | InlineJob: Created job wrapper or inline result.
#
def run_job(
    command: str,
    *args: Any,
    dependencies: Iterable[Job] = (),
    inline: bool = False,
    **kwargs: Any,
) -> Union[Job, Any]:
    if inline:
        mod_path, func_name = command.rsplit(".", 1)
        mod = __import__(mod_path, fromlist=[func_name])
        func = getattr(mod, func_name)
        return func(*args, **kwargs)
    else:
        default_fs = FileSystem(JOBS_ROOT)
        new_job = create_job(
            default_fs,
            command,
            dependencies=dependencies,
            args=args,
            kwargs=kwargs
        )
        # TODO: Ensure a worker is runnign (up to the max).
        # return _launch_worker(fs=fs, job_dir=job_dir)
        return new_job


# Host-local maintenance loop.
# 
# Steps:
#  - check active registrations for watchers
#  - register self if < max workers running (global config)
#  - verify that < max workers are running (kill latest to spawn, including self, if over)
#  - check for QUEUED jobs, execute "worker" directly on first job
#  - if none QUEUED, then look for WAITING with none RUNNING (kill those WAITING as unsatisfiable)
#  - if all empty, then exit
# 
def watcher(fs: Optional[FileSystem] = None) -> None:
    if fs is None:
        fs = FileSystem(JOBS_ROOT)


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
    with open(out_path, "ab") as so, open(err_path, "ab") as se:
        proc = subprocess.Popen(
            [sys.executable, "-c", launcher_code, job.command, payload_hex],
            stdout=so,
            stderr=se,
            close_fds=True,
        )
    # Update the job config.
    job.monitor_pid = os.getpid()
    job.executor_pid = proc.pid
    job._save()
    # --- monitoring loop ---
    res_file = Path(fs.join(job.path, "resources"))
    res_file.touch(exist_ok=True)
    mon_out = Path(fs.join(job.path, "stdout"))
    while proc.poll() is None:
        # include worker pid, recorded child pid, plus direct children
        pid_list = [proc.pid]
        if getattr(job, "child_pid", None):
            pid_list.append(int(job.child_pid))
        try:
            out = subprocess.check_output(["ps", "-o", "pid=", "--ppid", str(proc.pid)], text=True)
            for line in out.strip().splitlines():
                try:
                    pid_list.append(int(line.strip()))
                except Exception:
                    pass
        except Exception:
            pass
        snap = proc_usage(pid_list)
        try:
            with res_file.open("a", encoding="utf-8") as rf:
                rf.write(json.dumps({"ts": time.time(), **snap}) + "\n")
        except Exception:
            pass
        # Also write a lightweight heartbeat to stdout for UIs.
        try:
            with mon_out.open("a", encoding="utf-8") as mo:
                mo.write(f"[monitor] rss={snap['rss']//1_048_576:.1f}MiB cpu={snap['cpu_percent']:.1f}% gpu={snap['gpu_percent'] if snap['gpu_percent'] is not None else 'n/a'}\n")
        except Exception:
            pass
        time.sleep(5.0)
    exit_code = proc.returncode or 0
    # --- finalize job directory ---
    cfg_path = fs.join(job.path, "job_config")
    cfg = json.loads(fs.read(cfg_path).decode())
    cfg.update({
        "exit_code": exit_code,
        "end_ts": time.time(),
        "status": "SUCCEEDED" if exit_code == 0 else "FAILED",
    })
    fs.write(cfg_path, json.dumps(cfg).encode(), overwrite=True)
    dest_bucket = "SUCCEEDED" if exit_code == 0 else "failed"
    dest_dir = fs.join(dest_bucket, job.id)
    fs.rename(job.path, dest_dir)
    job = Job(fs=fs, path=dest_dir)
    # --- trigger downstreams ---
    next_dir = fs.join("next", job.id)
    if not fs.exists(next_dir):
        return job
    for did in fs.listdir(next_dir):
        wdir = fs.join("waiting", did)
        up_dir = fs.join(wdir, "upstream_jobs")
        try:
            fs.remove(fs.join(up_dir, job.id))
        except Exception as e:
            print(f"Failed to remove downstream wait-lock. {e}")
        # If the downstream job has no more upstreams, move it to queued.
        if len(list(fs.listdir(up_dir))) == 0:
            try:
                fs.rename(wdir, fs.join("queued", did))
                print("Moved ready downstream {did} to QUEUED state.")
            except Exception as e:
                print(f"Failed to move downstream into QUEUED state: {e}")
    # TODO: Log the downstreams into the config at time of launching and delete the next directory (since it will not be watched further).
    # # Remove the "next" directory indicating it has been processed.
    # fs.remove(next_dir)
    return job


if __name__ == "__main__":
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
    # Print results.
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
        time.sleep(0.01)
    print("done waiting on b", flush=True)
    print("Stdout:")
    print("", "job_a", repr(job_a.stdout))
    if job_b:
        print("", "job_b", repr(job_b.stdout))
