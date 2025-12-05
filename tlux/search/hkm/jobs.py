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
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

try:
    from .fs import FileSystem
except:
    # exec(open(os.path.join(os.path.dirname(__file__), "fs.py")).read())
    from tlux.search.hkm.fs import FileSystem

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


# Launch a worker subprocess for the specified job directory.
#
# Description:
#   Writes a temporary worker script and launches it as a subprocess,
#   passing filesystem and job directory arguments. Returns a Job
#   instance tracking the active worker process.
#
# Parameters:
#   fs (FileSystem): File-system abstraction.
#   job_dir (str): Path to the job directory to run.
#   module_path (str): Import path to the current module.
#
# Returns:
#   Job: Job object tracking the worker subprocess.
#
# Raises:
#   OSError: If process launch fails.
#
def _launch_worker(fs: FileSystem, job_dir: str, module_path: str = MODULE_PATH) -> 'Job':
    # Create the worker script as a string.
    repo_root = os.path.abspath(os.path.join(CODE_ROOT, os.pardir, os.pardir, os.pardir))
    script = (
        "import sys\n"
        "import traceback\n"
        "try:\n"
       f"    sys.path.append({repr(repo_root)})\n"
       f"    sys.path.append({repr(CODE_ROOT)})\n"
       f"    import {module_path} as mod\n"
        "    fs_cls = getattr(mod, 'FileSystem')\n"
        "    job_cls = getattr(mod, 'Job')\n"
        "    worker = getattr(mod, 'worker')\n"
        "    fs = fs_cls(sys.argv[2])\n"
        "    job = job_cls(fs, path=sys.argv[3])\n"
        "    worker(fs, job)\n"
        "except Exception:\n"
        "    traceback.print_exc()\n"
        "    sys.exit(1)\n"
    )
    # Write the script to a temporary file.
    temp_fd, temp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(temp_fd, "w") as f:
            f.write(script)
        proc = subprocess.Popen([
                sys.executable,
                temp_path,
                module_path,
                fs.root,
                job_dir,
            ],
            stderr=open(os.path.join(JOBS_ROOT, "launcher.stderr"), "w"),
            stdout=open(os.path.join(JOBS_ROOT, "launcher.stdout"), "w"),
            cwd=os.path.abspath(os.path.dirname(__file__)),
        )
        time.sleep(0.1)
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        return Job(fs=fs, path=job_dir, active_job=proc)
    except Exception as exc:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise exc


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
# Every *scan_interval* seconds this function:
# 
# 1. Marks orphaned RUNNING jobs as FAILED.
# 2. Promotes QUEUED jobs whose deps are satisfied and launches a worker.
# 3. Fires downstream activations for newly succeeded/failed jobs.
# 4. Prunes ids/, succeeded/, failed/ to the 100 newest entries.
# 
def watcher(fs: FileSystem, scan_interval: float = 10.0) -> None:

    def _process_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _load_cfg(jdir: str) -> Dict[str, Any]:
        return json.loads(fs.read(fs.join(jdir, "job_config")).decode())

    def _save_cfg(jdir: str, cfg: Dict[str, Any]) -> None:
        fs.write(fs.join(jdir, "job_config"),
                 json.dumps(cfg).encode(),
                 overwrite=True)

    def _finalize(jdir: str, status: str, reason: str) -> None:
        cfg = _load_cfg(jdir)
        cfg.update({"status": status, "exit_code": -1, "end_ts": time.time(), "fail_reason": reason})
        _save_cfg(jdir, cfg)
        fs.rename(jdir, fs.join(status.lower(), Path(jdir).name))
        _activate_downstreams(Path(jdir).name)

    def _activate_downstreams(jid: str) -> None:
        nxt = fs.join("next", jid)
        if not fs.exists(nxt):
            return
        for did in fs.listdir(nxt):
            wdir = fs.join("queued", did)
            up_dir = fs.join(wdir, "upstream_jobs")
            deps_remaining = [p for p in fs.listdir(up_dir)
                              if not p.startswith("_")] if fs.exists(up_dir) else []
            if deps_remaining:
                continue
            rdir = fs.join("running", did)
            # Launch worker for promoted job
            try:
                if fs.rename(wdir, rdir):
                    _launch_worker(fs=fs, job_dir=rdir)
            except Exception as e:
                print(f"Failed to move job directory and activate the downstream: {e}")
        fs.rename(nxt, fs.join("next", f"_{jid}_done"))

    while True:
        tick_start = time.time()

        # Orphan detection
        for jid in fs.listdir("running"):
            jdir = fs.join("running", jid)
            cfg_path = fs.join(jdir, "job_config")
            if not fs.exists(cfg_path):
                _finalize(jdir, "FAILED", "missing-config")
                continue
            cfg = json.loads(fs.read(cfg_path).decode())
            pid = cfg.get("pid")
            if pid is None:
                _finalize(jdir, "FAILED", "missing-pid")
                continue
            if not _process_exists(pid):
                _finalize(jdir, "FAILED", "proc-gone")

        # Downstream activation for jobs already succeeded/failed
        for bucket in ("succeeded", "failed"):
            for jid in fs.listdir(bucket):
                _activate_downstreams(jid)

        # Garbage-collect oldest entries
        for bucket in ("ids", "succeeded", "failed"):
            if not fs.exists(bucket):
                continue
            entries = sorted(fs.listdir(bucket))
            for old in entries[:-100]:
                gc_dir = fs.join(bucket, "_gc")
                fs.mkdir(gc_dir, exist_ok=True)
                fs.rename(fs.join(bucket, old), fs.join(gc_dir, old))

        # Sleep remainder of interval
        delay = scan_interval - (time.time() - tick_start)
        if delay > 0:
            time.sleep(delay)


# Run *job* in a subprocess, monitor resources, finalize state, trigger deps.
def worker(fs: FileSystem, job: Job) -> Job:
    # --- launch target -------------------------------------------------------
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
    job.job = proc
    job._save()

    # --- monitoring loop -----------------------------------------------------
    mem_limit = job.resource_config.get("max_rss")  # bytes or None
    cpu_target = job.resource_config.get("target_cpu")  # % or None
    grace_deadline: Optional[float] = None
    prev_cpu: Optional[Tuple[float, float]] = None  # (proc_time, wall_time)

    res_file = Path(fs.join(job.path, "resources"))
    res_file.touch(exist_ok=True)
    mon_out = Path(fs.join(job.path, "stdout"))

    def _gpu_percent(pid: int) -> float | None:
        """Best-effort GPU percent (nvidia-smi); returns None if unavailable."""
        nvsmi = shutil.which("nvidia-smi")
        if not nvsmi:
            return None
        try:
            out = subprocess.check_output(
                [
                    nvsmi,
                    "--query-compute-apps=pid,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 2:
                    continue
                if str(pid) == parts[0]:
                    return float(parts[1])
        except Exception:
            return None
        return None

    def sample(pids: list[int]) -> Dict[str, Any]:
        """Return {'rss','cpu_percent','gpu_percent'} for the hottest PID."""
        best = {"rss": 0, "cpu_percent": 0.0, "gpu_percent": None}
        for pid in pids:
            rss = 0
            cpu_pct = 0.0
            stat_path = Path(f"/proc/{pid}/stat")
            status_path = Path(f"/proc/{pid}/status")
            if stat_path.exists() and status_path.exists():
                with stat_path.open() as f:
                    parts = f.readline().split()
                    ut, st = int(parts[13]), int(parts[14])
                clk = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
                proc_time = (ut + st) / clk
                wall = time.time()
                nonlocal prev_cpu  # noqa: PLW0603
                if prev_cpu is not None:
                    dt_cpu = proc_time - prev_cpu[0]
                    dt_wall = wall - prev_cpu[1]
                    if dt_wall > 0:
                        cpu_pct = 100 * dt_cpu / dt_wall
                prev_cpu = (proc_time, wall)
                with status_path.open() as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss = int(line.split()[1]) * 1024
                            break
            else:
                try:
                    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
                    rss_kb = int(out.strip() or 0)
                    rss = rss_kb * 1024
                except Exception:
                    rss = 0
                try:
                    out = subprocess.check_output(["ps", "-o", "%cpu=", "-p", str(pid)], text=True)
                    cpu_pct = float(out.strip() or 0.0)
                except Exception:
                    cpu_pct = 0.0
            gpu_pct = _gpu_percent(pid)
            if rss > best["rss"]:
                best["rss"] = rss
            if cpu_pct > best["cpu_percent"]:
                best["cpu_percent"] = cpu_pct
            if gpu_pct is not None:
                if best["gpu_percent"] is None or gpu_pct > best["gpu_percent"]:
                    best["gpu_percent"] = gpu_pct
        return best

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
        snap = sample(pid_list)
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
        # Memory enforcement
        if mem_limit:
            if snap["rss"] > mem_limit:
                if grace_deadline is None:
                    os.kill(proc.pid, signal.SIGTERM)
                    grace_deadline = time.time() + 10
                elif time.time() >= grace_deadline and sample(proc.pid)["rss"] > mem_limit:
                    os.kill(proc.pid, signal.SIGKILL)
            else:
                grace_deadline = None  # reset if usage recovers
        # Simple CPU throttling
        if cpu_target and snap["cpu_percent"] > cpu_target * 1.2:
            os.kill(proc.pid, signal.SIGSTOP)
            time.sleep(2.0)
            os.kill(proc.pid, signal.SIGCONT)
        time.sleep(5.0)

    exit_code = proc.returncode or 0
    # --- finalize job directory ---------------------------------------------
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

    # --- trigger downstreams -------------------------------------------------
    next_dir = fs.join("next", job.id)
    if not fs.exists(next_dir):
        return job
    for did in fs.listdir(next_dir):
        wdir = fs.join("queued", did)
        up_dir = fs.join(wdir, "upstream_jobs")
        try:
            lock_dir = fs.join(up_dir, f"_{job.id}_done")
            fs.rename(fs.join(up_dir, job.id), lock_dir)
            fs.remove(lock_dir)
        except Exception as e:
            print(f"Failed to move job directory and activate the downstream: {e}")
        if not [p for p in fs.listdir(up_dir) if not p.startswith("_")]:
            rdir = fs.join("running", did)
            try:
                if fs.rename(wdir, rdir):
                    _launch_worker(fs=fs, job_dir=rdir)
            except Exception as e:
                print(f"Failed to move job directory and activate the downstream: {e}")
    fs.rename(next_dir, fs.join("next", f"_{job.id}_done"))

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
