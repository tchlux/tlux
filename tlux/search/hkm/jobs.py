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

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .fs import FileSystem

JOBS_ROOT: str = "jobs"
ID_WIDTH: int = 9
STATUS_VALUES: set[str] = {"QUEUED", "RUNNING", "FINISHED", "FAILED"}


# Wrapper for a single job stored on disk.
#
# Parameters:
#   fs (FileSystem): File-system abstraction.
#   path (str | Path | None): Path to existing job directory.
#   command (str | None): Import path to callable used when spawning.
#   args (Iterable[Any] | None): Positional arguments.
#   kwargs (dict | None): Keyword arguments.
#   resource_config (dict | None): Resource configuration.
#
# Attributes:
#   id (str), status (str), exit_code (int | None), submit_ts (float),
#   start_ts (float | None), end_ts (float | None), resource_config (dict)
# 
class Job:
    def __init__(self, fs: FileSystem, path: str | Path | None = None, *,
                 command: str | None = None, args: Iterable[Any] | None = None,
                 kwargs: Dict[str, Any] | None = None,
                 resource_config: Optional[Dict[str, Any]] = None) -> None:
        self._fs: FileSystem = fs

        if path is not None:
            self._load(Path(path))
            return

        if command is None:
            raise ValueError("Either 'path' or 'command' must be provided.")

        spawned: Job = create_new_job(fs, command, args=args, kwargs=kwargs,
                                      resource_config=resource_config)
        self.__dict__.update(spawned.__dict__)

    @property
    def command(self) -> str:
        return self._fs.read(self._fs.join(self.path, "exec_function")).decode()

    @property
    def arguments(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        blob: bytes = self._fs.read(self._fs.join(self.path, "exec_args"))
        payload: Dict[str, Any] = pickle.loads(blob)
        return tuple(payload.get("args", ())), payload.get("kwargs", {})

    @property
    def stdout(self) -> str:
        path = self._fs.join(self.path, "stdout")
        return self._fs.read(path).decode(errors="ignore") if self._fs.exists(path) else ""

    @property
    def stderr(self) -> str:
        path = self._fs.join(self.path, "stderr")
        return self._fs.read(path).decode(errors="ignore") if self._fs.exists(path) else ""

    def _save(self) -> None:
        cfg = {
            "id": self.id,
            "status": self.status,
            "exit_code": self.exit_code,
            "submit_ts": self.submit_ts,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "resource_config": self.resource_config,
        }
        self._fs.write(self._fs.join(self.path, "job_config"),
                       json.dumps(cfg).encode(), overwrite=True)

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
        self.exit_code = data.get("exit_code")
        self.submit_ts = data.get("submit_ts")
        self.start_ts = data.get("start_ts")
        self.end_ts = data.get("end_ts")
        self.resource_config = data.get("resource_config", {})
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
def create_new_job(fs: FileSystem, command: str, *,
                   dependencies: Iterable[Job] = (),
                   args: Iterable[Any] | None = None,
                   kwargs: Dict[str, Any] | None = None,
                   resource_config: Optional[Dict[str, Any]] = None) -> Job:
    # Reserve a unique job ID via atomic directory creation.
    #
    # Parameters:
    #   fs (FileSystem): File-system abstraction.
    #
    # Returns:
    #   str: Reserved job ID.
    # 
    def _reserve_id(fs: FileSystem) -> str:
        while True:
            tick = int(time.time() * 10_000)
            candidate = f"{tick % 10 ** ID_WIDTH:0{ID_WIDTH}d}"
            try:
                fs.mkdir(fs.join("ids", candidate), exist_ok=False)
                return candidate
            except OSError:
                time.sleep(0.0001)

    args = tuple(args or ())
    kwargs = dict(kwargs or {})
    resource_config = dict(resource_config or {})

    job_id = _reserve_id(fs)
    has_deps = bool(dependencies)
    bucket = "waiting" if has_deps else "running"
    job_dir = fs.join(bucket, job_id)
    fs.mkdir(job_dir, exist_ok=False)

    fs.write(fs.join(job_dir, "exec_function"), command.encode(), overwrite=True)
    fs.write(fs.join(job_dir, "exec_args"),
             pickle.dumps({"args": args, "kwargs": kwargs}), overwrite=True)

    now_ts = time.time()
    cfg: Dict[str, Any] = {
        "id": job_id,
        "status": "QUEUED" if has_deps else "RUNNING",
        "exit_code": None,
        "submit_ts": now_ts,
        "start_ts": None if has_deps else now_ts,
        "end_ts": None,
        "resource_config": resource_config,
    }
    fs.write(fs.join(job_dir, "job_config"), json.dumps(cfg).encode(), overwrite=True)

    if has_deps:
        up_root = fs.join(job_dir, "upstream_jobs")
        fs.mkdir(up_root, exist_ok=True)
        for dep in dependencies:
            fs.mkdir(fs.join(up_root, dep.id), exist_ok=True)
            next_root = fs.join("next", dep.id)
            fs.mkdir(next_root, exist_ok=True)
            fs.mkdir(fs.join(next_root, job_id), exist_ok=True)
    else:
        _launch_job(fs, job_dir, command, args, kwargs)

    return Job(fs, path=job_dir)


# Spawn a job using the default file system rooted at JOBS_ROOT.
#
# Parameters:
#   command (str): Import path to function.
#   dependencies (Iterable[Job]): Optional upstream jobs.
#   *args (Any): Positional arguments.
#   **kwargs (Any): Keyword arguments.
#
# Returns:
#   Job: The created job wrapper.
# 
def spawn_job(command: str, dependencies: Iterable[Job] = (), *args: Any,
              **kwargs: Any) -> Job:
    default_fs = FileSystem(JOBS_ROOT)
    return create_new_job(default_fs, command, dependencies=dependencies,
                          args=args, kwargs=kwargs)



# Host-local maintenance loop.
# 
# Every *scan_interval* seconds this function:
# 1. Marks orphaned RUNNING jobs as FAILED.
# 2. Promotes WAITING jobs whose deps are satisfied and launches a worker.
# 3. Fires downstream activations for newly finished/failed jobs.
# 4. Prunes ids/, finished/, failed/ to the 100 newest entries.
def watcher(fs: FileSystem, scan_interval: float = 10.0) -> None:
    module_path = __name__  # path to this module

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

    def _finalise(jdir: str, status: str, reason: str) -> None:
        cfg = _load_cfg(jdir)
        cfg.update({"status": status,
                    "exit_code": -1,
                    "end_ts": time.time(),
                    "fail_reason": reason})
        _save_cfg(jdir, cfg)
        fs.rename(jdir, fs.join(status.lower(), Path(jdir).name))
        _activate_downstreams(Path(jdir).name)

    def _activate_downstreams(jid: str) -> None:
        nxt = fs.join("next", jid)
        if not fs.exists(nxt):
            return
        for did in fs.listdir(nxt):
            wdir = fs.join("waiting", did)
            up_dir = fs.join(wdir, "upstream_jobs")
            deps_remaining = [p for p in fs.listdir(up_dir)
                              if not p.startswith("_")] if fs.exists(up_dir) else []
            if deps_remaining:
                continue
            rdir = fs.join("running", did)
            try:
                if fs.rename(wdir, rdir):
                    # Launch worker for promoted job
                    subprocess.Popen(
                        [
                            sys.executable, "-c",
                            (
                                "import importlib, sys;"
                                f"mod=importlib.import_module('{module_path}');"
                                "fs_cls=getattr(mod, 'FileSystem');"
                                "job_cls=getattr(mod, 'Job');"
                                "worker=getattr(mod, 'worker');"
                                "fs=fs_cls(sys.argv[1]);"
                                "worker(fs, job_cls(fs, path=sys.argv[2]))"
                            ),
                            fs.root, rdir
                        ],
                        close_fds=True,
                    )
            except OSError:
                pass
        fs.rename(nxt, fs.join("next", f"_{jid}_done"))

    while True:
        tick_start = time.time()

        # Orphan detection
        for jid in fs.listdir("running"):
            jdir = fs.join("running", jid)
            pid_file = fs.join(jdir, "hostname", "pid")
            if not fs.exists(pid_file):
                _finalise(jdir, "FAILED", "missing-pid")
                continue
            pid = int(fs.read(pid_file).decode())
            if not _process_exists(pid):
                _finalise(jdir, "FAILED", "proc-gone")

        # Downstream activation for jobs already finished/failed
        for bucket in ("finished", "failed"):
            for jid in fs.listdir(bucket):
                _activate_downstreams(jid)

        # Garbage-collect oldest entries
        for bucket in ("ids", "finished", "failed"):
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


def worker(fs: FileSystem, job: Job) -> None:
    """Run *job* in a subprocess, monitor resources, finalise state, trigger deps."""
    import os
    import subprocess
    import sys

    module_path = __name__

    # --- launch target -------------------------------------------------------
    args, kwargs = job.arguments
    payload_hex = pickle.dumps({"args": args, "kwargs": kwargs}).hex()

    launcher_code = (
        "import importlib, pickle, sys;"
        "mod_path, blob = sys.argv[1], sys.argv[2];"
        "mod = importlib.import_module('.'.join(mod_path.split('.')[:-1]));"
        "func = getattr(mod, mod_path.split('.')[-1]);"
        "payload = pickle.loads(bytes.fromhex(blob));"
        "ret = func(*payload['args'], **payload['kwargs']);"
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

    # Record PID
    host_dir = fs.join(job.path, "hostname")
    fs.mkdir(host_dir, exist_ok=True)
    fs.write(fs.join(host_dir, "pid"), str(proc.pid).encode())

    # --- monitoring loop -----------------------------------------------------
    mem_limit = job.resource_config.get("max_rss")  # bytes or None
    cpu_target = job.resource_config.get("target_cpu")  # % or None
    grace_deadline: Optional[float] = None
    prev_cpu: Optional[Tuple[float, float]] = None  # (proc_time, wall_time)

    res_file = Path(fs.join(job.path, "resources"))
    res_file.touch(exist_ok=True)

    def sample(pid: int) -> Dict[str, Any]:
        """Return {'rss','cpu_percent'} — zero if unsupported platform."""
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
        return {"rss": rss, "cpu_percent": cpu_pct}

    while proc.poll() is None:
        snap = sample(proc.pid)
        res_file.write_bytes(
            res_file.read_bytes() +
            (json.dumps({"ts": time.time(), **snap}) + "\n").encode()
        )

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

        time.sleep(1.0)

    exit_code = proc.returncode or 0
    # --- finalise job directory ---------------------------------------------
    cfg_path = fs.join(job.path, "job_config")
    cfg = json.loads(fs.read(cfg_path).decode())
    cfg.update({
        "exit_code": exit_code,
        "end_ts": time.time(),
        "status": "FINISHED" if exit_code == 0 else "FAILED",
    })
    fs.write(cfg_path, json.dumps(cfg).encode(), overwrite=True)
    dest_bucket = "finished" if exit_code == 0 else "failed"
    fs.rename(job.path, fs.join(dest_bucket, job.id))

    # --- trigger downstreams -------------------------------------------------
    next_dir = fs.join("next", job.id)
    if not fs.exists(next_dir):
        return
    for did in fs.listdir(next_dir):
        wdir = fs.join("waiting", did)
        up_dir = fs.join(wdir, "upstream_jobs")
        try:
            fs.rename(fs.join(up_dir, job.id),
                      fs.join(up_dir, f"_{job.id}_done"))
        except OSError:
            pass
        if not [p for p in fs.listdir(up_dir) if not p.startswith("_")]:
            rdir = fs.join("running", did)
            try:
                if fs.rename(wdir, rdir):
                    subprocess.Popen(
                        [
                            sys.executable, "-c",
                            (
                                "import importlib, sys;"
                                f"mod=importlib.import_module('{module_path}');"
                                "fs_cls=getattr(mod, 'FileSystem');"
                                "job_cls=getattr(mod, 'Job');"
                                "worker=getattr(mod, 'worker');"
                                "fs=fs_cls(sys.argv[1]);"
                                "worker(fs, job_cls(fs, path=sys.argv[2]))"
                            ),
                            fs.root, rdir
                        ],
                        close_fds=True,
                    )
            except OSError:
                pass
    fs.rename(next_dir, fs.join("next", f"_{job.id}_done"))


# Simple CLI for basic usage example.
if __name__ == "__main__":
    # Quick test: Can create and load a job definition.
    job = spawn_job("my_pkg.training.run", args=("config.yaml",))
    print(f"Spawned job ID: {job.id}")
