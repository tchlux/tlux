# Host-local worker that claims jobs from the `running/` bucket and executes
# them with subprocess isolation. Heartbeats live at `workers/<hostname>/<pid>.json`
# and concurrency is capped by `max_workers`.
# 
# Example:
#   from tlux.search.hkm.jobs_worker import start_worker
#   start_worker(fs_root="/tmp/hkm/jobs", max_workers=1, poll_interval=0.5)


import json
import os
import pickle
import socket
import subprocess
import time
import sys
import signal
from typing import Dict

from tlux.search.hkm.fs import FileSystem
from tlux.search.hkm.jobs import Job

HOSTNAME: str = socket.gethostname()


class WorkerDaemon:
    # Description:
    #   Host-local loop that claims jobs from the running bucket, executes
    #   them in subprocesses, tracks state, writes heartbeats, and kills
    #   children on shutdown.
    #
    # Parameters:
    #   fs (FileSystem): File-system abstraction rooted at jobs dir.
    #   max_workers (int): Maximum concurrent jobs.
    #   poll_interval (float): Seconds between loop iterations.
    #
    def __init__(self, fs: FileSystem, max_workers: int = 1,
                 poll_interval: float = 0.5) -> None:
        self.fs = fs
        self.max_workers = max(1, max_workers)
        self.poll_interval = poll_interval
        self._stop = False
        self._last_heartbeat = 0.0
        self._active: Dict[str, Dict[str, object]] = {}
        self._registration_path = fs.join("workers", HOSTNAME, f"{os.getpid()}.json")
        for bucket in ("workers", "running", "finished", "failed"):
            self.fs.mkdir(self.fs.join(bucket), exist_ok=True)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    # Description:
    #   Main loop: admit jobs, harvest completions, heartbeat, and
    #   honor shutdown flag set by signals.
    #
    def run_forever(self) -> None:
        while not self._stop:
            self._admit_jobs()
            self._harvest_jobs()
            self._heartbeat()
            time.sleep(self.poll_interval)
        self._kill_active()

    # Description:
    #   Claim jobs in running bucket up to capacity using a _claim file
    #   to prevent duplicate execution across workers.
    #
    def _admit_jobs(self) -> None:
        if len(self._active) >= self.max_workers:
            return
        try:
            running = list(self.fs.listdir("running"))
        except Exception:
            return
        for jid in running:
            if len(self._active) >= self.max_workers:
                break
            jdir = self.fs.join("running", jid)
            claim_path = self.fs.join(jdir, "_claim")
            try:
                self.fs.write(claim_path, str(os.getpid()).encode(), overwrite=False)
            except Exception:
                continue
            try:
                job = Job(self.fs, jdir)
            except Exception:
                try:
                    self.fs.remove(claim_path)
                except Exception:
                    pass
                continue
            proc = self._launch_job(job)
            self._active[jid] = {"proc": proc, "job": job, "claim": claim_path}

    # Description:
    #   Spawn the job function in a subprocess, wiring stdout/stderr to
    #   files in the job directory and persisting PIDs.
    #
    # Parameters:
    #   job (Job): Job wrapper with metadata and arguments.
    #
    # Returns:
    #   subprocess.Popen: Handle to launched child.
    #
    def _launch_job(self, job: Job) -> subprocess.Popen:
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
        out_path = self.fs.join(job.path, "stdout")
        err_path = self.fs.join(job.path, "stderr")
        proc = subprocess.Popen(
            [sys.executable, "-c", launcher_code, job.command, payload_hex],
            stdout=open(out_path, "ab"),
            stderr=open(err_path, "ab"),
            close_fds=True,
        )
        job.hostname = HOSTNAME
        job.job_pid = proc.pid
        job.child_pid = proc.pid
        job.status = "RUNNING"
        job._save()
        return proc

    # Description:
    #   Finalize completed processes, update job_config, move job dirs,
    #   and drop active tracking entries.
    #
    def _harvest_jobs(self) -> None:
        for jid, meta in list(self._active.items()):
            proc: subprocess.Popen = meta["proc"]  # type: ignore[assignment]
            job: Job = meta["job"]  # type: ignore[assignment]
            if proc.poll() is None:
                continue
            exit_code = proc.returncode or 0
            cfg_path = self.fs.join(job.path, "job_config")
            cfg = json.loads(self.fs.read(cfg_path).decode())
            cfg.update({
                "exit_code": exit_code,
                "end_ts": time.time(),
                "status": "FINISHED" if exit_code == 0 else "FAILED",
            })
            self.fs.write(cfg_path, json.dumps(cfg).encode(), overwrite=True)
            dest_bucket = "finished" if exit_code == 0 else "failed"
            dest_dir = self.fs.join(dest_bucket, jid)
            self.fs.rename(job.path, dest_dir)
            job.path = dest_dir
            job.status = "FINISHED" if exit_code == 0 else "FAILED"
            job.exit_code = exit_code
            claim = self._active[jid]["claim"]
            try:
                self.fs.remove(str(claim))
            except Exception:
                pass
            del self._active[jid]

    # Description:
    #   Signal handler: set stop flag and force-kill active children.
    #
    def _handle_shutdown(self, *_) -> None:
        self._stop = True
        self._kill_active()

    # Description:
    #   SIGKILL all still-alive child processes tracked as active.
    #
    def _kill_active(self) -> None:
        for meta in list(self._active.values()):
            proc: subprocess.Popen = meta["proc"]  # type: ignore[assignment]
            if proc.poll() is None:
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except Exception:
                    pass

    # Description:
    #   Write heartbeat file with pid, hostname, timestamp, active job ids.
    #
    def _heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat < 2.0:
            return
        payload = {
            "pid": os.getpid(),
            "hostname": HOSTNAME,
            "start_ts": now,
            "active_jobs": list(self._active.keys()),
        }
        self.fs.write(self._registration_path, json.dumps(payload).encode(), overwrite=True)
        self._last_heartbeat = now

# Description:
#   Entry point: start a worker loop for the given jobs root.
#
# Parameters:
#   fs_root (str): Jobs root path.
#   max_workers (int): Maximum concurrent jobs.
#   poll_interval (float): Loop sleep in seconds.
#
def start_worker(fs_root: str, max_workers: int = 1,
                 poll_interval: float = 0.5) -> None:
    fs = FileSystem(fs_root)
    WorkerDaemon(fs, max_workers=max_workers, poll_interval=poll_interval).run_forever()
