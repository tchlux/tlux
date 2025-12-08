# Test-only helper jobs for filesystem-backed scheduler.
# 
# Provides deterministic, lightweight callables referenced by run_job in tests.

import math
import time
from pathlib import Path


def job_ok() -> int:
    print("[job_ok] start")
    time.sleep(0.1)
    print("[job_ok] end")
    return 0


def job_with_output() -> int:
    for i in range(3):
        print(f"[job_with_output] tick {i}")
        time.sleep(0.1)
    return 0


def job_cpu_burner() -> int:
    end = time.time() + 5.0
    val = 0.0
    while time.time() < end:
        val = math.sin(val + 0.05)
    print("[job_cpu_burner] done", val)
    return 0


def run_default_worker_job(document_directory: str, output_directory: str) -> int:
    print("[worker] start")
    src = Path(document_directory)
    dst = Path(output_directory)
    dst.mkdir(parents=True, exist_ok=True)
    manifest = dst / "manifest.txt"
    names = sorted(p.name for p in src.glob("*") if p.is_file())
    manifest.write_text("\n".join(names), encoding="ascii")
    time.sleep(6.0)
    print("[worker] end")
    return 0
