"""
Helper functions used by worker import regression tests.
"""

import time
import json
from pathlib import Path


def job_ok() -> int:
    """Return zero to indicate success."""
    return 0


def job_with_output() -> int:
    """Emit several lines to stdout for log visibility tests."""
    for i in range(5):
        print(f"[job_with_output] tick {i}", flush=True)
        time.sleep(0.05)
    return 0


def job_cpu_burner() -> int:
    """Burn CPU briefly to produce non-zero cpu% in resources."""
    start = time.time()
    x = 0
    while time.time() - start < 0.2:
        x += 1
    print(f"[job_cpu_burner] iterations={x}", flush=True)
    return 0


def run_default_worker_job(docs_dir: str, out_dir: str) -> int:
    """Run tokenize_and_embed.default_worker on a small corpus."""
    from tlux.search.hkm.builder.tokenize_and_embed import default_worker
    manifest_path = Path(out_dir) / "manifest.json"
    manifest_path.write_text(json.dumps([str(p) for p in Path(docs_dir).glob("*.txt")]), encoding="utf-8")
    print(f"[runner] manifest at {manifest_path}", flush=True)
    default_worker(
        document_directory=docs_dir,
        output_directory=out_dir,
        worker_index=0,
        total_workers=1,
        manifest_path=str(manifest_path),
        fs_root=out_dir,
    )
    return 0
