import json
import os
import subprocess
from pathlib import Path

import pytest


# Run one expanded benchmark in a bounded, configurable subprocess.
def _run_expanded_benchmark(
    index: str,
    benchmark: str,
    require_gate: bool = False,
    require_precision: bool = False,
) -> dict:
    root = Path(__file__).parents[1]
    command = [
        str(root / "bin" / "hkm-language-benchmark"),
        index,
        "--benchmark", str(root / "plan" / benchmark),
        "--tool-mode", "semantic",
        "--top-k", "5",
        "--timeout", os.environ.get("HKM_EXPANDED_LM_TIMEOUT", "15"),
    ]
    if require_gate:
        command.append("--require-gate")
    if require_precision:
        command.extend([
            "--judgements", str(root / "plan" / "benchmark_fineweb_expanded_deep_judgements.json"),
            "--require-precision",
        ])
    model = os.environ.get("HKM_EXPANDED_MODEL")
    if model:
        command.extend([
            "--base-url", os.environ.get("HKM_EXPANDED_BASE_URL", "http://127.0.0.1:1234/v1"),
            "--model", model,
            "--model-first",
            "--always-refine",
            "--warmup",
        ])
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=float(os.environ.get("HKM_EXPANDED_TEST_TIMEOUT", "900")),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


# Run the expanded local benchmark only when its large ignored index is present.
def test_fineweb_expanded_deep_gate() -> None:
    index = os.environ.get("HKM_EXPANDED_INDEX")
    if not index:
        pytest.skip("set HKM_EXPANDED_INDEX to run the long FineWeb gate")
    report = _run_expanded_benchmark(
        index,
        "benchmark_fineweb_expanded_deep.md",
        require_gate=True,
        require_precision=True,
    )
    assert report["summary"]["cases"] == 24
    assert report["summary"]["gate_passed"] is True
    assert report["summary"]["precision_gate_passed"] is True


# Keep a known-hard floor visible without making it a perfect-recall gate.
def test_fineweb_expanded_stress_floor() -> None:
    index = os.environ.get("HKM_EXPANDED_INDEX")
    if os.environ.get("HKM_RUN_EXPANDED_STRESS") != "1" or not index:
        pytest.skip("set HKM_RUN_EXPANDED_STRESS=1 and HKM_EXPANDED_INDEX for the stress floor")
    report = _run_expanded_benchmark(index, "benchmark_fineweb_expanded_stress.md")
    assert report["summary"]["cases"] == 25
    assert report["summary"]["coherent_hits"] == 25
    assert report["summary"]["passed"] >= 13
