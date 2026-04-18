from tlux.search.hkm import jobs
from tlux.search.hkm.tests.support import enable_delayed_fs, run_watchers, setup_jobs_root, sync_fs


def test_delayed_fs_import_and_stdout(monkeypatch, tmp_path) -> None:
    old_root = jobs.JOBS_ROOT
    try:
        enable_delayed_fs(monkeypatch, listdir_delay=0.01)
        fs = setup_jobs_root(tmp_path / "jobs")
        job = jobs.run_job("tlux.search.hkm.tests.job_runner_helper.job_with_output")
        run_watchers(fs, watcher_count=1, max_workers=1)
        sync_fs(fs, settle=0.02)
        job.reload()
        assert job.status == "SUCCEEDED", job.stderr
        assert job.stdout.count("[job_with_output] tick 0") == 1
    finally:
        jobs.JOBS_ROOT = old_root


def test_delayed_fs_dependency_release_converges(monkeypatch, tmp_path) -> None:
    old_root = jobs.JOBS_ROOT
    try:
        enable_delayed_fs(monkeypatch, listdir_delay=0.01)
        fs = setup_jobs_root(tmp_path / "jobs")
        upstream = jobs.run_job("tlux.search.hkm.tests.job_runner_helper.job_ok")
        downstream = jobs.run_job(
            "tlux.search.hkm.tests.job_runner_helper.job_with_output",
            dependencies=[upstream],
        )
        run_watchers(fs, watcher_count=1, max_workers=1)
        sync_fs(fs, settle=0.02)
        upstream.reload()
        downstream.reload()
        assert upstream.status == "SUCCEEDED"
        assert downstream.status == "SUCCEEDED", downstream.stderr
        assert downstream.stdout.count("[job_with_output] tick 0") == 1
    finally:
        jobs.JOBS_ROOT = old_root


def test_delayed_fs_failure_propagates(monkeypatch, tmp_path) -> None:
    old_root = jobs.JOBS_ROOT
    try:
        enable_delayed_fs(monkeypatch, listdir_delay=0.01)
        fs = setup_jobs_root(tmp_path / "jobs")
        upstream = jobs.run_job("tlux.search.hkm.tests.job_runner_helper.job_fail")
        downstream = jobs.run_job(
            "tlux.search.hkm.tests.job_runner_helper.job_ok",
            dependencies=[upstream],
        )
        run_watchers(fs, watcher_count=1, max_workers=1)
        sync_fs(fs, settle=0.02)
        upstream.reload()
        downstream.reload()
        assert upstream.status == "FAILED"
        assert downstream.status == "FAILED"
        assert "Upstream job" in downstream.status_reason
    finally:
        jobs.JOBS_ROOT = old_root


def test_delayed_fs_watcher_oversubscription_has_no_duplicate_execution(monkeypatch, tmp_path) -> None:
    old_root = jobs.JOBS_ROOT
    try:
        enable_delayed_fs(monkeypatch, listdir_delay=0.01)
        fs = setup_jobs_root(tmp_path / "jobs")
        first = jobs.run_job("tlux.search.hkm.tests.job_runner_helper.job_with_output")
        second = jobs.run_job("tlux.search.hkm.tests.job_runner_helper.job_ok")
        run_watchers(fs, watcher_count=4, max_workers=4)
        sync_fs(fs, settle=0.02)
        first.reload()
        second.reload()
        assert first.status == "SUCCEEDED"
        assert second.status == "SUCCEEDED"
        assert first.stdout.count("[job_with_output] tick 0") == 1
        assert second.stdout.count("[job_ok] start") == 1
    finally:
        jobs.JOBS_ROOT = old_root


def test_delayed_fs_restart_after_watcher_kill_converges(monkeypatch, tmp_path) -> None:
    old_root = jobs.JOBS_ROOT
    try:
        enable_delayed_fs(monkeypatch, listdir_delay=0.01)
        fs = setup_jobs_root(tmp_path / "jobs")
        upstream = jobs.run_job("tlux.search.hkm.tests.job_runner_helper.job_cpu_burner")
        downstream = jobs.run_job(
            "tlux.search.hkm.tests.job_runner_helper.job_ok",
            dependencies=[upstream],
        )
        run_watchers(fs, watcher_count=1, max_workers=1, kill_one=True, timeout=15.0)
        sync_fs(fs, settle=0.02)
        upstream.reload()
        downstream.reload()
        assert upstream.status == "FAILED"
        assert downstream.status == "FAILED"
        assert not fs.listdir("waiting")
        assert not fs.listdir("queued")
        assert not fs.listdir("running")
    finally:
        jobs.JOBS_ROOT = old_root
