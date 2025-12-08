"""
HKM curses builder and explorer.

One-screen terminal UI to: (1) pick an input corpus path and index root,
launch the HKM builder, and watch all jobs progress; (2) browse the built
HKM tree with per-node statistics and preview snippets. Uses only curses
plus the existing job and builder modules. Designed for quick local demos
and smoke-tests of the index layout.

Example:
    python -m tlux.search.hkm.tui
"""

from __future__ import annotations

import curses
import json
import os
import shutil
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .builder.launcher import build_search_index
from .fs import FileSystem
from . import jobs
from .jobs import JOBS_ROOT, watcher

# Robust key codes (macOS curses lacks KEY_TAB)
KEY_TAB = getattr(curses, "KEY_TAB", 9)
KEY_BTAB = getattr(curses, "KEY_BTAB", 353)


# --------------------------------------------------------------------------- #
#  Small data holders                                                         #
# --------------------------------------------------------------------------- #

# Description:
#   Track a single form field used on the launch screen.
# 
# Parameters:
#   name (str): Field key.
#   label (str): Human label.
#   value (str): Current text.
#   kind (str): "path" | "int".
#   cursor (int): Cursor position inside value.
#   suggestions (List[str]): Path completions.
#   suggestion_idx (int): Index of highlighted suggestion.
# 
@dataclass
class FormField:
    name: str
    label: str
    value: str
    kind: str
    cursor: int = 0
    prev_value: str = ""
    last_input_ts: float = 0.0
    suggestions: List[str] = field(default_factory=list)
    suggestion_idx: int = 0


# Description:
#   Job status summary used to render the build monitor.
# 
# Parameters:
#   job_id (str): Identifier from the job directory.
#   status (str): WAITING | QUEUED | RUNNING | SUCCEEDED | FAILED.
#   command (str): Executed function path.
#   exit_code (int | None): Process return code.
#   runtime (float): Seconds from start_ts to now/end_ts.
#   start_ts (float | None): Start timestamp.
#   end_ts (float | None): End timestamp.
#   stdout_tail (List[str]): Last few stdout lines.
#   stderr_tail (List[str]): Last few stderr lines.
# 
@dataclass
class JobSummary:
    job_id: str
    status: str
    command: str
    exit_code: Optional[int]
    runtime: float
    start_ts: Optional[float]
    end_ts: Optional[float]
    pid: Optional[int] = None
    child_pid: Optional[int] = None
    rss_bytes: int = 0
    cpu_pct: float = 0.0
    gpu_pct: Optional[float] = None
    stdout_tail: List[str] = field(default_factory=list)
    stderr_tail: List[str] = field(default_factory=list)


# Description:
#   Snapshot of an HKM node for the browser pane.
# 
# Parameters:
#   path (Path): Node directory.
#   stats (Dict): Parsed stats.json if present.
#   children (List[Path]): Child cluster directories.
#   preview_random (List[str]): Human-readable preview rows.
#   preview_diverse (List[str]): Human-readable preview rows.
#   files (List[str]): Files present under the node.
# 
@dataclass
class NodeInfo:
    path: Path
    stats: Dict
    children: List[Path]
    preview_random: List[str]
    preview_diverse: List[str]
    files: List[str]


# --------------------------------------------------------------------------- #
#  Utility helpers                                                            #
# --------------------------------------------------------------------------- #

# Description:
#   Return a short, human-readable hh:mm:ss string for *seconds*.
# 
# Parameters:
#   seconds (float): Elapsed seconds.
# 
# Returns:
#   str: Formatted duration.
# 
def _fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


# Description:
#   Collect path suggestions for a partially typed value.
# 
# Parameters:
#   text (str): Current user input.
#   limit (int): Maximum number of suggestions.
# 
# Returns:
#   List[str]: Candidate absolute paths.
# 
def _path_suggestions(text: str, limit: int | None = None) -> List[str]:
    expanded = os.path.expanduser(text or ".")
    target = Path(expanded)
    base_dir = target if target.is_dir() else target.parent
    prefix = "" if target.is_dir() else target.name
    if not base_dir.exists():
        return []
    suggestions: List[str] = []
    try:
        for entry in sorted(base_dir.iterdir()):
            if entry.name.startswith(prefix):
                suggestions.append(str(entry))
            if limit is not None and len(suggestions) >= limit:
                break
    except OSError:
        return []
    return suggestions


# Description:
#   Read the tail of a small text file safely.
# 
# Parameters:
#   path (Path): File path.
#   lines (int): Number of lines to return from the end.
# 
# Returns:
#   List[str]: Tail lines stripped of newlines.
# 
def _tail(path: Path, lines: int = 4) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            buf = handle.readlines()
        return [ln.rstrip("\n") for ln in buf[-lines:]]
    except OSError:
        return []


# Description:
#   Load job summaries from the jobs root using FileSystem semantics.
# 
# Parameters:
#   fs (FileSystem): Job file system rooted at JOBS_ROOT.
#   tail_lines (int): Number of stdout/stderr lines to capture.
# 
# Returns:
#   List[JobSummary]: One entry per job directory.
# 
def _load_jobs(fs: FileSystem, tail_lines: int = 200) -> List[JobSummary]:
    jobs: List[JobSummary] = []
    buckets = ("waiting", "queued", "running", "succeeded", "failed")
    for bucket in buckets:
        bucket_path = Path(fs.join(bucket))
        if not bucket_path.exists():
            continue
        for jid in sorted(bucket_path.iterdir()):
            id_dir = Path(fs.join("ids", jid.name))
            cfg_path = id_dir / "job_config"
            cmd_path = id_dir / "exec_function"
            if not cfg_path.exists():
                continue
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            cmd = cmd_path.read_text(encoding="utf-8") if cmd_path.exists() else "unknown"
            start_ts = cfg.get("start_ts")
            end_ts = cfg.get("end_ts")
            now = time.time()
            runtime = 0.0
            if start_ts is not None:
                runtime = (end_ts or now) - float(start_ts)
            # Last resource sample
            rss = 0
            cpu = 0.0
            gpu: float | None = None
            res_path = id_dir / "resources"
            if res_path.exists():
                try:
                    lines = res_path.read_text().strip().splitlines()
                    if lines:
                        last = json.loads(lines[-1])
                        rss = int(last.get("rss", 0))
                        cpu = float(last.get("cpu_percent", 0.0))
                        if "gpu_percent" in last:
                            gpu = float(last.get("gpu_percent", 0.0))
                except Exception:
                    pass
            jobs.append(
                JobSummary(
                    job_id=cfg.get("id", jid.name),
                    status=cfg.get("status", bucket).upper(),
                    command=cmd.strip(),
                    exit_code=cfg.get("exit_code"),
                    runtime=runtime,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    pid=cfg.get("pid"),
                    child_pid=cfg.get("child_pid"),
                    rss_bytes=rss,
                    cpu_pct=cpu,
                    gpu_pct=gpu,
                    stdout_tail=_tail(id_dir / "stdout", lines=tail_lines),
                    stderr_tail=_tail(id_dir / "stderr", lines=tail_lines),
                )
            )
    return jobs


# Description:
#   Convert a preview array into printable rows with a few components.
# 
# Parameters:
#   preview_path (Path): Path to .npy file.
#   limit (int): Maximum rows to render.
# 
# Returns:
#   List[str]: Compact preview strings.
# 
def _preview_rows(preview_path: Path, limit: int = 3) -> List[str]:
    if not preview_path.exists():
        return []
    try:
        arr = np.load(preview_path, mmap_mode="r")
        if arr.ndim < 2 or arr.size == 0:
            return [f"{preview_path.name}: empty"]
        rows = min(limit, arr.shape[0])
        dims = arr.shape[1]
        out: List[str] = [f"{preview_path.name}: {arr.shape[0]}x{dims}"]
        for i in range(rows):
            vals = " ".join(f"{float(v):.3f}" for v in arr[i][: min(6, dims)])
            out.append(f"  {i:02d}: {vals}")
        return out
    except Exception as exc:
        return [f"{preview_path.name}: error {exc}"]


# Description:
#   Gather stats and child information for an HKM node directory.
# 
# Parameters:
#   node_path (Path): Directory representing the node.
# 
# Returns:
#   NodeInfo: Snapshot including previews and stats.
# 
def _load_node(node_path: Path) -> NodeInfo:
    stats: Dict = {}
    stats_path = node_path / "stats.json"
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            stats = {}

    children = [p for p in sorted(node_path.iterdir()) if p.is_dir() and p.name.startswith("cluster_")]
    preview_random = _preview_rows(node_path / "preview_random.npy")
    preview_diverse = _preview_rows(node_path / "preview_diverse.npy")
    files = sorted(p.name for p in node_path.iterdir() if p.is_file())

    return NodeInfo(
        path=node_path,
        stats=stats,
        children=children,
        preview_random=preview_random,
        preview_diverse=preview_diverse,
        files=files,
    )


# --------------------------------------------------------------------------- #
#  Main curses application                                                    #
# --------------------------------------------------------------------------- #

# Description:
#   Interactive curses UI orchestrating build launch and index browsing.
# 
# Parameters:
#   stdscr (curses.window): Root window provided by curses.wrapper.
# 
class HkmTuiApp:
    def __init__(self, stdscr: curses.window) -> None:
        self.stdscr = stdscr
        self.state: str = "form"
        self.message: str = ""
        self.fields: List[FormField] = [
            FormField("docs", "Docs directory", str(Path.cwd()), "path"),
            FormField("index", "Index root", str(Path.cwd() / "hkm_index"), "path"),
            FormField("workers", "Workers", "1", "int"),
        ]
        self.active_field: int = 0
        self.state_path = Path.home() / ".hkm_state.json"
        self.skip_paths = []
        self._load_state()
        self.job_cursor: int = 0
        self.jobs_root = Path(JOBS_ROOT)
        self.jobs_fs = FileSystem(root=str(self.jobs_root))
        self.job_table: List[JobSummary] = []
        self.build_thread: Optional[threading.Thread] = None
        self.build_error: Optional[str] = None
        self.watcher_thread: Optional[threading.Thread] = None
        self.browser_path: Optional[Path] = None
        self.browser_info: Optional[NodeInfo] = None
        self.browser_cursor: int = 0
        self.last_job_refresh: float = 0.0
        curses.curs_set(0)
        self.stdscr.nodelay(True)
        self._init_colors()

    def _load_state(self) -> None:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        except Exception:
            data = {}
        for field in self.fields:
            if field.name in data.get("fields", {}):
                field.value = str(data["fields"][field.name])
        self.skip_paths = data.get("skip_paths", [])

    def _persist_state(self) -> None:
        try:
            payload = {"fields": {f.name: f.value for f in self.fields}, "skip_paths": self.skip_paths}
            self.state_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    # Description:
    #   Set up color pairs for highlighting.
    # 
    def _init_colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_GREEN, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)
        curses.init_pair(6, curses.COLOR_BLACK, -1)  # dim/skip

    # Write text padded to clear the full line.
    def _write_line(self, y: int, text: str, *, color: int | None = None, attr: int | None = None) -> None:
        _, w = self.stdscr.getmaxyx()
        text = text.replace("\x00", "")
        if color:
            self.stdscr.attron(curses.color_pair(color))
        if attr:
            self.stdscr.attron(attr)
        self.stdscr.addstr(y, 2, (text[: w - 4]).ljust(w - 4))
        if attr:
            self.stdscr.attroff(attr)
        if color:
            self.stdscr.attroff(curses.color_pair(color))

    # Description:
    #   Primary event loop driving the UI.
    # 
    def run(self) -> None:
        while True:
            self._refresh_jobs()
            self._draw()
            try:
                key = self.stdscr.getch()
            except KeyboardInterrupt:
                self._kill_all_jobs()
                break
            if key == -1:
                time.sleep(0.05)
                continue
            if key in (ord("q"), ord("Q")):
                self._kill_all_jobs()
                break
            if self.state == "form":
                self._handle_form_key(key)
            elif self.state == "build":
                self._handle_build_key(key)
            else:
                self._handle_browser_key(key)

    # Description:
    #   Poll job files periodically to update the table.
    # 
    def _refresh_jobs(self) -> None:
        now = time.time()
        if self.state == "form":
            return
        if now - self.last_job_refresh < 0.5:
            return
        self.last_job_refresh = now
        raw_jobs = _load_jobs(self.jobs_fs, tail_lines=200)
        priority = {"RUNNING": 0, "QUEUED": 1, "WAITING": 1, "SUCCEEDED": 2, "FAILED": 3}
        self.job_table = sorted(raw_jobs, key=lambda j: (priority.get(j.status, 4), j.job_id))
        if self.state == "build" and self._build_finished():
            self.message = "Build finished. Press ENTER to open the browser."

    # Description:
    #   Draw the appropriate screen for the current state.
    # 
    def _draw(self) -> None:
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()
        title = "HKM Builder + Explorer"
        self.stdscr.attron(curses.color_pair(1))
        self.stdscr.addstr(0, 2, title)
        self.stdscr.attroff(curses.color_pair(1))
        if self.state == "form":
            self._draw_form(h, w)
        elif self.state == "build":
            self._draw_build(h, w)
        else:
            self._draw_browser(h, w)
        if self.message:
            self._write_line(h - 1, self.message, color=2)
        self.stdscr.refresh()

    # Description:
    #   Render the launch form with path autocomplete hints.
    # 
    def _draw_form(self, height: int, width: int) -> None:
        y = 2
        for idx, field in enumerate(self.fields):
            label = f"{field.label}: "
            val_display = field.value if field.value else "<empty>"
            if idx == self.active_field:
                self.stdscr.attron(curses.color_pair(3))
            self.stdscr.addstr(y, 4, label)
            is_skipped = val_display in self.skip_paths
            if is_skipped:
                self.stdscr.attron(curses.color_pair(6) | curses.A_DIM)
            self.stdscr.addstr(y, 4 + len(label), val_display[: width - 10 - len(label)])
            if is_skipped:
                self.stdscr.attroff(curses.color_pair(6) | curses.A_DIM)
            if idx == self.active_field:
                self.stdscr.attroff(curses.color_pair(3))
            y += 2

        active = self.fields[self.active_field]
        active.suggestions = _path_suggestions(active.value) if active.kind == "path" else []
        if active.suggestions:
            if active.suggestion_idx >= len(active.suggestions):
                active.suggestion_idx = 0
            total_sugs = len(active.suggestions)
            available_lines = max(0, height - (y + 2) - 2)  # start two rows below header, leave 2-line buffer
            display_max = min(total_sugs, available_lines)
            start = 0
            # Scroll window to keep cursor near bottom without hiding earlier entries.
            if active.suggestion_idx >= max(0, display_max - 2):
                start = active.suggestion_idx - (display_max - 2)
            start = max(0, min(start, max(0, total_sugs - display_max)))
            window = active.suggestions[start : start + display_max]

            # Show position / range inline with the Suggestions label.
            showing_end = start + len(window)
            header = (
                f"Suggestions {active.suggestion_idx + 1}/{total_sugs} "
                f"(showing {start + 1}-{showing_end})"
            )
            self._write_line(y + 1, header[: width - 4])
            for i, sug in enumerate(window):
                global_idx = start + i
                marker = "→ " if global_idx == active.suggestion_idx else "  "
                self.stdscr.addstr(y + 2 + i, 6, f"{marker}")
                # 0 white
                # 1 aqua
                # 2 mustard
                # 3 green
                # 4 red
                # 5 magenta
                # 6 dark gray
                if sug in self.skip_paths:
                    self.stdscr.attron(curses.color_pair(6))
                self.stdscr.addstr(y + 2 + i, 6 + len(marker), f"{sug[: width - 10]}")
                if sug in self.skip_paths:
                    self.stdscr.attroff(curses.color_pair(6))
            # Clear any unused rows within the available window to avoid stale text when shrinking.
            for i in range(len(window), available_lines):
                self._write_line(y + 2 + i, "")


        index_value = self._field_value("index")
        index_path = Path(os.path.expanduser(index_value)) if index_value else None
        warn_line = None
        if index_path and index_path.exists():
            warn_line = f"Warning: index path {index_path} exists and may be overwritten."
        jobs_path = index_path / ".hkm_jobs" if index_path else None
        if jobs_path and jobs_path.exists():
            warn_line = f"Warning: {jobs_path} will be cleared before running."
        if warn_line:
            self.stdscr.attron(curses.color_pair(4))
            lines = textwrap.wrap(warn_line, width - 4) or [warn_line]
            start_y = height - 3 - len(lines)
            for i, line in enumerate(lines):
                self.stdscr.addstr(start_y + i, 2, line)
            self.stdscr.attroff(curses.color_pair(4))
        else:
            start_y = height - 2
        if self.skip_paths:
            skip_text = "Skipping: " + ", ".join([
                "." + s[len(str(Path.cwd())):] for s in
                self.skip_paths[:3]
            ])
            if len(self.skip_paths) > 3:
                skip_text += f" (+{len(self.skip_paths)-3} more)"
            self.stdscr.addstr(start_y - 2, 2, skip_text[: width - 4])

        footer = "TAB next | ENTER build | up/down suggestions | right accept | left undo | q quit"
        self.stdscr.addstr(height - 2, 2, footer[: width - 4])

    # Description:
    #   Handle keypresses on the launch form.
    # 
    def _handle_form_key(self, key: int) -> None:
        field = self.fields[self.active_field]
        if key in (KEY_TAB, 9):
            self.active_field = (self.active_field + 1) % len(self.fields)
            return
        if key in (KEY_BTAB, 353):
            self.active_field = (self.active_field - 1) % len(self.fields)
            return
        if key in (ord("s"), ord("S")) and field.kind == "path":
            # Mark highlighted suggestion (or current value) to skip.
            target = None
            if field.suggestions and field.suggestion_idx < len(field.suggestions):
                target = field.suggestions[field.suggestion_idx]
            else:
                target = field.value
            if target:
                if target not in self.skip_paths:
                    self.skip_paths.append(target)
                else:
                    self.skip_paths.remove(target)
                self._persist_state()
            return
        if key in (curses.KEY_UP, curses.KEY_DOWN) and field.kind == "int":
            try:
                current = int(field.value) if field.value.strip() else 0
            except ValueError:
                current = 0
            delta = 1 if key == curses.KEY_UP else -1
            field.value = str(max(1, current + delta))
            field.cursor = len(field.value)
            field.last_input_ts = time.time()
            self._persist_state()
            return
        if key in (curses.KEY_UP, curses.KEY_DOWN) and field.kind == "path":
            if field.suggestions:
                if key == curses.KEY_UP:
                    field.suggestion_idx = (field.suggestion_idx - 1) % len(field.suggestions)
                else:
                    field.suggestion_idx = (field.suggestion_idx + 1) % len(field.suggestions)
            return
        if key == curses.KEY_RIGHT and field.kind == "path" and field.suggestions:
            field.prev_value = field.value
            field.value = field.suggestions[field.suggestion_idx]
            field.cursor = len(field.value)
            self._persist_state()
            return
        if key == curses.KEY_LEFT and field.kind == "path":
            if field.prev_value:
                field.value = field.prev_value
                field.cursor = len(field.value)
                field.prev_value = ""
                self._persist_state()
            return
        if key in (curses.KEY_ENTER, 10, 13):
            self._start_build()
            return
        if key in (curses.KEY_BACKSPACE, 127, 8):
            if field.value:
                field.value = field.value[:-1]
                field.cursor = max(0, field.cursor - 1)
                self._persist_state()
            return
        if 32 <= key <= 126:
            ch = chr(key)
            if field.kind == "int" and not ch.isdigit():
                return
            now = time.time()
            if field.kind == "int" and (now - field.last_input_ts) > 3.0:
                field.value = ""
                field.cursor = 0
            field.value = field.value[: field.cursor] + ch + field.value[field.cursor :]
            field.cursor += 1
            field.last_input_ts = now
            self._persist_state()

    # Description:
    #   Launch the build in a background thread and start the watcher.
    # 
    def _start_build(self) -> None:
        docs_dir = self._field_value("docs")
        index_root = self._field_value("index")
        workers = int(self._field_value("workers") or "1")
        if not Path(docs_dir).exists():
            self.message = "Docs directory does not exist."
            return
        self.message = "Launching build jobs..."

        # Choose a safe FileSystem root that encloses docs and index.
        try:
            candidate_root = os.path.commonpath([os.path.abspath(docs_dir), os.path.abspath(index_root)])
        except Exception:
            candidate_root = os.path.abspath(index_root)
        # Keep roots scoped to the work tree; fall back to index_root if commonpath is '/'.
        fs_root = candidate_root if candidate_root not in (os.sep, "") else os.path.abspath(index_root)

        # Isolate jobs under the chosen index root to avoid stale test jobs.
        jobs_root = Path(index_root) / ".hkm_jobs"
        if jobs_root.exists():
            shutil.rmtree(jobs_root, ignore_errors=True)
        jobs_root.mkdir(parents=True, exist_ok=True)
        for bucket in ("ids", "queued", "running", "succeeded", "failed", "next"):
            (jobs_root / bucket).mkdir(exist_ok=True)
        jobs.JOBS_ROOT = str(jobs_root)
        self.jobs_root = jobs_root
        self.jobs_fs = FileSystem(root=str(jobs_root))
        self.job_table = []
        self.watcher_thread = None
        self._ensure_watcher()

        def _run_build() -> None:
            try:
                build_search_index(
                    FileSystem(root=fs_root),
                    docs_dir,
                    index_root,
                    workers,
                    fs_root=fs_root,
                    skip_paths=self.skip_paths,
                )
            except Exception as exc:
                self.build_error = f"Build failed: {exc}"

        self.build_thread = threading.Thread(target=_run_build, daemon=True)
        self.build_thread.start()
        self.state = "build"

    # Description:
    #   Return current value for a named form field.
    # 
    def _field_value(self, name: str) -> str:
        for f in self.fields:
            if f.name == name:
                return f.value
        return ""

    # Description:
    #   Ensure a watcher thread is running to advance queued jobs.
    # 
    def _ensure_watcher(self) -> None:
        if self.watcher_thread is not None:
            return
        # Ensure expected buckets exist to avoid watcher listdir errors.
        for bucket in ("ids", "queued", "running", "succeeded", "failed", "next"):
            try:
                self.jobs_fs.mkdir(self.jobs_fs.join(bucket), exist_ok=True)
            except Exception:
                pass
        self.watcher_thread = threading.Thread(
            target=watcher,
            args=(self.jobs_fs, 1.5),
            daemon=True,
        )
        self.watcher_thread.start()

    # Description:
    #   Kill all running jobs in the active jobs root (best-effort).
    # 
    def _kill_all_jobs(self) -> None:
        try:
            running_dir = Path(self.jobs_fs.join("running"))
        except Exception:
            return
        if not running_dir.exists():
            return
        for job_dir in running_dir.iterdir():
            try:
                job = jobs.Job(self.jobs_fs, path=str(job_dir))
                job.kill()
            except Exception:
                continue
        self.job_table = _load_jobs(self.jobs_fs, tail_lines=12)

    # Description:
    #   Draw the job monitor view.
    # 
    def _draw_build(self, height: int, width: int) -> None:
        summary = self._job_summary()
        self._write_line(2, summary)
        columns = "ID         STATUS    RUNTIME    CPU% GPU%  RSS(MB)  COMMAND"
        self._write_line(4, columns)
        # Show at most half the screen for the table to keep detail visible.
        max_rows = max(3, min(len(self.job_table), max(3, height // 2 - 4)))
        if not self.job_table:
            self._write_line(6, "No jobs yet. Waiting for launcher...")
        for i, job in enumerate(self.job_table[: max_rows]):
            row_y = 5 + i
            state_colors = {
                "SUCCEEDED": 3,
                "FAILED": 4,
                "QUEUED": 1,
                "WAITING": 1,
                "RUNNING": 0,
            }
            color = curses.color_pair(state_colors.get(job.status, 0))
            if i == self.job_cursor:
                self.stdscr.attron(curses.A_REVERSE)
            self.stdscr.attron(color)
            runtime = _fmt_seconds(job.runtime)
            gpu_disp = "n/a" if job.gpu_pct is None or job.gpu_pct < 0 else f"{job.gpu_pct:>4.1f}"
            line = (
                f"{job.job_id:<10} {job.status:<9} {runtime:<9} "
                f"{job.cpu_pct:>5.1f} {gpu_disp:>4} {job.rss_bytes/1_048_576:>7.1f} "
                f"   {job.command}"
            )
            self._write_line(row_y, line)
            self.stdscr.attroff(color)
            if i == self.job_cursor:
                self.stdscr.attroff(curses.A_REVERSE)

        detail_y = 6 + max_rows
        self._write_line(detail_y, "Selected job details (stdout/stderr tail):")
        if self.job_table:
            job = self.job_table[min(self.job_cursor, len(self.job_table) - 1)]
            gpu_disp = "n/a" if job.gpu_pct is None or job.gpu_pct < 0 else f"{job.gpu_pct:.1f}%"
            # Build chronological-ish list: meta lines then stdout then stderr tails.
            all_lines: List[Tuple[str, Optional[int]]] = []
            named_log_lines: Dict[str, int] = {}
            meta = [
                f"Exit code: {job.exit_code} | Start: {job.start_ts} | End: {job.end_ts}",
                f"pid: {job.pid} child: {job.child_pid} rss: {job.rss_bytes/1_048_576:.2f} MiB cpu: {job.cpu_pct:.1f}% gpu: {gpu_disp}",
                "",
                "STDOUT" if job.stdout_tail else "STDOUT (empty)",
            ]
            all_lines.extend((m, None) for m in meta)
            for ln in job.stdout_tail or [""]:
                if (ln.strip().startswith("[") and "]" in ln):
                    log_line_name = ln[ln.index('['):ln.index(']')]
                    if (log_line_name in named_log_lines) and (named_log_lines[log_line_name] == len(all_lines)-1):
                        all_lines[named_log_lines[log_line_name]] = (ln, None)
                        ln = None
                if ln is not None:
                    all_lines.append((ln, None))  # default color (white)
            all_lines.append(("", None))
            all_lines.append(("STDERR" if job.stderr_tail else "STDERR (empty)", 4))
            for ln in job.stderr_tail or ["(no stderr captured)"]:
                all_lines.append((ln, 4))

            wrap_width = max(10, width - 6)
            flat_lines: List[Tuple[str, Optional[int]]] = []
            for text, col in all_lines:
                wrapped = textwrap.wrap(text, wrap_width) or [text]
                for wline in wrapped:
                    flat_lines.append((wline, col))

            available = max(1, height - detail_y - 3)  # leave footer space
            flat_lines = flat_lines[-available:]  # keep most recent
            start_row = detail_y + 1
            # bottom-align within available space
            start_row = detail_y + 1 + (available - len(flat_lines))
            row = start_row
            for line, col in flat_lines:
                self._write_line(row, "  " + line, color=col)
                row += 1

        # Spacer then footer
        self._write_line(height - 3, "")
        footer = "ENTER: browse when finished | arrows: move | TAB: refresh | q: quit"
        self._write_line(height - 2, footer)
        if self.build_error:
            err_lines = textwrap.wrap(self.build_error, width - 4) or [self.build_error]
            start = height - 4 - len(err_lines[-2:])
            for i, line in enumerate(err_lines[-2:]):
                self._write_line(start + i, line, color=4)

    # Description:
    #   Handle keypresses on the build monitor.
    # 
    def _handle_build_key(self, key: int) -> None:
        if key in (curses.KEY_DOWN, ord("j")) and self.job_table:
            self.job_cursor = min(self.job_cursor + 1, len(self.job_table) - 1)
        elif key in (curses.KEY_UP, ord("k")) and self.job_table:
            self.job_cursor = max(0, self.job_cursor - 1)
        elif key in (curses.KEY_ENTER, 10, 13):
            if self._build_finished():
                self._enter_browser()
            else:
                self.message = "Build still running; wait for jobs to finish."
        elif key in (KEY_TAB, 9):
            self.last_job_refresh = 0.0  # force refresh

    # Description:
    #   Compute a compact textual summary of job counts.
    # 
    def _job_summary(self) -> str:
        counts: Dict[str, int] = {"QUEUED": 0, "WAITING": 0, "RUNNING": 0, "SUCCEEDED": 0, "FAILED": 0}
        for job in self.job_table:
            counts[job.status] = counts.get(job.status, 0) + 1
        parts = [f"{k.lower()}={v}" for k, v in counts.items()]
        return "Jobs: " + " | ".join(parts)

    # Description:
    #   Check whether building appears to be complete.
    # 
    def _build_finished(self) -> bool:
        if not self.job_table:
            return False
        pending = any(j.status in {"QUEUED", "WAITING", "RUNNING"} for j in self.job_table)
        if pending:
            return False
        index_root = Path(self._field_value("index"))
        hkm_root = index_root / "hkm"
        return hkm_root.exists()

    # Description:
    #   Switch into the browser view once build completes.
    # 
    def _enter_browser(self) -> None:
        index_root = Path(self._field_value("index"))
        hkm_root = index_root / "hkm"
        if not hkm_root.exists():
            self.message = "Index root missing hkm/ directory. Build may have failed."
            return
        self.browser_path = hkm_root
        self.browser_info = _load_node(hkm_root)
        self.browser_cursor = 0
        self.state = "browse"
        self.message = "Browse mode: arrows navigate, ENTER descend, BACKSPACE up."

    # Description:
    #   Draw the index browser view with stats and previews.
    # 
    def _draw_browser(self, height: int, width: int) -> None:
        if self.browser_info is None:
            self.stdscr.addstr(2, 2, "No index loaded.")
            return
        node = self.browser_info
        self.stdscr.addstr(2, 2, f"Node: {str(node.path)[: width - 6]}")
        self.stdscr.addstr(3, 2, "Children (ENTER to descend, BACKSPACE to ascend):")
        max_children = max(3, min(len(node.children), height - 12))
        for i in range(max_children):
            label = node.children[i].name if i < len(node.children) else ""
            if i == self.browser_cursor:
                self.stdscr.attron(curses.A_REVERSE)
            self.stdscr.addstr(4 + i, 4, label[: width - 8])
            if i == self.browser_cursor:
                self.stdscr.attroff(curses.A_REVERSE)

        stats_y = 4
        stats_x = width // 2
        stats_lines = [
            f"doc_count: {node.stats.get('doc_count', 'n/a')}",
            f"emb_count: {node.stats.get('emb_count', 'n/a')}",
            f"depth: {node.stats.get('depth', 'n/a')}",
            f"leaf: {node.stats.get('leaf', 'n/a')}",
            f"files: {', '.join(node.files[:4])}",
        ]
        for i, line in enumerate(stats_lines):
            self.stdscr.addstr(stats_y + i, stats_x, line[: width - stats_x - 2])

        preview_y = stats_y + len(stats_lines) + 1
        self.stdscr.addstr(preview_y, stats_x, "Preview random:")
        for i, line in enumerate(node.preview_random[:3]):
            self.stdscr.addstr(preview_y + 1 + i, stats_x, line[: width - stats_x - 2])
        div_y = preview_y + 1 + max(1, len(node.preview_random[:3])) + 1
        self.stdscr.addstr(div_y, stats_x, "Preview diverse:")
        for i, line in enumerate(node.preview_diverse[:3]):
            self.stdscr.addstr(div_y + 1 + i, stats_x, line[: width - stats_x - 2])

        if not node.children and not node.files and not node.stats:
            self.stdscr.addstr(preview_y, 2, "Empty node (no stats or children found).")

        footer = "UP/DOWN: select child | ENTER: descend | BACKSPACE: up | TAB: refresh | q: quit"
        self.stdscr.addstr(height - 2, 2, footer[: width - 4])

    # Description:
    #   Handle keypresses inside the browser view.
    # 
    def _handle_browser_key(self, key: int) -> None:
        if self.browser_info is None:
            return
        node = self.browser_info
        if key in (curses.KEY_DOWN, ord("j")) and node.children:
            self.browser_cursor = min(self.browser_cursor + 1, len(node.children) - 1)
        elif key in (curses.KEY_UP, ord("k")) and node.children:
            self.browser_cursor = max(0, self.browser_cursor - 1)
        elif key in (curses.KEY_ENTER, 10, 13) and node.children:
            self.browser_path = node.children[self.browser_cursor]
            self.browser_info = _load_node(self.browser_path)
            self.browser_cursor = 0
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            parent = self.browser_path.parent if self.browser_path else None
            if parent and (parent / "stats.json").exists():
                self.browser_path = parent
                self.browser_info = _load_node(parent)
                self.browser_cursor = 0
        elif key in (KEY_TAB, 9):
            if self.browser_path:
                self.browser_info = _load_node(self.browser_path)


# --------------------------------------------------------------------------- #
#  Entrypoint                                                                 #
# --------------------------------------------------------------------------- #

# Description:
#   Launch curses app; kept short per repository CLI guidelines.
# 
def main() -> None:
    curses.wrapper(lambda stdscr: HkmTuiApp(stdscr).run())


if __name__ == "__main__":  # pragma: no cover
    main()
