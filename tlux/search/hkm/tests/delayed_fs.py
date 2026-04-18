from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List

from tlux.search.hkm.fs import FileSystem


class DelayedFileSystem:
    def __init__(
        self,
        root: str = "/tmp/hkm_index",
        write_delay: float | None = None,
        rename_delay: float | None = None,
        remove_delay: float | None = None,
        listdir_delay: float | None = None,
        exists_delay: float | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.root = root
        self.write_delay = self._delay("HKM_FS_WRITE_DELAY", write_delay)
        self.rename_delay = self._delay("HKM_FS_RENAME_DELAY", rename_delay)
        self.remove_delay = self._delay("HKM_FS_REMOVE_DELAY", remove_delay)
        self.listdir_delay = self._delay("HKM_FS_LISTDIR_DELAY", listdir_delay)
        self.exists_delay = self._delay("HKM_FS_EXISTS_DELAY", exists_delay)
        self._now = now or time.monotonic
        self._fs = FileSystem(root)
        self._meta = Path(f"{os.path.abspath(root)}.__delayed_fs__")
        self._pending = self._meta / "pending"
        self._staging = self._meta / "staging"
        self._lock = self._meta / "lock"
        self._meta.mkdir(parents=True, exist_ok=True)
        self._pending.mkdir(exist_ok=True)
        self._staging.mkdir(exist_ok=True)

    # Read one delay value from the environment or constructor.
    #
    # Arguments:
    #   env_name (str): Environment variable name
    #   value (float | None): Optional explicit override
    #
    # Returns:
    #   (float): Delay in seconds
    #
    def _delay(self, env_name: str, value: float | None) -> float:
        if value is not None:
            return max(0.0, float(value))
        raw = os.environ.get(env_name, "").strip()
        return max(0.0, float(raw or 0.0))

    # Resolve a path inside the visible root.
    #
    # Arguments:
    #   path (str): Relative or absolute path
    #
    # Returns:
    #   (str): Absolute path inside the visible root
    #
    def _resolve(self, path: str) -> str:
        return self._fs._resolve(path)

    # Join root-relative path components.
    #
    # Arguments:
    #   parts (str): Path components
    #
    # Returns:
    #   (str): Joined absolute path
    #
    def join(self, *parts: str) -> str:
        return self._fs.join(*parts)

    # Convert an absolute path into a root-relative key.
    #
    # Arguments:
    #   path (str): Absolute path under root
    #
    # Returns:
    #   (str): Root-relative path
    #
    def _rel(self, path: str) -> str:
        return os.path.relpath(path, self._resolve("."))

    # Acquire the filesystem coordination lock.
    #
    # Arguments:
    #   ().
    #
    # Returns:
    #   (None): Lock is held until _unlock is called
    #
    def _lock_fs(self) -> None:
        while True:
            try:
                self._lock.mkdir(exist_ok=False)
                return
            except FileExistsError:
                time.sleep(0.001)

    # Release the coordination lock.
    #
    # Arguments:
    #   ().
    #
    # Returns:
    #   (None): Lock released
    #
    def _unlock_fs(self) -> None:
        self._lock.rmdir()

    # Load all pending events from disk.
    #
    # Arguments:
    #   ().
    #
    # Returns:
    #   (List[Dict[str, object]]): Pending operations
    #
    def _events(self) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        for path in sorted(self._pending.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_path"] = str(path)
            events.append(data)
        return events

    # Return pending events targeting an exact relative path.
    #
    # Arguments:
    #   rel_path (str): Root-relative path
    #
    # Returns:
    #   (List[Dict[str, object]]): Matching events
    #
    def _path_events(self, rel_path: str) -> List[Dict[str, object]]:
        return [event for event in self._events() if event.get("path") == rel_path]

    # Persist a new delayed event on disk.
    #
    # Arguments:
    #   event (Dict[str, object]): Serializable event description
    #
    # Returns:
    #   (None): Event file written
    #
    def _schedule(self, event: Dict[str, object]) -> None:
        name = f"{int(self._now() * 1e9):020d}_{os.getpid()}_{time.time_ns()}.json"
        (self._pending / name).write_text(json.dumps(event, separators=(",", ":")), encoding="utf-8")

    # Apply all events whose visibility deadline has passed.
    #
    # Arguments:
    #   ().
    #
    # Returns:
    #   (None): Visible root synchronized
    #
    def _sync(self) -> None:
        self._lock_fs()
        try:
            for event in self._events():
                if float(event["at"]) > self._now():
                    continue
                path = self._resolve(str(event["path"]))
                kind = str(event["kind"])
                if kind == "write":
                    stage = str(event["stage"])
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    elif os.path.exists(path):
                        os.remove(path)
                    os.replace(self._staging / stage, path)
                elif kind == "rename":
                    source = self._resolve(str(event["source"]))
                    if os.path.exists(source):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        if os.path.exists(path):
                            shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)
                        shutil.move(source, path)
                elif kind == "remove" and os.path.exists(path):
                    shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)
                Path(str(event["_path"])).unlink(missing_ok=True)
        finally:
            self._unlock_fs()

    # Return True when a path has a staged future write.
    #
    # Arguments:
    #   rel_path (str): Root-relative path
    #
    # Returns:
    #   (bool): Whether an unreadable future write exists
    #
    def _pending_write(self, rel_path: str) -> bool:
        return any(event.get("kind") == "write" for event in self._path_events(rel_path))

    # Create a directory immediately in the visible root.
    #
    # Arguments:
    #   path (str): Directory path
    #   exist_ok (bool): Allow an existing directory
    #
    # Returns:
    #   (str): Created directory path
    #
    def mkdir(self, path: str, exist_ok: bool = False) -> str:
        self._sync()
        path = self._resolve(path)
        os.makedirs(path, exist_ok=exist_ok)
        return path

    # Check whether a path exists now or is a staged exact-path write.
    #
    # Arguments:
    #   path (str): Path to check
    #
    # Returns:
    #   (bool): Path existence
    #
    def exists(self, path: str) -> bool:
        self._sync()
        path = self._resolve(path)
        if os.path.exists(path):
            return True
        return self._pending_write(self._rel(path))

    # List directory contents from the currently visible root.
    #
    # Arguments:
    #   path (str): Directory path
    #
    # Returns:
    #   (List[str]): Visible directory entries
    #
    def listdir(self, path: str) -> List[str]:
        self._sync()
        path = self._resolve(path)
        return os.listdir(path)

    # Read file contents from the visible root or a staged exact-path write.
    #
    # Arguments:
    #   path (str): File path
    #
    # Returns:
    #   (bytes): File contents
    #
    def read(self, path: str) -> bytes:
        self._sync()
        path = self._resolve(path)
        if os.path.exists(path):
            with open(path, "rb") as handle:
                return handle.read()
        rel_path = self._rel(path)
        writes = [event for event in self._path_events(rel_path) if event.get("kind") == "write"]
        if not writes:
            raise FileNotFoundError(path)
        with open(self._staging / str(writes[-1]["stage"]), "rb") as handle:
            return handle.read()

    # Stage a file write that becomes visible after write_delay.
    #
    # Arguments:
    #   path (str): Target file path
    #   data (bytes): File bytes
    #   overwrite (bool): Allow replacing an existing file
    #   mkdir (bool): Create parent directories at visibility time
    #
    # Returns:
    #   (None): Write scheduled
    #
    def write(self, path: str, data: bytes, overwrite: bool = True, mkdir: bool = True) -> None:
        self._sync()
        path = self._resolve(path)
        if not overwrite and self.exists(path):
            raise RuntimeError(f"Refusing to overwrite existing contents at '{path}'.")
        stage = f"{int(self._now() * 1e9):020d}_{os.getpid()}_{time.time_ns()}.bin"
        with open(self._staging / stage, "wb") as handle:
            handle.write(data)
        self._lock_fs()
        try:
            for event in self._path_events(self._rel(path)):
                if event.get("kind") == "write":
                    Path(str(event["_path"])).unlink(missing_ok=True)
                    (self._staging / str(event["stage"])).unlink(missing_ok=True)
            self._schedule({
                "kind": "write",
                "path": self._rel(path),
                "stage": stage,
                "mkdir": bool(mkdir),
                "at": self._now() + max(self.write_delay, self.exists_delay, self.listdir_delay),
            })
        finally:
            self._unlock_fs()

    # Reserve and delay a rename so duplicate claims fail immediately.
    #
    # Arguments:
    #   source (str): Source path
    #   destination (str): Destination path
    #
    # Returns:
    #   (bool): True on successful reservation
    #
    def rename(self, source: str, destination: str) -> bool:
        self._sync()
        source = self._resolve(source)
        destination = self._resolve(destination)
        self._lock_fs()
        try:
            rel_source = self._rel(source)
            if not os.path.exists(source):
                return False
            if any(event.get("kind") in {"rename", "remove"} and event.get("source", event.get("path")) == rel_source for event in self._events()):
                return False
            self._schedule({
                "kind": "rename",
                "source": rel_source,
                "path": self._rel(destination),
                "at": self._now() + max(self.rename_delay, self.exists_delay, self.listdir_delay),
            })
            return True
        finally:
            self._unlock_fs()

    # Delay removal so paths stay visible until remove_delay elapses.
    #
    # Arguments:
    #   path (str): Path to remove
    #   recursive (bool): Remove directories recursively
    #
    # Returns:
    #   (None): Removal scheduled
    #
    def remove(self, path: str, recursive: bool = True) -> None:
        self._sync()
        path = self._resolve(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path '{path}' does not exist.")
        if os.path.isdir(path) and (not recursive) and any(Path(path).iterdir()):
            raise RuntimeError(f"Directory '{path}' is not empty or cannot be removed without recursive=True.")
        self._lock_fs()
        try:
            rel_path = self._rel(path)
            if any(event.get("kind") in {"rename", "remove"} and event.get("source", event.get("path")) == rel_path for event in self._events()):
                return
            self._schedule({
                "kind": "remove",
                "path": rel_path,
                "at": self._now() + max(self.remove_delay, self.exists_delay, self.listdir_delay),
            })
        finally:
            self._unlock_fs()
