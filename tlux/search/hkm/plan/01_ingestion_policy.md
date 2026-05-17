# Ingestion Policy

Goal: make indexing safe and predictable over messy directories.

Why it matters:

The first real HKM source build wasted time embedding generated tokenizer JSON
files. Large heterogeneous datasets will contain caches, model artifacts,
binary blobs, generated indexes, virtual environments, and broken text files.
HKM needs explicit ingestion rules before it can be trusted on large trees.

Scope:

- Add repeatable `--skip` path support to `hkm-index`.
- Add include and exclude glob support.
- Add default skips for `.git`, `.env`, `__pycache__`, generated indexes,
  virtual environments, caches, model weights, tokenizer files, and common
  binary artifacts.
- Add max file size and max token limits.
- Report skipped files by reason.
- Report per-file ingest failures instead of silently hiding them.
- Keep the library API accepting explicit `skip_paths`.

Done when:

- A source tree can be indexed from `hkm-index` without custom Python.
- Generated environments and model artifacts are excluded by default.
- The build summary shows indexed, skipped, and failed file counts.
- Existing tests still pass.
