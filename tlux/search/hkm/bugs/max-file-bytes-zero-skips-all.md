# `--max-file-bytes 0` Skips Every Non-Empty File

Status: open
Severity: medium

## Summary

The plan used `--max-file-bytes 0` as an "unlimited" value. Current behavior
treats zero as a literal maximum of zero bytes, so every non-empty document is
skipped and the build raises `ValueError("No documents found to index.")`.

## Reproduction

```bash
env HF_HUB_OFFLINE=1 HKM_EMBEDDER=drama HKM_MAX_WORKERS=4 \
  bin/hkm-index data/fourth_wing_hkm_index data/fourth_wing_markdown \
  --workers 4 --full-rebuild --include '*.md' --max-file-bytes 0
```

Observed failure:

```text
ValueError: No documents found to index.
```

## Expected

Either:

- `0` should be documented and implemented as unlimited, or
- the CLI should reject `--max-file-bytes 0` with a clear validation error that
  tells the user to pass a positive byte ceiling.

## Actual

The ingestion planner applies `path.stat().st_size > max_file_bytes`, so any
non-empty file is skipped when `max_file_bytes` is zero.

## Impact

This is easy to hit because many CLIs use zero to mean "disabled" or
"unlimited". The resulting error says no documents exist, which points the user
at the corpus rather than the flag semantics.

## Workaround Used

Use a large explicit ceiling:

```bash
--max-file-bytes 20000000
```

## Likely Fix Area

Validate `max_file_bytes` in `builder/launcher.py`. Prefer the smallest fix:
reject zero with a specific error unless the project wants zero to mean
unlimited. Update CLI help and tests accordingly.
