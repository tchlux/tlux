# Relative Index Paths Can Publish Worker Artifacts Under The Wrong Root

Status: open
Severity: high

## Summary

Running `bin/hkm-index` with relative corpus and index paths caused worker
output to be rooted under `data/data/fourth_wing_hkm_index/...` while the
canonical index metadata and job root were under `data/fourth_wing_hkm_index`.

## Reproduction

From `tlux/search/hkm`, run:

```bash
env HF_HUB_OFFLINE=1 HKM_EMBEDDER=drama HKM_MAX_WORKERS=4 \
  bin/hkm-index data/fourth_wing_hkm_index data/fourth_wing_markdown \
  --workers 4 --full-rebuild --include '*.md' --max-file-bytes 20000000
```

The tokenizer job logged:

```text
output_dir=/Users/thomaslux/Git/tlux/tlux/search/hkm/data/data/fourth_wing_hkm_index/docs/worker_0000
```

The expected output root was:

```text
/Users/thomaslux/Git/tlux/tlux/search/hkm/data/fourth_wing_hkm_index/docs/worker_0000
```

## Expected

Relative CLI paths should resolve once to absolute paths before job arguments
are serialized. Worker output, manifests, index metadata, and job paths should
all agree on one index root.

## Actual

The worker received a relative `output_directory` plus a filesystem root derived
from relative input paths. `FileSystem.join()` then prepended the root again,
creating a duplicated `data/data/...` path.

## Impact

This can make a build appear to have a valid job root and `index.json` while
workers publish artifacts somewhere search and later build stages do not expect.
It also makes cleanup confusing because failed artifacts are split across
multiple generated trees.

## Likely Fix Area

Normalize `docs_dir`, `index_root`, `jobs_root`, `docs_root_out`, `hkm_root`,
worker `output_directory`, and manifest paths to absolute paths at the CLI/API
boundary in `builder/launcher.py`. Add a regression test that invokes
`bin/hkm-index` with relative paths and asserts artifacts are only written under
the intended index root.
