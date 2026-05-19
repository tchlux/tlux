# Long Documents Can Be Silently Skipped By The Token Limit

Status: open
Severity: high

## Summary

The Fourth Wing Markdown document tokenized to 249,068 tokens. The default
`--max-tokens 200000` caused the worker to skip the only planned document. The
build pipeline still completed successfully, producing an index with zero active
documents.

## Reproduction

Build the one-file corpus with defaults except file size:

```bash
env HF_HUB_OFFLINE=1 HKM_EMBEDDER=drama HKM_MAX_WORKERS=4 HKM_DISABLE_WATCHER_LAUNCH=1 \
  bin/hkm-index /abs/index /abs/fourth_wing_markdown \
  --workers 4 --full-rebuild --include '*.md' --max-file-bytes 20000000
```

Then drain jobs. Observed `manifests/ingest_summary.json`:

```json
{
  "planned": 1,
  "indexed": 0,
  "skipped": 1,
  "failed": 0,
  "skip_reasons": {
    "max_tokens": 1
  }
}
```

`source_snapshot.json` had zero documents, and the root job still succeeded.

## Expected

When all planned files are skipped by worker-stage token limits, the build
should end in an explicit failure or warning state that is visible from the
root job and CLI output. A zero-document index should not look like a successful
build unless the user explicitly requested an allow-empty mode.

## Actual

The build succeeded with no active documents. The only signal was inside
`manifests/ingest_summary.json`.

## Impact

Large single documents can appear to have built successfully while search has
nothing to query. This is especially misleading for long local books, PDFs, or
exported manuals.

## Workaround Used

Rerun with a higher token limit:

```bash
--max-tokens 1000000
```

## Likely Fix Area

After worker reports are merged, fail the build if `planned > 0`,
`indexed == 0`, and all documents were skipped or failed. At minimum, surface
this condition in the root job status reason and CLI output. Add a test with a
single file over `max_tokens`.
