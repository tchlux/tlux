# CLI Auto-Launched Watchers Can Be Orphaned After Enqueue

Status: open
Severity: high

## Summary

`bin/hkm-index` enqueues jobs and returns immediately. Its automatic watcher
launch can leave the worker monitor process orphaned after the CLI process exits.
During the Fourth Wing run, the tokenizer job stayed marked `RUNNING` after both
recorded PIDs had disappeared, and a later watcher reconciliation failed the
whole build as stale.

## Reproduction

Run a build through `bin/hkm-index` without a separate foreground drain:

```bash
env HF_HUB_OFFLINE=1 HKM_EMBEDDER=drama HKM_MAX_WORKERS=4 \
  bin/hkm-index data/fourth_wing_hkm_index data/fourth_wing_markdown \
  --workers 4 --full-rebuild --include '*.md' \
  --max-file-bytes 20000000 --max-tokens 1000000
```

Observed job config for the tokenizer job:

```json
{
  "status": "FAILED",
  "status_reason": "Watcher disappeared while the job was RUNNING.",
  "exit_code": -9
}
```

The job stderr was empty. The process PIDs recorded in `job_config` were gone.

## Expected

The CLI should either:

- keep workers supervised until the root job reaches a terminal state, or
- document and expose an explicit command to drain/watch the job root, or
- avoid launching fragile background watchers from a short-lived enqueue command.

## Actual

The enqueue command returned after printing the root job id. The watcher process
was not reliably kept alive for long embedding work. Reconciliation then failed
the tokenizer and propagated failure to downstream jobs.

## Workaround Used

Disable automatic watcher launch and run a foreground drain:

```bash
env HF_HUB_OFFLINE=1 HKM_EMBEDDER=drama HKM_MAX_WORKERS=4 HKM_DISABLE_WATCHER_LAUNCH=1 \
  bin/hkm-index /abs/index /abs/docs --workers 4 --full-rebuild ...

env HF_HUB_OFFLINE=1 HKM_EMBEDDER=drama HKM_MAX_WORKERS=4 HKM_DISABLE_WATCHER_LAUNCH=1 \
  bin/hkm-python -c "from tlux.search.hkm import drain_jobs; from tlux.search.hkm.fs import FileSystem; drain_jobs(FileSystem(root='/abs/index/.hkm_jobs'), max_workers=4)"
```

## Likely Fix Area

Clarify the `hkm-index` contract. If the CLI is intended to build, not only
enqueue, it should drain or supervise until terminal. If it is intended to
enqueue only, provide a companion `hkm-drain` or documented `--wait` option.
Add a test that a CLI-launched long job does not fail solely because the enqueue
process exited.
