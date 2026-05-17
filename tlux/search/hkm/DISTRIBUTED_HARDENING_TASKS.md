# HKM Distributed Hardening Tasks

This file is the persistent task queue for proving HKM is hardened for shared-filesystem and distributed-style execution.

Rules for future conversations:
- Always work the first task in `Open Tasks`.
- When you finish a task, delete it from `Open Tasks`.
- If the work reveals follow-up tasks, append them to the bottom of `Open Tasks`.
- Keep tasks concrete and testable.
- Keep `Recent Notes` short and factual.

## Open Tasks

1. Add one repeatable command for the delayed-filesystem scheduler/build slice.
   Done when:
   - There is a single documented command that runs the delayed-FS HKM/job hardening tests.
   - The command uses the fixed HKM interpreter.

2. Add a repeated delayed-FS scheduler stress harness.
   Scope:
   - Multiple watchers.
   - More watchers than jobs.
   - Repeated watcher restarts.
   - Delayed `write`, `rename`, `remove`, stale `listdir`.
   Done when:
   - The harness can run the high-value scheduler cases many times in one command.
   - It asserts no duplicate execution and no stuck nonterminal jobs in the successful path.

3. Add a repeated delayed-FS HKM build stress harness.
   Scope:
   - Real HKM build.
   - Watcher churn.
   - Mid-build watcher kill and restart.
   - Post-build token and semantic queries.
   Done when:
   - The harness can run many iterations.
   - Each successful iteration proves the built index is queryable.
   - Each failed iteration ends in an explicit inspectable terminal state.

4. Add a built-index publication audit.
   Scope:
   - Walk every `node.json`.
   - Assert every referenced artifact exists before the node is considered valid.
   - Assert non-leaf nodes have `centroids.npy`.
   - Assert leaf nodes have readable chunk roots.
   Done when:
   - The audit runs automatically in the delayed-FS HKM stress path.
   - Failures point at the exact broken node/artifact pair.

5. Add deterministic crash-point coverage for the scheduler.
   Crash points:
   - After `queued -> running` claim.
   - Before worker pid metadata is saved.
   - During downstream release.
   - During node finalization.
   Done when:
   - Each crash point has a dedicated test or harness scenario.
   - The system converges to either clean success or clean explicit failure.

6. Add a duplicate-execution audit path.
   Scope:
   - Record per-job execution count in stress scenarios.
   - Fail if any job body runs more than once.
   Done when:
   - Scheduler stress runs prove no duplicate execution under delayed visibility.

7. Run the hardened stress harness against a real shared filesystem.
   Preferred targets:
   - NFS.
   - SMB.
   - Another real shared mount available on this machine or LAN.
   Done when:
   - The exact environment and commands are recorded in `Recent Notes`.
   - Results are converted into concrete follow-up tasks if anything fails.

8. Write a short hardening report.
    Scope:
    - What was simulated locally.
    - What was validated on a real shared filesystem.
    - Remaining unproven failure modes.
    Done when:
    - There is one concise markdown report under `tlux/search/hkm/`.

9. Add source-index skip controls to `bin/hkm-index`.
   Current state: real source builds that need `skip_paths` must call the Python API directly.
   Done when:
   - The CLI accepts repeatable skip paths.
   - A source build can exclude generated envs, model artifacts, old indexes, and caches without a custom Python command.

10. Make cached `drama` CLI search work without network.
    Current state: `hkm-search` may contact Hugging Face while loading a cached `drama` model unless `HF_HUB_OFFLINE=1` is set.
    Done when:
    - Token and semantic CLI searches over a built `drama` index succeed offline after the model is cached.
    - The required environment behavior is documented or encoded.

## Recent Notes

- 2026-04-18: Standardized HKM test runs on `tlux/search/hkm/bin/hkm-python`, because the shell-level `PYTHONPATH` injects `~/Library/Python/3.9/...` ahead of newer interpreters; the wrapper unsets `PYTHONPATH`, prefers the local `.env` Python 3.12, and falls back to the newest interpreter on this machine that can `import numpy`.
- 2026-04-18: Verified `tlux/search/hkm/bin/hkm-python -c "import sys, numpy; print(sys.version.split()[0], numpy.__version__)"` succeeds on Python 3.12 and `tlux/search/hkm/bin/hkm-python -m pytest tlux/search/hkm/tests/test_job_imports.py` passes without using `python3.9`.
- 2026-04-18: Added delayed-FS scheduler and HKM build tests plus minimal scheduler/publication hardening.
- 2026-04-18: Verified the touched HKM slice only, not the full repo or full HKM suite.
- 2026-04-18: `python3.12`, `python3.11`, and `python3.10` currently fail on `import numpy`; `python3.9` was used only because it was the first interpreter on this machine that could run the HKM tests.
- 2026-05-17: `bin/hkm-python -m pytest tests/` from `tlux/search/hkm` passes: 22 passed, 1 pytest cache warning, 51.84s.
- 2026-05-17: Fixed delayed-FS false failure by persisting `monitor_pid` before publishing RUNNING, trusting `job_config` in `job_status`, and increasing orphan grace to 1.0s.
- 2026-05-17: `env PYTHONPATH=/bad/path bin/hkm-index --help`, `bin/hkm-search --help`, and `bin/hkm-tui --help` all succeed through `bin/hkm-python`.
- 2026-05-17: Built a `drama` index over `tlux/search/hkm` source at `tmp_hkm_source_index`: root job 383491160 SUCCEEDED, 76 docs, 6795 embeddings, token and semantic searches returned results with `HF_HUB_OFFLINE=1`.

## Next Prompt

Use this in the next conversation:

`Solve the next task in /Users/thomaslux/Git/tlux/tlux/search/hkm/DISTRIBUTED_HARDENING_TASKS.md. Follow the file's rules exactly: complete the first open task, update Recent Notes, delete the finished task, and append any new concrete follow-up tasks that the work reveals.`
