# HKM Distributed Hardening Tasks

This file is the persistent task queue for proving HKM is hardened for shared-filesystem and distributed-style execution.

Rules for future conversations:
- Always work the first task in `Open Tasks`.
- When you finish a task, delete it from `Open Tasks`.
- If the work reveals follow-up tasks, append them to the bottom of `Open Tasks`.
- Keep tasks concrete and testable.
- Keep `Recent Notes` short and factual.

## Open Tasks

1. Fix the HKM test runtime to use `python3.12` or the latest viable interpreter on this machine.
   Current state: `python3.12`, `python3.11`, and `python3.10` fail to import NumPy, and the traceback shows those interpreters loading a broken package tree from `~/Library/Python/3.9/...`.
   Done when:
   - `python3.12 -c "import numpy"` succeeds, or there is a documented reason to standardize on a different newer interpreter on this machine.
   - HKM tests can run without falling back to `python3.9`.

2. Run the full existing HKM test suite unchanged on the normal filesystem under the fixed interpreter.
   Done when:
   - The exact command and results are recorded in `Recent Notes`.
   - Any failures are converted into concrete follow-up tasks.

3. Add one repeatable command for the delayed-filesystem scheduler/build slice.
   Done when:
   - There is a single documented command that runs the delayed-FS HKM/job hardening tests.
   - The command uses the fixed interpreter from task 1.

4. Add a repeated delayed-FS scheduler stress harness.
   Scope:
   - Multiple watchers.
   - More watchers than jobs.
   - Repeated watcher restarts.
   - Delayed `write`, `rename`, `remove`, stale `listdir`.
   Done when:
   - The harness can run the high-value scheduler cases many times in one command.
   - It asserts no duplicate execution and no stuck nonterminal jobs in the successful path.

5. Add a repeated delayed-FS HKM build stress harness.
   Scope:
   - Real HKM build.
   - Watcher churn.
   - Mid-build watcher kill and restart.
   - Post-build token and semantic queries.
   Done when:
   - The harness can run many iterations.
   - Each successful iteration proves the built index is queryable.
   - Each failed iteration ends in an explicit inspectable terminal state.

6. Add a built-index publication audit.
   Scope:
   - Walk every `node.json`.
   - Assert every referenced artifact exists before the node is considered valid.
   - Assert non-leaf nodes have `centroids.npy`.
   - Assert leaf nodes have readable chunk roots.
   Done when:
   - The audit runs automatically in the delayed-FS HKM stress path.
   - Failures point at the exact broken node/artifact pair.

7. Add deterministic crash-point coverage for the scheduler.
   Crash points:
   - After `queued -> running` claim.
   - Before worker pid metadata is saved.
   - During downstream release.
   - During node finalization.
   Done when:
   - Each crash point has a dedicated test or harness scenario.
   - The system converges to either clean success or clean explicit failure.

8. Add a duplicate-execution audit path.
   Scope:
   - Record per-job execution count in stress scenarios.
   - Fail if any job body runs more than once.
   Done when:
   - Scheduler stress runs prove no duplicate execution under delayed visibility.

9. Run the hardened stress harness against a real shared filesystem.
   Preferred targets:
   - NFS.
   - SMB.
   - Another real shared mount available on this machine or LAN.
   Done when:
   - The exact environment and commands are recorded in `Recent Notes`.
   - Results are converted into concrete follow-up tasks if anything fails.

10. Write a short hardening report.
    Scope:
    - What was simulated locally.
    - What was validated on a real shared filesystem.
    - Remaining unproven failure modes.
    Done when:
    - There is one concise markdown report under `tlux/search/hkm/`.

## Recent Notes

- 2026-04-18: Added delayed-FS scheduler and HKM build tests plus minimal scheduler/publication hardening.
- 2026-04-18: Verified the touched HKM slice only, not the full repo or full HKM suite.
- 2026-04-18: `python3.12`, `python3.11`, and `python3.10` currently fail on `import numpy`; `python3.9` was used only because it was the first interpreter on this machine that could run the HKM tests.

## Next Prompt

Use this in the next conversation:

`Solve the next task in /Users/thomaslux/Git/tlux/tlux/search/hkm/DISTRIBUTED_HARDENING_TASKS.md. Follow the file's rules exactly: complete the first open task, update Recent Notes, delete the finished task, and append any new concrete follow-up tasks that the work reveals.`
