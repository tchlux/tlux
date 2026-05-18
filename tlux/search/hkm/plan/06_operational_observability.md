# Operational Observability

Status: next.

Goal: make long builds and searches inspectable.

Why it matters:

Long-running search builds need visible progress and clear failure states.
Without summaries, users cannot tell whether a build is healthy, stuck, or
polluting the index with bad inputs.

Scope:

- Add a build summary report.
- Track per-stage job counts and durations.
- Track indexed, skipped, failed, and oversized files.
- Add an index audit command.
- Report search latency by stage.
- Surface job stderr and resource usage through CLI/TUI.

Done when:

- A completed build has a concise human-readable summary.
- Failed ingestion paths are inspectable.
- Search and build latency can be diagnosed without manually walking job dirs.
