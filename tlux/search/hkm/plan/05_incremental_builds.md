# Incremental Builds

Goal: avoid full rebuilds when only part of a dataset changed.

Why it matters:

Large datasets make full rebuilds expensive. A useful search product must reuse
unchanged token and embedding work, detect deletions, and update the index with
bounded effort.

Scope:

- Store source snapshot metadata in the index manifest.
- Hash file contents during ingestion.
- Reuse unchanged document shards.
- Detect new, changed, and deleted files.
- Rebuild affected HKM branches.
- Keep full rebuild as the simple fallback path.

Done when:

- Re-running an index over mostly unchanged data skips unchanged files.
- Changed and deleted files are reflected in search results.
- Incremental behavior is tested against a small synthetic corpus.
