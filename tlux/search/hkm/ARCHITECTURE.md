# HKM Architecture Direction

This document describes the intended destination for HKM. The current supported surface is documented in `README.md`.

## Product shape

HKM is intended to be:

- an installable Python library
- backed by a shared-filesystem job manager
- usable locally by default
- able to scale to multiple hosts when the same filesystem is shared and workers are started on each host
- consumed by multiple interfaces, with the TUI as the default client today

The UI is not the architecture center. The library and job model are.

## Stable architecture boundaries

The intended long-term boundaries are:

- `jobs.py`: queue ownership, worker lifecycle, status, resource tracking
- build pipeline: tokenize/embed, consolidate, partition, recurse
- index format: chunk-directory storage plus HKM cluster tree
- query layer: load an existing index, execute search, return stable result records
- clients: TUI now, future web/app interfaces later

Current cleanup work should preserve those boundaries even when the implementation remains simple.

## Current compromises

The current implementation is intentionally narrower than the long-term target:

- search currently supports token and semantic text queries, plus low-level token-sequence and embedding inputs
- the chunk-directory format is the canonical storage contract for now
- query-time traversal is described by `index.json` and per-node `node.json` manifests
- result records now carry `source_path`, `preview_text`, and `query_mode` so UIs do not need chunk internals
- the default embedder backend is single-choice in the docs even though the interface permits swapping
- local execution is still the easiest path, with distributed execution relying on the same shared-filesystem job model

These are acceptable as long as the code keeps one job-managed execution path and one coherent storage format.

## Distributed destination

The intended distributed model is:

- one shared index root
- one shared jobs root, usually `<index_root>/.hkm_jobs`
- workers started on each participating host
- every worker seeing the same job directories and claiming work through the filesystem
- no second orchestration system or alternate execution path

That means local and distributed operation should differ by deployment topology, not by library semantics.

## Design rules for future changes

- Do not add a second execution model that bypasses the job manager.
- Do not add a second canonical index format without migrating all layers together.
- Keep UI-specific logic out of the core build/search APIs.
- Prefer narrow extension points over multiple equally documented implementations.
- Document future intent here, not in the README, unless the behavior already exists.
