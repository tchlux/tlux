# HKM Product Plan

These notes track the work needed to make HKM useful as a search product over
large heterogeneous datasets.

Recommended order:

1. Ingestion policy.
2. Stable document model.
3. Search result quality.
4. Query API.
5. Incremental builds.
6. Operational observability.

The first concrete task should be adding `hkm-index` include/skip controls and a
default source-code indexing profile, then rebuilding `tlux/search/hkm` without a
custom Python command.
