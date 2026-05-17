# HKM Product Plan

These notes track the work needed to make HKM useful as a search product over
large heterogeneous datasets.

Recommended order:

1. Ingestion policy. Done.
2. Stable document model. Done.
3. Search result quality. Done.
4. Query API. Next.
5. Incremental builds.
6. Operational observability.

The next concrete task should be Query API: stabilize the JSON request and
response schema now that hybrid ranking, deduplication, snippets, and match
reasons are available on top of stable document records.
