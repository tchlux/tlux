# HKM Product Plan

These notes track the work needed to make HKM useful as a search product over
large heterogeneous datasets.

Recommended order:

1. Ingestion policy. Done.
2. Stable document model. Done.
3. Search result quality. Next.
4. Query API.
5. Incremental builds.
6. Operational observability.

The next concrete task should be search result quality: hybrid ranking,
deduplication, better snippets, and match reasons on top of the stable document
records now attached to every hit.
