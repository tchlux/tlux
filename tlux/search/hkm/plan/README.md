# HKM Product Plan

These notes track the work needed to make HKM useful as a search product over
large heterogeneous datasets.

Recommended order:

1. Ingestion policy. Done.
2. Stable document model. Done.
3. Search result quality. Done.
4. Query API. Done.
5. Incremental builds. Next.
6. Operational observability.

The next concrete task should be Incremental Builds: reuse unchanged source
documents now that ingestion, document identity, ranking, and the public query
schema are stable.
