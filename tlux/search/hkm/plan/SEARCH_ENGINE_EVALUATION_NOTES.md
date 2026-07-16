# Search engine evaluation notes

The expanded FineWeb stress set is useful for diagnosis but is not yet a clean
recall gate. Several rows use aliases that are not present in the source text,
including `electric curler` for `electric hair curler`, `stop` for `cease`,
`startup` for `restart`, `WWE champion` for `WWE champ`, `fixed channel` for
`one channel`, and `darker` for `darkens`. The Montessori name is also fused
into a URL path. These rows should be repaired in the benchmark oracle before
engine regressions are inferred from them.

The real engine failures are source-window failures: one source can contain
the required clues across adjacent indexed windows, while the first page is
filled by duplicate or semantically similar windows. I tested lexical rescue
lanes, local clue masks, overlap penalties, and source-level aggregation. Those
changes improved the broad stress score but regressed the canonical deep gate,
so they were reverted. The current implementation remains the last known
deep-gate baseline; future fixes must preserve that gate before optimizing the
noisy stress set.

The current deterministic expanded-corpus runs are 24/24 on the deep gate
(group recall 1.0, coherent hits 24/24, judged precision 1.0: 71 relevant
and zero negative hits) and 18/25 on the broader stress set (mean group
coverage 0.9207, coherent hits 25/25). A bounded source-filtered token rescue
now recovers the missing project and bases-loaded windows (LQ-52 and LQ-73),
while a named-entity contrast lane recovers the Montessori context window
(LQ-72). A final replay on the current tree remains 18/25 with all 25 cases
coherent and the deep gate still perfect.

The retained implementation is intentionally narrow: it adds at most six
adjacent token lanes for a low-confidence, non-negated missing-entity query,
and one named-anchor/tail token lane for a low-confidence contrast query. Every
rescue hit is restricted to source paths already present in the first semantic
page. This raised the stress diagnostic from 13/25 to 16/25 in two replays
before the payload fix, without changing the quality-benchmark means or the
deep precision gate.
Missing-entity reranking now expands all seven condition concept groups rather
than only the initial funny/room groups. This rewards structural, crowd, and
watch/horror evidence when the remembered subject or object is omitted. The
change passes the six-case Fourth Wing gate at 6/6, keeps the expanded deep gate
at 24/24 with 71/71 judged relevant, and leaves the stress diagnostic at 18/25.
The persistent JSONL agent now includes the indexed `document_preview` in each
hit's compact `document` field, restoring evidence that was present in the
index but hidden from downstream agents. This fixes LQ-65 and LQ-66 without
changing ranking, raising stress from 16/25 to 18/25. The full regression suite
remains green at 156 passed and 2 skipped. Broader compact-term/bundle rescues
and a broader missing-entity detector were tested and reverted: they preserved
the deep gate but displaced evidence windows and lowered stress coverage.

The payload now also exposes a bounded `document.source_context` read from the
indexed source span. It returns the full span up to 2,048 bytes, or a bounded
head/tail pair for larger spans, and rejects paths outside the configured source
root. This leaves ranking unchanged while letting a downstream LM inspect clues
that fall after the fixed 512-byte document preview; the LQ-63 source context
contains vitamins A/C/E, melanin, and exfoliation evidence. Deep remains 24/24
and stress remains 18/25 after this API-only change.
The LM planner tool payload now includes the same indexed 512-byte document
preview, so a local model can inspect metadata evidence during refinement rather
than receiving only the shorter hit snippet. This is bounded to eight planner
hits and does not change search ranking.

The remaining stress misses are oracle defects or mixed strict-alias cases:
LQ-53, LQ-54, LQ-56, LQ-62, LQ-70, and LQ-74 need alias/morphology repairs;
LQ-63 needs both alias repairs and a better visible window. LQ-65 and LQ-66
are now covered through the returned document preview. The `file returns`
alias in LQ-56 does not match the source's `filing returns`. These should be
repaired by the benchmark owner before further engine ranking changes.
An offline normalized-oracle replay adding source-equivalent forms (`other
people's`, `electric hair curler`, `filing returns`, `cease`, `vitamins A, C
and E`, `one channel`, `time zones`, and `darkens`) reaches 25/25 with the
current payload and zero negative hits. The direct LQ-63 rescue query
`twice a week exfoliation vitamins melanin` retrieves the complete target span,
but shipping that lane before oracle repair makes the strict score appear worse
by removing an unrelated `twice weekly` hit; it remains a future evidence-window
test rather than a production ranking change.
The missing-entity detector also does not yet recognize every natural phrasing
(for example, `forgetting the author` and `names are missing`); a direct attempt
to enable those phrases left the stress score flat but collapsed one deep query
from two distinct judged evidence entries to one source-local cluster, so it was
reverted pending a safer clause-aware rescue lane.
Replaying the stress set with `--always-refine` and with `--tool-mode hybrid`
does not improve the pre-rescue baseline, confirming that the original gap was
passage selection rather than missing query-generation rounds.

The committed quality benchmark is a separate synthetic protocol. Its current
seeded baseline is lexical recall@5 0.6944 / precision@5 0.1778 and agentic
recall@5 0.6667 / precision@5 0.1667. Most returned hits are unjudged, so these
numbers are diagnostic rather than a production-quality precision gate.
The required unseeded `--k 5` run measured lexical, sparse, and late-interaction
recall@5 0.7222, hybrid and agentic recall@5 0.6667, with only 16--18 judged
items out of 90 per method.
The supplied LM Studio endpoint was not reachable from this shell during this
pass, so the language gates below use the deterministic local agent path.

The required standard-benchmark harness was exercised against the downloaded
SciFact archive with a 100-document, 10-query smoke run. It completed in
46.4 s of build time, but all 100 retrieved items were unjudged because the
truncated corpus did not include the relevant documents. The exact unlimited
SciFact command built all 5,183 documents but expanded to about 2.1 GB and
3,476 filesystem jobs before a watcher visibility timeout prevented publishing
the index; it was stopped without recording a score. No MIRACL or TREC data
root is present locally; both commands fail before indexing with a missing
benchmark-file error.

The six-case Fourth Wing language gate currently passes 6/6 coverage and
coherence cases, but its judged precision is only 0.75 (5 relevant and 4
negative hits). LQ-02 is the clearest precision miss: the default agent stops
after a partial lexical-confidence pass and returns four broad wall-related
negative snippets, while a direct semantic query for `large wall structure`
finds the intended structure-over-crowd passage. Forcing extra concept lanes
lets keyword/token decoys dominate and did not improve precision, so no
unverified reranking change was retained. A no-answer query for `lunar battery
gardening` likewise returns five unrelated documents; agent-level abstention
needs a guarded policy that does not sacrifice vague-query recall.

The compact quality evaluator has a separate accounting defect: with
`policy='ignore'`, unjudged ranks are removed before AP, reciprocal rank, and
nDCG enumerate positions. For example, `['unknown', 'a']` against `{'a': 1}`
reports AP and reciprocal rank 1.0 despite the relevant item being rank two.
This is benchmark-owned and should not be used to justify a production search
change.

The first direct standard-corpus audit exposed a real hybrid-fusion gap that
the compact synthetic quality suite cannot see. On a qrels-only SciFact index
containing all 283 documents judged by the 300 test queries, semantic search
recalled 0.9833 at 10 (MAP 0.9020), while the original hybrid weights recalled
0.8302 (MAP 0.7062). Lexical overlaps and metadata bonuses displaced semantic
targets. The retained production candidate changes the fusion weights from
0.45 semantic / 0.55 token to 0.90 semantic / 0.10 token. For non-exact lexical
support on a semantic hit, the score is semantic plus a small token tie-breaker;
exact phrases and metadata-only matches retain their stronger evidence bonuses.
The exact replay improves hybrid recall to 0.9860, MAP to 0.8984, and MRR to
0.9011, with 334 judged hits out of 3,000 retrieved. A conditional
metadata-scale experiment did not improve the replay and broke an exact
metadata-ranking regression, so it was reverted. End-to-end search tests remain
green.

A trial relation bonus for phrases such as `towers over` was reverted. It did
not improve LQ-02's judged precision and reduced the six-case coverage gate to
5/6. The intended giant-wall passage was labeled negative because the expanded
source context also contained the sidecar's unrelated `climb`/`cliff`/`wall`
negative clause, so this is not a safe production ranking signal.

The retained language-agent precision guard now abstains only when a multi-term
request returns a token-only page whose best indexed text covers less than one
third of the non-filler query terms. This fixes the observed `lunar battery
gardening` false-positive page (grounded=false, zero docs) while preserving the
Fourth Wing 6/6 gate, the FineWeb expanded deep 24/24 gate with 71/71 judged
relevant hits, and the unchanged 18/25 stress replay. Semantic evidence is
never rejected by this guard; it is intentionally a narrow out-of-corpus
defense rather than a general confidence threshold.

The five remaining direct SciFact top-10 misses are relation-level gaps rather than simple
fusion errors. The judged targets are q437 (MDS/genomic alterations -> glioma iPSC model),
q560 (Th17/iTregs -> JAM-A intestinal compensation), q1199 (colchicine/statins -> cardiovascular
event risk), q1213 (monocyte activation -> KLF2 atherosclerosis), and q975 (cytokine mediators,
which is already recovered at hybrid rank 5). Exhaustive candidate search places the other
targets at hybrid ranks 121, 33, 16, and 20 respectively; larger probe counts do not change
those ranks. Bounded pseudo-relevance feedback that appends the titles of the first two semantic
hits to the original claim recovers q437, q560, q975, and q1199 into the top 10, but not q1213.
This is a promising LM-agent refinement path, not a safe default reranker: the first hits are
not guaranteed relevant and naive lane fusion displaced targets. A full 300-query replay confirms
the risk: title feedback falls from hybrid recall@10 0.9867 (296/300 queries) to 0.9667 (290/300)
and micro precision from 0.1113 to 0.1097. It is therefore rejected for default fusion; the
retained change only tells an available LM how to use feedback during its own bounded refinement.

The cached Contriever ONNX backend was also rebuilt as a controlled comparison. Its inference
wrapper previously allowed the caller's 1,024-token window length to exceed Contriever's 512-token
ONNX input; clamping the body length to the model limit fixes that latent crash. A qrels-only
SciFact index using Contriever is runnable, but its direct replay is materially worse than DRAMA:
semantic recall@10 0.9193 and hybrid recall@10 0.8217 (versus 0.9833 and 0.9860). The clamp is
retained as a correctness fix, while Contriever is not selected as the default backend.

The full 5,183-document SciFact standard generation was repaired after its queue drained. A
watcher disappearance had left eight depth-four child manifests unpublished even though their
parent assignments and data were present. Re-running only those recursive jobs in an isolated
queue made audit_index pass; no documents were rebuilt. On the complete corpus, semantic and
hybrid target ranks are much worse than the qrels-only replay (for example, q437 ranks 2,410
semantic and 1,908 hybrid), confirming that the qrels-only score is an optimistic diagnostic.

The language agent now has one guarded compact-concept path for short, simple, one-concept
requests. It searches the compact concept lane first, forces one bounded second round with the
original wording, and disables the generic coverage bonus for that request so extra token decoys
cannot displace the semantic lane. The Fourth Wing base gate remains 6/6 with the reviewed wall
target at rank 1 (the displayed 0.75 judged precision is a known source-context sidecar artifact),
conditionals remain 7/7 with 1.0 judged precision, and variants remain 4/4 with 0.7778 judged
precision. The focused agent/language tests pass (66 tests). FineWeb and stress results are
unchanged in the offline replay, so this rule is intentionally narrow.

The delayed-filesystem regression also exposed a scheduler race: a stale ``running`` directory
could be reaped as failed before a completed worker's delayed terminal-status write was visible.
The scheduler now gives terminal status publication a short grace period, ignores already-terminal
job metadata during reap, and the recursive drain waits through one stable idle interval before
auditing. The delayed job/build tests pass 7/7 on repeated runs; this prevents false dependency
cascades without weakening deliberate watcher-kill recovery.

After the changes, the required compact quality benchmark is unchanged: lexical, sparse, and
late-interaction recall@5 are 0.7222 with judged precision 0.1889; BM25 and reranker are 0.6944 /
0.1778; dense, hybrid, and agentic are 0.6667 / 0.1667. It judges only 16--18 of 90 returned
items per method, so it remains a diagnostic fixture rather than a broad precision gate. The
complete repository suite is green at 157 passed and 2 skipped.

A full-corpus title/preview rerank was tested without production edits. Adding title overlap to
the semantic score improved a 25-query sample slightly (recall .63 to .64) and preserved a
50-query sample's .84 recall, but did not move any of the five known hard targets into the top 10;
preview overlap moved q437 from rank 2,410 to 1,797 but still missed the cutoff and regressed at a
larger weight. It is therefore a ranking polish experiment, not a recall solution, and remains
rejected as a default change. Deterministic language-agent query variants likewise failed to
recover the five hard targets on the complete corpus; reliable recovery still requires a capable
LM query-refinement pass or a stronger embedding model.

The default LM Studio request timeout is now 30 seconds rather than 1 second, leaving room for a
small local model to complete bounded query planning. The previously supplied endpoint was
unreachable at the time of that test; the current localhost endpoint is now verified separately
and is the preferred live path.

An optional `gemma` embedder now targets an OpenAI-compatible local endpoint (`HKM_GEMMA_ENDPOINT`).
The cached 300M EmbeddingGemma GGUF could not load through the older local `llama_cpp` Python
wheel, but the LM Studio llama-server 2.24 binary loaded it successfully. On the complete 5,183-
document SciFact corpus, exhaustive canonical EmbeddingGemma cosine retrieval scored 0.9219
recall@10, 0.7468 MAP, 0.7564 MRR, and 0.104 precision versus exhaustive DRAMA window retrieval
at 0.7558 / 0.5404 / 0.5500 / 0.0853. A 283-document Gemma HKM build also completed successfully; semantic recall was
0.9833 and hybrid recall 0.9867 with MAP 0.9643. These are optional-backend experiments, not a
default switch: they require a compatible embedding server and a rebuilt index, and full-corpus
traversal still needs a dedicated production benchmark.
On that same 283-document index, bounded hybrid probing measured recall@10 of 0.7166 at
`probe_count=1`, 0.8793 at 2, and 0.9733 at 4, versus 0.9867 with exhaustive probing. This
quantifies the cost/recall tradeoff instead of assuming that a stronger embedder alone makes
search cheap.

Benchmark audit: a full SciFact run with `HKM_FAKE_EMBEDDER=1` is plumbing-only, not a prose
retrieval result. The fake tokenizer only parses numeric whitespace tokens, so prose queries and
documents collapse to empty/constant vectors; its observed recall is therefore an expected
artifact. The recognized-benchmark tests likewise use two fake documents and k=1, so they verify
loader/build plumbing but not corpus recall or precision. Keep fake runs labeled as smoke tests
and use the real DRAMA or Gemma endpoints for quality claims.

The full SciFact generation should be opened through its generation directory
(`/tmp/hkm-beir/.hkm_builds/20260715T092620Z-2587c289`); the parent scratch root has no canonical
`index.json`. Current full-corpus DRAMA replay confirms qids 437, 560, 975, 1199, and 1213 have
their positive targets absent from the top ten, while q873 recovers only part of its five positive
targets. The historical standard report is useful for comparison, but it treats more than 90% of
retrieved rows as nonrelevant even though BEIR qrels are sparse: judged-only precision is 1.0 for
the returned judged rows, while unknown precision cannot be inferred. Standard runs also do not
expose `probe_count`, so they measure exhaustive traversal latency rather than the cheap-search
recall/latency tradeoff.

An additional 50-query qrels-positive smoke run reported hybrid recall 1.0 and MAP 0.9411, but it
selected the first query-file rows and omitted known hard IDs 437, 560, 873, 975, 1199, and 1213;
it is not evidence of generalization. The recognized-benchmark loader now rejects malformed qrels,
invalid grades, and query IDs absent from the query file, and the runner rejects negative corpus or
query limits. Unknown document IDs still require corpus-level validation before precision can be
interpreted.

The repaired full-corpus DRAMA replay is now exact on all 300 SciFact queries: semantic recall@10
is 0.8345 with 47 incomplete queries, while current hybrid is 0.8273 with 48 incomplete queries;
hybrid MAP/MRR/precision are 0.6844/0.6980/0.0933. The hybrid rerank therefore loses a small
amount of recall relative to semantic-only traversal on this corpus, an actionable regression to
revisit after the Gemma tree comparison.

The full-corpus audit identified 57 active documents in an internal ``data`` directory that had
non-leaf manifests with empty child nodes; semantic traversal could never reach them. The builder
now publishes each routed child as a searchable leaf until recursive refinement replaces it, and
the searcher performs a bounded embedding scan for unrouted active documents in older indexes.
On the repaired DRAMA generation this recovered q49 at rank 1 and q238 at rank 2 (12-miss replay;
warm mean 4.34 seconds/query). Audit now rejects populated non-leaf nodes whose children are all
empty. Hybrid metadata bonuses are also disabled for semantic-only hits: an offline hard-query
check recovered q1019 at rank 2 and q649 at rank 5 while preserving q1, q1175, q660, and q691.

An early 5,183-document Gemma HKM benchmark process stalled with queued recursive jobs and was
stopped before query evaluation. The resulting generation was later verified directly: all 5,183
documents were indexed, no documents were unrouted, and the audit passed. Direct full-corpus
evaluation of that generation measured semantic recall@10 0.8876 (MAP 0.7168, MRR 0.7270,
precision 0.1007) and hybrid recall@10 0.8986 (MAP 0.7276, MRR 0.7380, precision 0.1013), with
28 hybrid misses. This is now the authoritative Gemma HKM baseline; the exhaustive canonical
EmbeddingGemma baseline remains stronger at 0.9219 recall@10 and identifies ranking headroom.

After the routing repair, audit invariant, qrels validation, Gemma coverage, and tiered hybrid
guard changes, the full repository suite is green at 165 passed and 2 skipped. The compact quality
benchmark remains unchanged on recall/precision (hybrid 0.6667/0.1667 at k=5); its agentic AP is
0.6003. Full SciFact DRAMA replay remains the broad baseline, with the targeted repairs measured
on hard misses rather than conflated with a new full-corpus aggregate.

The recognized benchmark runner now compares qrels document IDs with the active indexed corpus
after each build. Judgments outside a bounded corpus are excluded from the recall denominator and
reported as ``unknown_judgments`` plus ``queries_without_indexed_judgments``; this prevents a
10,000-document MIRACL/TREC run from treating documents beyond the selected bound as recoverable
search failures. The qrels loader still rejects malformed rows, invalid grades, and unknown query
IDs at load time.

The current LM Studio endpoint is reachable at ``http://127.0.0.1:4321``. Its structured
language plan was previously truncated by the 64-token response budget; the planner now allows
128 tokens and the default request timeout is 30 seconds. A live Gemma-4 E4B run produced
alternative result-conditioned lanes and antipatterns for the full SciFact hard queries. A third
round is available through ``--language-rounds 3`` for deeper iterative search. The two tested
hard queries (q437 and q560) still missed their qrels targets, so this is an observed capability
improvement, not a recall claim.

Language-agent fusion now deduplicates multiple chunks from one document and scores each candidate
against the query lane that found it. This prevents an expansion-only hit from being discarded
because it shares few words with the original phrasing. A focused regression test covers this
failure mode; full-corpus testing is still required to quantify precision impact.

The self-contained active-search prototype now rotates base, paraphrase, HyDE, and follow-up
lanes before and after the first positive. No-positive probes target original lane ranks rather
than filtered-list positions; HyDE reaches a deep rank early enough to test missed facets. On
the Fourth Wing hard suite, the final wall run found massive-wall, outer-wall, and
structure-over-crowd facets by calls 7-22. The entry-plus-humor run found its first explicit
positive at call 11 and a second office variant at call 38, with judged precision 1.0 on its
first five positive windows. These are small-LM labels only; the larger model remains outside
runtime search.

The language audit sidecar was corrected after trace review: incidental ``Archives`` and
``walls ... behind`` phrases had been overriding otherwise relevant windows. The sidecar now
uses conjunctions that represent the actual exclusions and accepts surface synonyms such as
``kidding``, ``massive``, and ``enormous``. Sidecar changes are evaluation artifacts and are
never imported by runtime search.
