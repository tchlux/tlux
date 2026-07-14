# Language-query challenge set

This is a small, hand-reviewed benchmark for natural-language requests over
the Fourth Wing index. It tests paraphrases, conjunctions, conditionals, and
queries where the remembered subject or object is missing. The cases are
intentionally different from the raw-passage benchmark: there is no single
golden passage, so relevance is measured by coverage of observable concept
groups in the returned evidence.

## Query cases

Each `group` contains acceptable surface forms for one remembered condition.
The evaluator should match a group against the normalized `preview_text` (and
the document preview when available) of any top-k hit. The aliases are not
answers; they are an auditable relevance oracle that can survive a rebuild
with different document IDs.

| ID | Query | Required groups | Soft antipatterns |
|---|---|---|---|
| LQ-01 | A scenario where something funny is said | `funny, humorous, amusing, joke, laugh` | none |
| LQ-02 | A large wall stands over us | `large, giant, huge, towering` + `wall, tower, structure, cliff, parapet` | none |
| LQ-03 | Crowds gathered to watch in horror | `crowd, people, gathered, circle` + `watch, watching, see, spectators` + `horror, horrified, terror, fear, scream` | none |
| LQ-04 | When night falls, someone climbs a large structure while a crowd gathers below to watch in horror | `night, nighttime, midnight, dark` + `climb, climbs, climbing, ascending, scaling` + `structure, wall, tower, cliff, chimney` + `crowd, people, gathers, below` + `watch, watching, horror, fear` | none |
| LQ-05 | I remember a person climbing something tall at night while people watched below, but I forgot who the person was and what the structure was called | `night, nighttime, midnight, dark` + `climb, climbs, climbing, ascending, scaling` + `tall, large, tower, wall, structure, cliff` + `people, crowd, gathers, below` + `watch, watching, fear, horror` | do not require a character or place name |
| LQ-06 | Find the scene where, after someone enters a room, something amusing is said, even though it is not the later discussion about old research | `enter, entered, door, room, office` + `funny, humorous, amusing, joke, laugh` | `research, archives, old` |

For LQ-04 and LQ-05, the conditions may occur in separate nearby passages.
The test therefore measures coverage across the top-k set as well as a
coherence check for at least one hit covering two or more groups. A query must
not be made easier by inventing the missing person or object.

## Metrics and acceptance gate

For each case and `k=5`, report:

- `group_coverage@5`: required groups represented by at least one top-five hit,
  divided by the number of required groups.
- `coherent_hit@5`: whether one hit covers at least two required groups (or the
  only group for LQ-01).
- `top_hit_groups`: number of required groups represented by the rank-one hit.
- `anti_rank`: rank of the first hit containing a soft antipattern, when one is
  provided. This is diagnostic only; antipatterns are a penalty, never a hard
  filter, so recall cannot be lost just because a useful passage contains a
  common term.

The optional `plan/benchmark_language_judgements.json` sidecar adds
conservative all-term clauses for known relevant and negative evidence. The
evaluator reports `judged_precision@5` using only labeled hits; unknown hits
are counted separately and excluded from its denominator. Use
`--require-precision` to apply this separate gate after the sidecar grows to
cover the cases being compared. The existing coverage gate is unchanged.

The initial quality gate is `group_coverage@5 == 1.0` for every case and
`coherent_hit@5` true for every case. `top_hit_groups` and `anti_rank` are
reported rather than gated until a larger judged set exists. A result that is
non-empty but has zero group coverage is an ungrounded language result and
must be treated as a failure.

## Reproducible run

The persistent agent exposes the query trace, antipatterns, round count, hit
previews, and timings needed to audit these metrics. Run the deterministic
baseline first, then the LM Studio planner:

    printf '%s\n' \
      '{"text":"A scenario where something funny is said"}' \
      '{"text":"A large wall stands over us"}' \
      '{"text":"Crowds gathered to watch in horror"}' \
      '{"text":"When night falls, someone climbs a large structure while a crowd gathers below to watch in horror"}' \
      '{"text":"I remember a person climbing something tall at night while people watched below, but I forgot who the person was and what the structure was called"}' \
      '{"text":"Find the scene where, after someone enters a room, something amusing is said, even though it is not the later discussion about old research"}' \
    | bin/hkm-agent data/fourth_wing_hkm_index --language-query \
        --tool-mode hybrid --top-k 5 --warmup

Save the same six-line JSONL input as `queries.jsonl`, then run it with the
live LM Studio planner:

    bin/hkm-agent data/fourth_wing_hkm_index --language-query \
        --base-url http://192.168.8.222:1234/v1 \
        --model google/gemma-3-4b --model-first --tool-mode hybrid \
        --top-k 5 --warmup < queries.jsonl

For an auditable gate, `bin/hkm-language-benchmark` reads the table above,
reuses one persistent agent, and emits per-case group coverage, coherent-hit,
antipattern rank, and wall/model/search latency. Add `--jsonl PATH` to retain
the raw case reports and `--require-gate` for a nonzero exit when any case
misses full coverage or coherence:

    bin/hkm-language-benchmark data/fourth_wing_hkm_index \
      --base-url http://192.168.8.222:1234/v1 --model google/gemma-3-4b \
      --model-first --tool-mode semantic --top-k 5 --warmup \
      --jsonl /tmp/language_queries.jsonl --require-gate

The first run is a deterministic regression, not a quality claim for the
language model. Save the JSONL output beside a dated report when recording a
live run; do not treat model-generated antipatterns as labels without manual
review.

## Current Fourth Wing anchor observations

Direct semantic searches provide useful anchors for judging the agent lanes:

- A funny-saying search retrieves the Chapter 25 office scene (doc 248) near
  rank one, while a separate old-research passage (doc 241) is a plausible
  antipattern for LQ-06.
- Wall/structure searches retrieve the giant outer-wall passage (doc 38) and
  the structure towering over candidates (doc 15).
- Crowd/horror searches retrieve the crowd-circle passage (doc 174), the
  explicit watch-in-horror passage (doc 375), and the crowd beneath a towering
  structure (doc 15).
- Climbing-at-night searches retrieve distinct climbing passages (docs 101 and
  115). The subject, object, crowd, and fear conditions are distributed, which
  is why LQ-04 and LQ-05 test iterative alternatives rather than one exact
  phrase.

These IDs are diagnostic examples only. The concept-group oracle above is the
portable criterion; changing chunk sizes or rebuilding the source must not
turn a valid answer into a failure solely because a document ID moved.

## Live diagnostic: 2026-07-14

A warmed six-request run through LM Studio Gemma 3 (`--timeout 5`,
`--model-first`, `--tool-mode hybrid`) returned non-empty JSON for every case,
but it did not pass the language quality gate. LQ-01 returned the funny
sentence in the old-research passage (doc 241), LQ-03 did not put the crowd or
horror anchors in the top five, and LQ-04/LQ-05 split the climb/night evidence
without recovering the crowd condition. LQ-06 also ranked doc 241 despite its
research/archives antipattern. LQ-02 had a wall-related hit but not a stable
rank-one wall answer. Two requests used deterministic recovery after the
bounded model call.

This is a useful failure baseline, not a model-quality claim. Direct semantic
searches for shorter clauses do retrieve the reviewed anchors (docs 174, 375,
15 for crowd/horror and docs 101, 115 for climbing), so the next change should
improve language-lane mode selection and result merging rather than add more
corpus data. In particular, the current run used hybrid for the original
natural-language query while direct semantic search is stronger for these
paraphrases; model-generated one-word alternatives also need a lane/condition
coverage penalty so they cannot outrank a coherent clause.

## Current gate result: 2026-07-14

After preserving the complete first-pass candidate page through refinement,
the executable evaluator passed all six cases through the active Gemma 3
LM Studio server (`--model-first --tool-mode semantic --timeout 20`): six of
six had full group coverage and a coherent hit in the top five. Three fresh
runs were identical, with median wall/agent latency of 2.94-3.02 seconds and
p95 latency of 3.50-3.55 seconds; search itself was 0.96-1.04 seconds median
and 2.03-2.09 seconds p95. LQ-06 recovered both the room-entry and
funny-office evidence while the old-research antipattern was rank five. This
is a repeatable quality gate for the reviewed corpus, not yet a universal
precision or service-level guarantee; repeat the command after changing the
model, index, or ranking policy.

## Latest rerun: 2026-07-14

After adding explicit negative-clause filtering to contrast reranking, the
warmed Gemma 3 model-first run (`--timeout 5`, semantic mode) again passed all
six coverage/coherence cases. It measured 0.75 macro judged precision over 13
labeled hits, with a 3.53 second median wall time (5.17 second p95) and 0.86
second median search time (2.12 second p95). The separate four-case contrast
diagnostic passed 4/4 and reached 0.92 macro judged precision over six labeled
hits. These are useful regression measurements, not a claim that all unknown
hits are relevant.

The executable evaluator was added to make this gate repeatable. A warmed
Gemma 3 run on 2026-07-14 (`--model-first --tool-mode semantic --timeout 10`)
passed LQ-01 through LQ-03, but only 3/6 cases overall: mean group coverage
was 0.90, minimum coverage 0.60, and coherent-hit coverage was 5/6. Median
wall/agent latency was 3.82 seconds (p95 4.57 seconds); search itself was
1.04 seconds median (p95 1.23 seconds). LQ-04/LQ-05 missed night or
watch/horror groups, and LQ-06 had an old-research passage at rank one despite
the antipattern. The report was saved as `/tmp/language_eval_live.jsonl`.

## Post-fix smoke

After semantic routing and morphology-aware antipattern filtering, a warmed
Gemma 3 JSONL smoke (`--timeout 10`, `--model-first`,
`--tool-mode semantic`) recovered the reviewed anchors in the top five for the
four short examples: doc 248 (funny office scene) at rank 1, doc 174 (crowd
circle and screaming student) at rank 1, doc 38 (large outer wall) at rank 3,
and doc 115 (climb/chimney/rope) at rank 3 for the missing-name query. The
agent still spent roughly 4.7-15.3 seconds per request on this server and one
planner call used deterministic recovery. This is a smoke result, not the
six-case acceptance gate; run the full JSONL command above after every model,
index, or ranking change.

## Strict agentic trace: 2026-07-14

The opt-in `--always-refine` mode forces the planner/refinement round even when
the first semantic pass is already confident. A live six-case Gemma 3 run
used two rounds and five recorded query lanes for every case, proving the
inspect-then-search loop. It reached 5/6 group-coverage cases, 0.933 mean
coverage, and 0.750 judged precision over eight labeled hits. Median agent and
search latency were 3.31/0.66 seconds (p95 4.71/1.71 seconds). The strict
trace is therefore an autonomy diagnostic, not yet the default quality gate;
adaptive routing remains the lower-latency production path.

## Known limitation

The current language runner uses a bounded two-round plan and soft penalties.
It does not yet synthesize a final natural-language answer or prove that every
returned hit is relevant. The sidecar is an initial conservative precision
sample, not a B2B service-level target; expand it with manually reviewed
negative snippets before treating `--require-precision` as a broad guarantee.
