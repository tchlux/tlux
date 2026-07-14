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

## Post-fix smoke

After semantic routing, first-lane anchoring, and morphology-aware antipattern
filtering, a warmed Gemma 3 JSONL smoke (`--timeout 10`, `--model-first`,
`--tool-mode semantic`) recovered the reviewed anchors in the top five for the
four short examples: doc 248 (funny office scene) at rank 1, doc 174 (crowd
circle and screaming student) at rank 1, doc 38 (large outer wall) at rank 3,
and doc 115 (climb/chimney/rope) at rank 3 for the missing-name query. The
agent still spent roughly 4.7-15.3 seconds per request on this server and one
planner call used deterministic recovery. This is a smoke result, not the
six-case acceptance gate; run the full JSONL command above after every model,
index, or ranking change.

## Known limitation

The current language runner uses a bounded two-round plan and soft penalties.
It does not yet synthesize a final natural-language answer or prove that every
returned hit is relevant. This benchmark is deliberately a recall-oriented
gate for remembered conditions. Add judged negatives and a precision@k review
before using it as a B2B service-level target.
