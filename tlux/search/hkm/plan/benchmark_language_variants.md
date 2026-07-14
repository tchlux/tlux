# Language-query contrast variants

This diagnostic set extends the six-case gate with named subjects, missing
entities, and explicit contrasts. It is intentionally separate from the
initial acceptance gate until each returned hit has a manual relevance label.
Run it with the same evaluator and the companion judgement sidecar:

    bin/hkm-language-benchmark data/fourth_wing_hkm_index \
      --benchmark plan/benchmark_language_variants.md \
      --judgements plan/benchmark_language_variants_judgements.json \
      --top-k 5

| ID | Query | Required groups | Soft antipatterns |
|---|---|---|---|
| LQ-07 | Xaden climbs the side of a large structure at night | `Xaden` + `climb, climbs, climbing` + `structure, wall, cliff, chimney` + `night, dark` | none |
| LQ-08 | If someone climbs a chimney after dark while people below watch, find it even though I forgot the climber's name | `night, dark` + `climb, climbing, chimney` + `people, crowd, below` + `watch, horror, fear` | none |
| LQ-09 | I remember the office doorway and a funny remark, not the Archives research passage | `enter, door, office, room` + `funny, humorous, amusing, joke` | `archives, research` |
| LQ-10 | Not the crowd watching dragons; the crowd watching a horrified attacker | `crowd, people, gathers` + `watch, watched, watching` + `horror, horrified, fear, scream` | `dragons` |

LQ-07 deliberately tests a possibly incorrect remembered subject: the climb
evidence is present, but the named character may only appear in nearby context.
LQ-09 and LQ-10 test that the excluded scene does not become the answer. A
successful coverage result is not by itself a claim that every top-five hit is
relevant; use the sidecar precision labels and inspect unknowns.

## Run result: 2026-07-14

The deterministic runner and the warmed Gemma 3 runner both passed all four
variants for group coverage and coherent evidence. The live model-first run
(`--timeout 5`) had 0.92 macro judged precision over six labeled hits and a
3.71 second median wall time (4.62 second p95); the precision gate remains
intentionally open because LQ-07 still includes a known climbing decoy. LQ-09
returned the office/funny target at rank two, and LQ-10 returned the explicit
non-dragon contrast target at rank one. Unknown top-five hits are excluded from
the precision denominator and still need manual review before a service target
is claimed.
