# Language conditional challenge set

This seven-case diagnostic extends the six-case language gate with one through
four condition markers, alternate conditional phrasing, and forgotten names.
Each case is grounded in a scene that exists in the current Fourth Wing index;
the required groups therefore test a plausible memory request rather than an
impossible conjunction of unrelated scenes.

Run it with the evaluator and conservative judgement sidecar:

    bin/hkm-language-benchmark data/fourth_wing_hkm_index \
      --benchmark plan/benchmark_language_conditionals.md \
      --judgements plan/benchmark_language_conditionals_judgements.json \
      --tool-mode semantic --top-k 5 --jsonl /tmp/language_conditionals.jsonl \
      --require-gate

| ID | Query | Required groups | Soft antipatterns |
|---|---|---|---|
| LQ-11 | If someone climbs a chimney with a rope, find that passage | `climb, climbing` + `chimney` + `rope` | none |
| LQ-12 | After someone climbs a chimney with a rope, even if I forgot their name, find the passage | `climb, climbing` + `chimney` + `rope` | none |
| LQ-13 | While people watch, if someone climbs a chimney with a rope, return that memory; I do not remember who | `watch, watching` + `climb, climbing` + `chimney` + `rope` | none |
| LQ-14 | Find the crowd circle where someone screamed, even though I do not know who it was | `crowd, people` + `circle` + `scream, screamed` | none |
| LQ-15 | When darkness falls, if someone watches in horror as a tall attacker emerges, find it even if the name is forgotten | `dark, darkness` + `watch, watching` + `horror, horrified` + `tall` | none |
| LQ-16 | After someone enters the commanding office, a funny remark is made; unless this is Archives research, return it | `enter, entered, office, door` + `funny, amusing` | `archives, research` |
| LQ-17 | Whether the name is forgotten or not, locate the scene where a crowd forms a circle while someone screamed, provided it is not dragon spectators | `crowd, people` + `circle` + `scream, screamed` | `dragons, spectators` |

The latest deterministic run on 2026-07-14 passed all seven cases for group
coverage and coherent-hit at five results. It measured 0.929 judged precision
over 11 labeled hits, with 0.96 second median and 4.79 second p95 agent
latency. This is a coverage gate, not a precision guarantee: LQ-16 surfaced
an Archives/research antipattern at rank two, and LQ-17 surfaced a
dragon/spectator antipattern at rank three. The sidecar labels only a small
set of manually inspected patterns; unknown hits remain outside the precision
denominator.
