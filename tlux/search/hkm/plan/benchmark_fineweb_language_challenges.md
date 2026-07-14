# FineWeb language challenge set

This ten-case fixture extends the random language benchmark with hand-reviewed
queries from the heterogeneous FineWeb profile corpus. Cases cover vague
memory, multiple conditionals, and forgotten subjects or objects. Every case
was sampled from indexed source text, then rewritten while retaining grounded
evidence clues.

## Provenance

- Source corpus: `data/fineweb_profile/docs`
- Source size: 512 files, 1,359,407 bytes, 227,435 words, 509 domains
- Index used for the recorded run: `/private/tmp/hkm_fineweb_profile_drama`
- Cases LQ-21 through LQ-28 came from random seed `20260714` failures.
- Cases LQ-29 and LQ-30 came from random seed `20260715` failures.
- Target document IDs below are diagnostic only; the evaluator uses aliases so
  the fixture remains valid after a rebuild.

## Query cases

| ID | Query | Required groups | Soft antipatterns |
|---|---|---|---|
| LQ-21 | When discussion of coverings and disagreement mentions translation theories, find the passage even if missionary teams are elsewhere | `coverings` + `disagreement` + `translation` + `missionary, teams` | none |
| LQ-22 | If Geewa's platform has attracted players and fans, find the passage even if the game name is forgotten | `Geewa` + `platform` + `attracted` + `players` | none |
| LQ-23 | I remember a platform attracting players and fans every day, but not who made it or what game was mentioned | `platform` + `attract, attracted` + `players` + `fans` + `every day` | none |
| LQ-24 | If a company's executive team discusses intelligence and fraud systems, find that passage even if the business name is forgotten | `company` + `executive` + `intelligence` + `fraud` + `systems` | none |
| LQ-25 | When copies of a new book appear at a Bookseller, find that passage even if the title is forgotten | `copies` + `new` + `book` + `Bookseller` | none |
| LQ-26 | I remember new golds and a future hedge, plus a spoiler about Skill Points, but not who was involved | `golds` + `future` + `hedge` + `spoiler` + `Skill Points` | none |
| LQ-27 | I remember a restaurant striving for perfection and keeping the highest standards at Gavroche, but not the speaker | `restaurant` + `perfection` + `standards` + `Gavroche` | none |
| LQ-28 | If three guardians undertake quests and face tribulations, find the passage even if the names are forgotten | `three` + `guardians` + `quests` + `tribulations` | none |
| LQ-29 | A meeting in Northwest Ohio where representatives discuss farmland preservation | `meeting` + `Northwest Ohio` + `representatives` + `farmland` + `preservation` | none |
| LQ-30 | After creating a list of software vendors, find the market-research passage even if the Internet wording differs | `list` + `software` + `vendors` + `market research` + `Internet` | none |

The target evidence sources and diagnostic document IDs are:

| Cases | Source file | Document ID |
|---|---|---:|
| LQ-21 | `00428_cc-main-2019-18_http-haretranslation-com-2018-11-19-should-i-wait-for-the-id.txt` | 88 |
| LQ-22, LQ-23 | `00252_cc-main-2015-32_http-venturebeatprofiles-com-company-profile-geewa-opinion.txt` | 727 |
| LQ-24 | `00134_cc-main-2017-47_http-birnbachcom-com-clients-client-quotes-shtml.txt` | 51 |
| LQ-25 | `00228_cc-main-2015-32_http-communityadvocate-com-2013-11-12-author-green-to-sign-c.txt` | 372 |
| LQ-26 | `00337_cc-main-2018-13_http-forum-skullgirlsmobile-com-threads-official-1-5-0-updat.txt` | 192 |
| LQ-27 | `00216_cc-main-2015-32_http-www-nzherald-co-nz-recipes-news-article-cfm-c-id-300-ob.txt` | 54 |
| LQ-28 | `00125_cc-main-2019-39_https-ramblings-ajaxed-net-category-books-page-2.txt` | 38 |
| LQ-29 | `00504_cc-main-2016-22_http-www-presspublications-com-from-the-press-1216-land-pres.txt` | 208 |
| LQ-30 | `00118_cc-main-2019-39_https-www-mysurveylab-com-en-blog-choose-best-survey-softwar.txt` | 240 |

The complete target aliases are retained in the sidecar judgement file; IDs
are diagnostic only and are not used by the evaluator.

## Reproducible run

```text
bin/hkm-language-benchmark /private/tmp/hkm_fineweb_profile_drama \
  --benchmark plan/benchmark_fineweb_language_challenges.md \
  --judgements plan/benchmark_fineweb_language_challenges_judgements.json \
  --tool-mode semantic --top-k 5 --jsonl /tmp/fineweb_language_challenges.jsonl \
  --require-gate
```

The evaluator requires complete group coverage and one coherent hit in the
top five. The sidecar labels only target evidence with conservative all-term
clauses; unknown hits remain outside the precision denominator.
