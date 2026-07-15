# FineWeb extended language challenge set

This twenty-case gate broadens the ten-case FineWeb fixture with unrelated
domains and harder memory phrasing. It mixes multiple conditionals, vague
descriptions, and requests that omit a person, place, product, or other entity.
Each case comes from one random active passage in the heterogeneous FineWeb
profile index and retains several independently checkable evidence clues.

## Provenance

- Source corpus: `data/fineweb_profile/docs`
- Source size: 512 files, 1,359,407 bytes, 227,435 words, 509 domains
- Index used for the recorded run: `/private/tmp/hkm_fineweb_profile_drama`
- Cases LQ-31 through LQ-50 are drawn with `sample_passages(..., seed=20260716)`
  and `sample_passages(..., seed=20260717)`; each seed contributes ten cases.
- Target IDs and paths are diagnostic only. The evaluator uses aliases so a
  rebuild with different document IDs remains comparable.

The first dry run exposed two useful retrieval diagnostics. The election source
says that the claims were "true" only in the phrase "none of it was true", so
LQ-32 accepts both `false` and `true` as the remembered condition. A generic
tax-policy paraphrase initially missed LQ-40; retaining the grounded
medications context makes that query recover the Senator Booker passage while
still omitting the speaker's name.

## Query cases

| ID | Query | Required groups | Soft antipatterns |
|---|---|---|---|
| LQ-31 | I remember an American Cocker Spaniel playing in mud at a dog park with clear views of San Francisco, but not where the park was | `Cocker Spaniel` + `mud` + `dog park` + `San Francisco` | none |
| LQ-32 | If the campaign knew a claim was false but still used Karl Rove tactics to make Obama seem dangerous, find the election passage even if I forgot the campaign name | `campaign` + `false, true` + `Karl Rove` + `Obama` | none |
| LQ-33 | When authors resolve disagreements through discussion and rely on published rationales for a Top 5 list of patient services, find the study even if the journal is forgotten | `authors` + `disagreements` + `discussion` + `patient diagnosis` | none |
| LQ-34 | I remember a property with building plans, an enclosed porch, foyer, and office, but not its address | `building plans` + `enclosed porch` + `foyer` + `office` | none |
| LQ-35 | If a complete conversation is an MP3 of Edgar Allan Poe readings recorded in Charlottesville, find it even if the speaker is forgotten | `conversation` + `MP3` + `Edgar Allan Poe` + `Charlottesville` | none |
| LQ-36 | When a team records four sacks and an interception, and Nebraska has the special-teams edge from its kicker, find the preview even if the opponent is unknown | `sacks` + `interception` + `Nebraska` + `special teams` | none |
| LQ-37 | I remember a word that was uncommon until an about-turn in meaning, entering the language around 1570, but not the word itself | `about-turn` + `1570` + `language` + `common` | none |
| LQ-38 | A tented camp where Zimbabwe and South Africa meet, with two rivers and luxury relaxation, is the memory I need even if the camp name is missing | `tented camp` + `Zimbabwe, Zimbabwean` + `South African` + `rivers` + `luxurious` | none |
| LQ-39 | If a black band tee is made from black cotton and aimed at fans, find the merchandise passage even if the design name is forgotten | `Black Band Tee` + `black cotton` + `fan` + `tees` | none |
| LQ-40 | When drug companies use a tax windfall to benefit shareholders rather than patients, find the passage about medications even if the speaker is unknown | `drug companies` + `tax windfall` + `shareholders` + `patients` | none |
| LQ-41 | I remember the Reds facing Petrojet in an Egyptian Premier League game after someone said the team was not legitimate, but not who said it | `Reds` + `Petrojet` + `Egyptian Premier League` + `legitimate` | none |
| LQ-42 | If Zexion surprises Demyx by waking up straddling him, find the scene even if the relationship is forgotten | `Zexion` + `Demyx` + `straddling` + `emotion` | none |
| LQ-43 | When outgoing and incoming requests need geolocation and a proxy server port, find the Windows setup article even if the address is forgotten | `outgoing` + `incoming` + `geolocation` + `proxy server` | none |
| LQ-44 | I remember a Denver Paper Fashion Show where paper and scissors transform designs with ingenuity, but not the designers | `Denver Paper Fashion` + `paper` + `scissors` + `transformations` | none |
| LQ-45 | If someone waits two months for a BioVisions post that never happens and could have written a similar post on their own terms, find it even if the person name is missing | `BioVisions` + `two months` + `waiting` + `post` | none |
| LQ-46 | A list of animal clips includes a panda exhibit, a leopard roar, and a cheetah ambush; find it even if I forgot which site | `panda` + `leopard` + `cheetah` + `ambushed` | none |
| LQ-47 | When a German division pays in euros, a forex transaction influences EUR/USD in the buying and selling market, find the passage even if the trader is forgotten | `German division` + `euros` + `forex` + `EUR/USD` | none |
| LQ-48 | I remember Zexion avoiding eye contact while Demyx asks what he is doing, but not who spoke first | `Zexion` + `Demyx` + `eye contact` + `voice` | none |
| LQ-49 | A royal park offers palaces, museums, ancient trees, and swimming from a private rock; find the travel description even if its name escapes me | `royal palaces` + `museums` + `ancient trees` + `private rock` | none |
| LQ-50 | If an airline says economy demand returned to year-ago levels but business trips remain lower and cost cuts are not being considered, find the report even if the carrier is forgotten | `economy` + `business trips` + `cost cuts` + `demand` | none |

## Target evidence

| ID | Source file | Document ID |
|---|---|---:|
| LQ-31 | `00198_cc-main-2015-32_http-www-allisonacres-org-dppics1-html.txt` | 297 |
| LQ-32 | `00357_cc-main-2018-13_https-www-huffingtonpost-com-sean-hartofilis-an-appeal-to-in.txt` | 152 |
| LQ-33 | `00154_cc-main-2017-47_https-jamanetwork-com-journals-jama-fullarticle-1857323.txt` | 117 |
| LQ-34 | `00405_cc-main-2019-18_http-www-hubcityrealty-ca-listing-m121001-227-dominion-st-mo.txt` | 313 |
| LQ-35 | `00481_cc-main-2015-06_http-jacket2-org-commentary-jerome-mcgann-close-listening.txt` | 276 |
| LQ-36 | `00017_cc-main-2013-20_http-nfldraft-rivals-com-content-asp-cid-1124611.txt` | 130 |
| LQ-37 | `00191_cc-main-2018-47_http-www-worldwidewords-org-nl-pwnq-htm.txt` | 431 |
| LQ-38 | `00454_cc-main-2015-06_http-www-groupon-co-za-deals-johannesburg-a-2-night-stay-or-.txt` | 656 |
| LQ-39 | `00014_cc-main-2013-20_http-civilcivic-com-merch.txt` | 331 |
| LQ-40 | `00269_cc-main-2020-05_https-www-hcacfoundation-org-drug-companies-using-tax-windfa.txt` | 384 |
| LQ-41 | `00171_cc-main-2017-47_https-www-kingfut-com-2015-01-10-garrido-launches-attack-off.txt` | 278 |
| LQ-42 | `00205_cc-main-2015-32_https-www-fanfiction-net-s-4736739-1-christmas-rain.txt` | 411 |
| LQ-43 | `00390_cc-main-2019-18_https-proxy-am-en-articles-nastrojka-proxy-server-na-windows.txt` | 489 |
| LQ-44 | `00120_cc-main-2019-39_https-bellatory-com-fashion-industry-denver-paper-fashion-sh.txt` | 81 |
| LQ-45 | `00001_cc-main-2013-20_http-endogenousretrovirus-blogspot-com-2007-11-if-you-have-s.txt` | 596 |
| LQ-46 | `00078_cc-main-2018-39_http-scrappybook-com-board-cute-animals-38.txt` | 776 |
| LQ-47 | `00415_cc-main-2019-18_http-forexturtle-com-where-do-forex-signals-come-from-can-fo.txt` | 478 |
| LQ-48 | `00205_cc-main-2015-32_https-www-fanfiction-net-s-4736739-1-christmas-rain.txt` | 413 |
| LQ-49 | `00363_cc-main-2018-13_http-www-nationalstadsparken-se-default-aspx-id-2141-ptid-0.txt` | 261 |
| LQ-50 | `00142_cc-main-2017-47_https-centreforaviation-com-news-all-nippon-airways-maintain.txt` | 361 |

## Reproducible run

```text
bin/hkm-language-benchmark /private/tmp/hkm_fineweb_profile_drama \
  --benchmark plan/benchmark_fineweb_language_challenges_extended.md \
  --judgements plan/benchmark_fineweb_language_challenges_extended_judgements.json \
  --tool-mode semantic --top-k 5 --jsonl /tmp/fineweb_language_challenges_extended.jsonl \
  --require-gate
```

The gate requires every group to appear in the top five and at least one hit to
cover two or more groups. The sidecar labels conservative target evidence only;
unknown hits remain outside the precision denominator.
