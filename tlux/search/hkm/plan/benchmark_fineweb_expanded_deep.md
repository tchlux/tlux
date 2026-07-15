# FineWeb expanded deep language-query benchmark

This fixed set stresses paraphrase, missing entities, reordered clues, negation,
and several conditionals while keeping each expected answer grounded in a local
FineWeb passage. The required groups are content clues, not source IDs.

| ID | Query | Required groups | Soft antipatterns |
| --- | --- | --- | --- |
| LQ-101 | I remember a coupe with four wheel drive and a diesel engine but not the model. Find the review with the automatic gearbox and M Sport trim. | BMW 435d xDrive + 308bhp diesel + eight-speed automatic + M Sport | archives |
| LQ-102 | If the title belongs to a claimant ruling an island, and the heir is a princess, find the Game of Thrones passage even if I forgot the claimant's name. | Stannis Baratheon + Dragonstone + Princess Shireen + Targaryen | archives |
| LQ-103 | Find the equipment-hire business where mini excavators and tipper trucks are offered, unless the result is only about selling cars. | mini excavators + tipper trucks + labour hire | cars |
| LQ-104 | I cannot recall the musician's stage name: search for the German born artist linked to Can, krautrock, and pioneering sampling. | Holger Czukay + Can + krautrock + sampling | archives |
| LQ-105 | When two neighboring countries repair a tense relationship after a constitution dispute, and the plan includes connectivity, trade, and agriculture, what passage is this? | India + Nepal + constitution, bilateral ties + connectivity | China |
| LQ-106 | A green formal-wear page mentions a dress that can be worn several ways. If it also says emerald and multi-way, return that page, not a generic wedding story. | green bridesmaid dresses + convertible + multi-way + emerald | wedding story |
| LQ-107 | I remember an economist saying mortgage arrears need not cause panic, but I forgot the country and speaker. Find the item mentioning Australians and the Reserve Bank. | Australians + mortgage repayments + Reserve Bank of Australia + Jonathan Kearns | home design |
| LQ-108 | Something struck a home as a blue ball of fire. Unless this is a generic astronomy article, find the report that places the explosion in Monte Grande and mentions gunpowder. | meteorite + Monte Grande + blue ball of fire + gunpowder | astronomy |
| LQ-109 | If a transit executive is promoted to oversee fixed-route services, and another person becomes assistant director, identify the Metro announcement. | Timothy TJ Thorn + Shawn Donaghy + Transit Operations + fixed-route | subway map |
| LQ-110 | Find the mortgage example with a 30-year conventional fixed-rate loan. I forgot the builder, but Greenwood Village and DTC Boulevard are in the text. | 30-year Conventional fixed-rate loan + Greenwood Village + DTC Boulevard + down payment, monthly payment | refinance |
| LQ-111 | I remember a photograph exhibit where a famous Associated Press photographer spoke with a curator at Los Angeles City Hall, but not their names. | Nick Ut + Sara Cannon + Los Angeles City Hall + Bridge Gallery | wedding |
| LQ-112 | A security writer says the Internet is not the real world. If the same passage discusses firewall software, whois, and an IP address, find it rather than a generic cybersecurity guide. | Internet isn't the real world + firewall + whois + IP address | cybersecurity guide |
| LQ-113 | Find the study where radiation sterilizes male Aedes aegypti to limit dengue and Zika. It must include 70 Gy and sterile insect technique, not a vaccine trial. | Aedes aegypti, Aedesaegypti, Ae.aegypti + dengue + Zika + 70 Gy + sterile insect technique | vaccine trial |
| LQ-114 | I remember a telecom provider using Telcordia for prepaid GSM charging while serving tens of millions of CDMA subscribers. What company was it? | Tata Teleservices + Telcordia + GSM + CDMA + 40 million | smartphones |
| LQ-115 | If the illness is West Nile and the proposed treatment uses RNAs and autoimmune interference, find the Texas report even though I forgot the scientist. | West Nile + Lubbock + Manjunath Swamy + RNAs + autoimmune interference | influenza |
| LQ-116 | A travel note says a Croatian city is UNESCO listed and shows Diocletian's Palace, Fruit square, and Marjan hill. Find the city page. | Split + Croatia + UNESCO + Fruit square + Diocletian's Palace + Marjan hill | Dubrovnik |
| LQ-117 | Which home-design advice calls wall-to-wall carpeting outdated and recommends hardwood for renovations and resale value? | wall-to-wall carpet + outdated + renovations + resale value | mortgage |
| LQ-118 | I remember someone with a PhD in Medicine leaving Sydney for Europe to work and travel, but not the person's name. | PhD in Medicine + Sydney + Europe + medical researcher | university ranking |
| LQ-119 | If a sequel is praised as funny and profound by Esquire UK and also mentioned by GQ UK, identify the book and its author. | Generation A + Douglas Coupland, Coupland + Generation X + Esquire UK + GQ UK | Generation Z |
| LQ-120 | A Mexican international-relations program emphasizes negotiation and conflict resolution. I forgot the university name; find the Anáhuac page. | Anáhuac + Mexico + International Relations + negotiation + conflict resolution | accounting |
| LQ-121 | Reorder the clues: a four-wheel-drive coupe, 308bhp diesel, Sport Auto, and torque advantage should lead to the same BMW review, even if the headline is omitted. | BMW 435d xDrive + four-wheel-drive + 308bhp diesel + Sport Auto | motorcycle |
| LQ-122 | Do not return a generic Game of Thrones character list. If the evidence combines Blackwater Bay, the Stormlands, and a Targaryen heir's traditional title, find the relevant lord entry. | Blackwater Bay + Stormlands + Targaryen heir, Targaryen, Stannis Baratheon + Lord of Dragonstone | character list |
| LQ-123 | Unless the passage is about ordinary machinery sales, find the plant-and-labour-hire description that records regular services and service histories. | plant and labour hire, plant labour hire + regular services + service histories + mini excavators | machinery sales |
| LQ-124 | I remember a satellite-or-meteorite dispute, witnesses seeing fire from above, and a destroyed pizza oven. Find the account, even if the victim's name is missing. | Monte Grande, Buenos Aires + blue ball of fire, sky blue + explosion + pizza oven | satellite launch |
