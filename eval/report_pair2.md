# Evaluation Report — Pairing 2: Section + Hybrid

_Generated: 2026-10-24T19:47:12+00:00_

Queries evaluated: **32**

## Configuration

| Setting | Value |
|---|---|
| LLM provider | openai |
| LLM model | qwen/qwen3.7-flash |
| Device | cuda (NVIDIA GeForce RTX 3050) |
| Chunk stats source | hybrid vector+BM25 |
| Index dir | ./data/indexes/pair2 |

## 1. Retrieval-Level Metrics

|   k |   Recall@k |   Precision@k |   nDCG@k |   Recall@k (section) |
|----:|-----------:|--------------:|---------:|---------------------:|
|   1 |    0.75    |      0.75     | 0.75     |              0.75    |
|   3 |    0.75    |      0.520833 | 0.744983 |              0.75    |
|   5 |    0.8125  |      0.46875  | 0.758885 |              0.8125  |
|  10 |    0.96875 |      0.409375 | 0.813085 |              0.96875 |

MRR (answer-term): **0.7888** | MRR (section): **0.7888**

### Recall@5 by Query Category

| category         |   Recall@5 |   n |
|:-----------------|-----------:|----:|
| annex_lookup     |     0.5    |   2 |
| comparison       |     0.5    |   2 |
| cross_reference  |     0.5    |   2 |
| definition       |     0.8667 |  15 |
| parameter_lookup |     0.8    |   5 |
| procedure        |     1      |   6 |

## 2. Chunking-Specific Metrics

|   chunk_count |   avg_chunk_size_chars |   min_chunk_size_chars |   max_chunk_size_chars | relevance_density   | boundary_coherence   |
|--------------:|-----------------------:|-----------------------:|-----------------------:|:--------------------|:---------------------|
|          2128 |                678.199 |                     33 |                   4018 |                     |                      |

## 3. End-to-End Generation Metrics

| faithfulness   | answer_relevance   | answer_correctness   |   citation_accuracy |
|:---------------|:-------------------|:---------------------|--------------------:|
|                |                    |                      |            0.911389 |

## 4. Efficiency / Cost Metrics

|   indexing_time_ms |   avg_retrieval_ms |   avg_generation_ms |   avg_e2e_ms |   index_size_bytes |   llm_calls_per_query |   tokens_per_query |
|-------------------:|-------------------:|--------------------:|-------------:|-------------------:|----------------------:|-------------------:|
|             166719 |            2390.55 |             14280.4 |      16390.8 |         6.1655e+07 |                     1 |            5974.31 |

## Flagged / Unavailable Metrics

| metric                        | reason                                                                        |
|:------------------------------|:------------------------------------------------------------------------------|
| generation.faithfulness       | judge unavailable or not applicable (e.g. no cited sections / no gold answer) |
| generation.answer_relevance   | judge unavailable or not applicable (e.g. no cited sections / no gold answer) |
| generation.answer_correctness | judge unavailable or not applicable (e.g. no cited sections / no gold answer) |

## Judge Coverage

|   queries_judged |   metrics_attempted |
|-----------------:|--------------------:|
|                0 |                   0 |

## Per-Query Appendix

| id   | category         |   recall@5 |   mrr |   citation_acc | answer                                                                           |
|:-----|:-----------------|-----------:|------:|---------------:|:---------------------------------------------------------------------------------|
| q01  | parameter_lookup |          0 | 0     |          1     | Based on the provided context, ISO 15118-2 does not specify a single fixed maxim |
| q02  | procedure        |          1 | 1     |          1     | Based on the provided ISO 15118-3 context, the SLAC (System Level Automatic Coup |
| q03  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.1, an actor is defined as an **"entity whi |
| q04  | definition       |          1 | 1     |          1     | Based on the provided context, according to **ISO 15118-1, Section 3.1.2**, anci |
| q05  | definition       |          0 | 0.167 |          0.8   | Based on the provided context, authentication in ISO 15118 is addressed across m |
| q06  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.8, a battery management system (BMS) is an |
| q07  | definition       |          1 | 1     |          1     | Based on the provided context, basic signalling is defined as follows:  * **Defi |
| q08  | definition       |          0 | 0.167 |          1     | According to ISO 15118-1, Section 3.1.11, a certificate is defined as an **"elec |
| q09  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.12, "charge" is defined as the action to " |
| q10  | definition       |          1 | 1     |          1     | Based on the provided context, according to ISO 15118-1, Section 3.1.13, a charg |
| q11  | definition       |          1 | 1     |          1     | Based on the provided context, a **communication session** is defined across ISO |
| q12  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.16, a **contactor** is defined as an *"ele |
| q13  | definition       |          1 | 1     |          1     | Based on the provided context, a credential is defined in two ways across the IS |
| q14  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.18, **demand and prognosis** is defined as |
| q15  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.20, distributed energy resources (DER) are |
| q16  | parameter_lookup |          1 | 1     |        nan     | Based on the provided context, there is no specific value defined for the nomina |
| q17  | parameter_lookup |          1 | 1     |          1     | Based on the provided context, the voltage limits defined for the EV are specifi |
| q18  | procedure        |          1 | 1     |          1     | Based on the provided context, a communication session is established through th |
| q19  | procedure        |          1 | 1     |          1     | Based on the provided context, the matching procedure (referred to as the "match |
| q20  | procedure        |          1 | 1     |          1     | Based on the provided context, I don't have enough information to explain how bi |
| q21  | cross_reference  |          1 | 1     |          1     | Based on the provided context, **Clause 5** is referenced in relation to Annex A |
| q22  | cross_reference  |          0 | 0.125 |          1     | Based on the provided context, **ISO 15118-2, Section 8.2.3** references Annex C |
| q23  | annex_lookup     |          1 | 1     |          0     | Based on the provided context, there is no complete description of what Annex A  |
| q24  | annex_lookup     |          0 | 0.167 |        nan     | Based on the provided context, **Annex C** describes the **normative schema defi |
| q25  | comparison       |          0 | 0.167 |          1     | Based on the provided ISO 15118-2 context, AC and DC charging can be compared ac |
| q26  | comparison       |          1 | 0.2   |          1     | Based on the provided ISO 15118 specifications, here is a comparison between Plu |
| q27  | parameter_lookup |          1 | 0.25  |          1     | Based on the provided context, there is no single fixed value for the maximum ch |
| q28  | procedure        |          1 | 1     |          0.875 | Based on the provided context, the SECC communication setup follows a defined se |
| q29  | definition       |          1 | 1     |          1     | Based on the provided context:  According to **ISO 15118-1, Section 3.1.31**, th |
| q30  | definition       |          1 | 1     |          0     | Based on the provided context, I don't have enough information. The supplied exc |
| q31  | parameter_lookup |          1 | 1     |          1     | Based on the provided context, the control pilot signal is described through the |
| q32  | procedure        |          1 | 1     |          0.667 | Based on the provided ISO 15118-3 context, PLC communication operates through th |
