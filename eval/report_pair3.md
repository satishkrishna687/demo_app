# Evaluation Report — Pairing 3: Advanced ISO + Graph

_Generated: 2026-10-25T03:22:07+00:00_

Queries evaluated: **32**

## Configuration

| Setting | Value |
|---|---|
| LLM provider | openai |
| LLM model | qwen/qwen3.7-flash |
| Device | cuda (NVIDIA GeForce RTX 3050) |
| Chunk stats source | graph-enhanced hybrid + rerank |
| Index dir | ./data/indexes/pair3 |

## 1. Retrieval-Level Metrics

|   k |   Recall@k |   Precision@k |   nDCG@k |   Recall@k (section) |
|----:|-----------:|--------------:|---------:|---------------------:|
|   1 |    0.90625 |      0.90625  | 0.90625  |              0.90625 |
|   3 |    0.9375  |      0.729167 | 0.914349 |              0.96875 |
|   5 |    0.96875 |      0.65625  | 0.921983 |              1       |
|  10 |    0.96875 |      0.328125 | 0.921983 |              1       |

MRR (answer-term): **0.9245** | MRR (section): **0.9401**

### Recall@5 by Query Category

| category         |   Recall@5 |   n |
|:-----------------|-----------:|----:|
| annex_lookup     |        1   |   2 |
| comparison       |        1   |   2 |
| cross_reference  |        1   |   2 |
| definition       |        1   |  15 |
| parameter_lookup |        0.8 |   5 |
| procedure        |        1   |   6 |

## 2. Chunking-Specific Metrics

|   chunk_count |   avg_chunk_size_chars |   min_chunk_size_chars |   max_chunk_size_chars | relevance_density   | boundary_coherence   |
|--------------:|-----------------------:|-----------------------:|-----------------------:|:--------------------|:---------------------|
|          2007 |                709.643 |                     33 |                   3989 |                     |                      |

## 3. End-to-End Generation Metrics

| faithfulness   | answer_relevance   | answer_correctness   |   citation_accuracy |
|:---------------|:-------------------|:---------------------|--------------------:|
|                |                    |                      |            0.977833 |

## 4. Efficiency / Cost Metrics

|   indexing_time_ms |   avg_retrieval_ms |   avg_generation_ms |   avg_e2e_ms |   index_size_bytes |   llm_calls_per_query |   tokens_per_query |
|-------------------:|-------------------:|--------------------:|-------------:|-------------------:|----------------------:|-------------------:|
|             115655 |            4318.45 |             13205.2 |        17232 |        6.25652e+07 |                     1 |            4847.12 |

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
| q01  | parameter_lookup |          0 | 0     |        nan     | Based on the provided context, according to ISO 15118-2, Section D.2.2 (page 287 |
| q02  | procedure        |          1 | 1     |          1     | Based on the provided context from ISO 15118-3, the SLAC communication setup is  |
| q03  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.1, an **actor** is defined as an `"entity  |
| q04  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.2, ancillary services are defined as **"se |
| q05  | definition       |          1 | 1     |          1     | Based on the provided context, authentication in ISO 15118 is defined and implem |
| q06  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.8, a battery management system (BMS) is an |
| q07  | definition       |          1 | 1     |          1     | Based on the provided context, basic signalling is defined and characterized as  |
| q08  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.11, a **certificate** is defined as an *"e |
| q09  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.12, **"charge"** is defined as storing ele |
| q10  | definition       |          1 | 1     |          1     | Based on the provided context, according to ISO 15118-1, Section 3.1.13, a charg |
| q11  | definition       |          1 | 1     |          1     | Based on the provided context, a communication session is defined across the ISO |
| q12  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.16, a contactor is an **electrically contr |
| q13  | definition       |          1 | 1     |          1     | Based on the provided context, a credential is defined in two ways across the st |
| q14  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.18, **demand and prognosis** is defined as |
| q15  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.20, distributed energy resources (DER) are |
| q16  | parameter_lookup |          1 | 1     |        nan     | Based on the provided context, I don't have enough information. The text does no |
| q17  | parameter_lookup |          1 | 1     |          1     | Based on the provided context, the following voltage limit elements are defined  |
| q18  | procedure        |          1 | 1     |          1     | Based on the provided context, a communication session is established through th |
| q19  | procedure        |          1 | 1     |          1     | Based on the provided context from ISO 15118-3, the matching procedure can be de |
| q20  | procedure        |          1 | 1     |          1     | Based on the provided context, bidirectional power transfer (BPT) is defined as  |
| q21  | cross_reference  |          1 | 1     |          1     | Based on the provided context, I don't have enough information. The excerpt only |
| q22  | cross_reference  |          1 | 1     |          1     | Based on the provided context, **Section 8.2.3 of ISO 15118-2** references Annex |
| q23  | annex_lookup     |          1 | 1     |          0.5   | Based on the provided context, Annex A describes the following:  * **Technology- |
| q24  | annex_lookup     |          1 | 1     |          1     | Based on the provided context, Annex C describes the complete XML Schema definit |
| q25  | comparison       |          1 | 1     |          1     | Based on the provided ISO 15118-2 excerpts, AC and DC charging are differentiate |
| q26  | comparison       |          1 | 0.333 |          1     | I don't have enough information to provide a direct comparison between Plug and  |
| q27  | parameter_lookup |          1 | 0.25  |          1     | Based on the provided context, there is no single fixed maximum charging current |
| q28  | procedure        |          1 | 1     |          0.857 | Based on the provided context from ISO 15118-2, the SECC communication setup inv |
| q29  | definition       |          1 | 1     |          1     | According to ISO 15118-1, Section 3.1.31, the **EVCC** (Electric Vehicle Communi |
| q30  | definition       |          1 | 1     |        nan     | Based on the provided context, I don't have enough information to define what th |
| q31  | parameter_lookup |          1 | 1     |          1     | Based on the provided context, the control pilot signal is characterized as foll |
| q32  | procedure        |          1 | 1     |          1     | Based on the provided context from ISO 15118-3, PLC communication operates throu |
