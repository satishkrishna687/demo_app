# Evaluation Report — Pairing 1: Recursive + Parent-Child

_Generated: 2026-10-24T19:28:14+00:00_

Queries evaluated: **32**

## Configuration

| Setting | Value |
|---|---|
| LLM provider | openai |
| LLM model | qwen/qwen3.7-flash |
| Device | cuda (NVIDIA GeForce RTX 3050) |
| Chunk stats source | corpus_chunks = parent chunks (children are internal) |
| Index dir | ./data/indexes/pair1 |

## 1. Retrieval-Level Metrics

|   k |   Recall@k |   Precision@k |   nDCG@k | Recall@k (section)   |
|----:|-----------:|--------------:|---------:|:---------------------|
|   1 |    0.71875 |       0.71875 | 0.71875  |                      |
|   3 |    0.875   |       0.5625  | 0.806085 |                      |
|   5 |    0.90625 |       0.4875  | 0.818065 |                      |
|  10 |    0.90625 |       0.24375 | 0.818065 |                      |

MRR (answer-term): **0.7943** | MRR (section): **unavailable**

### Recall@5 by Query Category

| category         |   Recall@5 |   n |
|:-----------------|-----------:|----:|
| annex_lookup     |     1      |   2 |
| comparison       |     1      |   2 |
| cross_reference  |     0.5    |   2 |
| definition       |     0.8667 |  15 |
| parameter_lookup |     1      |   5 |
| procedure        |     1      |   6 |

## 2. Chunking-Specific Metrics

|   chunk_count |   avg_chunk_size_chars |   min_chunk_size_chars |   max_chunk_size_chars | relevance_density   | boundary_coherence   |
|--------------:|-----------------------:|-----------------------:|-----------------------:|:--------------------|:---------------------|
|          2084 |                709.475 |                     15 |                   1024 |                     |                      |

## 3. End-to-End Generation Metrics

| faithfulness   | answer_relevance   | answer_correctness   |   citation_accuracy |
|:---------------|:-------------------|:---------------------|--------------------:|
|                |                    |                      |                   0 |

## 4. Efficiency / Cost Metrics

|   indexing_time_ms |   avg_retrieval_ms |   avg_generation_ms |   avg_e2e_ms |   index_size_bytes |   llm_calls_per_query |   tokens_per_query |
|-------------------:|-------------------:|--------------------:|-------------:|-------------------:|----------------------:|-------------------:|
|             109763 |            2684.63 |             11395.3 |      14079.9 |        1.46446e+08 |                     1 |            4160.75 |

## Flagged / Unavailable Metrics

| metric                        | reason                                                                        |
|:------------------------------|:------------------------------------------------------------------------------|
| retrieval_section             | not applicable: retrieved chunks carry no section_number (recursive chunking) |
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
| q01  | parameter_lookup |          1 | 0.25  |            nan | I don't have enough information. The provided context does not specify a maximum |
| q02  | procedure        |          1 | 1     |            nan | Based on the provided excerpts from **ISO 15118-3 (pages 44 and 47)**, the SLAC  |
| q03  | definition       |          0 | 0     |            nan | I don't have enough information. The provided context does not define or mention |
| q04  | definition       |          1 | 1     |              0 | According to ISO 15118-1, Section 3.1.2, **ancillary services** are defined as:  |
| q05  | definition       |          1 | 1     |            nan | Based on the provided context, I don't have enough information to define or expl |
| q06  | definition       |          1 | 1     |              0 | According to ISO 15118-1, Section 3.1.8 (page 8), a **battery management system  |
| q07  | definition       |          1 | 1     |              0 | Based on the provided context, **basic signalling** is defined and described as  |
| q08  | definition       |          1 | 1     |            nan | According to ISO 15118-1 (page 9), a certificate is defined as an **"electronic  |
| q09  | definition       |          1 | 1     |              0 | According to **ISO 15118-1:2019, Section 3.1.12**, **"charge"** is defined as: > |
| q10  | definition       |          1 | 1     |              0 | According to ISO 15118-1, Section 3.1.13, a charger is defined as a **"power con |
| q11  | definition       |          1 | 1     |              0 | According to ISO 15118-1, Section 3.1.15, a **communication session** is defined |
| q12  | definition       |          1 | 1     |              0 | According to ISO 15118-1, Section 3.1.16, a **contactor** is defined as an elect |
| q13  | definition       |          1 | 0.5   |            nan | Based on the provided context, I don't have enough information. The term "creden |
| q14  | definition       |          0 | 0     |            nan | Based on the provided context, there is no mention of "demand" or "prognosis." I |
| q15  | definition       |          1 | 1     |              0 | According to ISO 15118-1, Section 3.1.20, **distributed energy resources (DER)** |
| q16  | parameter_lookup |          1 | 0.5   |            nan | Based on the provided context from ISO 15118-2 (page 294), the nominal AC chargi |
| q17  | parameter_lookup |          1 | 0.5   |              0 | Based on the provided context, the only voltage limit defined for the EV is the  |
| q18  | procedure        |          1 | 0.333 |              0 | Based on the provided context, a V2G Communication Session is established by beg |
| q19  | procedure        |          1 | 1     |              0 | Based on the provided excerpts from ISO 15118-3, the matching procedure can be s |
| q20  | procedure        |          1 | 1     |              0 | Based on the provided context, **I don't have enough information** to explain ho |
| q21  | cross_reference  |          1 | 1     |            nan | Based on the provided context, I don't have enough information. The excerpts onl |
| q22  | cross_reference  |          0 | 0     |            nan | I don't have enough information. The provided context does not mention Annex C.  |
| q23  | annex_lookup     |          1 | 1     |            nan | Based on the provided context, Annex A describes different content depending on  |
| q24  | annex_lookup     |          1 | 1     |            nan | Based on the provided context, Annex C describes the **schema definition**. This |
| q25  | comparison       |          1 | 1     |            nan | Based on the provided context from ISO 15118-2, AC and DC charging are different |
| q26  | comparison       |          1 | 0.333 |            nan | Based on the provided context, **I don't have enough information** to compare Pl |
| q27  | parameter_lookup |          1 | 1     |            nan | Based on the provided context from ISO 15118-2, the maximum value for current-re |
| q28  | procedure        |          1 | 1     |            nan | Based on the provided context from ISO 15118-2, the SECC communication setup inv |
| q29  | definition       |          1 | 1     |            nan | According to ISO 15118-2, page 13, EVCC stands for **Electric Vehicle Communicat |
| q30  | definition       |          1 | 1     |            nan | Based on the provided context, there is no explicit definition or explanation of |
| q31  | parameter_lookup |          1 | 1     |              0 | Based on the provided context from ISO 15118-3, the control pilot signal is a bi |
| q32  | procedure        |          1 | 1     |              0 | Based on the provided context, PLC (Power Line Communication) operates using the |
