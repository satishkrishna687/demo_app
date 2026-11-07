# MASTER PROMPT: Generate an M.Tech Phase 1 Project Report

## 1. ROLE AND TASK

Act as an academic technical writer, research assistant, and software engineering documentation specialist experienced in M.Tech Computer Science, Data Science, Artificial Intelligence, Retrieval-Augmented Generation (RAG), information retrieval, and Natural Language Inference (NLI).

Your task is to generate a formal, detailed, academically written **M.Tech Phase 1 Project Report** based on the supplied project summary file, `stage1_1_ppt.md`.

The report must follow the chapter organization, academic writing style, depth of explanation, and documentation conventions described in this prompt.

The supplied project summary is the primary source of truth for the project's scope, objectives, proposed architecture, technologies, algorithms, and expected outcomes.

Do not simply convert each presentation slide into a paragraph. Expand the technical content into a coherent academic report, explaining the motivation, problem, design decisions, methodology, algorithms, implementation plan, and evaluation strategy.

## 2. PROJECT INFORMATION

Use the following project information.

**Project title:** Grounded RAG QA System over ISO 15118

**Subtitle:** Enhancing Retrieval-Augmented Generation with NLI-based Evaluation and Adaptive Hybrid Search

**Academic program:** M.Tech

**Review stage:** Phase 1 / First Review

**Student name:** [STUDENT NAME]

**University:** [UNIVERSITY NAME]

**Department:** [DEPARTMENT NAME]

**Project guide:** [GUIDE NAME]

**Submission date:** [SUBMISSION DATE]

Replace placeholders only when the actual information is supplied. Never invent student details, university details, guide names, signatures, certificates, or institutional approvals.

## 3. PROJECT OVERVIEW

The project aims to develop a completely local Retrieval-Augmented Generation question-answering system for technical documents related to ISO 15118, which specifies communication interfaces used in vehicle-to-grid and electric vehicle charging contexts.

The system should support technical questions involving definitions, procedures, message names, field identifiers, clause references, and structured tables.

The central research problem is that conventional RAG systems may retrieve irrelevant passages, lose relationships between clauses and tables during document chunking, or generate answers that are not adequately supported by retrieved evidence.

The proposed system investigates the combination of:

1. Structured PDF parsing using Docling.
2. Clause-aligned, table-aware document chunking.
3. Dense semantic retrieval using `bge-small-en-v1.5`.
4. Sparse keyword retrieval using BM25.
5. Reciprocal Rank Fusion (RRF) for combining retrieval rankings.
6. Adaptive-k context selection based on score gaps, bounded between 1 and 8 chunks.
7. Local answer generation using `Llama-3.2-3B-Instruct`.
8. Sentence-level Natural Language Inference evaluation using `cross-encoder/nli-deberta-v3-base`.
9. Separate evaluation of retrieval quality and answer faithfulness.
10. A local Streamlit interface for document selection, question answering, and source-passage inspection.

The proposed system will be evaluated against a baseline using hybrid retrieval with fixed top-k selection, where k = 5, and generation without the proposed grounding-validation stage.

All processing is intended to run locally on Apple Silicon/M1 hardware. Treat complete local operation and data privacy as design objectives, not as experimentally verified achievements unless implementation evidence is provided.

## 4. PRIMARY SOURCE AND EVIDENCE RULES

Use `stage1_1_ppt.md` as the primary project source.

Preserve the project's terminology, stated objectives, named models, algorithms, architecture, and experimental plan.

Do not change the project's scope or silently replace its selected technologies with alternatives.

Distinguish explicitly between:

* Established background information.
* Methods proposed for implementation.
* Work already implemented.
* Experiments actually completed.
* Expected results and hypotheses.
* Limitations and future work.

Never claim that an algorithm has been implemented, an experiment has been executed, or a result has been measured unless the supplied materials support that claim.

If implementation evidence is unavailable, write the relevant section as a proposed design, planned implementation, experimental protocol, or expected outcome.

Use clearly marked placeholders such as `[RESULTS TO BE ADDED AFTER EXPERIMENTS]` wherever actual data is required.

Do not fabricate accuracy percentages, latency measurements, hardware benchmarks, dataset sizes, experiment outcomes, citations, publications, page numbers, or bibliographic information.

Do not claim that the system eliminates hallucinations. Describe the NLI component as a mechanism for detecting or flagging potential unsupported statements, subject to its limitations.

Do not reproduce extensive copyrighted ISO 15118 text. Explain technical concepts in original words, use appropriately brief references to clauses where permitted, and distinguish general explanations from verified statements in the licensed source documents.

If the source materials do not provide sufficient information, identify the gap instead of guessing.

## 5. REQUIRED REPORT STRUCTURE

Create the report using the following seven main chapters and preliminary pages.

### Preliminary Pages

Generate the following in order:

1. Title page.
2. Certificate page template.
3. Abstract.
4. Acknowledgement template.
5. Table of contents.
6. List of figures.
7. List of tables.
8. List of abbreviations and acronyms.

Use placeholders for official signatures, seals, approvals, and personal details. Clearly label institutional documents as templates requiring authorized completion.

The abstract should explain the domain, research problem, objectives, proposed methodology, baseline comparison, evaluation strategy, and expected contribution. Do not present expected results as completed results.

### Chapter 1: Introduction

Include:

1.1 Background

1.2 Retrieval-Augmented Generation

1.3 Technical Challenges in Engineering Standards

1.4 Motivation

1.5 Problem Statement

1.6 Objectives

1.7 Scope of the Project

1.8 Proposed Solution

1.9 Contributions of the Project

1.10 Organization of the Report

Explain why technical standards are difficult to process using conventional LLMs. Discuss long documents, exact technical identifiers, structured tables, clause dependencies, retrieval errors, and unsupported generated statements.

Explain the difference between retrieval failure and generation failure.

Define the project's specific problem and scope. Explain what the proposed system will do and what it will not attempt to do.

### Chapter 2: Literature Survey

Include:

2.1 Introduction

2.2 Retrieval-Augmented Generation

2.3 Dense Semantic Retrieval

2.4 Sparse Keyword Retrieval and BM25

2.5 Hybrid Information Retrieval

2.6 Reciprocal Rank Fusion

2.7 Document Parsing and Structure-Aware Chunking

2.8 Adaptive Context Selection

2.9 Natural Language Inference for Factual Consistency Evaluation

2.10 Evaluation of RAG Systems

2.11 Comparative Summary of Existing Approaches

2.12 Research Gap

2.13 Chapter Summary

Discuss the following foundational references identified in the project summary:

* Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
* Robertson et al. (2009), The Probabilistic Relevance Framework: BM25 and Beyond.
* Cormack et al. (2009), Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods.
* Honovich et al. (2022), TRUE: Re-evaluating Factual Consistency Evaluation.

Explain each work's relevance to this project, the concept it contributes, its relationship to the proposed architecture, and any limitations relevant to the project.

Verify all bibliographic details against available trustworthy references before presenting them as confirmed. If external research is unavailable, mark unverified details for verification.

Do not invent additional publications or misattribute findings to existing papers.

Include a comparative table with columns such as:

* Approach or publication.
* Main technique.
* Contribution.
* Limitation relevant to this project.
* Relevance to the proposed system.

Conclude by identifying the research gap addressed by the combination of structure-aware parsing, hybrid retrieval, adaptive-k selection, and independent NLI-based evaluation.

### Chapter 3: System Requirements Specification

Include:

3.1 System Overview

3.2 User Requirements

3.3 Functional Requirements

3.4 Non-Functional Requirements

3.5 Hardware Requirements

3.6 Software Requirements

3.7 System Constraints

3.8 Assumptions and Dependencies

3.9 Test Case Specification

3.10 Requirements Traceability Matrix

Document the functional requirements for:

* Loading supported ISO 15118 PDF documents.
* Parsing document text, headings, clauses, and tables.
* Creating clause-aligned chunks with appropriate metadata.
* Building dense and BM25 indexes.
* Accepting user queries.
* Performing hybrid retrieval.
* Fusing rankings using RRF.
* Selecting context using adaptive-k.
* Generating answers with the local LLM.
* Evaluating generated statements using NLI.
* Displaying answers and source passages.
* Recording evaluation results for baseline and proposed methods.

Define non-functional requirements relating to local execution, privacy, usability, reproducibility, traceability, performance measurement, and maintainability.

Distinguish measurable requirements from qualitative goals. Do not invent numerical performance thresholds unless provided or explicitly labelled as proposed acceptance criteria.

Include a requirements table with unique IDs such as FR-01, FR-02, NFR-01, and NFR-02.

Include a test-case table containing test ID, requirement ID, test scenario, input, expected outcome, and status.

Include a traceability matrix connecting requirements, system modules, and planned test cases.

Mark unexecuted tests as "Not yet executed" rather than "Pass".

### Chapter 4: Proposed Methodology

Include:

4.1 Overall System Architecture

4.2 System Workflow

4.3 Document Ingestion

4.4 PDF Parsing Using Docling

4.5 Clause-Aligned and Table-Aware Chunking

4.6 Metadata Enrichment

4.7 Dense Vector Indexing

4.8 BM25 Keyword Indexing

4.9 Hybrid Retrieval

4.10 Reciprocal Rank Fusion

4.11 Adaptive-k Selection

4.12 Answer Generation Using the Local LLM

4.13 NLI-Based Grounding Evaluation

4.14 Baseline and Proposed Pipelines

4.15 Evaluation Metrics

4.16 Algorithms and Pseudocode

4.17 Methodology Summary

Explain the full workflow step by step.

**Document ingestion:** Explain how the source PDF is parsed into structured text and tables. Explain the importance of retaining clause identifiers and source-document metadata.

**Chunking:** Describe clause-aligned chunking and the goal of preserving complete tables. The proposed text chunk limit is approximately 512 tokens; this limit does not apply to tables, which should remain intact where feasible. Explain how oversized tables may need a separately documented strategy without silently splitting away important context.

**Metadata:** Describe fields such as `clause_number`, `source_document`, and `chunk_type`.

**Dense retrieval:** Explain semantic similarity and the use of `bge-small-en-v1.5`.

**Sparse retrieval:** Explain BM25 and why exact identifiers, message names, acronyms, and field names benefit from lexical matching.

**Hybrid retrieval:** Explain how the dense and sparse rankings are combined.

**RRF:** Provide the correct conceptual equation and explain how rank-based fusion works. Define any parameters used and identify tunable values that still need to be selected.

**Adaptive-k:** Explain the proposed selection of a variable number of chunks based on score gaps, bounded between 1 and 8 chunks. Describe the baseline fixed-k = 5 approach. If the exact score-gap formula is not defined in the supplied materials, propose a clearly labelled candidate algorithm and explain that it requires experimental validation.

**Generation:** Explain how the query and retrieved evidence are passed to `Llama-3.2-3B-Instruct`. Specify that answers should remain grounded in retrieved passages and identify source clauses.

**NLI evaluation:** Explain how `cross-encoder/nli-deberta-v3-base` can evaluate whether individual answer statements are entailed by, contradicted by, or unsupported by the retrieved evidence. Clarify the model's label mapping, threshold selection, and aggregation rules as implementation details requiring validation.

Explain that an NLI score is an estimate, not proof of correctness.

Include readable pseudocode for the retrieval, fusion, adaptive-k, generation, and evaluation steps.

Provide a system architecture diagram in Mermaid syntax showing:

PDF documents → Docling parser → clause-aligned/table-aware chunker → metadata enrichment → dense and BM25 indexes → hybrid retrieval → RRF → adaptive-k → local LLM → NLI evaluation → grounded answer with citations.

Show the baseline path separately, including fixed-k = 5 and generation without the proposed NLI grounding filter.

### Chapter 5: Implementation Details

Include:

5.1 Development Environment

5.2 Software and Tools

5.3 Project Directory Structure

5.4 Document Parsing Module

5.5 Chunking and Metadata Module

5.6 Indexing Module

5.7 Hybrid Retrieval Module

5.8 RRF and Adaptive-k Module

5.9 Answer Generation Module

5.10 NLI Evaluation Module

5.11 Streamlit User Interface

5.12 Integration of Modules

5.13 Implementation Status

Describe the intended implementation using Python, Docling, Chroma, BM25, Streamlit, the specified embedding model, the specified local LLM, and the specified NLI model.

Explain the purpose, inputs, outputs, and interactions of each module.

Include a proposed directory tree, module-level responsibilities, and pseudocode or short illustrative code only when useful.

Do not invent an existing directory structure or claim that code exists unless the supplied materials establish that it does.

Provide an implementation status table with columns:

* Component.
* Planned technology.
* Intended responsibility.
* Status.
* Evidence or outstanding work.

If actual source code, dependency versions, or execution logs are not supplied, mark these items as "To be confirmed" or "Not yet implemented".

### Chapter 6: Intermediate Results and Discussion

Include:

6.1 Experimental Objectives

6.2 Experimental Environment

6.3 Dataset and Evaluation Query Design

6.4 Golden Question-Answer Dataset

6.5 Baseline Configuration

6.6 Proposed Configuration

6.7 Evaluation Metrics

6.8 Retrieval Results

6.9 Adaptive-k Analysis

6.10 NLI and Hallucination Analysis

6.11 Context Size and Latency Analysis

6.12 Comparative Discussion

6.13 Limitations

6.14 Chapter Summary

Design an experiment that compares:

**Baseline:** Hybrid BM25 + dense retrieval, RRF ranking, fixed top-k = 5, local LLM generation, and no proposed post-generation NLI grounding filter.

**Proposed system:** The same general retrieval foundation with adaptive-k selection bounded between 1 and 8 chunks, local LLM generation, and NLI-based grounding evaluation.

Describe a planned evaluation dataset of 50–100 manually verified question-answer-clause triples. Include definitional, procedural, cross-reference, and tabular question types.

Define and explain appropriate metrics, including:

* Context Recall or an explicitly defined retrieval-coverage measure.
* Answer faithfulness or a carefully defined hallucination rate.
* Answer correctness against the golden answers.
* Citation or clause-reference correctness.
* Number of retrieved chunks.
* Context token count.
* Query latency.
* NLI processing overhead.

Specify metric formulas where their definitions are sufficiently established. State clearly how the evaluation set, annotations, and denominators will be constructed.

Distinguish retrieval metrics from generation metrics. Explain why a correct answer can be generated from poor retrieval by chance and why good retrieval can still be followed by a hallucinated answer.

**CRITICAL RESULTS RULE:** No experimental results have been supplied in the project summary. Do not create fictional tables containing measured values, claim improvements, or write that the proposed method outperformed the baseline.

Instead, provide empty result-table templates, experimental procedures, interpretation criteria, and placeholders for future measurements.

For each planned result, explain what it would indicate if the proposed system performs better, similarly, or worse than the baseline.

Discuss limitations such as parsing failures, broken table relationships, incorrect retrieval, false NLI classifications, model resource constraints, and limited evaluation coverage.

### Chapter 7: Conclusion and Future Work

Include:

7.1 Conclusion

7.2 Summary of Proposed Contributions

7.3 Current Limitations

7.4 Future Work

Summarize the problem, proposed architecture, selected algorithms, planned baseline comparison, and intended evaluation strategy.

Clearly distinguish the design contribution from any implementation or experimental contribution that has actually been completed.

Describe future work such as implementation completion, evaluation with a verified query set, tuning adaptive-k, testing alternative chunking methods, improving citation accuracy, and measuring local performance.

Do not claim successful results that have not been measured.

### References

Include a properly formatted academic reference list.

Use the four foundational references named in the project summary as starting points.

Add further references only when they are verifiable and relevant. Ensure that in-text citations and bibliography entries correspond.

Use a consistent IEEE-style numeric citation format, such as [1], [2], and [3], throughout the report.

Do not retain presentation-specific markers such as `[cite: 1]` or `[cite: 2]`. Convert verified citations into normal academic citations.

Do not fabricate DOIs, publication venues, volume numbers, page ranges, or URLs.

## 6. ACADEMIC WRITING STYLE

Write in formal academic English suitable for an M.Tech project report.

Use connected paragraphs to explain concepts, followed by lists, tables, algorithms, or diagrams where appropriate.

Avoid writing the entire report as bullet points.

For every major algorithm or design decision, explain:

1. What it is.
2. Why it is needed.
3. How it works.
4. How it fits into this project.
5. What its limitations are.
6. How it will be evaluated.

Introduce technical terminology before using abbreviations repeatedly.

Use consistent names for modules, models, algorithms, metrics, and system components.

Use numbered chapters, sections, subsections, tables, figures, and equations.

Write informative figure captions and table captions.

Include transitions between sections so the report reads as one coherent dissertation.

Avoid repetition, generic filler, unsupported claims, and excessively complicated language.

Do not copy text from the sample report. Use it only as a guide for academic structure and presentation.

## 7. FIGURES, TABLES, AND EQUATIONS

Provide meaningful diagrams and tables throughout the report.

At a minimum, include:

1. Overall system architecture.
2. Document ingestion and chunking workflow.
3. Baseline versus proposed retrieval pipeline.
4. Adaptive-k selection workflow.
5. NLI-based evaluation workflow.
6. Functional and non-functional requirements table.
7. Requirements traceability matrix.
8. Literature comparison table.
9. Test-case specification table.
10. Software and hardware environment table.
11. Module implementation status table.
12. Golden evaluation dataset template.
13. Baseline versus proposed experiment configuration table.
14. Experimental results template.
15. Limitations and future work table, if useful.

Use Mermaid for diagrams when supported. If Mermaid rendering is unavailable, provide clean text-based diagrams and figure descriptions.

Use Markdown tables with clear headings.

Use LaTeX for mathematical equations when useful, and define every symbol.

Generate a list of figures and tables with placeholder page numbers if final pagination is unavailable. Never invent precise page numbers.

## 8. EXPECTED LENGTH AND LEVEL OF DETAIL

Target approximately 30–45 pages after formatting, excluding any institutionally required appendices, if the available evidence supports that length.

Prioritize technical depth and completeness over page count. Do not artificially inflate the report with repeated explanations.

Every chapter must have a clear purpose and sufficient technical detail for an academic project review.

The report should be substantially more detailed than the supplied presentation summary but must remain faithful to its scope and evidence.

## 9. OUTPUT FORMAT

Generate the report as a single Markdown document.

Use a logical filename such as:

`Grounded_RAG_QA_ISO15118_Phase1_Report.md`

Use this formatting convention:

* `#` for the report title.
* `##` for main chapters.
* `###` for numbered subsections.
* Markdown tables for requirements, comparisons, configurations, and results.
* Fenced code blocks for pseudocode and Mermaid diagrams.
* LaTeX for equations where appropriate.
* Academic numeric citations such as `[1]`.

Include a Markdown table of contents with links to the report sections if supported.

If the model cannot generate the entire report in one response, generate it in sequential parts. Keep a stable chapter structure, consistent terminology, and a cumulative list of figures, tables, abbreviations, and references.

Do not omit chapters merely because of output-length restrictions. Stop only at a complete section boundary and state exactly which chapter should be generated next.

## 10. FINAL QUALITY CHECK

Before completing the report, verify that:

* All seven chapters are present.
* The report is about the specified ISO 15118 RAG project, not the sample report's unrelated subject.
* The baseline and proposed approaches are clearly distinguished.
* Dense retrieval, BM25, RRF, adaptive-k, local LLM generation, and NLI evaluation are explained.
* Requirements map to modules and planned test cases.
* The experimental design separates retrieval errors from generation errors.
* All expected results are identified as expectations, not measured facts.
* No implementation status, benchmark, accuracy, or performance result has been fabricated.
* All references are cited consistently and are either verified or explicitly marked for verification.
* The report contains appropriate diagrams, tables, equations, and figure captions.
* Placeholders remain for missing personal, institutional, implementation, and experimental information.
* The report is suitable for review by an M.Tech project supervisor.

Now generate the complete Phase 1 report based on the supplied `stage1_1_ppt.md` project summary, following all the instructions above.
