# Evaluation-Driven RAG Pipeline Methodology — Shared Across All 9 Groups

**This is the single most important rule of this capstone:** no design decision may be justified by "it seemed to work better" or "this is what the tutorial used." Every choice must be backed by a number, measured on your own evaluation set, presented in a table, with a winner picked in writing.

Analogy: choosing a chunking strategy without measuring it is like buying a car without a test drive because you liked the color. You might get lucky. A real ML engineer doesn't get lucky — they measure.

> **Note on metric naming:** the required metric set is R@1, R@3, **MRR@10** (Mean Reciprocal Rank, top 10), and DCG@3/NDCG@3 — confirmed. ("10 MCC" in early drafts was a mishearing of MRR@10; Matthews Correlation Coefficient is a classification metric and never belonged in this ranked-retrieval set.)

---

## Part A — Metrics Glossary (know what you're measuring before you measure it)

### Retrieval-stage metrics (measure: "did we find the right chunk?")
| Metric | Plain-English definition | How to compute |
|---|---|---|
| **R@1 (Recall@1)** | Did the correct source chunk appear as the #1 retrieved result? | For each eval query: 1 if correct chunk is rank 1, else 0. Average across all queries. |
| **R@3 (Recall@3)** | Did the correct source chunk appear anywhere in the top 3 retrieved results? | 1 if correct chunk is in ranks 1–3, else 0. Average across all queries. |
| **MRR@10 (Mean Reciprocal Rank)** | If the correct chunk is found within the top 10, how *high* was it ranked? Rewards ranking it 1st over ranking it 8th, unlike plain Recall. | For each query: `1 / rank_of_first_correct_chunk` (0 if not found in top 10). Average across all queries. |
| **DCG@3 / NDCG@3 (Discounted Cumulative Gain)** | Like Recall, but rewards *graded* relevance (a highly relevant chunk at rank 1 scores more than a marginally relevant one at rank 1) and penalizes relevant results appearing lower in the list. NDCG normalizes against the best-possible ranking so scores compare cleanly across queries (0 to 1). | Requires you to grade each retrieved chunk 0/1/2/3 for relevance, not just yes/no. Use the standard DCG formula: `DCG@3 = sum(relevance_i / log2(rank_i + 1))` for i=1..3. NDCG@3 = DCG@3 / ideal-DCG@3. |

### Generation-stage metrics (measure: "was the final answer any good?")
| Framework | Metrics to report | What each one catches |
|---|---|---|
| **RAGAS** | Faithfulness, Answer Relevancy, Context Precision, Context Recall | Faithfulness = is the answer actually supported by the retrieved text (catches hallucination). Answer Relevancy = does the answer address what was asked. Context Precision/Recall = did retrieval fetch the right stuff, and only the right stuff. |
| **DeepEval** | G-Eval (custom LLM-as-judge criteria), Hallucination, Answer Relevancy, Faithfulness | Use as a **second judge** alongside RAGAS — if RAGAS and DeepEval agree, you can trust the score more. If they disagree sharply, that's a signal to investigate, not to pick whichever number looks better. |

---

## Part B — Step 1: Build Your Evaluation Set BEFORE Touching Any Design Decision

Every group must build a labeled evaluation set of **at least 20 real questions** before optimizing anything, each with:
- The question text
- The ground-truth source document (and page/chunk if possible)
- A short ideal answer (for RAGAS/DeepEval generation scoring)

This is your ruler. You cannot pick a "best" chunking strategy or embedding model without it — that's the whole point of this methodology. Each group's `REQUIREMENT.md` includes 5–8 seed questions from its own real documents; expand these to 20+ by writing more questions against the actual PDFs/data in your folder.

---

## Part C — The Ablation Matrix: One Required Experiment Per Pipeline Stage

For **each** stage below, you must: (1) implement at least 2 real alternatives, (2) run your full eval set through each, holding everything else constant, (3) report retrieval metrics in a table, (4) pick a winner and justify it **in writing, citing the actual numbers**.

### Stage 1 — Parsing strategy
Compare at least 2: `pypdf` vs `pdfplumber` vs `PyMuPDF (fitz)`.
Report: % of pages with clean extracted text (no garbled/missing characters), table-extraction success if relevant to your documents, and downstream **R@3** when each parser's output is fed into the same chunking/embedding pipeline.

| Parser | Clean text % | R@3 (downstream) |
|---|---|---|
| pypdf | | |
| pdfplumber | | |
| PyMuPDF | | |

### Stage 2 — Chunking strategy
Compare at least 2: fixed-size (no overlap) vs fixed-size with overlap vs recursive/sentence-aware vs semantic chunking. Vary chunk size (e.g., 300 / 500 / 800 tokens) and overlap (e.g., 0 / 50 / 100).

| Strategy | Chunk size | Overlap | R@1 | R@3 | MRR@10 | NDCG@3 |
|---|---|---|---|---|---|---|

### Stage 3 — Embedding model
Compare at least 2: `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `text-embedding-3-small`, `multi-qa-mpnet-base-dot-v1`, or similar.

| Embedding model | Dimensions | R@1 | R@3 | MRR@10 | NDCG@3 | Latency (ms/query) |
|---|---|---|---|---|---|---|

### Stage 4 — Vector database
Compare at least 2: ChromaDB vs FAISS (or others). **Note:** retrieval *quality* (R@3 etc.) usually won't change much between vector DBs for the same embeddings — the real justification here is indexing speed, persistence, metadata-filter support, and ease of deployment. Still report R@3 to confirm parity, then justify on the operational axes.

| Vector DB | R@3 (parity check) | Index build time | Metadata filtering? | Persistence? |
|---|---|---|---|---|

### Stage 5 — Retrieval mode: dense vs sparse vs hybrid
Compare all 3: dense-only (embeddings), sparse-only (BM25), hybrid. **Critical:** break results out by query type — exact-term/keyword-style queries vs paraphrased/semantic queries. This breakdown is the actual proof for whether hybrid is worth the added complexity.

| Mode | R@1 (all) | R@3 (all) | R@3 (keyword queries) | R@3 (semantic queries) | MRR@10 | NDCG@3 |
|---|---|---|---|---|---|---|
| Dense only | | | | | | |
| Sparse (BM25) only | | | | | | |
| Hybrid | | | | | | |

### Stage 6 — Hybrid merge method and weighting (only if Stage 5 picked hybrid)
Compare: Reciprocal Rank Fusion (RRF, `score = Σ 1/(k + rank)`, k≈60) vs weighted linear combination (`score = α·dense_norm + (1-α)·sparse_norm`). If weighted, sweep at least 3 α values.

| Merge method | α (if weighted) | R@3 | MRR@10 | NDCG@3 |
|---|---|---|---|---|

State explicitly: **what % weight did you land on, and why** (cite the row that won).

### Stage 7 — Reranking
Compare: no reranker vs cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) applied to top-20 retrieved, reranked down to top-3/5.

| Config | R@3 | NDCG@3 | Added latency (ms) |
|---|---|---|---|
| No reranker | | | 0 |
| + Cross-encoder rerank | | | |

Justify whether the accuracy gain is worth the latency cost for this bot's use case.

### Stage 8 — LLM for generation
Compare at least 2 real LLMs (e.g. Gemini 2.5/2.0 Flash, GPT-4o-mini, Claude Haiku, a local Ollama model). **Fix retrieval — vary only the generator** so you isolate the LLM's contribution.

| LLM | Faithfulness (RAGAS) | Answer Relevancy (RAGAS) | Context Precision (RAGAS) | Hallucination (DeepEval) | G-Eval score (DeepEval) | Cost/query | Latency |
|---|---|---|---|---|---|---|---|

---

## Part D — Final Deliverable: The Evaluation Report

Every group submits a report (notebook or markdown) containing:
1. **All 8 ablation tables above, filled with real measured numbers.** A placeholder like "we ran out of time" scores zero for that stage — this is non-negotiable, it's the whole assignment.
2. **Final chosen configuration**, with a synthesis paragraph that cites the numbers from every table above, e.g.: *"We chose recursive chunking at 500 tokens/50 overlap (R@3 = 0.81 vs. 0.68 for fixed-size no-overlap); hybrid retrieval merged via RRF (R@3 = 0.84 vs. 0.79 dense-only, 0.62 sparse-only); and Gemini 2.0 Flash for generation (Faithfulness = 0.91, the highest of the 3 candidates, at 1/4 the latency of the next-best option)."*
3. **End-to-end RAGAS + DeepEval scores on the final pipeline** — this is the headline number for the whole project.
4. **A short "what we'd try next" section** — tie back to the course's M06 "when RAG fails" material: what's the next failure mode you'd hunt for if you had another week?

---

## Part E — Grading Weight (applies to all 9 groups)

| Component | Weight |
|---|---|
| Evaluation rigor — ablation tables complete, real numbers, clear winner justification per stage | **40%** |
| Correctness on this group's specific acceptance-test questions (in `REQUIREMENT.md`) | 30% |
| End-to-end RAGAS + DeepEval scores on the final chosen pipeline | 20% |
| Code quality / app usability | 10% |

A chatbot that "just works" with no evaluation evidence caps at the 30% correctness line — it cannot pass this capstone on functionality alone.
