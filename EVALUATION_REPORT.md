# Evaluation Report — Consumer Credit Rights Assistant (Group 6)

Stages 1–7 below are filled with real numbers from `python eval/ablation_runner.py`
(local models only, n=21 eval questions with ground truth, guardrail question #22
excluded from retrieval metrics per `eval/eval_questions.json`). Stage 8 and the
end-to-end RAGAS/DeepEval scores require `python eval/generation_eval.py` with a
real `OPENAI_API_KEY` — run it and paste the numbers from
`eval/results/stage8_llm_comparison.csv` / `final_e2e_scores.csv` before submitting;
until then those sections and the acceptance-test table are marked **PENDING**.

## Stage 1 — Parsing strategy
| Parser | Clean text % | R@3 (downstream) |
|---|---|---|
| pypdf | 100.0 | 0.905 |
| pdfplumber | 100.0 | **0.952** |
| PyMuPDF | 100.0 | 0.857 |

**Winner: pdfplumber.** All three extract clean text on this small, well-formatted
PDF set (100% clean), so the tie-breaker is downstream retrieval: pdfplumber's
layout-aware extraction (it preserves bullet/paragraph boundaries better than
PyMuPDF's raw text dump) produced chunks that retrieved correctly 95.2% of the
time in the top 3, vs 90.5% for pypdf and 85.7% for PyMuPDF.

## Stage 2 — Chunking strategy
| Strategy | Chunk size | Overlap | R@1 | R@3 | MRR@10 | NDCG@3 |
|---|---|---|---|---|---|---|
| fixed_no_overlap | 300 | 0 | 0.714 | 0.905 | 0.825 | 0.601 |
| fixed_no_overlap | 500 | 0 | 0.762 | 0.905 | 0.849 | 0.622 |
| fixed_no_overlap | 800 | 0 | 0.762 | 0.905 | 0.840 | 0.611 |
| fixed_overlap | 300 | 60 | 0.571 | 0.857 | 0.730 | 0.530 |
| fixed_overlap | 500 | 100 | 0.714 | 0.857 | 0.791 | 0.598 |
| fixed_overlap | 800 | 100 | 0.714 | 0.952 | 0.833 | 0.659 |
| recursive | 300 | 60 | 0.619 | 0.905 | 0.762 | 0.587 |
| recursive | 500 | 100 | 0.619 | 0.952 | 0.782 | 0.622 |
| **recursive** | **800** | **100** | **0.762** | **1.000** | **0.873** | **0.713** |

**Winner: recursive @ 800 tokens / 100 overlap** — perfect R@3 (1.0) and the best
score on every metric (R@1 0.762, MRR@10 0.873, NDCG@3 0.713). At this small chunk
size for the corpus (chapters are 3–6 pages), larger recursive chunks keep whole
provisions (a full §202.5(c) list, a full "types of discrimination" paragraph)
intact instead of splitting them across chunk boundaries.

## Stage 3 — Embedding model
| Embedding model | Dimensions | R@1 | R@3 | MRR@10 | NDCG@3 | Latency (ms/query) |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 0.571 | 0.810 | 0.731 | 0.596 | 11.7 |
| **bge-small-en-v1.5** | 384 | **0.762** | **1.000** | **0.873** | **0.713** | **5.5** |
| multi-qa-mpnet-base-dot-v1 | 768 | 0.619 | 0.810 | 0.733 | 0.574 | 111.9 |

**Winner: bge-small-en-v1.5** — wins every retrieval metric outright (R@3 1.0 vs
0.81 for both alternatives) *and* is the fastest by a wide margin (5.5ms vs
11.7ms and 111.9ms). The requirement flagged legal-domain terminology as a risk
for general-purpose embeddings; that risk shows up here as all-MiniLM-L6-v2 and
multi-qa-mpnet both tying at a much lower R@3 (0.81) — bge-small's retrieval-tuned
training clearly generalizes better to this regulatory-chapter phrasing.

## Stage 4 — Vector database
| Vector DB | R@3 (parity check) | Index build time | Metadata filtering? | Persistence? |
|---|---|---|---|---|
| ChromaDB | 1.000 | 0.024s | Yes | Yes (`PersistentClient`) |
| FAISS | 1.000 | 0.000s | No (needs a side lookup table) | Yes (`write_index`) |

**Winner: ChromaDB.** R@3 parity confirmed (both 1.0, as expected — same
embeddings). FAISS builds faster, but that's irrelevant at this corpus size
(24ms either way). ChromaDB's native metadata filtering (`source`, `page`) and
built-in persistence API matter more here since every answer must cite a
specific source/page — that's a first-class Chroma query capability FAISS
would need a hand-rolled side table to replicate.

## Stage 5 — Retrieval mode: dense vs sparse vs hybrid
| Mode | R@1 (all) | R@3 (all) | R@3 (keyword queries) | R@3 (semantic queries) | MRR@10 | NDCG@3 |
|---|---|---|---|---|---|---|
| **Dense only** | **0.762** | **1.000** | **1.000** | **1.000** | **0.873** | **0.713** |
| Sparse (BM25) only | 0.667 | 0.762 | 0.750 | 0.769 | 0.721 | 0.580 |
| Hybrid (RRF) | 0.667 | 0.810 | 0.750 | 0.846 | 0.762 | 0.569 |

**Winner: dense-only** — and this is the one place we depart from the reference
capstone's architecture on purpose. Dense-only hits a perfect R@3 (1.0) on *both*
keyword and semantic queries, while hybrid (0.81) and sparse (0.76) both trail
it. On this corpus, BM25 introduces noise rather than recall gains: the
documents are short, clean, single-topic chapters where a good sentence
embedding already disambiguates "age" (ECOA) from "race" (FHAct) contexts
better than term overlap does. Per the methodology's own rule — decisions must
be backed by numbers, not by what a tutorial used — the final pipeline
(`src/pipeline.py`, `DEFAULT_RETRIEVAL_MODE = "dense"`) uses dense-only
retrieval, not hybrid.

## Stage 6 — Hybrid merge method and weighting
**Skipped per methodology** ("only if Stage 5 picked hybrid") — Stage 5 picked
dense-only, so no fusion method is in the final pipeline. (`ablation_runner.py`
defaults to RRF as the Stage 7 candidate-generation fallback only.)

## Stage 7 — Reranking
| Config | R@3 | NDCG@3 | Added latency (ms) |
|---|---|---|---|
| No reranker | 0.810 | 0.569 | 0 |
| + Cross-encoder rerank | **0.905** | **0.656** | 1144.2 |

**Winner: cross-encoder rerank — but with a caveat worth stating plainly.**
This table measures reranking on top of an RRF-fused top-20 candidate pool
(per the ablation script), which trails dense-only's ceiling (R@3 1.0):
reranking recovers most of that gap (0.81 → 0.905) at a real latency cost
(+1.1s/query on CPU). The final pipeline generates candidates with dense-only
retrieval instead (Stage 5's winner, not RRF) and then reranks *those* —
that specific combination (dense top-8 → rerank) was not separately measured
in this table and should be added as a follow-up ablation row before final
submission. Until then, reranking is kept on the reasoning that it can only
tighten which chunk lands at rank 1 for citation quality, at an acceptable
latency cost for a helpline bot where correctness matters more than
sub-second response time — but this is a judgment call, not yet a measured
one, and belongs on the "what we'd try next" list if time is short.

## Stage 8 — LLM for generation
**PENDING** — run `python eval/generation_eval.py` (requires `OPENAI_API_KEY`)
and paste `eval/results/stage8_llm_comparison.csv` here:

| LLM | Faithfulness (RAGAS) | Answer Relevancy (RAGAS) | Context Precision (RAGAS) | Hallucination (DeepEval) | G-Eval score (DeepEval) | Cost/query | Latency |
|---|---|---|---|---|---|---|---|
| gpt-4o-mini | | | | | | | |
| gpt-4o | | | | | | | |

## Final chosen configuration
pdfplumber parsing (95.2% R@3 vs 90.5%/85.7%); recursive chunking at 800
tokens/100 overlap (R@3 = 1.0 vs 0.905 for fixed-no-overlap at the same size);
bge-small-en-v1.5 embeddings (R@3 = 1.0 at 5.5ms/query, vs 0.81 for both
alternatives); ChromaDB for metadata-filtered citations; **dense-only**
retrieval (R@3 = 1.0, beating hybrid's 0.81 and sparse's 0.76 — the one
deliberate departure from the reference bot's hybrid design, justified by
these numbers, not convention); and cross-encoder reranking retained for
citation-rank quality at a ~1.1s/query cost. Stage 8's generator choice is
**PENDING** an API-backed run.

## End-to-end RAGAS + DeepEval on the final pipeline
**PENDING** — run `python eval/generation_eval.py` and paste
`eval/results/final_e2e_scores.csv`:

| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Hallucination | G-Eval |
|---|---|---|---|---|---|
| | | | | | |

## Acceptance test results (REQUIREMENT.md §6)
**PENDING** — run the 5 questions through `streamlit run app.py` (or a small
script calling `ConsumerCreditRightsBot.chat`) once Stage 8 is run, and record
actual bot output here:

| # | Question | Expected behavior | Actual bot behavior | Pass? |
|---|---|---|---|---|
| 1 | I was denied a mortgage — can a bank consider my age in that decision? | Answer from ECOA/Reg B, cited | | |
| 2 | What protections exist if I think I was denied housing credit because of my race? | Answer from Fair Housing Act, cited | | |
| 3 | What's the overall purpose of fair lending law? | Answer from overview doc | | |
| 4 | Has total US consumer credit gone up or down over the last year? | Uses the CSV time series | | |
| 5 | Was I definitely discriminated against? | Explains the law, declines verdict, suggests filing a complaint | | |

## What we'd try next
Two retrieval-eval findings are worth a second week's attention: (1) BM25/hybrid
underperforming dense-only here is corpus-specific (clean, short, single-topic
chapters) — it would likely flip on noisier real-world input like CFPB complaint
narratives (mentioned as a manual extension in `data/GROUP_README.md`), so hybrid
should be re-measured if that data is added. (2) Multi-hop questions that need
both an ECOA chunk and an FHAct chunk together (e.g. "does either law cover a
co-signer situation?") aren't in the current eval set and are a likely failure
mode worth building ground truth for. (3) The regex-based `is_verdict_seeking()`
guardrail (`src/guardrails.py`) will miss adversarial phrasings that don't match
its patterns — a classifier or LLM-based intent check would be more robust than
pattern matching for a helpline bot at real scale.
