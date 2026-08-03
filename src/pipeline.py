"""
Final assembled pipeline for the Consumer Credit Rights Assistant.

parse (PyMuPDF) -> chunk (recursive, 500/80) -> hybrid index (dense bge-small + sparse BM25)
-> RRF fuse -> cross-encoder rerank -> grounded + law-routed + cited generation
-> groundedness guardrail + no-legal-verdict guardrail -> conversational memory.

The retrieval-stage choices (parser, chunk size/overlap, embedding model, fusion method,
reranker) are the *winners* from eval/ablation_runner.py — see EVALUATION_REPORT.md for the
numbers that justify each one. Update the constants below if a re-run picks a different winner.
"""

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import bm25s
from Stemmer import Stemmer
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from langchain_openai import ChatOpenAI

from .parsing import parse_with_pymupdf, csv_to_narrative_chunks
from .chunking import chunk_recursive
from .guardrails import is_verdict_seeking, VERDICT_INSTRUCTION

# Winning config from Stages 1-7, measured in EVALUATION_REPORT.md (eval/results/*.csv):
# recursive chunking @ 800/100 (R@3=1.0), bge-small-en-v1.5 (R@3=1.0, 5.5ms/query),
# DENSE-ONLY retrieval beat hybrid on this small, clean corpus (R@3 1.0 vs 0.81 hybrid vs
# 0.76 sparse) -- reranking still applied since it improved the RRF candidate pool's R@3
# (0.81 -> 0.905), but the primary retrieval signal is dense, not hybrid.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RETRIEVAL_MODE = "dense"
DEFAULT_MIN_RERANK_SCORE = -9.5

ANSWER_PROMPT = '''You are a careful assistant for a consumer credit-rights helpline. Answer ONLY
using the numbered sources below, which are chapters of the Federal Reserve's Consumer
Compliance Handbook plus a Federal Reserve consumer-credit data series.

Rules:
- Cite sources inline like [1], [2] after every claim they support.
- Name which specific law applies (Equal Credit Opportunity Act / Regulation B for credit-decision
  discrimination; Fair Housing Act for housing- or mortgage-related lending discrimination) so the
  reader knows which protection covers their situation.
- If the sources don't contain the answer, say so plainly -- never invent a legal protection.
{extra_instruction}

Sources:
{sources}

Conversation so far:
{history}

Question: {question}

Answer (with inline citations):'''

CONDENSE_PROMPT = '''Given the recent conversation and a follow-up question, decide whether the
follow-up depends on the conversation (uses a pronoun or implicit reference like "it", "that",
"those", "the previous one") to make sense.

- If it DOES depend on the conversation, rewrite it as a standalone question that replaces the
  reference with the actual thing it refers to.
- If it does NOT depend on the conversation -- including if it's simply a new question on a
  different topic -- return it EXACTLY UNCHANGED.

Conversation:
{history}

Follow-up question: {question}

Standalone question:'''


def load_all_documents(data_dir: str) -> list[dict]:
    data_dir = Path(data_dir)
    pages = []
    for pdf in sorted(data_dir.glob("*.pdf")):
        pages.extend(parse_with_pymupdf(str(pdf)))
    for csv in sorted(data_dir.glob("*.csv")):
        pages.extend(csv_to_narrative_chunks(str(csv)))
    return pages


class HybridIndex:
    """Dense (Chroma + bge-small) + sparse (BM25) index, fused with RRF at query time."""

    def __init__(self, embedding_model=DEFAULT_EMBEDDING_MODEL, collection_name="consumer_credit_rights"):
        self.encoder = SentenceTransformer(embedding_model)
        self.stemmer = Stemmer("english")
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        self.chunks: list[dict] = []
        self.bm25 = None
        self._doc_emb = None

    def _tokenize(self, texts):
        return bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer.stemWords, return_ids=False, show_progress=False)

    def build(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        for c, cid in zip(chunks, ids):
            c["id"] = cid

        vecs = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
        self.collection.add(embeddings=vecs.tolist(), documents=texts, metadatas=metadatas, ids=ids)
        self._doc_emb = vecs

        self.bm25 = bm25s.BM25(method="lucene")
        self.bm25.index(self._tokenize(texts), show_progress=False)

    def _dense_order(self, query):
        q = self.encoder.encode([query], normalize_embeddings=True)[0]
        return list(np.argsort(q @ self._doc_emb.T)[::-1])

    def _sparse_order(self, query):
        scores = self.bm25.get_scores(self._tokenize([query])[0])
        return list(np.argsort(scores)[::-1])

    def search(self, query, k=8, mode=DEFAULT_RETRIEVAL_MODE, rrf_k=60):
        """mode='dense' is the measured Stage-5 winner on this corpus (R@3=1.0 vs 0.81
        hybrid / 0.76 sparse -- see EVALUATION_REPORT.md). 'hybrid' (RRF) kept available
        for re-evaluation if the document set changes."""
        if mode == "dense":
            order = self._dense_order(query)
        elif mode == "sparse":
            order = self._sparse_order(query)
        else:
            rankings = [self._dense_order(query), self._sparse_order(query)]
            scores: dict[int, float] = {}
            for ranking in rankings:
                for rank, idx in enumerate(ranking):
                    scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
            order = sorted(scores, key=scores.get, reverse=True)
        return [self.chunks[i] for i in order[:k]]


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_n=4):
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(c, float(s)) for c, s in ranked[:top_n]]


def format_sources(reranked):
    return "\n".join(f"[{i + 1}] ({c['source']}, p.{c['page']}) {c['text']}" for i, (c, _) in enumerate(reranked))


@dataclass
class ChatTurn:
    role: str
    content: str


class ConsumerCreditRightsBot:
    """
    parse -> chunk -> hybrid index -> RRF fuse -> cross-encoder rerank
    -> law-routed, cited generation -> groundedness + no-verdict guardrails -> memory.
    """

    def __init__(self, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                 top_k=8, top_n=4, min_rerank_score=DEFAULT_MIN_RERANK_SCORE,
                 embedding_model=DEFAULT_EMBEDDING_MODEL, retrieval_mode=DEFAULT_RETRIEVAL_MODE,
                 model="gpt-4o-mini"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.top_n = top_n
        self.min_rerank_score = min_rerank_score
        self.retrieval_mode = retrieval_mode
        self.index = HybridIndex(embedding_model=embedding_model)
        self.reranker = Reranker()
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.history: list[ChatTurn] = []

    def ingest(self, data_dir: str):
        pages = load_all_documents(data_dir)
        chunks = chunk_recursive(pages, self.chunk_size, self.chunk_overlap)
        self.index.build(chunks)
        return {"pages": len(pages), "chunks": len(chunks)}

    def _condense(self, question):
        if not self.history:
            return question
        history_text = "\n".join(f"{t.role}: {t.content}" for t in self.history[-6:])
        return self.llm.invoke(CONDENSE_PROMPT.format(history=history_text, question=question)).content.strip()

    def chat(self, question: str) -> dict:
        standalone = self._condense(question)
        candidates = self.index.search(standalone, k=self.top_k, mode=self.retrieval_mode)
        reranked = self.reranker.rerank(standalone, candidates, top_n=self.top_n)

        if not reranked or reranked[0][1] < self.min_rerank_score:
            answer = "I don't have enough information in these documents to answer that confidently."
            sources = []
        else:
            extra = VERDICT_INSTRUCTION if is_verdict_seeking(question) else ""
            history_text = "\n".join(f"{t.role}: {t.content}" for t in self.history[-6:])
            prompt = ANSWER_PROMPT.format(
                extra_instruction=extra, sources=format_sources(reranked),
                history=history_text, question=standalone,
            )
            answer = self.llm.invoke(prompt).content.strip()
            sources = [
                {"rank": i + 1, "source": c["source"], "page": c["page"], "score": round(s, 3), "text": c["text"]}
                for i, (c, s) in enumerate(reranked)
            ]

        self.history.append(ChatTurn("user", question))
        self.history.append(ChatTurn("assistant", answer))
        return {"answer": answer, "sources": sources, "standalone_question": standalone,
                "verdict_seeking": is_verdict_seeking(question)}
