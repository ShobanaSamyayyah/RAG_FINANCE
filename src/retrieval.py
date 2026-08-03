"""
Stages 5 & 6 (retrieval mode + hybrid fusion) — dense-only, sparse-only (BM25), and
hybrid (RRF or weighted-linear) ranking over a fixed set of chunk embeddings.

Everything here operates on chunk *indices* into a shared `chunks` list, so the ablation
runner can compute R@k / MRR / NDCG against ground-truth chunk indices directly.
"""

import bm25s
import numpy as np
from Stemmer import Stemmer

_stemmer = Stemmer("english")


def _tokenize(texts: list[str]):
    return bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer.stemWords, return_ids=False, show_progress=False)


def build_bm25(texts: list[str]):
    bm25 = bm25s.BM25(method="lucene")
    bm25.index(_tokenize(texts), show_progress=False)
    return bm25


def dense_order(query_vec: np.ndarray, doc_vecs: np.ndarray) -> list[int]:
    scores = doc_vecs @ query_vec
    return list(np.argsort(scores)[::-1])


def sparse_order(bm25, query: str) -> list[int]:
    scores = bm25.get_scores(_tokenize([query])[0])
    return list(np.argsort(scores)[::-1])


def rrf_fuse(rankings: list[list[int]], k: int, rrf_k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]


def weighted_fuse(dense_rank: list[int], sparse_rank: list[int], alpha: float, k: int) -> list[int]:
    """score = alpha * dense_norm + (1 - alpha) * sparse_norm, ranks normalized to [0,1] by position."""
    n_d, n_s = len(dense_rank), len(sparse_rank)
    dense_norm = {idx: 1 - rank / max(n_d - 1, 1) for rank, idx in enumerate(dense_rank)}
    sparse_norm = {idx: 1 - rank / max(n_s - 1, 1) for rank, idx in enumerate(sparse_rank)}
    all_idx = set(dense_norm) | set(sparse_norm)
    scores = {i: alpha * dense_norm.get(i, 0.0) + (1 - alpha) * sparse_norm.get(i, 0.0) for i in all_idx}
    return sorted(scores, key=scores.get, reverse=True)[:k]


def retrieve(mode: str, query: str, query_vec: np.ndarray, doc_vecs: np.ndarray, bm25, k: int,
             fusion: str = "rrf", alpha: float = 0.5) -> list[int]:
    if mode == "dense":
        return dense_order(query_vec, doc_vecs)[:k]
    if mode == "sparse":
        return sparse_order(bm25, query)[:k]
    if mode == "hybrid":
        d_rank, s_rank = dense_order(query_vec, doc_vecs), sparse_order(bm25, query)
        return rrf_fuse([d_rank, s_rank], k) if fusion == "rrf" else weighted_fuse(d_rank, s_rank, alpha, k)
    raise ValueError(f"unknown retrieval mode: {mode!r}")
