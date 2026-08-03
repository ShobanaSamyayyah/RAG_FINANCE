"""Part A retrieval metrics: R@1, R@3, MRR@10, DCG@3/NDCG@3."""

from __future__ import annotations

import math


def recall_at_k(ranked_idx: list[int], relevant_idx: set[int], k: int) -> float:
    return 1.0 if set(ranked_idx[:k]) & relevant_idx else 0.0


def mrr_at_10(ranked_idx: list[int], relevant_idx: set[int]) -> float:
    for rank, idx in enumerate(ranked_idx[:10], start=1):
        if idx in relevant_idx:
            return 1.0 / rank
    return 0.0


def dcg_at_3(ranked_idx: list[int], relevance: dict[int, int]) -> float:
    """relevance: {chunk_idx: graded relevance 0/1/2/3}."""
    return sum(relevance.get(idx, 0) / math.log2(rank + 1) for rank, idx in enumerate(ranked_idx[:3], start=1))


def ndcg_at_3(ranked_idx: list[int], relevance: dict[int, int]) -> float:
    dcg = dcg_at_3(ranked_idx, relevance)
    ideal_order = sorted(relevance, key=relevance.get, reverse=True)
    idcg = dcg_at_3(ideal_order, relevance)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(ranked_idx: list[int], relevant_idx: set[int], relevance: dict[int, int] | None = None) -> dict:
    relevance = relevance or {i: 1 for i in relevant_idx}
    return {
        "r@1": recall_at_k(ranked_idx, relevant_idx, 1),
        "r@3": recall_at_k(ranked_idx, relevant_idx, 3),
        "mrr@10": mrr_at_10(ranked_idx, relevant_idx),
        "ndcg@3": ndcg_at_3(ranked_idx, relevance),
    }
