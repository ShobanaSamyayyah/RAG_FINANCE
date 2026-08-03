"""Stage 7 (reranking) — no-reranker baseline vs cross-encoder second stage."""

from sentence_transformers import CrossEncoder

_cache: dict[str, CrossEncoder] = {}


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    if model_name not in _cache:
        _cache[model_name] = CrossEncoder(model_name)
    return _cache[model_name]


def rerank(query: str, candidates: list[dict], top_n: int, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    model = get_reranker(model_name)
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(c, float(s)) for c, s in ranked[:top_n]]
