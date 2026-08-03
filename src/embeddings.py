"""Stage 3 (embedding model) — thin wrapper so the ablation runner can swap models by name."""

from sentence_transformers import SentenceTransformer

MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "multi-qa-mpnet-base-dot-v1": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
}

_cache: dict[str, SentenceTransformer] = {}


def get_encoder(name: str) -> SentenceTransformer:
    if name not in _cache:
        _cache[name] = SentenceTransformer(MODELS.get(name, name))
    return _cache[name]


def encode(name: str, texts: list[str]):
    model = get_encoder(name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
