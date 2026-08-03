"""Stage 4 (vector DB) — ChromaDB vs FAISS behind one interface. Same embeddings in, so
retrieval quality should be at parity; the real comparison axes are build time / persistence
/ metadata filtering, timed by the ablation runner around these calls."""

import uuid

import numpy as np


class ChromaStore:
    name = "chromadb"
    supports_metadata_filter = True
    persistent = True  # via PersistentClient in production; in-memory Client used here for the ablation

    def __init__(self, collection_name="ablation"):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    def build(self, vecs: np.ndarray, texts: list[str], metadatas: list[dict]):
        ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.add(embeddings=vecs.tolist(), documents=texts, metadatas=metadatas, ids=ids)

    def search(self, query_vec: np.ndarray, k: int) -> list[int]:
        res = self.collection.query(query_embeddings=[query_vec.tolist()], n_results=k)
        # Chroma returns document text; map back to index via documents order isn't guaranteed,
        # so callers needing index positions should use FaissStore or the HybridIndex in pipeline.py.
        return res


class FaissStore:
    name = "faiss"
    supports_metadata_filter = False  # not natively; needs a side lookup table
    persistent = True  # via index.write_index; in-memory here for the ablation

    def __init__(self):
        import faiss
        self._faiss = faiss
        self.index = None
        self._dim = None

    def build(self, vecs: np.ndarray, texts: list[str], metadatas: list[dict]):
        self._dim = vecs.shape[1]
        self.index = self._faiss.IndexFlatIP(self._dim)  # inner product == cosine on normalized vecs
        self.index.add(vecs.astype("float32"))

    def search(self, query_vec: np.ndarray, k: int) -> list[int]:
        _, idx = self.index.search(query_vec.astype("float32").reshape(1, -1), k)
        return [i for i in idx[0] if i != -1]


STORES = {"chromadb": ChromaStore, "faiss": FaissStore}
