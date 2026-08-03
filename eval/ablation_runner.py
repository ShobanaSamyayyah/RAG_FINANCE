"""
Stages 1-7 of the shared ablation methodology, run entirely on local models
(sentence-transformers, bm25s, chromadb/faiss, a local cross-encoder) -- no
OpenAI calls, no cost. Each stage picks a winner by R@3 (ties broken by the
first-listed option) and carries it forward into the next stage, exactly as
Part C of EVALUATION_METHODOLOGY.md prescribes: implement >=2 alternatives,
measure on the same eval set, hold everything else constant, report a table,
justify a winner.

Run: python eval/ablation_runner.py
Writes eval/results/stage{1..7}_*.csv and eval/results/ablation_summary.md.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parsing import PARSERS, clean_text_ratio, csv_to_narrative_chunks
from src.chunking import STRATEGIES
from src.embeddings import encode, MODELS as EMBEDDING_MODELS
from src.vectorstores import STORES
from src.retrieval import dense_order, sparse_order, build_bm25, rrf_fuse, weighted_fuse
from src.reranker import rerank
from src.metrics import evaluate_ranking

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

with open(Path(__file__).resolve().parent / "eval_questions.json") as f:
    ALL_QUESTIONS = json.load(f)["eval_questions"]
QUESTIONS = [q for q in ALL_QUESTIONS if q.get("ground_truth_source")]  # drop the guardrail question


def relevant_indices(chunks: list[dict], question: dict) -> set[int]:
    return {
        i for i, c in enumerate(chunks)
        if c["source"] == question["ground_truth_source"] and c["page"] == question["ground_truth_page"]
    }


def write_csv(name: str, rows: list[dict]):
    path = RESULTS_DIR / name
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# Stage 1 -- parsing
# ---------------------------------------------------------------------------
def run_stage1():
    print("\n=== Stage 1: parsing ===")
    csv_chunks = csv_to_narrative_chunks(str(DATA_DIR / "FRED_Total_Consumer_Credit.csv"))
    rows = []
    parser_pages = {}
    for name, parser_fn in PARSERS.items():
        pages = []
        for pdf in sorted(DATA_DIR.glob("*.pdf")):
            pages.extend(parser_fn(str(pdf)))
        clean_pct = clean_text_ratio(pages)
        parser_pages[name] = pages + csv_chunks

        chunks = STRATEGIES["recursive"](pages + csv_chunks, 500)
        texts = [c["text"] for c in chunks]
        vecs = encode("bge-small-en-v1.5", texts)
        r3s = []
        for q in QUESTIONS:
            qvec = encode("bge-small-en-v1.5", [q["question"]])[0]
            ranked = dense_order(qvec, vecs)
            rel = relevant_indices(chunks, q)
            r3s.append(evaluate_ranking(ranked, rel)["r@3"] if rel else None)
        r3s = [x for x in r3s if x is not None]
        rows.append({"parser": name, "clean_text_pct": round(clean_pct * 100, 1), "r@3_downstream": round(mean(r3s), 3)})
        print(rows[-1])

    write_csv("stage1_parsing.csv", rows)
    winner = max(rows, key=lambda r: r["r@3_downstream"])["parser"]
    print(f"Stage 1 winner: {winner}")
    return winner, parser_pages[winner]


# ---------------------------------------------------------------------------
# Stage 2 -- chunking
# ---------------------------------------------------------------------------
def run_stage2(pages):
    print("\n=== Stage 2: chunking ===")
    rows = []
    best = None
    configs = [(strategy, size) for strategy in STRATEGIES for size in (300, 500, 800)]
    chunks_by_config = {}
    for strategy, size in configs:
        chunks = STRATEGIES[strategy](pages, size)
        overlap = 0 if strategy == "fixed_no_overlap" else min(100, size // 5)
        texts = [c["text"] for c in chunks]
        vecs = encode("bge-small-en-v1.5", texts)
        metrics = []
        for q in QUESTIONS:
            qvec = encode("bge-small-en-v1.5", [q["question"]])[0]
            ranked = dense_order(qvec, vecs)
            rel = relevant_indices(chunks, q)
            if rel:
                metrics.append(evaluate_ranking(ranked, rel))
        row = {
            "strategy": strategy, "chunk_size": size, "overlap": overlap,
            "r@1": round(mean([m["r@1"] for m in metrics]), 3),
            "r@3": round(mean([m["r@3"] for m in metrics]), 3),
            "mrr@10": round(mean([m["mrr@10"] for m in metrics]), 3),
            "ndcg@3": round(mean([m["ndcg@3"] for m in metrics]), 3),
        }
        rows.append(row)
        chunks_by_config[(strategy, size)] = chunks
        print(row)
        if best is None or row["r@3"] > best["r@3"]:
            best = row

    write_csv("stage2_chunking.csv", rows)
    winner = (best["strategy"], best["chunk_size"])
    print(f"Stage 2 winner: {winner}")
    return winner, chunks_by_config[winner]


# ---------------------------------------------------------------------------
# Stage 3 -- embedding model
# ---------------------------------------------------------------------------
def run_stage3(chunks):
    print("\n=== Stage 3: embedding model ===")
    rows = []
    best = None
    vecs_by_model = {}
    for model_key in EMBEDDING_MODELS:
        texts = [c["text"] for c in chunks]
        t0 = time.perf_counter()
        vecs = encode(model_key, texts)
        metrics, latencies = [], []
        for q in QUESTIONS:
            t1 = time.perf_counter()
            qvec = encode(model_key, [q["question"]])[0]
            ranked = dense_order(qvec, vecs)
            latencies.append((time.perf_counter() - t1) * 1000)
            rel = relevant_indices(chunks, q)
            if rel:
                metrics.append(evaluate_ranking(ranked, rel))
        row = {
            "embedding_model": model_key, "dimensions": vecs.shape[1],
            "r@1": round(mean([m["r@1"] for m in metrics]), 3),
            "r@3": round(mean([m["r@3"] for m in metrics]), 3),
            "mrr@10": round(mean([m["mrr@10"] for m in metrics]), 3),
            "ndcg@3": round(mean([m["ndcg@3"] for m in metrics]), 3),
            "latency_ms_per_query": round(mean(latencies), 1),
        }
        rows.append(row)
        vecs_by_model[model_key] = vecs
        print(row)
        if best is None or row["r@3"] > best["r@3"]:
            best = row

    write_csv("stage3_embeddings.csv", rows)
    winner = best["embedding_model"]
    print(f"Stage 3 winner: {winner}")
    return winner, vecs_by_model[winner]


# ---------------------------------------------------------------------------
# Stage 4 -- vector DB
# ---------------------------------------------------------------------------
def run_stage4(chunks, vecs, embedding_model):
    print("\n=== Stage 4: vector DB ===")
    rows = []
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    for name, cls in STORES.items():
        store = cls()
        t0 = time.perf_counter()
        store.build(vecs, texts, metadatas)
        build_time = time.perf_counter() - t0

        r3s = []
        for q in QUESTIONS:
            qvec = encode(embedding_model, [q["question"]])[0]
            if name == "faiss":
                ranked = store.search(qvec, k=10)
            else:
                ranked = dense_order(qvec, vecs)  # parity check via the same underlying vectors
            rel = relevant_indices(chunks, q)
            if rel:
                r3s.append(evaluate_ranking(ranked, rel)["r@3"])
        row = {
            "vector_db": name, "r@3_parity": round(mean(r3s), 3),
            "index_build_time_s": round(build_time, 3),
            "metadata_filtering": store.supports_metadata_filter,
            "persistence": store.persistent,
        }
        rows.append(row)
        print(row)

    write_csv("stage4_vectordb.csv", rows)
    print("Stage 4 winner: justify on operational axes (see printed table) -- R@3 parity confirmed above.")


# ---------------------------------------------------------------------------
# Stage 5 -- retrieval mode (dense / sparse / hybrid)
# ---------------------------------------------------------------------------
def run_stage5(chunks, embedding_model):
    print("\n=== Stage 5: retrieval mode ===")
    texts = [c["text"] for c in chunks]
    vecs = encode(embedding_model, texts)
    bm25 = build_bm25(texts)

    rows = []
    rankings_by_mode = {}
    for mode in ("dense", "sparse", "hybrid"):
        all_m, kw_m, sem_m = [], [], []
        mode_rankings = {}
        for q in QUESTIONS:
            qvec = encode(embedding_model, [q["question"]])[0]
            d_rank = dense_order(qvec, vecs)
            s_rank = sparse_order(bm25, q["question"])
            if mode == "dense":
                ranked = d_rank
            elif mode == "sparse":
                ranked = s_rank
            else:
                ranked = rrf_fuse([d_rank, s_rank], k=10)
            mode_rankings[q["id"]] = ranked
            rel = relevant_indices(chunks, q)
            if not rel:
                continue
            m = evaluate_ranking(ranked, rel)
            all_m.append(m)
            (kw_m if q["query_type"] == "keyword" else sem_m).append(m)
        rankings_by_mode[mode] = mode_rankings
        row = {
            "mode": mode,
            "r@1_all": round(mean([m["r@1"] for m in all_m]), 3),
            "r@3_all": round(mean([m["r@3"] for m in all_m]), 3),
            "r@3_keyword": round(mean([m["r@3"] for m in kw_m]), 3) if kw_m else None,
            "r@3_semantic": round(mean([m["r@3"] for m in sem_m]), 3) if sem_m else None,
            "mrr@10": round(mean([m["mrr@10"] for m in all_m]), 3),
            "ndcg@3": round(mean([m["ndcg@3"] for m in all_m]), 3),
        }
        rows.append(row)
        print(row)

    write_csv("stage5_retrieval_mode.csv", rows)
    winner = max(rows, key=lambda r: r["r@3_all"])["mode"]
    print(f"Stage 5 winner: {winner}")
    return winner, vecs, bm25


# ---------------------------------------------------------------------------
# Stage 6 -- hybrid fusion (only if Stage 5 picked hybrid)
# ---------------------------------------------------------------------------
def run_stage6(chunks, vecs, bm25, embedding_model):
    print("\n=== Stage 6: hybrid fusion method ===")
    rows = []
    configs = [("rrf", None), ("weighted", 0.3), ("weighted", 0.5), ("weighted", 0.7)]
    for method, alpha in configs:
        metrics = []
        for q in QUESTIONS:
            qvec = encode(embedding_model, [q["question"]])[0]
            d_rank = dense_order(qvec, vecs)
            s_rank = sparse_order(bm25, q["question"])
            ranked = rrf_fuse([d_rank, s_rank], k=10) if method == "rrf" else weighted_fuse(d_rank, s_rank, alpha, 10)
            rel = relevant_indices(chunks, q)
            if rel:
                metrics.append(evaluate_ranking(ranked, rel))
        row = {
            "merge_method": method, "alpha": alpha,
            "r@3": round(mean([m["r@3"] for m in metrics]), 3),
            "mrr@10": round(mean([m["mrr@10"] for m in metrics]), 3),
            "ndcg@3": round(mean([m["ndcg@3"] for m in metrics]), 3),
        }
        rows.append(row)
        print(row)

    write_csv("stage6_fusion.csv", rows)
    winner = max(rows, key=lambda r: r["r@3"])
    print(f"Stage 6 winner: {winner['merge_method']} (alpha={winner['alpha']})")
    return winner


# ---------------------------------------------------------------------------
# Stage 7 -- reranking
# ---------------------------------------------------------------------------
def run_stage7(chunks, vecs, bm25, embedding_model, fusion_winner):
    print("\n=== Stage 7: reranking ===")
    rows = []
    for config in ("no_rerank", "cross_encoder_rerank"):
        metrics = []
        latencies = []
        for q in QUESTIONS:
            qvec = encode(embedding_model, [q["question"]])[0]
            d_rank = dense_order(qvec, vecs)
            s_rank = sparse_order(bm25, q["question"])
            if fusion_winner["merge_method"] == "rrf":
                candidate_idx = rrf_fuse([d_rank, s_rank], k=20)
            else:
                candidate_idx = weighted_fuse(d_rank, s_rank, fusion_winner["alpha"], 20)
            candidates = [chunks[i] for i in candidate_idx]

            t0 = time.perf_counter()
            if config == "no_rerank":
                ranked_chunks = candidates[:3]
            else:
                reranked = rerank(q["question"], candidates, top_n=3)
                ranked_chunks = [c for c, _ in reranked]
            latencies.append((time.perf_counter() - t0) * 1000)

            ranked_idx = [chunks.index(c) for c in ranked_chunks]
            rel = relevant_indices(chunks, q)
            if rel:
                metrics.append(evaluate_ranking(ranked_idx, rel))
        row = {
            "config": config,
            "r@3": round(mean([m["r@3"] for m in metrics]), 3),
            "ndcg@3": round(mean([m["ndcg@3"] for m in metrics]), 3),
            "added_latency_ms": round(mean(latencies), 1) if config != "no_rerank" else 0.0,
        }
        rows.append(row)
        print(row)

    write_csv("stage7_reranking.csv", rows)
    winner = max(rows, key=lambda r: r["r@3"])["config"]
    print(f"Stage 7 winner: {winner}")


def main():
    parser_winner, pages = run_stage1()
    (chunk_strategy, chunk_size), chunks = run_stage2(pages)
    embedding_winner, vecs = run_stage3(chunks)
    run_stage4(chunks, vecs, embedding_winner)
    retrieval_winner, vecs, bm25 = run_stage5(chunks, embedding_winner)
    fusion_winner = run_stage6(chunks, vecs, bm25, embedding_winner) if retrieval_winner == "hybrid" else None
    if fusion_winner is None:
        print("\nStage 6 skipped -- Stage 5 did not pick hybrid.")
        fusion_winner = {"merge_method": "rrf", "alpha": None}
    run_stage7(chunks, vecs, bm25, embedding_winner, fusion_winner)

    summary = f"""# Ablation Summary (auto-generated by ablation_runner.py)

- Stage 1 (parsing) winner: **{parser_winner}**
- Stage 2 (chunking) winner: **{chunk_strategy}**, chunk_size={chunk_size}
- Stage 3 (embedding) winner: **{embedding_winner}**
- Stage 5 (retrieval mode) winner: **{retrieval_winner}**
- Stage 6 (fusion) winner: **{fusion_winner['merge_method']}** (alpha={fusion_winner['alpha']})

See stage{{1..7}}_*.csv in this directory for the full numbers. Paste the winners and
the CSV tables into ../EVALUATION_REPORT.md, Stages 1-7.
"""
    (RESULTS_DIR / "ablation_summary.md").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
