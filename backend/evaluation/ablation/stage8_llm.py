"""
Stage 8 ablation — LLM for generation. Fixes retrieval to the Stage 1-7
winning config (pdfplumber + sentence chunks @250/50, dense-only Pinecone
retrieval, alpha=1.0, no reranker) and varies ONLY the generator, so the
comparison isolates the LLM's contribution to answer quality. See
../../EVALUATION_METHODOLOGY.md Part C, Stage 8.

For each golden question: retrieve once (shared across every LLM variant,
via retrieval.dense_retriever.retrieve_dense + retrieval.context_builder.
build_context — the exact same call orchestration.chains.OrchestrationService
.retrieve_context() makes in production), then call each candidate LLM with
that identical prompt/context and record RAGAS + DeepEval scores, cost, and
latency.

Candidates come from orchestration.llm_client.LLMClient — the same
multi-provider client the chat endpoint uses — so this measures the actual
production code path, not a reimplementation of it. A model is skipped
(not silently scored as a bad model) if its API key/local server isn't
configured; skip reasons are printed so the table's absence of a row is
never mistaken for that model losing.

RAGAS/DeepEval metrics (Faithfulness, Answer Relevancy, Context Precision,
Context Recall, Hallucination, G-Eval) are LLM-as-judge metrics scored by
a separate judge call (OpenAI gpt-4o-mini per the ragas/deepeval default) —
if those packages aren't importable in this environment (see requirements.txt
[evaluation] section) this falls back to a single-call structured judge
prompt covering the same four RAGAS-named dimensions, so the ablation can
still run end-to-end; the fallback is clearly labeled in the printed table
and should be replaced with real ragas/deepeval scores once that dependency
chain is resolved (see module-level NOTE below).

NOTE on the ragas/deepeval dependency conflict encountered while wiring this
stage: ragas==0.4.3 imports langchain_community.chat_models.vertexai, which
requires the separate `langchain-google-vertexai` package; installing that
package downgrades `langchain-core` below what this project's pinned
`langchain==1.3.14`/`langgraph==1.2.10`/`langchain-openai==1.4.1` require,
breaking the orchestration layer's own imports. Do not `pip install
langchain-google-vertexai` in this project's venv to fix this — either
install ragas/deepeval in an isolated venv for offline scoring, or wait for
an upstream ragas release that drops the vertexai import at module load
time.

Run from backend/:  python -m evaluation.ablation.stage8_llm
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from api.core.config import Settings, get_settings
from evaluation.retrieval_metrics import load_golden_dataset
from orchestration.chains import build_prompt
from orchestration.llm_client import LLMClient
from retrieval.context_builder import ContextResult, build_context
from retrieval.dense_retriever import retrieve_dense
from retrieval.embed_query import embed_query

# Stage 1-7 winners (see EVALUATION_METHODOLOGY.md Part C table) — retrieval
# is held fixed here; ingestion must already have been run against this
# config so the live Pinecone index reflects it.
TOP_K = 5  # Settings.rag_top_k default; retrieval mode/alpha/rerank are
           # index-side/query-side choices already baked into retrieve_dense.

# Candidate generators: (label, llm_model name understood by LLMClient's
# _infer_provider, required-config check). Real API keys/local server
# required — see module docstring.
CANDIDATES: list[tuple[str, str]] = [
    ("gemini-2.0-flash", "gemini-2.0-flash"),
    ("gpt-4o-mini", "gpt-4o-mini"),
    ("claude-3-haiku", "claude-3-haiku-20240307"),
    ("ollama-local", "llama3"),
]

# Approximate per-1K-token pricing (USD) as of the provider's published
# rates — Ollama is $0 (local/self-hosted). Used only for the cost/query
# column; update if pricing changes.
_COST_PER_1K_TOKENS: dict[str, tuple[float, float]] = {  # (prompt, completion)
    "gemini-2.0-flash": (0.000075 * 1, 0.0003 * 1),
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    "llama3": (0.0, 0.0),
}


@dataclass
class GeneratedSample:
    query_id: str
    question: str
    reference_answer: str
    context_text: str
    answer: str
    token_usage: dict[str, int]
    latency_s: float
    cost_usd: float


def _is_configured(model_name: str, settings: Settings) -> tuple[bool, str]:
    """Mirror LLMClient._infer_provider's routing so we skip (not silently
    fall back and mis-score) a candidate whose key/server isn't set up."""
    name = model_name.lower()
    if "gemini" in name:
        return bool(settings.gemini_api_key), "gemini_api_key not set"
    if "claude" in name:
        return bool(settings.anthropic_api_key), "anthropic_api_key not set"
    if "gpt" in name or name.startswith("o1") or name.startswith("o3"):
        return bool(settings.openai_api_key), "openai_api_key not set"
    # Ollama: always "configured" per LLMClient's own catch-all, but confirm
    # the local server actually responds so a dead server doesn't silently
    # skew results — fail the check rather than let the request hang.
    import httpx

    try:
        httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
        return True, ""
    except Exception:
        return False, f"ollama server unreachable at {settings.ollama_base_url}"


def _build_retrieval_contexts(settings: Settings) -> list[tuple, ContextResult]:
    """Retrieve once per golden question — shared across every LLM
    candidate so the comparison varies only the generator."""
    golden = load_golden_dataset()
    contexts = []
    for query in golden:
        if not query.retrieval_eval:
            continue
        query_vector = embed_query(query.question, settings=settings)
        chunks = retrieve_dense(query_vector, top_k=TOP_K, settings=settings)
        context_result = build_context(chunks, settings=settings)
        contexts.append((query, context_result))
    return contexts


def _cost(model_name: str, token_usage: dict[str, int]) -> float:
    rates = _COST_PER_1K_TOKENS.get(model_name)
    if rates is None:
        return 0.0
    prompt_rate, completion_rate = rates
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    return (prompt_tokens / 1000) * prompt_rate + (completion_tokens / 1000) * completion_rate


async def _generate_for_candidate(
    model_name: str,
    contexts: list,
    settings: Settings,
) -> list[GeneratedSample]:
    candidate_settings = settings.model_copy(update={"llm_model": model_name})
    client = LLMClient()
    samples: list[GeneratedSample] = []

    for query, context_result in contexts:
        prompt = build_prompt(
            question=query.question,
            context_text=context_result.context_text,
        )
        start = time.perf_counter()
        answer, token_usage = await client.generate(prompt=prompt, settings=candidate_settings)
        latency_s = time.perf_counter() - start

        samples.append(
            GeneratedSample(
                query_id=query.id,
                question=query.question,
                reference_answer=query.reference_answer,
                context_text=context_result.context_text,
                answer=answer,
                token_usage=token_usage,
                latency_s=latency_s,
                cost_usd=_cost(model_name, token_usage),
            )
        )
    return samples


# ---------------------------------------------------------------------------
# RAGAS + DeepEval scoring, with a same-shape fallback judge if those
# packages aren't importable in this environment (see module docstring).
# ---------------------------------------------------------------------------

def _try_ragas_deepeval(samples: list[GeneratedSample], settings: Settings) -> dict | None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    except Exception as exc:
        print(f"  (ragas unavailable, using fallback judge: {exc})")
        return None

    try:
        from deepeval.metrics import GEval, HallucinationMetric
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except Exception as exc:
        print(f"  (deepeval unavailable, using fallback judge: {exc})")
        return None

    dataset = Dataset.from_dict(
        {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [[s.context_text] for s in samples],
            "ground_truth": [s.reference_answer for s in samples],
        }
    )
    ragas_result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
    )
    ragas_scores = ragas_result.to_pandas().mean(numeric_only=True).to_dict()

    hallucination_metric = HallucinationMetric()
    geval_metric = GEval(
        name="Answer Quality",
        criteria="Assess whether the answer correctly and completely addresses the question "
                 "using only the provided context, without asserting a legal verdict.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
    )
    hallucination_scores, geval_scores = [], []
    for s in samples:
        case = LLMTestCase(
            input=s.question,
            actual_output=s.answer,
            context=[s.context_text],
            expected_output=s.reference_answer,
        )
        hallucination_metric.measure(case)
        hallucination_scores.append(hallucination_metric.score)
        geval_metric.measure(case)
        geval_scores.append(geval_metric.score)

    return {
        "faithfulness": ragas_scores.get("faithfulness", 0.0),
        "answer_relevancy": ragas_scores.get("answer_relevancy", 0.0),
        "context_precision": ragas_scores.get("context_precision", 0.0),
        "hallucination": sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0,
        "g_eval": sum(geval_scores) / len(geval_scores) if geval_scores else 0.0,
        "judge": "ragas+deepeval",
    }


_FALLBACK_JUDGE_PROMPT = """You are grading a RAG system's answer against ground truth.

Question: {question}
Retrieved context: {context}
Reference answer: {reference}
Generated answer: {answer}

Score each on a 0.0-1.0 scale and return ONLY a JSON object with these keys:
- "faithfulness": does the answer state only things supported by the retrieved context?
- "answer_relevancy": does the answer actually address the question asked?
- "context_precision": was the retrieved context relevant (not noisy) for this question?
- "hallucination": 1.0 if the answer invents facts not in the context, else 0.0 (lower is better)
- "g_eval": overall answer quality vs. the reference answer

Return only the JSON object, no other text."""


def _fallback_judge(samples: list[GeneratedSample], settings: Settings) -> dict:
    """Single structured-judge call per sample using OpenAI (same judge
    model choice as Stage 7's LLM-as-reranker), covering the same four
    RAGAS-named dimensions plus DeepEval's Hallucination/G-Eval so the
    ablation table has a consistent column set with or without the real
    ragas/deepeval packages installed."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": [], "hallucination": [], "g_eval": []}

    for s in samples:
        prompt = _FALLBACK_JUDGE_PROMPT.format(
            question=s.question,
            context=s.context_text[:2000],
            reference=s.reference_answer,
            answer=s.answer,
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = resp.choices[0].message.content.strip().strip("`").removeprefix("json").strip()
            parsed = json.loads(text)
            for key in scores:
                scores[key].append(float(parsed.get(key, 0.0)))
        except Exception as exc:
            print(f"  (fallback judge failed for {s.query_id}: {exc})")

    out = {k: (sum(v) / len(v) if v else 0.0) for k, v in scores.items()}
    out["judge"] = "fallback-gpt-4o-mini"
    return out


def _score_candidate(model_name: str, samples: list[GeneratedSample], settings: Settings) -> dict:
    quality = _try_ragas_deepeval(samples, settings) or _fallback_judge(samples, settings)
    avg_latency = sum(s.latency_s for s in samples) / len(samples) if samples else 0.0
    avg_cost = sum(s.cost_usd for s in samples) / len(samples) if samples else 0.0
    return {
        "config": model_name,
        "n_queries": len(samples),
        **quality,
        "cost_per_query_usd": round(avg_cost, 6),
        "latency_s": round(avg_latency, 3),
    }


def run() -> list[dict]:
    settings = get_settings()
    contexts = _build_retrieval_contexts(settings)
    print(f"Retrieved shared context for {len(contexts)} golden queries (fixed across all LLM variants).\n")

    rows = []
    for label, model_name in CANDIDATES:
        configured, reason = _is_configured(model_name, settings)
        if not configured:
            print(f"Skipping {label} ({model_name}): {reason}")
            continue
        print(f"Generating with {label} ({model_name})...")
        samples = asyncio.run(_generate_for_candidate(model_name, contexts, settings))
        rows.append(_score_candidate(label, samples, settings))

    return rows


def print_table(rows: list[dict]) -> None:
    print("\nStage 8 - LLM for generation comparison (retrieval fixed at Stage 1-7 winners)")
    if not rows:
        print("No candidates were scored — configure at least 2 provider API keys (or a local Ollama "
              "server) in .env and re-run.")
        return

    judge = rows[0].get("judge", "unknown")
    header = f"{'Model':16s} {'Faith':>6s} {'AnsRel':>7s} {'CtxPrec':>8s} {'Halluc':>7s} {'G-Eval':>7s} {'Cost/q($)':>10s} {'Latency(s)':>11s}"
    print(header)
    for r in rows:
        print(
            f"{r['config']:16s} {r['faithfulness']:6.3f} {r['answer_relevancy']:7.3f} "
            f"{r['context_precision']:8.3f} {r['hallucination']:7.3f} {r['g_eval']:7.3f} "
            f"{r['cost_per_query_usd']:10.6f} {r['latency_s']:11.3f}"
        )

    # Lower hallucination is better; every other quality metric here is
    # higher-is-better. Rank primarily on faithfulness (the guardrail's
    # "never invent a legal protection" requirement maps directly onto it),
    # tie-broken by G-Eval, then penalize hallucination.
    winner = max(rows, key=lambda r: (r["faithfulness"], r["g_eval"], -r["hallucination"]))
    print(
        f"\nJudge: {judge}. Winner: {winner['config']} — highest faithfulness "
        f"({winner['faithfulness']:.3f}) among {len(rows)} generators scored on "
        f"{winner['n_queries']} golden queries, ${winner['cost_per_query_usd']:.6f}/query, "
        f"{winner['latency_s']:.3f}s avg latency. Faithfulness is weighted first because this "
        "domain's guardrail explicitly forbids inventing a legal protection not present in the "
        "retrieved documents — a cheaper or faster model that hallucinates more is not an "
        "acceptable trade-off for a fair-lending compliance assistant. Re-run with real "
        "ragas/deepeval installed (see module docstring) before finalizing this choice for "
        "production; the fallback judge is a single gpt-4o-mini call per sample, not a "
        "substitute for the peer-reviewed RAGAS/DeepEval metric implementations."
    )


if __name__ == "__main__":
    results = run()
    print_table(results)
