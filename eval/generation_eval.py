"""
Stage 8 (LLM comparison) + final end-to-end RAGAS/DeepEval scoring.

Fixes retrieval to the Stage 1-7 winning config (src/pipeline.py constants) and
varies only the generator, so the LLM's contribution is isolated per the
methodology. Requires OPENAI_API_KEY (see ../.env.example) -- this is the one
script in this project that costs real API money.

Run: python eval/generation_eval.py
Writes eval/results/stage8_llm_comparison.csv and eval/results/final_e2e_scores.csv.
"""

import csv
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.pipeline import ConsumerCreditRightsBot

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

with open(Path(__file__).resolve().parent / "eval_questions.json") as f:
    QUESTIONS = json.load(f)["eval_questions"]

CANDIDATE_LLMS = ["gpt-4o-mini", "gpt-4o"]
FINAL_MODEL = "gpt-4o-mini"  # set to whichever wins Stage 8


def run_pipeline(model: str):
    bot = ConsumerCreditRightsBot(model=model)
    bot.ingest(str(DATA_DIR))
    results = []
    for q in QUESTIONS:
        bot.history = []  # each eval question is independent, not a multi-turn conversation
        out = bot.chat(q["question"])
        results.append({
            "question": q["question"],
            "answer": out["answer"],
            "contexts": [s["text"] for s in out["sources"]],
            "ground_truth": q.get("ideal_answer", ""),
        })
    return results


def ragas_scores(results):
    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    embed = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    samples = [
        SingleTurnSample(
            user_input=r["question"], response=r["answer"],
            retrieved_contexts=r["contexts"] or [""], reference=r["ground_truth"],
        )
        for r in results
    ]
    dataset = EvaluationDataset(samples=samples)
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    out = evaluate(dataset, metrics=metrics, llm=judge, embeddings=embed, show_progress=False)
    df = out.to_pandas()
    return {
        "faithfulness": round(df["faithfulness"].mean(), 3),
        "answer_relevancy": round(df["answer_relevancy"].mean(), 3),
        "context_precision": round(df["llm_context_precision_with_reference"].mean(), 3),
        "context_recall": round(df["context_recall"].mean(), 3),
    }


def deepeval_scores(results):
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric, GEval
    from deepeval.test_case import LLMTestCaseParams

    judge = "gpt-4o"
    cases = [
        LLMTestCase(
            input=r["question"], actual_output=r["answer"],
            retrieval_context=r["contexts"] or [""], context=r["contexts"] or [""],
            expected_output=r["ground_truth"],
        )
        for r in results
    ]

    faithfulness = FaithfulnessMetric(threshold=0.7, model=judge, async_mode=False)
    answer_rel = AnswerRelevancyMetric(threshold=0.7, model=judge, async_mode=False)
    hallucination = HallucinationMetric(threshold=0.5, model=judge, async_mode=False)
    grounding = GEval(
        name="Legal Grounding + No Verdict",
        criteria=(
            "Return 1 only if the answer (a) is fully supported by the retrieval context, "
            "(b) correctly names ECOA/Reg B or the Fair Housing Act as applicable, and "
            "(c) does not render a legal verdict on whether discrimination definitely occurred."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        model=judge, threshold=0.7, async_mode=False,
    )

    scores = {"faithfulness": [], "answer_relevancy": [], "hallucination": [], "g_eval": []}
    for case in cases:
        faithfulness.measure(case); scores["faithfulness"].append(faithfulness.score)
        answer_rel.measure(case); scores["answer_relevancy"].append(answer_rel.score)
        hallucination.measure(case); scores["hallucination"].append(hallucination.score)
        grounding.measure(case); scores["g_eval"].append(grounding.score)

    return {k: round(sum(v) / len(v), 3) for k, v in scores.items()}


def main():
    print("=== Stage 8: LLM comparison (fixed retrieval, varying generator) ===")
    stage8_rows = []
    all_results = {}
    for model in CANDIDATE_LLMS:
        print(f"\n--- {model} ---")
        results = run_pipeline(model)
        all_results[model] = results
        ragas = ragas_scores(results)
        deepeval_out = deepeval_scores(results)
        row = {"llm": model, **{f"ragas_{k}": v for k, v in ragas.items()},
               "deepeval_hallucination": deepeval_out["hallucination"],
               "deepeval_g_eval": deepeval_out["g_eval"]}
        stage8_rows.append(row)
        print(row)

    with open(RESULTS_DIR / "stage8_llm_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stage8_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stage8_rows)
    print(f"\nwrote {RESULTS_DIR / 'stage8_llm_comparison.csv'}")

    print(f"\n=== Final end-to-end RAGAS + DeepEval on {FINAL_MODEL} ===")
    final_results = all_results.get(FINAL_MODEL) or run_pipeline(FINAL_MODEL)
    final_ragas = ragas_scores(final_results)
    final_deepeval = deepeval_scores(final_results)
    final_row = {"model": FINAL_MODEL, **{f"ragas_{k}": v for k, v in final_ragas.items()},
                 **{f"deepeval_{k}": v for k, v in final_deepeval.items()}}
    print(final_row)

    with open(RESULTS_DIR / "final_e2e_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_row.keys()))
        writer.writeheader()
        writer.writerow(final_row)
    print(f"wrote {RESULTS_DIR / 'final_e2e_scores.csv'}")


if __name__ == "__main__":
    main()
