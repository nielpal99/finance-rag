"""
Finance RAG evaluation suite — powered by Braintrust.

Loads evals/eval_set.json, runs each question through the full RAG pipeline
with hard doc_type filtering, scores with autoevals LLMClassifier, and
pushes results to a Braintrust experiment.

Usage:
    python3 evals/run_eval.py                        # all 20 questions
    python3 evals/run_eval.py --id 9                 # single question
    python3 evals/run_eval.py --difficulty medium    # filter by difficulty
    python3 evals/run_eval.py --experiment finance-rag-v2
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "query"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic
import braintrust
from retriever import retrieve, rerank, RERANK_PER_NS, RERANK_FINAL_N
from prompt_builder import build_prompt

# ── config ────────────────────────────────────────────────────────────────────

MODEL               = "claude-sonnet-4-20250514"
TOP_K               = 8
ALL_TICKERS         = ["NVDA", "AMD", "AVGO", "TSM", "ANET", "MU", "CRWV"]
EVAL_SET_PATH       = Path(__file__).parent / "eval_set.json"
BRAINTRUST_PROJECT  = "finance-rag"

RETRIEVAL_PASS_THRESHOLD  = 0.5
GENERATION_PASS_THRESHOLD = 0.6

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── judge ─────────────────────────────────────────────────────────────────────

_CHOICE_SCORES = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.2, "E": 0.0}

_JUDGE_PROMPT = """\
You are evaluating whether a RAG system's generated answer captures the key \
facts in an expected answer. Score strictly on factual coverage — not writing \
style or length.

Expected answer:
{expected}

Generated answer:
{output}

Pick the single letter that best describes factual coverage:
(A) All key facts from the expected answer are present and accurate
(B) Most key facts present, minor gaps or slight inaccuracies
(C) Some key facts present, significant gaps
(D) Few key facts, mostly missing or incorrect
(E) No relevant facts or completely incorrect

Answer with only a single letter: A, B, C, D, or E."""


def factual_coverage(output: str, expected: str, **_) -> float:
    prompt = _JUDGE_PROMPT.format(output=output, expected=expected)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4,
        messages=[{"role": "user", "content": prompt}],
    )
    letter = msg.content[0].text.strip().upper()[:1]
    return _CHOICE_SCORES.get(letter, 0.0)

# ── retrieval ─────────────────────────────────────────────────────────────────

def _run_retrieval(
    question: str,
    relevant_tickers: list,
    preferred_doc_type: str = None,
) -> dict:
    namespaces = relevant_tickers if relevant_tickers else ALL_TICKERS
    all_results = []
    for ticker in namespaces:
        chunks = retrieve(
            question,
            namespace=ticker,
            top_k=TOP_K,
            rerank_top_n=RERANK_PER_NS,
            doc_type_filter=preferred_doc_type,
        )
        all_results.extend(chunks)

    top = rerank(question, all_results, top_n=RERANK_FINAL_N)

    retrieved_tickers  = list({r.get("ticker", r.get("id", "").split("_")[0]) for r in top})
    retrieved_sections = list({r.get("section", "") for r in top if r.get("section")})

    relevant_set    = set(t.upper() for t in relevant_tickers)
    hit_set         = set(t.upper() for t in retrieved_tickers) & relevant_set
    retrieval_score = len(hit_set) / len(relevant_set) if relevant_set else 0.0

    return {
        "results":            top,
        "retrieval_score":    round(retrieval_score, 4),
        "tickers_hit":        sorted(hit_set),
        "tickers_missed":     sorted(relevant_set - hit_set),
        "retrieved_tickers":  sorted(retrieved_tickers),
        "retrieved_sections": retrieved_sections,
    }


# ── generation ────────────────────────────────────────────────────────────────

def _run_generation(question: str, retrieval_results: list) -> tuple:
    """Returns (answer_text, full_prompt)."""
    prompt = build_prompt(question, retrieval_results)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text, prompt


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",         type=int, default=None)
    parser.add_argument("--difficulty", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="finance-rag-v1")
    args = parser.parse_args()

    bt_key = os.environ.get("BRAINTRUST_API_KEY", "")
    if not bt_key:
        print("ERROR: BRAINTRUST_API_KEY not set in .env")
        return

    eval_set = json.loads(EVAL_SET_PATH.read_text())
    if args.id:
        eval_set = [q for q in eval_set if q["id"] == args.id]
    if args.difficulty:
        eval_set = [q for q in eval_set if q["difficulty"] == args.difficulty]
    if not eval_set:
        print("No questions matched the filter.")
        return

    print(f"Running {len(eval_set)} question(s) → Braintrust '{args.experiment}' …\n")

    experiment = braintrust.init(
        project=BRAINTRUST_PROJECT,
        experiment=args.experiment,
        api_key=bt_key,
    )

    passed = 0

    for i, q in enumerate(eval_set, 1):
        t0 = time.time()
        print(f"[{i}/{len(eval_set)}] Q{q['id']} ({q['difficulty']}): {q['question'][:70]}…")

        preferred_doc_type = q.get("preferred_doc_type")

        # 1 — retrieval
        ret = _run_retrieval(q["question"], q["relevant_tickers"], preferred_doc_type)

        # 2 — generation
        generated, full_prompt = _run_generation(q["question"], ret["results"])

        # 3 — judge
        expected  = q.get("expected_answer") or q.get("expected", "")
        gen_score = round(factual_coverage(output=generated, expected=expected), 4)

        is_pass = (
            ret["retrieval_score"] >= RETRIEVAL_PASS_THRESHOLD and
            gen_score              >= GENERATION_PASS_THRESHOLD
        )
        if is_pass:
            passed += 1

        elapsed = round(time.time() - t0, 1)
        print(
            f"  → {'PASS' if is_pass else 'FAIL'}  "
            f"ret={ret['retrieval_score']:.2f}  gen={gen_score:.2f}  ({elapsed}s)"
        )

        # 4 — log to Braintrust
        experiment.log(
            input={"question": q["question"]},
            output=generated,
            expected=expected,
            scores={
                "factual_coverage": gen_score,
                "retrieval_score":  ret["retrieval_score"],
            },
            metadata={
                "id":                 q["id"],
                "difficulty":         q["difficulty"],
                "relevant_tickers":   q["relevant_tickers"],
                "preferred_doc_type": preferred_doc_type,
                "tickers_hit":        ret["tickers_hit"],
                "tickers_missed":     ret["tickers_missed"],
                "retrieved_sections": ret["retrieved_sections"],
                "retrieved_chunks": [
                    {
                        "ticker":       r.get("ticker"),
                        "doc_type":     r.get("filing_type") or r.get("doc_type"),
                        "section":      r.get("section") or r.get("fiscal_quarter"),
                        "score":        round(float(r.get("score", 0)), 4),
                        "text_preview": r.get("text", "")[:300],
                    }
                    for r in ret["results"]
                ],
                "full_prompt": full_prompt,
                "pass":        is_pass,
                "elapsed_s":   elapsed,
                "model":       MODEL,
            },
        )

    experiment.close()

    total = len(eval_set)
    print(f"\n{'═'*60}")
    print(f"Results: {passed}/{total} passed ({100*passed/total:.1f}%)")
    try:
        summary = experiment.summarize(summarize_scores=False)
        print(f"\nBraintrust experiment URL:")
        print(f"  {summary.experiment_url}")
    except Exception:
        print(f"\nBraintrust project: finance-rag  |  experiment: {args.experiment}")


if __name__ == "__main__":
    main()
