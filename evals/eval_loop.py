"""
Agentic eval runner with root-cause diagnostics.

Runs the full eval set, pushes to Braintrust, then reads results back
and produces a per-failure diagnostic report with suggested fixes.

Usage:
    python3 evals/eval_loop.py
    python3 evals/eval_loop.py --difficulty medium
    python3 evals/eval_loop.py --experiment finance-rag-v3
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
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
LOGS_DIR            = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

RETRIEVAL_PASS_THRESHOLD  = 0.5
GENERATION_PASS_THRESHOLD = 0.6
RETRIEVAL_GAP_THRESHOLD   = 0.7   # below this → retrieval_gap
GENERATION_GAP_THRESHOLD  = 0.6   # below this (with perfect ret) → generation_gap

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── logging ───────────────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path  = LOGS_DIR / f"eval_loop_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

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


def factual_coverage(output: str, expected: str) -> float:
    prompt = _JUDGE_PROMPT.format(output=output, expected=expected)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4,
        messages=[{"role": "user", "content": prompt}],
    )
    letter = msg.content[0].text.strip().upper()[:1]
    return _CHOICE_SCORES.get(letter, 0.0)

# ── retrieval + generation ────────────────────────────────────────────────────

def _run_retrieval(question: str, relevant_tickers: list, doc_type: str = None) -> dict:
    all_results = []
    for ticker in ALL_TICKERS:
        chunks = retrieve(
            question,
            namespace=ticker,
            top_k=TOP_K,
            rerank_top_n=RERANK_PER_NS,
            doc_type_filter=doc_type,
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
        "retrieved_sections": retrieved_sections,
    }


def _run_generation(question: str, results: list) -> tuple:
    prompt = build_prompt(question, results)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text, prompt

# ── root cause classifier ─────────────────────────────────────────────────────

def classify_failure(ret_score: float, gen_score: float, q: dict) -> tuple[str, str]:
    """Return (failure_type, suggested_fix)."""

    if ret_score < RETRIEVAL_GAP_THRESHOLD:
        missed = q.get("_tickers_missed", [])
        if missed:
            fix = (
                f"Missing chunks from {missed}. "
                f"Check Pinecone namespace for those tickers. "
                f"Consider ingesting additional filings or transcripts."
            )
        else:
            fix = (
                "Low retrieval score despite all tickers present. "
                "Try tweaking the query or increasing TOP_K / RERANK_FINAL_N."
            )
        return "retrieval_gap", fix

    if gen_score < GENERATION_GAP_THRESHOLD:
        preferred = q.get("preferred_doc_type", "both")
        if preferred == "filing" and gen_score < 0.4:
            fix = (
                "Filing chunks present but generation quality is low. "
                "The answer may require synthesis across multiple sections. "
                "Consider increasing RERANK_FINAL_N or switching to 'both' doc types."
            )
        else:
            fix = (
                "Retrieval is strong but Claude is not synthesising correctly. "
                "Review the grounding prompt — the required fact may be in a table "
                "or require arithmetic not currently permitted."
            )
        return "generation_gap", fix

    # retrieval ok, gen borderline — likely chunk boundary issue
    return "chunk_boundary", (
        "Factual answer exists in context but spans a chunk boundary. "
        "Consider reducing CHUNK_SIZE or increasing CHUNK_OVERLAP in the ingestor."
    )

# ── diagnostic report ─────────────────────────────────────────────────────────

def _report_failure(q: dict, ret: dict, gen_score: float, failure_type: str, fix: str):
    sep = "─" * 60
    log.info(sep)
    log.info(f"FAIL  Q{q['id']} ({q['difficulty']}): {q['question'][:80]}…")
    log.info(f"  failure_type : {failure_type}")
    log.info(f"  ret_score    : {ret['retrieval_score']:.2f}  |  gen_score: {gen_score:.2f}")
    log.info(f"  tickers_hit  : {ret['tickers_hit']}")
    log.info(f"  tickers_miss : {ret['tickers_missed']}")
    log.info("  top chunks:")
    for r in ret["results"][:3]:
        log.info(
            f"    [{r.get('ticker')}] {r.get('section') or r.get('fiscal_quarter', '—')} "
            f"(score={r.get('score', 0):.3f}) — {r.get('text', '')[:120]}…"
        )
    log.info(f"  suggested_fix: {fix}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=f"finance-rag-loop-{timestamp}")
    args = parser.parse_args()

    bt_key = os.environ.get("BRAINTRUST_API_KEY", "")
    if not bt_key:
        log.error("BRAINTRUST_API_KEY not set in .env")
        return

    eval_set = json.loads(EVAL_SET_PATH.read_text())
    if args.difficulty:
        eval_set = [q for q in eval_set if q["difficulty"] == args.difficulty]
    if not eval_set:
        log.error("No questions matched the filter.")
        return

    log.info(f"Running {len(eval_set)} question(s) → Braintrust '{args.experiment}'")
    log.info(f"Log: {log_path}\n")

    experiment = braintrust.init(
        project=BRAINTRUST_PROJECT,
        experiment=args.experiment,
        api_key=bt_key,
    )

    passed   = 0
    failures = []

    for i, q in enumerate(eval_set, 1):
        t0 = time.time()
        log.info(f"[{i}/{len(eval_set)}] Q{q['id']} ({q['difficulty']}): {q['question'][:70]}…")

        preferred_doc_type = q.get("preferred_doc_type")

        ret = _run_retrieval(q["question"], q["relevant_tickers"], preferred_doc_type)
        generated, full_prompt = _run_generation(q["question"], ret["results"])
        gen_score = round(factual_coverage(output=generated, expected=q["expected_answer"]), 4)

        is_pass = (
            ret["retrieval_score"] >= RETRIEVAL_PASS_THRESHOLD and
            gen_score              >= GENERATION_PASS_THRESHOLD
        )
        if is_pass:
            passed += 1

        elapsed = round(time.time() - t0, 1)
        log.info(
            f"  → {'PASS' if is_pass else 'FAIL'}  "
            f"ret={ret['retrieval_score']:.2f}  gen={gen_score:.2f}  ({elapsed}s)"
        )

        experiment.log(
            input={"question": q["question"]},
            output=generated,
            expected=q["expected_answer"],
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

        if not is_pass:
            q["_tickers_missed"] = ret["tickers_missed"]
            failure_type, fix = classify_failure(ret["retrieval_score"], gen_score, q)
            failures.append({
                "q": q, "ret": ret, "gen_score": gen_score,
                "failure_type": failure_type, "fix": fix,
            })

    experiment.close()

    total = len(eval_set)
    log.info(f"\n{'═'*60}")
    log.info(f"Results: {passed}/{total} passed ({100*passed/total:.1f}%)")

    # ── diagnostic report ──────────────────────────────────────────────────────
    if failures:
        log.info(f"\n{'═'*60}")
        log.info(f"DIAGNOSTIC REPORT — {len(failures)} failure(s)")
        log.info(f"{'═'*60}")

        by_type: dict[str, list] = {}
        for f in failures:
            by_type.setdefault(f["failure_type"], []).append(f)

        for ftype, items in by_type.items():
            log.info(f"\n▸ {ftype.upper()} ({len(items)} question(s))")
            for f in items:
                _report_failure(f["q"], f["ret"], f["gen_score"], f["failure_type"], f["fix"])

        log.info(f"\nSummary by failure type:")
        for ftype, items in by_type.items():
            log.info(f"  {ftype:<20} {len(items)} question(s): "
                     f"{[f['q']['id'] for f in items]}")

    try:
        summary = experiment.summarize(summarize_scores=False)
        log.info(f"\nBraintrust URL: {summary.experiment_url}")
    except Exception:
        log.info(f"\nBraintrust project: {BRAINTRUST_PROJECT}  experiment: {args.experiment}")

    log.info(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
