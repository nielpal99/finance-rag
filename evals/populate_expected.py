"""
Populate expected answers for eval questions where expected == "TBD".

Runs each TBD question through the full RAG pipeline and writes Claude's
answer back to evals/eval_set.json as the expected answer.

Usage:
    python3 evals/populate_expected.py           # all TBD questions
    python3 evals/populate_expected.py --dry-run # preview without writing
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "query"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic
from retriever import retrieve, rerank, RERANK_PER_NS, RERANK_FINAL_N
from prompt_builder import build_prompt

MODEL         = "claude-sonnet-4-20250514"
TOP_K         = 8
ALL_TICKERS   = ["NVDA", "AMD", "AVGO", "TSM", "ANET", "MU", "CRWV"]
EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def run_pipeline(question: str, relevant_tickers: list, doc_type: str = None) -> str:
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

    top    = rerank(question, all_results, top_n=RERANK_FINAL_N)
    prompt = build_prompt(question, top)
    msg    = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print answers without writing")
    args = parser.parse_args()

    data = json.loads(EVAL_SET_PATH.read_text())

    tbd = [q for q in data if q.get("expected") == "TBD" or q.get("expected_answer") == "TBD"]
    if not tbd:
        print("No TBD questions found.")
        return

    print(f"Found {len(tbd)} TBD question(s). {'(dry run)' if args.dry_run else 'Writing to eval_set.json'}\n")

    for i, q in enumerate(tbd, 1):
        qid  = q.get("id")
        diff = q.get("difficulty", "")
        print(f"[{i}/{len(tbd)}] {qid} ({diff}): {q['question'][:70]}…")

        answer = run_pipeline(
            q["question"],
            q.get("relevant_tickers", ALL_TICKERS),
            q.get("preferred_doc_type"),
        )

        print(f"  → {answer[:120]}…\n" if len(answer) > 120 else f"  → {answer}\n")

        if not args.dry_run:
            # Update whichever key the question uses
            if "expected_answer" in q:
                q["expected_answer"] = answer
            else:
                q["expected"] = answer

    if not args.dry_run:
        EVAL_SET_PATH.write_text(json.dumps(data, indent=2))
        print(f"✅ Wrote {len(tbd)} expected answer(s) to {EVAL_SET_PATH}")
    else:
        print("Dry run complete — no changes written.")


if __name__ == "__main__":
    main()
