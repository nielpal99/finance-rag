"""
Stress tester: sample random chunks from Pinecone, generate questions from
them with Claude, run through the full RAG pipeline, then judge whether the
answer is grounded in the source chunk or hallucinated.

Pushes results to Braintrust as experiment "stress-test-{timestamp}".

Usage:
    python3 evals/stress_test.py
    python3 evals/stress_test.py --chunks 20       # sample 20 chunks
    python3 evals/stress_test.py --ticker NVDA     # single namespace
"""

import os
import sys
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "query"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic
import braintrust
from pinecone import Pinecone
from retriever import retrieve, rerank, RERANK_PER_NS, RERANK_FINAL_N
from prompt_builder import build_prompt

# ── config ────────────────────────────────────────────────────────────────────

MODEL           = "claude-sonnet-4-20250514"
ALL_TICKERS     = ["NVDA", "AMD", "AVGO", "TSM", "ANET", "MU", "CRWV"]
INDEX_NAME      = "finance-rag"
EMBED_DIM       = 1024          # bge-large-en-v1.5
TOP_K           = 8
BRAINTRUST_PROJECT = "finance-rag"

client    = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── prompts ───────────────────────────────────────────────────────────────────

_QUESTION_GEN_PROMPT = """\
You are a financial analyst. Read the following excerpt from a SEC filing or \
earnings transcript and write ONE specific, answerable question whose answer \
is clearly contained in the excerpt. The question should require finding a \
specific fact, number, or statement — not general knowledge.

Excerpt:
{text}

Write only the question. No preamble."""


_GROUNDING_JUDGE_PROMPT = """\
You are evaluating whether a RAG system's answer is grounded in a source chunk \
or contains hallucinated information.

Source chunk (ground truth):
{source_chunk}

Generated answer:
{answer}

Does the answer contradict or introduce facts NOT present in the source chunk?

Answer with exactly one word: GROUNDED or HALLUCINATED."""

# ── helpers ───────────────────────────────────────────────────────────────────

def _sample_random_chunks(tickers: list[str], n: int) -> list[dict]:
    """Fetch n random chunks from Pinecone using a zero vector query."""
    pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    # zero vector returns lowest-similarity results — effectively random sample
    # We oversample then shuffle to get variety across namespaces
    per_ns   = max(n, 5)
    sampled  = []

    for ticker in tickers:
        try:
            resp = index.query(
                vector=[0.0] * EMBED_DIM,
                top_k=per_ns,
                namespace=ticker,
                include_metadata=True,
            )
            for match in resp.matches:
                text = match.metadata.get("text", "")
                if len(text) > 100:   # skip very short chunks
                    sampled.append({
                        "id":       match.id,
                        "ticker":   ticker,
                        "text":     text,
                        "section":  match.metadata.get("section") or match.metadata.get("fiscal_quarter", ""),
                        "doc_type": match.metadata.get("filing_type") or match.metadata.get("doc_type", ""),
                    })
        except Exception as e:
            print(f"  [sample] {ticker} error: {e}")

    random.shuffle(sampled)
    return sampled[:n]


def _generate_question(chunk_text: str) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": _QUESTION_GEN_PROMPT.format(text=chunk_text[:800])}],
    )
    return msg.content[0].text.strip()


def _run_rag(question: str) -> tuple[str, list[dict]]:
    all_results = []
    for ticker in ALL_TICKERS:
        chunks = retrieve(
            question,
            namespace=ticker,
            top_k=TOP_K,
            rerank_top_n=RERANK_PER_NS,
        )
        all_results.extend(chunks)

    top    = rerank(question, all_results, top_n=RERANK_FINAL_N)
    prompt = build_prompt(question, top)
    msg    = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text, top


def _judge_grounding(source_chunk: str, answer: str) -> float:
    prompt = _GROUNDING_JUDGE_PROMPT.format(
        source_chunk=source_chunk[:800],
        answer=answer,
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = msg.content[0].text.strip().upper()
    return 1.0 if "GROUNDED" in verdict else 0.0

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=10, help="Number of chunks to sample")
    parser.add_argument("--ticker", type=str, default=None, help="Restrict to one ticker")
    args = parser.parse_args()

    bt_key = os.environ.get("BRAINTRUST_API_KEY", "")
    if not bt_key:
        print("ERROR: BRAINTRUST_API_KEY not set in .env")
        return

    tickers     = [args.ticker.upper()] if args.ticker else ALL_TICKERS
    experiment_name = f"stress-test-{timestamp}"

    print(f"Sampling {args.chunks} chunks from {tickers} …")
    chunks = _sample_random_chunks(tickers, args.chunks)
    if not chunks:
        print("No chunks sampled — check Pinecone connectivity.")
        return

    print(f"Sampled {len(chunks)} chunks. Running stress test → Braintrust '{experiment_name}'\n")

    experiment = braintrust.init(
        project=BRAINTRUST_PROJECT,
        experiment=experiment_name,
        api_key=bt_key,
    )

    grounded_count = 0
    hallucinated   = []

    for i, chunk in enumerate(chunks, 1):
        t0 = time.time()
        print(f"[{i}/{len(chunks)}] {chunk['ticker']} / {chunk['section'] or chunk['doc_type']}")

        # 1 — generate question from source chunk
        question = _generate_question(chunk["text"])
        print(f"  Q: {question[:80]}…" if len(question) > 80 else f"  Q: {question}")

        # 2 — run full RAG pipeline
        answer, sources = _run_rag(question)

        # 3 — judge grounding against source chunk
        score = _judge_grounding(chunk["text"], answer)
        label = "GROUNDED" if score == 1.0 else "HALLUCINATED"
        elapsed = round(time.time() - t0, 1)

        print(f"  → {label}  ({elapsed}s)")

        if score == 1.0:
            grounded_count += 1
        else:
            hallucinated.append({"chunk": chunk, "question": question, "answer": answer})

        # 4 — log to Braintrust
        experiment.log(
            input={"question": question, "source_chunk_id": chunk["id"]},
            output=answer,
            expected=chunk["text"],      # source chunk as ground truth
            scores={"grounding": score},
            metadata={
                "source_ticker":  chunk["ticker"],
                "source_section": chunk["section"],
                "source_doc_type": chunk["doc_type"],
                "source_text_preview": chunk["text"][:300],
                "retrieved_chunks": [
                    {
                        "ticker":  r.get("ticker"),
                        "section": r.get("section") or r.get("fiscal_quarter"),
                        "score":   round(float(r.get("score", 0)), 4),
                    }
                    for r in sources[:5]
                ],
                "elapsed_s": elapsed,
                "model":     MODEL,
            },
        )

    experiment.close()

    total = len(chunks)
    print(f"\n{'═'*60}")
    print(f"Results: {grounded_count}/{total} grounded ({100*grounded_count/total:.1f}%)")

    if hallucinated:
        print(f"\n⚠ {len(hallucinated)} hallucinated answer(s):")
        for h in hallucinated:
            print(f"  [{h['chunk']['ticker']}] {h['question'][:70]}…")
            print(f"    Source: {h['chunk']['text'][:120]}…")
            print(f"    Answer: {h['answer'][:120]}…\n")

    try:
        summary = experiment.summarize(summarize_scores=False)
        print(f"Braintrust URL: {summary.experiment_url}")
    except Exception:
        print(f"Braintrust project: {BRAINTRUST_PROJECT}  experiment: {experiment_name}")


if __name__ == "__main__":
    main()
