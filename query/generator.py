"""
Generate a grounded answer from retrieved chunks using the Claude API.

Usage:
    python3 query/generator.py "What are NVDA's main revenue segments?"
    python3 query/generator.py "What is NVDA's data center revenue?" --namespace NVDA
"""

import os
import argparse
from pathlib import Path

from dotenv import load_dotenv
import anthropic

from retriever import retrieve
from prompt_builder import build_prompt

# ── config ────────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

# ── generator ─────────────────────────────────────────────────────────────────

def generate(
    query: str,
    namespace: str = "NVDA",
    top_k: int = 5,
) -> dict:
    """Retrieve chunks, build prompt, call Claude, return answer + sources.

    Returns:
        {
            "answer":   str,           # Claude's response text
            "sources":  list[dict],    # the retriever results used
            "query":    str,
            "namespace": str,
        }
    """
    results = retrieve(query, namespace=namespace, top_k=top_k)
    prompt = build_prompt(query, results)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": message.content[0].text,
        "sources": results,
        "query": query,
        "namespace": namespace,
    }


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG answer generator.")
    parser.add_argument("query", help="Natural language question")
    parser.add_argument("--namespace", default="NVDA", help="Ticker namespace")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    result = generate(args.query, namespace=args.namespace, top_k=args.top_k)

    print(f"\n{'═' * 60}")
    print(f"Query : {result['query']}")
    print(f"Model : {MODEL}  |  namespace: {result['namespace']}")
    print(f"{'═' * 60}\n")
    print(result["answer"])
    print(f"\n{'─' * 60}")
    print("Sources:")
    for i, s in enumerate(result["sources"], start=1):
        print(
            f"  [{i}] {s.get('ticker')} {s.get('filing_type')} "
            f"filed {s.get('filed_at')} | {s.get('section')} "
            f"(score: {s.get('score', 0):.4f})"
        )
