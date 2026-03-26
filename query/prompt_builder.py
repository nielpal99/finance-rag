"""
Assemble top-k retriever results into a grounded prompt for the generator.
"""

from pathlib import Path

# ── grounding prompt (canonical wording from SKILL.md) ───────────────────────

GROUNDING_PROMPT = (
    "You are a financial research assistant analyzing SEC filings and\n"
    "earnings call transcripts for AI infrastructure companies.\n\n"
    "Rules:\n"
    "- Answer ONLY from the provided context chunks. Do not use prior knowledge.\n"
    '- If the answer is not present in the provided context, say exactly:\n'
    '  "I couldn\'t find this in the available documents."\n'
    "- Always cite your sources: include the company ticker, document type,\n"
    "  and filing date for every claim.\n"
    "- You MAY synthesize and compute straightforward totals or comparisons\n"
    "  (e.g. summing line items, computing year-over-year growth) when all\n"
    "  the underlying numbers are explicitly present in the retrieved chunks.\n"
    "  Show your arithmetic inline so the reader can verify it.\n"
    "- Never infer, speculate, or fill gaps with outside knowledge.\n"
    "  If context is ambiguous or incomplete, say so.\n"
    "- When comparing companies, only make claims that are directly supported\n"
    "  by retrieved text from each company."
)

# ── builder ───────────────────────────────────────────────────────────────────

def build_prompt(query: str, results: list[dict]) -> str:
    """Return a fully-assembled prompt string ready to send to Claude.

    Args:
        query:   The user's natural language question.
        results: List of dicts returned by retriever.retrieve() — each has
                 score, id, section, chunk_index, filed_at, period_of_report,
                 ticker, filing_type, accession_number, source_url, text.
    """
    passages = []
    for i, r in enumerate(results, start=1):
        header = (
            f"[Result {i}] "
            f"{r.get('ticker')} {r.get('filing_type')} | "
            f"Period: {r.get('period_of_report')} | "
            f"Section: {r.get('section')} | "
            f"Score: {r.get('score', 0):.4f}"
        )
        passages.append(f"{header}\n{r.get('text', '').strip()}")

    context_block = "\n\n".join(passages)

    return (
        f"{GROUNDING_PROMPT}\n\n"
        f"---\n\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"Question: {query}"
    )
