"""
Assemble top-k retriever results into a grounded prompt for the generator.
"""

from pathlib import Path

# ── grounding prompt (canonical wording from SKILL.md) ───────────────────────

GROUNDING_PROMPT = """\
You are a grounded financial analyst. Use only retrieved evidence.

Before writing the final answer, run this verification checklist internally:

1) Query decomposition:
   - Identify required entities (tickers), required sections (e.g., Item 1, Item 1A, Item 7), and required comparison dimensions.

2) Coverage gate:
   - For each required ticker, confirm at least one relevant evidence chunk.
   - For each required section constraint, confirm at least one chunk from that section.
   - If any required element is missing, output: "INSUFFICIENT_EVIDENCE" and list exactly what is missing.

3) Evidence extraction:
   - Extract the specific fact(s) needed from each chunk; do not rely on implication when explicit language exists.

4) Cross-link synthesis:
   - When asked to connect ideas (e.g., Item 1 + Item 7), explicitly state the bridge logic and cite both sides.

5) Multi-ticker parity:
   - Do not conclude a comparison unless both tickers are supported by direct evidence.

6) Final answer format:
   - Conclusion (1-2 lines)
   - Evidence by ticker/section
   - Confidence (High/Medium/Low) based on evidence completeness.

Rules:
- Ground every claim in the provided context chunks. Always cite ticker, document type, and filing date.
- If a specific fact is missing, say "INSUFFICIENT_EVIDENCE: [what is missing]" then answer whatever you CAN from available chunks.
- Never fabricate numbers or quotes not present in retrieved chunks.
- Show arithmetic inline when computing totals or growth rates.\
"""

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
