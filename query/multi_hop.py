"""
Multi-hop query engine using LlamaIndex SubQuestionQueryEngine.

Wraps the existing retrieve() function as a set of per-ticker
QueryEngineTools, then uses SubQuestionQueryEngine with Claude Sonnet
to decompose complex queries into sub-questions, run them in parallel
across namespaces, and synthesize a final answer.

Usage:
    python3 query/multi_hop.py "Compare NVDA and CRWV on GPU infrastructure revenue"
    python3 query/multi_hop.py "How do NVDA and AMD describe AI demand risk?" --tickers NVDA AMD
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

try:
    import streamlit as st
    def get_secret(key: str, default: str = "") -> str:
        return st.secrets.get(key) or os.environ.get(key, default)
except ImportError:
    def get_secret(key: str, default: str = "") -> str:
        return os.environ.get(key, default)

from llama_index.core import Settings
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.core.base.response.schema import Response

# ── config ────────────────────────────────────────────────────────────────────

MODEL       = "claude-sonnet-4-20250514"
ALL_TICKERS = ["NVDA", "AMD", "AVGO", "TSM", "ANET", "MU", "CRWV"]

TICKER_DESCRIPTIONS = {
    "NVDA": "NVIDIA — GPU chips, AI data center platforms, CUDA ecosystem, gaming",
    "AMD":  "AMD — CPUs, GPUs, EPYC server processors, Instinct AI accelerators",
    "AVGO": "Broadcom — networking ASICs, infrastructure software, VMware",
    "TSM":  "TSMC — semiconductor foundry, advanced process nodes (3nm, 5nm), fab capacity",
    "ANET": "Arista Networks — cloud networking switches, EOS software, AI fabric",
    "MU":   "Micron Technology — DRAM, NAND flash, HBM memory for AI",
    "CRWV": "CoreWeave — GPU cloud infrastructure, AI compute rental, NVIDIA clusters",
}

# ── custom LLM wrapper (avoids llama-index-llms-anthropic SDK conflicts) ──────

class _ClaudeLLM(CustomLLM):
    """Thin CustomLLM wrapper around the project's anthropic client."""

    context_window: int = 200000
    num_output: int = 1024
    model_name: str = MODEL

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        import anthropic
        client = anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model=self.model_name,
            max_tokens=self.num_output,
            messages=[{"role": "user", "content": prompt}],
        )
        return CompletionResponse(text=msg.content[0].text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        # SubQuestionQueryEngine doesn't use streaming; delegate to complete
        response = self.complete(prompt, **kwargs)
        yield response


# ── per-ticker query engine wrapper ──────────────────────────────────────────

class _TickerQueryEngine(BaseQueryEngine):
    """LlamaIndex QueryEngine backed by the existing retrieve() function."""

    def __init__(self, ticker: str, top_k: int = 5):
        from llama_index.core.callbacks import CallbackManager
        self.ticker = ticker
        self.top_k  = top_k
        super().__init__(callback_manager=CallbackManager())

    def _query(self, query_bundle: QueryBundle) -> Response:
        from query.retriever import retrieve, RERANK_PER_NS
        query_str = query_bundle.query_str
        chunks = retrieve(
            query_str,
            namespace=self.ticker,
            top_k=self.top_k,
            rerank_top_n=RERANK_PER_NS,
        )
        if not chunks:
            return Response(response=f"No relevant chunks found for {self.ticker}.")

        passages = []
        for i, r in enumerate(chunks, 1):
            header = (
                f"[{self.ticker} {r.get('filing_type') or r.get('doc_type', '')} | "
                f"Period: {r.get('period_of_report', '')} | "
                f"Section: {r.get('section', '')}]"
            )
            passages.append(f"{header}\n{r.get('text', '').strip()}")

        return Response(response="\n\n".join(passages))

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        return self._query(query_bundle)

    def _get_prompt_modules(self) -> dict:
        return {}


# ── public API ────────────────────────────────────────────────────────────────

def multi_hop_query(query: str, tickers: Optional[list] = None) -> str:
    """Run a multi-hop query across the given ticker namespaces.

    Args:
        query:   Natural language question, may span multiple companies.
        tickers: List of ticker symbols to search. Defaults to ALL_TICKERS.

    Returns:
        Synthesized answer string from SubQuestionQueryEngine.
    """
    if tickers is None:
        tickers = ALL_TICKERS

    tickers = [t.upper() for t in tickers]

    llm = _ClaudeLLM()
    Settings.llm = llm
    Settings.chunk_size = 512

    tools = []
    for ticker in tickers:
        engine = _TickerQueryEngine(ticker=ticker)
        desc   = TICKER_DESCRIPTIONS.get(ticker, ticker)
        tools.append(
            QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name=f"search_{ticker}",
                    description=(
                        f"Search SEC filings and earnings transcripts for {desc}. "
                        f"Use this tool to find facts about {ticker}."
                    ),
                ),
            )
        )

    question_gen = LLMQuestionGenerator.from_defaults(llm=llm)

    engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools,
        question_gen=question_gen,
        llm=llm,
        verbose=True,
    )

    response = engine.query(query)
    return str(response)


# ── iterative retrieval ───────────────────────────────────────────────────────

_GAP_PROMPT = """\
You retrieved the following chunks for this question: {query}

Chunks:
{formatted_chunks}

Identify in one sentence: what specific information is missing that would be \
needed to fully answer this question?
If nothing is missing, say SUFFICIENT.
If something is missing, describe it as a short search query (10 words max).\
"""


def iterative_query(query: str, tickers: Optional[list] = None) -> str:
    """Two-round retrieval with gap detection between rounds.

    Round 1: retrieve top 8 chunks per ticker, rerank globally.
    Gap detection: ask Claude what's missing; if SUFFICIENT, go straight to synthesis.
    Round 2 (if gap): re-retrieve using the gap description as a targeted query,
                      merge + deduplicate by chunk id, keep top 12.
    Synthesis: build prompt via prompt_builder and call Claude.

    Args:
        query:   Original user question.
        tickers: Ticker namespaces to search. Defaults to ALL_TICKERS.

    Returns:
        Final synthesised answer string.
    """
    import anthropic
    from query.retriever import retrieve, rerank, RERANK_FINAL_N
    from query.prompt_builder import build_prompt

    if tickers is None:
        tickers = ALL_TICKERS
    tickers = [t.upper() for t in tickers]

    client = anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))

    # ── round 1 ───────────────────────────────────────────────────────────────
    _SECTION_FILTER = {
        "section": {"$in": [
            "Item 1. Business",
            "ITEM 1. BUSINESS",
            "Item 1",
            "ITEM 1",
            "Business",
            "Our Company",
            "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operat",
            "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS",
            "Item 7",
            "ITEM 7",
            "Item 7. Management",
            "Management's Discussion",
            "Item 1A. Risk Factors",
            "ITEM 1A. RISK FACTORS",
        ]}
    }

    r1_query = f"{query} 10-K annual filing"
    print(f"  [iterative] round 1 | query='{r1_query[:80]}' | tickers={tickers}")
    r1_raw = []
    for ticker in tickers:
        r1_raw.extend(retrieve(
            r1_query, namespace=ticker, top_k=8, rerank_top_n=2,
            doc_type_filter="filing",
            metadata_filter=_SECTION_FILTER,
        ))
    r1_top = rerank(query, r1_raw, top_n=RERANK_FINAL_N)
    print(f"  [iterative] round 1 retrieved {len(r1_top)} chunks")

    # ── gap detection ─────────────────────────────────────────────────────────
    formatted_chunks = "\n\n".join(
        f"[{r.get('ticker')} {r.get('filing_type') or r.get('doc_type', '')} | "
        f"Period: {r.get('period_of_report', '')} | Section: {r.get('section', '')}]\n"
        f"{r.get('text', '').strip()[:400]}"
        for r in r1_top
    )

    gap_msg = client.messages.create(
        model=MODEL,
        max_tokens=64,
        messages=[{
            "role": "user",
            "content": _GAP_PROMPT.format(query=query, formatted_chunks=formatted_chunks),
        }],
    )
    gap_response = gap_msg.content[0].text.strip()
    print(f"  [iterative] gap detected: '{gap_response}'")

    # ── round 2 (conditional) ─────────────────────────────────────────────────
    if gap_response.upper() != "SUFFICIENT":
        print(f"  [iterative] round 2 | gap_query='{gap_response[:80]}'")
        r2_raw = []
        for ticker in tickers:
            r2_raw.extend(retrieve(
                gap_response, namespace=ticker, top_k=8, rerank_top_n=2,
                doc_type_filter="filing",
            ))

        # merge and deduplicate by chunk id
        seen_ids = {r["id"] for r in r1_top}
        merged = list(r1_top)
        for r in r2_raw:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                merged.append(r)

        final_chunks = rerank(query, merged, top_n=12)
        new_chunks = len(final_chunks) - len(r1_top)
        print(f"  [iterative] round 2 added {new_chunks} new chunks (total {len(final_chunks)})")
    else:
        final_chunks = r1_top
        print("  [iterative] evidence sufficient — skipping round 2")

    # ── synthesis ─────────────────────────────────────────────────────────────
    prompt = build_prompt(query, final_chunks)
    synth_msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return synth_msg.content[0].text


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Multi-hop query to run")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Ticker symbols to search (default: all)"
    )
    args = parser.parse_args()

    print(f"\nQuery   : {args.query}")
    print(f"Tickers : {args.tickers or ALL_TICKERS}\n")
    print("=" * 60)

    answer = multi_hop_query(args.query, tickers=args.tickers)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print(answer)
