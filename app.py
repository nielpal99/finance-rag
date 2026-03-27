"""
Streamlit UI for the Finance RAG pipeline.

Run with:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

# Make query/ importable
sys.path.insert(0, str(Path(__file__).parent / "query"))

import anthropic
import streamlit as st

from retriever import retrieve, rerank, RERANK_PER_NS, RERANK_FINAL_N
from prompt_builder import build_prompt
from query.tavily_search import tavily_search, should_use_tavily

sys.path.insert(0, str(Path(__file__).parent / ".streamlit"))
from query_router import route_query, detect_ticker, query_snowflake

# ── config ────────────────────────────────────────────────────────────────────

MODEL      = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
ALL_TICKERS = ["NVDA", "AMD", "AVGO", "TSM", "ANET", "MU"]
TOP_K       = 8

_DOC_TYPE_MAP = {
    "Filings":     "filing",
    "Transcripts": "transcript",
    "Both":        None,
}

NOT_FOUND_PHRASE = "I couldn't find this in the available documents"

_RISK_KEYWORDS = {"risk", "risk factor", "disclosed"}

def preprocess_query(query: str) -> str:
    """Append an Item 1A section hint for risk-related queries."""
    lower = query.lower()
    if any(kw in lower for kw in _RISK_KEYWORDS):
        return query + " Item 1A risk factors"
    return query

# ── Anthropic client (cached) ─────────────────────────────────────────────────

@st.cache_resource
def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    return anthropic.Anthropic(api_key=api_key)

# ── RAG pipeline ──────────────────────────────────────────────────────────────

def run_rag(query: str, tickers: list[str], doc_type_filter) -> dict:
    """Retrieve across selected tickers, rerank globally, stream answer."""
    retrieval_query = preprocess_query(query)
    all_results = []
    for ticker in tickers:
        chunks = retrieve(
            retrieval_query,
            namespace=ticker,
            top_k=TOP_K,
            rerank_top_n=RERANK_PER_NS,
            doc_type_filter=doc_type_filter,
        )
        all_results.extend(chunks)

    top = rerank(query, all_results, top_n=RERANK_FINAL_N)
    prompt = build_prompt(query, top)
    return {"sources": top, "prompt": prompt}


def stream_answer(prompt: str):
    """Yield text chunks from Claude via the streaming API."""
    client = get_client()
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text

# ── page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Infrastructure Research",
    page_icon="📊",
    layout="wide",
)

st.title("AI Infrastructure Research")
st.caption("Grounded answers from SEC filings and earnings transcripts — NVDA · AMD · AVGO · TSM · ANET · MU")

# ── session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []   # list of {query, answer, sources}

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")

    st.subheader("Companies")
    all_selected = st.checkbox("All Companies", value=True)

    ticker_checks = {}
    for t in ALL_TICKERS:
        ticker_checks[t] = st.checkbox(t, value=all_selected, disabled=all_selected)

    selected_tickers = ALL_TICKERS if all_selected else [t for t, v in ticker_checks.items() if v]

    st.divider()

    st.subheader("Document type")
    doc_label = st.radio("Source", ["Filings", "Transcripts", "Both"], index=2)
    doc_type_filter = _DOC_TYPE_MAP[doc_label]

    st.divider()

    show_chunks = st.toggle("Show retrieved chunks", value=False)

    st.divider()

    st.subheader("Recent queries")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"- {item['query'][:60]}{'…' if len(item['query']) > 60 else ''}")
    else:
        st.caption("None yet")

# ── main area ─────────────────────────────────────────────────────────────────

if not selected_tickers:
    st.warning("Select at least one company in the sidebar.")
    st.stop()

query = st.text_area(
    "Question",
    placeholder="e.g. How does NVDA's data center revenue growth compare to AMD's?",
    height=90,
    label_visibility="collapsed",
)

submitted = st.button("Ask", type="primary", use_container_width=False)

if submitted and query.strip():
    clean_query = query.strip()
    route = route_query(clean_query)
    ticker = detect_ticker(clean_query)

    if route == "snowflake" and ticker:
        st.divider()
        with st.spinner("Querying structured financial data…"):
            sf_answer = query_snowflake(clean_query, ticker)
        st.markdown(sf_answer)
        st.caption("📊 Source: Snowflake XBRL (SEC EDGAR structured data)")
        st.session_state.history.append({
            "query": clean_query,
            "answer": sf_answer,
            "sources": [],
        })
        st.session_state.history = st.session_state.history[-5:]
    else:
        with st.spinner("Retrieving…"):
            rag = run_rag(clean_query, selected_tickers, doc_type_filter)

        sources  = rag["sources"]
        prompt   = rag["prompt"]

        st.divider()

        # ── stream answer ──────────────────────────────────────────────────────
        answer_box = st.empty()
        full_answer = ""

        with st.spinner("Generating…"):
            for chunk in stream_answer(prompt):
                full_answer += chunk
                answer_box.markdown(full_answer + "▌")

        answer_box.markdown(full_answer)

        # Tavily fallback — trigger if RAG couldn't find answer or confidence is low
        if should_use_tavily(full_answer, sources):
            with st.spinner("Searching latest news and filings…"):
                tavily_answer = tavily_search(clean_query, selected_tickers)
            if tavily_answer:
                st.divider()
                st.markdown("### 🌐 Live Web Results")
                st.markdown(tavily_answer)
                st.caption("Source: Tavily search")
        elif NOT_FOUND_PHRASE in full_answer:
            st.info(
                "The pipeline couldn't find relevant information in the selected sources. "
                "Try switching the document type filter or broadening your company selection.",
                icon="ℹ️",
            )

        # ── save to history ────────────────────────────────────────────────────
        st.session_state.history.append({
            "query":   clean_query,
            "answer":  full_answer,
            "sources": sources,
        })
        st.session_state.history = st.session_state.history[-5:]

        # ── sources ────────────────────────────────────────────────────────────
        st.divider()
        with st.expander(f"Sources ({len(sources)} chunks retrieved)", expanded=show_chunks):
            for i, r in enumerate(sources, 1):
                ticker    = r.get("ticker", "—")
                doc_type  = r.get("filing_type") or r.get("doc_type", "—")
                section   = r.get("section") or r.get("fiscal_quarter", "—")
                date      = r.get("filed_at") or r.get("filing_date") or r.get("period_of_report", "—")
                score     = r.get("score", 0.0)

                col1, col2, col3, col4, col5 = st.columns([1, 2, 3, 2, 1])
                col1.markdown(f"**{ticker}**")
                col2.caption(doc_type)
                col3.caption(section)
                col4.caption(date)
                col5.caption(f"{score:.3f}")

                if show_chunks:
                    st.markdown(
                        f"<div style='background:#f8f9fa;border-left:3px solid #dee2e6;"
                        f"padding:8px 12px;margin:4px 0 12px 0;font-size:0.85em;"
                        f"border-radius:2px'>{r.get('text','')[:500]}</div>",
                        unsafe_allow_html=True,
                    )

elif submitted and not query.strip():
    st.warning("Please enter a question.")
