# Finance RAG

A retrieval-augmented generation pipeline for researching AI infrastructure companies. Ask natural language questions across SEC filings and earnings call transcripts for NVDA, AMD, AVGO, TSM, ANET, MU, and CRWV — and get grounded, cited answers.

Built for analysts who want to query 10-Ks and earnings transcripts the way they'd query a database.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    Streamlit UI                      │
│         (sidebar: ticker filter, doc type)           │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   Query Router   │
          └────────┬────────┘
                   │
       ┌───────────┼────────────────────┐
       ▼           ▼                    ▼
  Snowflake    is_complex_query()    Standard RAG
  XBRL route   (multi-ticker,        (single-ticker,
  (metrics,    compare/vs/item       factual lookup)
  financials)  keywords)                  │
       │            │                    │
       │            ▼                    ▼
       │    LlamaIndex                Pinecone
       │    SubQuestion-           vector search
       │    QueryEngine            + Cohere rerank
       │    (per-ticker               │
       │     sub-questions)           ▼
       │            │           prompt_builder.py
       │            │           (grounding prompt
       │            │            + checklist)
       │            │                  │
       └────────────┴──────────────────┘
                    │
                    ▼
             Claude Sonnet
          (grounded synthesis
           with citations)
                    │
                    ▼
            Tavily fallback
         (if INSUFFICIENT_EVIDENCE
          or low-confidence answer)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Generation | Claude claude-sonnet-4-20250514 (Anthropic) |
| Embeddings | bge-large-en-v1.5 (BAAI, via SentenceTransformers) |
| Vector DB | Pinecone (single index, namespace per ticker) |
| Reranking | Cohere Rerank v3.5 |
| Multi-hop | LlamaIndex SubQuestionQueryEngine |
| Structured data | Snowflake + SEC EDGAR XBRL API |
| Web fallback | Tavily search |
| Ingestion compute | Modal (serverless A10G GPU) |

---

## Coverage

| Ticker | Company | Filing type | Chunks |
|---|---|---|---|
| NVDA | NVIDIA | 10-K | 744 |
| AMD | Advanced Micro Devices | 10-K | 648 |
| AVGO | Broadcom | 10-K | 659 |
| TSM | Taiwan Semiconductor | 20-F | 1,168 |
| ANET | Arista Networks | 10-K | 812 |
| MU | Micron Technology | 10-K | 494 |
| CRWV | CoreWeave | 10-K | 341 |
| **Total** | | | **4,866** |

---

## Eval Results — V2

Evaluated on 30 questions across 4 difficulty tiers using a factual coverage judge (Claude-as-judge, A–E scoring).

**Overall: 21/30 (70.0%)**

| Tier | Score | Notes |
|---|---|---|
| Easy (Q1–Q8) | 7/8 (87.5%) | Single-ticker factual lookups |
| Medium (Q9–Q15) | 5/7 (71.4%) | Multi-ticker, risk themes |
| Hard (Q16–Q20) | 3/5 (60.0%) | Cross-section synthesis, arithmetic |
| Stress Test (ST01–ST10) | 6/10 (60.0%) | Missing-link, multi-ticker traps |

Retrieval score is 1.00 on all failing questions — remaining failures are generation-side synthesis gaps, not retrieval gaps.

---

## V2 Features

### Snowflake XBRL Integration
Metric questions ("what was NVDA revenue", "AMD operating income") are routed to a Snowflake table loaded from SEC EDGAR's XBRL API. Covers US-GAAP and IFRS concepts across all 7 tickers.

### LlamaIndex Multi-Hop
Complex questions (multi-ticker comparisons, item-number cross-references) are routed through a LlamaIndex `SubQuestionQueryEngine`. It decomposes the query into per-ticker sub-questions, retrieves in parallel, and synthesizes a unified answer.

### Tavily Web Fallback
If the RAG pipeline returns `INSUFFICIENT_EVIDENCE` or low-confidence chunks, Tavily search fetches live web results as a supplementary source.

### CoreWeave (CRWV)
CoreWeave's 10-K (filed 2026-03-02, period ending 2025-12-31) is fully ingested — 341 chunks covering their GPU cloud platform, revenue trajectory ($229M → $1.9B → $5.1B), and infrastructure strategy.

---

## Setup

### Prerequisites
- Python 3.9+
- API keys: `ANTHROPIC_API_KEY`, `PINECONE_API_KEY`, `COHERE_API_KEY`, `BRAINTRUST_API_KEY`, `TAVILY_API_KEY`
- Optional: `SNOWFLAKE_*` credentials, `MODAL_TOKEN_*` for re-ingestion

### Install

```bash
git clone https://github.com/nielpal99/finance-rag
cd finance-rag
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Fill in your API keys
```

Or for Streamlit Cloud, add keys to `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "..."
PINECONE_API_KEY  = "..."
COHERE_API_KEY    = "..."
TAVILY_API_KEY    = "..."
```

### Run locally

```bash
streamlit run app.py
```

### Run evals

```bash
# Full eval set (30 questions)
python3 evals/run_eval.py

# Single question
python3 evals/run_eval.py --id 7

# By difficulty
python3 evals/run_eval.py --difficulty hard

# Agentic eval with root-cause diagnostics
python3 evals/eval_loop.py

# Hallucination stress test
python3 evals/stress_test.py --chunks 20
```

### Re-ingest a ticker

```bash
# Standard tickers (Modal GPU)
modal run ingestion/embedder.py --ticker NVDA

# CoreWeave (local)
python3 ingestion/ingest_coreweave.py | tee logs/ingest_crwv.log
```

---

## Project Structure

```
finance-rag/
├── ingestion/
│   ├── edgar_client.py        # EDGAR fetch + section chunking
│   ├── embedder.py            # Modal GPU embedding + Pinecone upsert
│   ├── transcript_scraper.py  # Earnings transcript ingestion
│   └── ingest_coreweave.py    # CoreWeave local ingestion
├── query/
│   ├── retriever.py           # Pinecone search + Cohere rerank
│   ├── prompt_builder.py      # Grounding prompt assembly
│   ├── generator.py           # Claude generation
│   ├── multi_hop.py           # LlamaIndex SubQuestionQueryEngine
│   └── tavily_search.py       # Tavily web fallback
├── evals/
│   ├── eval_set.json          # 30 questions (Easy/Medium/Hard/ST)
│   ├── run_eval.py            # Braintrust eval runner
│   ├── eval_loop.py           # Agentic eval with diagnostics
│   ├── stress_test.py         # Hallucination stress test
│   └── populate_expected.py   # Auto-fill TBD expected answers
├── api/
│   └── endpoint.py            # (future) REST API
├── app.py                     # Streamlit UI
├── CLAUDE.md                  # Project context for Claude Code
└── SKILL.md                   # Architecture decisions
```
