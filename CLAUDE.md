# Finance RAG — Claude Code Context

## What this is
A RAG pipeline for researching AI infrastructure companies using SEC
filings and earnings call transcripts. Cross-company analysis across
6 primary tickers: NVDA, AMD, AVGO, TSM, ANET, MU.

## Stack
- Compute: Modal (serverless, GPU on demand)
- Embeddings: bge-large-en-v1.5 on Modal A10G
- Vector DB: Pinecone (single index, namespace per ticker)
- Generation: Claude API (claude-sonnet-4-20250514)
- Reranking: Cohere Rerank (optional, improves accuracy)

## Current phase
Phase 1 — Ingestion pipeline.

Completed:
- edgar_client.py — fetches 10-K and 20-F for any ticker, chunks by
  section. CIK map validated for all 6 tickers. TSM 20-F fix applied.
- embedder.py — Modal A10G GPU, bge-large-en-v1.5, Pinecone upsert
- retriever.py — embeds query, searches Pinecone, returns top-5 chunks
- prompt_builder.py — assembles chunks with grounding prompt
- generator.py — Claude API call, grounded answer with citations
- All 6 tickers ingested: NVDA(199) AMD(235) AVGO(216)
  TSM(619) ANET(228) MU(216) — 1,713 total chunks

Current task: Current task: Phase 3 — transcript ingestion in progress.
- transcript_scraper.py built and validated (dry-run passed for NVDA)
- NVDA transcript ingestion failed silently — re-running with logging
- Remaining tickers (AMD, AVGO, TSM, ANET, MU) not yet ingested
- All ingestion output must write to logs/ folder


## File structure
finance-rag/
├── ingestion/
│   ├── edgar_client.py
│   ├── transcript_scraper.py
│   ├── chunker.py
│   └── embedder.py
├── query/
│   ├── retriever.py
│   ├── prompt_builder.py
│   └── generator.py
├── api/
│   └── endpoint.py
├── evals/
│   └── eval_set.json
├── modal_app.py
├── SKILL.md
└── CLAUDE.md

## Key conventions
- Always attach metadata to every chunk (see SKILL.md for schema)
- Chunk size: 400 tokens, 50 token overlap
- 10-Ks: chunk by section. Transcripts: chunk by speaker turn
- Never skip the grounding prompt — see SKILL.md for exact wording
- EDGAR requests must include User-Agent header with name + email
- Never hardcode API keys — use .env file

## Environment variables needed
PINECONE_API_KEY, ANTHROPIC_API_KEY, COHERE_API_KEY, MODAL_TOKEN_ID,
MODAL_TOKEN_SECRET — all in .env, never hardcoded

## How to work with me
- Tell me which phase we're on and what done looks like
- Reference SKILL.md for full architecture decisions
- Build one script at a time, prove it works before moving on
- Always log ingestion results to a file: 
  modal run ingestion/embedder.py | tee logs/ingest_$(date +%Y%m%d).log