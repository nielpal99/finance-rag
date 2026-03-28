"""
Fetch, embed, and upsert CoreWeave (CRWV) S-1 chunks into Pinecone.

Runs locally (no Modal). Uses bge-large-en-v1.5 via SentenceTransformers
and upserts to the 'CRWV' namespace in the finance-rag Pinecone index.

Usage:
    python3 ingestion/ingest_coreweave.py
    python3 ingestion/ingest_coreweave.py | tee logs/ingest_crwv.log
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from ingestion.edgar_client import iter_chunks

TICKER      = "CRWV"
INDEX_NAME  = "finance-rag"
EMBED_BATCH  = 64
UPSERT_BATCH = 100


def _embed(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    print("  [embed] loading bge-large-en-v1.5 …")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    all_embs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embs.extend(embs.tolist())
        print(f"  [embed] {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")
    return all_embs


def main():
    from pinecone import Pinecone

    print(f"[{TICKER}] connecting to Pinecone index '{INDEX_NAME}' …")
    pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    print(f"[{TICKER}] fetching chunks from EDGAR …")
    chunks = list(iter_chunks(TICKER))
    print(f"[{TICKER}] {len(chunks)} chunks fetched")

    if not chunks:
        print(f"[{TICKER}] no chunks found — check EDGAR for CRWV filings")
        return

    # Print filing metadata from first chunk
    c0 = chunks[0]
    print(f"[{TICKER}] filing_type={c0.filing_type}  period={c0.period_of_report}  filed={c0.filed_at}")
    print(f"[{TICKER}] source: {c0.source_url}\n")

    texts          = [c.text for c in chunks]
    all_embeddings = _embed(texts)

    records = []
    for chunk, emb in zip(chunks, all_embeddings):
        vec_id   = f"{chunk.ticker}_{chunk.accession_number}_{chunk.chunk_index}"
        metadata = asdict(chunk)
        records.append({"id": vec_id, "values": emb, "metadata": metadata})

    print(f"\n[{TICKER}] upserting {len(records)} vectors to namespace '{TICKER}' …")
    for i in range(0, len(records), UPSERT_BATCH):
        batch = records[i : i + UPSERT_BATCH]
        index.upsert(vectors=batch, namespace=TICKER)
        print(f"[{TICKER}] upserted {min(i + UPSERT_BATCH, len(records))}/{len(records)}")

    print(f"\n[{TICKER}] done — {len(records)} chunks in Pinecone namespace '{TICKER}'")


if __name__ == "__main__":
    main()
