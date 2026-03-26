"""
Modal function: embed 10-K chunks and upsert to Pinecone with full metadata.

Embedding model is selected via the EMBEDDING_MODEL env var (read from .env):
  EMBEDDING_MODEL=bge-large-en-v1.5   (default) local model on A10G GPU
  EMBEDDING_MODEL=voyage-finance-2              Voyage AI API (VOYAGE_API_KEY required)

Usage:
    modal run ingestion/embedder.py              # defaults to NVDA
    modal run ingestion/embedder.py --ticker AMD
    modal run ingestion/embedder.py --ticker NVDA --index-name finance-rag
"""

import os
import sys
from dataclasses import asdict

import modal

# ── paths ─────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../ingestion/
_ROOT = os.path.dirname(_HERE)                        # .../finance-rag/

# ── image ─────────────────────────────────────────────────────────────────────

def _download_model():
    """Bake bge-large-en-v1.5 into the image layer at build time."""
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("BAAI/bge-large-en-v1.5")
    print("Model cached.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers==3.3.1",
        "pinecone==7.3.0",
        "torch==2.5.1",
        "beautifulsoup4==4.12.3",
        "lxml==5.3.0",
        "tiktoken==0.8.0",
        "requests==2.32.3",
        "voyageai==0.3.2",
    )
    .run_function(_download_model)
    # add_local_* must come last — Modal injects these at container start,
    # not at image build time, so local changes don't force a full rebuild.
    .add_local_dir(_HERE, remote_path="/root/ingestion")
)

# ── app ───────────────────────────────────────────────────────────────────────

app = modal.App("finance-rag-embedder", image=image)

# ── constants ─────────────────────────────────────────────────────────────────

EMBED_BATCH  = 64    # texts per encode call — fits A10G VRAM (bge) and Voyage API limits
UPSERT_BATCH = 100   # Pinecone max recommended batch size
VOYAGE_MODEL = "voyage-finance-2"

# ── embedding helpers ─────────────────────────────────────────────────────────

def _embed_bge(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    print("  [embed] bge-large-en-v1.5 loaded")
    all_embs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embs.extend(embs.tolist())
        print(f"  [embed] bge {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")
    return all_embs


def _embed_voyage(texts: list[str]) -> list[list[float]]:
    import voyageai
    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    print("  [embed] voyage-finance-2 client ready")
    all_embs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        result = vo.embed(batch, model=VOYAGE_MODEL, input_type="document")
        all_embs.extend(result.embeddings)
        print(f"  [embed] voyage {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")
    return all_embs


def _embed(texts: list[str]) -> list[list[float]]:
    """Dispatch to the configured embedding model."""
    model = os.environ.get("EMBEDDING_MODEL", "bge-large-en-v1.5")
    print(f"  [embed] EMBEDDING_MODEL={model}")
    if model == "voyage-finance-2":
        return _embed_voyage(texts)
    return _embed_bge(texts)


# ── GPU function ──────────────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    secrets=[modal.Secret.from_dotenv(_ROOT)],
    timeout=600,
)
def embed_ticker(ticker: str, index_name: str = "finance-rag") -> dict:
    """Fetch, embed, and upsert all chunks for one ticker's latest 10-K.

    Uses EMBEDDING_MODEL env var to select bge-large-en-v1.5 (default,
    runs on GPU) or voyage-finance-2 (Voyage AI API, GPU idle but harmless).
    """
    sys.path.insert(0, "/root")
    from ingestion.edgar_client import iter_chunks  # noqa: E402
    from pinecone import Pinecone                   # noqa: E402

    ticker = ticker.upper()
    print(f"[{ticker}] connecting to Pinecone index '{index_name}' …")
    pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)

    # ── 1. fetch & collect all chunks ─────────────────────────────────────────
    print(f"[{ticker}] fetching chunks from EDGAR …")
    chunks = list(iter_chunks(ticker))
    print(f"[{ticker}] {len(chunks)} chunks ready")

    # ── 2. embed in batches ───────────────────────────────────────────────────
    texts          = [c.text for c in chunks]
    all_embeddings = _embed(texts)

    # ── 3. build Pinecone records ──────────────────────────────────────────────
    # Vector ID is deterministic → re-runs are idempotent (upsert overwrites)
    records = []
    for chunk, emb in zip(chunks, all_embeddings):
        vec_id   = f"{chunk.ticker}_{chunk.accession_number}_{chunk.chunk_index}"
        metadata = asdict(chunk)
        records.append({"id": vec_id, "values": emb, "metadata": metadata})

    # ── 4. upsert to Pinecone (namespace = ticker) ────────────────────────────
    for i in range(0, len(records), UPSERT_BATCH):
        batch = records[i : i + UPSERT_BATCH]
        index.upsert(vectors=batch, namespace=ticker)
        print(f"[{ticker}] upserted {min(i + UPSERT_BATCH, len(records))}/{len(records)}")

    summary = {
        "ticker":           ticker,
        "index":            index_name,
        "namespace":        ticker,
        "embedding_model":  os.environ.get("EMBEDDING_MODEL", "bge-large-en-v1.5"),
        "chunks_upserted":  len(records),
        "embedding_dim":    len(records[0]["values"]) if records else 0,
    }
    print(f"[{ticker}] done → {summary}")
    return summary


# ── local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(ticker: str = "NVDA", index_name: str = "finance-rag"):
    result = embed_ticker.remote(ticker, index_name=index_name)
    print("\nResult:", result)
