"""
Embed a query and retrieve top-k chunks from Pinecone.

Embedding model is selected via the EMBEDDING_MODEL env var:
  EMBEDDING_MODEL=bge-large-en-v1.5   (default) local SentenceTransformer
  EMBEDDING_MODEL=voyage-finance-2              Voyage AI API (VOYAGE_API_KEY required)

If COHERE_API_KEY is set, retrieved chunks are reranked with Cohere Rerank
and the top RERANK_TOP_N results are returned; otherwise falls back to
Pinecone cosine similarity ordering.

Usage:
    python3 query/retriever.py "What are NVDA's main revenue segments?"
    python3 query/retriever.py "What is NVDA's data center revenue?" --namespace NVDA --top-k 5
"""

import os
import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pinecone import Pinecone

try:
    import streamlit as st
    def get_secret(key: str, default: str = "") -> str:
        return st.secrets.get(key) or os.environ.get(key, default)
except ImportError:
    def get_secret(key: str, default: str = "") -> str:
        return os.environ.get(key, default)

# ── config ────────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent.parent / ".env")

# BGE asymmetric retrieval: documents embedded with no prefix (see embedder.py),
# queries use this prefix to improve passage-matching accuracy.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

INDEX_NAME        = "finance-rag"
DEFAULT_NAMESPACE = "NVDA"
DEFAULT_TOP_K     = 8    # chunks fetched from Pinecone per namespace
RERANK_TOP_N      = 5    # default chunks returned after single-namespace rerank
RERANK_PER_NS     = 2    # chunks kept per namespace in multi-namespace rerank
RERANK_FINAL_N    = 8    # final chunks after global rerank across all namespaces
COHERE_MODEL      = "rerank-v3.5"
VOYAGE_MODEL      = "voyage-finance-2"

EMBEDDING_MODEL   = get_secret("EMBEDDING_MODEL", "bge-large-en-v1.5")

# ── BGE model (lazy init, used when EMBEDDING_MODEL=bge-large-en-v1.5) ────────

try:
    from sentence_transformers import SentenceTransformer as _ST
except ImportError:
    _ST = None  # type: ignore

_bge_model: Optional[object] = None


def _get_bge():
    global _bge_model
    if _bge_model is None:
        if _ST is None:
            raise ImportError("sentence-transformers is required for bge-large-en-v1.5")
        print("Loading bge-large-en-v1.5 …")
        _bge_model = _ST("BAAI/bge-large-en-v1.5")
    return _bge_model


# ── Voyage client (lazy init, used when EMBEDDING_MODEL=voyage-finance-2) ─────

_voyage_client: Optional[object] = None


def _get_voyage():
    global _voyage_client
    if _voyage_client is not None:
        return _voyage_client
    api_key = get_secret("VOYAGE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "VOYAGE_API_KEY is not set — required for EMBEDDING_MODEL=voyage-finance-2"
        )
    import voyageai
    _voyage_client = voyageai.Client(api_key=api_key)
    return _voyage_client


# ── Cohere reranker (lazy init, optional) ─────────────────────────────────────

_cohere_client = None


def _get_cohere():
    """Return a Cohere client if COHERE_API_KEY is set, else None."""
    global _cohere_client
    if _cohere_client is not None:
        return _cohere_client
    api_key = get_secret("COHERE_API_KEY")
    if not api_key:
        return None
    import cohere
    _cohere_client = cohere.ClientV2(api_key=api_key)
    return _cohere_client


def rerank(query: str, results: list[dict], top_n: int) -> list[dict]:
    """Rerank *results* with Cohere and return the top *top_n* entries.

    The 'score' field is replaced with the Cohere relevance score so
    downstream consumers see a consistent interface. Falls back to the
    original ordering if Cohere is unavailable or the API call fails.
    """
    co = _get_cohere()
    if co is None:
        return results[:top_n]

    docs = [r.get("text", "") for r in results]
    try:
        response = co.rerank(
            model=COHERE_MODEL,
            query=query,
            documents=docs,
            top_n=top_n,
        )
        reranked = []
        for item in response.results:
            chunk = dict(results[item.index])
            chunk["score"] = item.relevance_score
            reranked.append(chunk)
        return reranked
    except Exception as e:
        print(f"  [rerank] Cohere error, falling back to cosine order: {e}")
        return results[:top_n]


# ── core functions ────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """Embed a single query string using the configured EMBEDDING_MODEL.

    BGE: prepends the asymmetric query prefix before encoding.
    Voyage: uses input_type='query' for native asymmetric retrieval.
    """
    if EMBEDDING_MODEL == "voyage-finance-2":
        vo = _get_voyage()
        result = vo.embed([query], model=VOYAGE_MODEL, input_type="query")
        return result.embeddings[0]
    else:
        model = _get_bge()
        vec = model.encode(BGE_QUERY_PREFIX + query, normalize_embeddings=True)
        return vec.tolist()


def apply_filing_boost(
    results: list[dict],
    multiplier: float = 1.2,
) -> list[dict]:
    """Boost scores of 10-K/20-F chunks before global rerank.

    Chunks from SEC filings carry a 'filing_type' metadata field;
    transcript chunks do not.  Multiplying their score by *multiplier*
    before the Cohere global rerank tilts the final top-N toward
    regulatory filing language when both doc types are present.
    """
    boosted = []
    for r in results:
        entry = dict(r)
        if entry.get("filing_type") in ("10-K", "20-F"):
            entry["score"] = entry["score"] * multiplier
        boosted.append(entry)
    return boosted


# Pinecone metadata filters for each doc_type mode
_DOC_TYPE_FILTERS: dict[str, dict] = {
    "filing":     {"filing_type": {"$in": ["10-K", "20-F"]}},
    "transcript": {"doc_type":    {"$eq":  "earnings_transcript"}},
}


def retrieve(
    query: str,
    namespace: str = DEFAULT_NAMESPACE,
    top_k: int = DEFAULT_TOP_K,
    rerank_top_n: int = RERANK_TOP_N,
    doc_type_filter: Optional[str] = None,
) -> list[dict]:
    """Return chunks for *query* from *namespace*.

    Fetches top_k chunks from Pinecone by cosine similarity. If
    COHERE_API_KEY is set, reranks and returns the top rerank_top_n;
    otherwise returns all top_k in cosine order.

    doc_type_filter:
        None         — search both filings and transcripts (default)
        "filing"     — 10-K / 20-F chunks only
        "transcript" — earnings transcript chunks only

    For multi-namespace cross-company queries, pass rerank_top_n=RERANK_PER_NS
    (2) here and then call rerank() on the merged results for a second
    global pass.  Call apply_filing_boost() on the merged list before
    the global rerank to keep 10-K language competitive.

    Each result dict contains:
        score, id, and all chunk metadata fields
        (ticker, filing_type, period_of_report, filed_at,
         accession_number, section, chunk_index, source_url, text)
    """
    query_vec = embed_query(query)

    pc = Pinecone(api_key=get_secret("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    pinecone_filter = _DOC_TYPE_FILTERS.get(doc_type_filter) if doc_type_filter else None

    query_kwargs = dict(
        vector=query_vec,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    if pinecone_filter:
        query_kwargs["filter"] = pinecone_filter

    response = index.query(**query_kwargs)

    results = []
    for match in response.matches:
        results.append({
            "score": match.score,
            "id": match.id,
            **match.metadata,
        })

    if _get_cohere() is not None:
        results = rerank(query, results, top_n=rerank_top_n)

    return results


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve top-k chunks from Pinecone.")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Ticker namespace")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    matches = retrieve(args.query, namespace=args.namespace, top_k=args.top_k)

    print(f"\n{'═' * 60}")
    print(f"Query     : {args.query}")
    print(f"Namespace : {args.namespace}  |  top-{args.top_k}")
    print(f"{'═' * 60}\n")

    for i, m in enumerate(matches):
        print(f"── Result {i + 1}  (score: {m['score']:.4f}) {'─' * 30}")
        print(f"  section  : {m.get('section')}")
        print(f"  chunk    : {m.get('chunk_index')}")
        print(f"  filed_at : {m.get('filed_at')}")
        print(f"  period   : {m.get('period_of_report')}")
        print(f"  text     :\n")
        print(m.get("text", "")[:400])
        print()
