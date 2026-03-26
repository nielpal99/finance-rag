"""
Read .txt transcripts from transcripts/, parse speaker turns, embed on
Modal A10G with bge-large-en-v1.5, and upsert to Pinecone.

Filename format: TICKER_Q{N}_{YEAR}.txt  e.g. NVDA_Q4_2025.txt

Usage:
    modal run ingestion/transcript_ingestor.py                   # all files in transcripts/
    modal run ingestion/transcript_ingestor.py --ticker NVDA     # filter by ticker
    modal run ingestion/transcript_ingestor.py --file transcripts/NVDA_Q4_2025.txt
"""

import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import modal
import tiktoken
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── paths ─────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../ingestion/
_ROOT = os.path.dirname(_HERE)                        # .../finance-rag/

# ── Modal image ───────────────────────────────────────────────────────────────

def _download_model():
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("BAAI/bge-large-en-v1.5")
    print("Model cached.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers==3.3.1",
        "pinecone==7.3.0",
        "torch==2.5.1",
        "tiktoken==0.8.0",
        "python-dotenv==1.0.1",
    )
    .run_function(_download_model)
)

app = modal.App("finance-rag-transcript-ingestor", image=image)

# ── constants ─────────────────────────────────────────────────────────────────

INDEX_NAME    = "finance-rag"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50
EMBED_BATCH   = 64
UPSERT_BATCH  = 100

ENC = tiktoken.get_encoding("cl100k_base")

# ── filename pattern ──────────────────────────────────────────────────────────

_FNAME_RE = re.compile(r"^([A-Z]+)_Q(\d)_(\d{4})\.txt$", re.IGNORECASE)


def _parse_filename(path: Path) -> Optional[tuple[str, str]]:
    """Return (ticker, fiscal_quarter) from e.g. NVDA_Q4_2025.txt, or None."""
    m = _FNAME_RE.match(path.name)
    if not m:
        return None
    ticker, q, year = m.group(1).upper(), m.group(2), m.group(3)
    return ticker, f"Q{q} {year}"


# ── speaker-turn parsing (mirrors transcript_scraper.py) ─────────────────────

_MF_SPEAKER = re.compile(
    r"\*\*([^*]{2,60})\*\*(?:\s*--\s*\*([^*]{2,80})\*)?"
)
_INLINE_SPEAKER = re.compile(
    r"^([A-Z][A-Za-z\.\s\-\']{1,50}?)\s*(?:\(([^)]{2,60})\))?\s*:\s",
    re.MULTILINE,
)


def _extract_after_header(text: str) -> str:
    for marker in ("## Prepared Remarks", "Prepared Remarks:", "Operator\n", "**Operator**"):
        idx = text.find(marker)
        if idx != -1:
            return text[idx:]
    return text


def _parse_motley_fool(text: str) -> list[dict]:
    turns = []
    parts = _MF_SPEAKER.split(text)
    i = 1
    while i + 1 < len(parts):
        name    = (parts[i]     or "").strip()
        role    = (parts[i + 1] or "").strip()
        content = (parts[i + 2] or "").strip()
        if name and content:
            turns.append({"speaker": name, "role": role, "text": content})
        i += 3
    return turns


def _parse_inline(text: str) -> list[dict]:
    matches = list(_INLINE_SPEAKER.finditer(text))
    if not matches:
        return []
    turns = []
    for idx, m in enumerate(matches):
        name    = m.group(1).strip()
        role    = (m.group(2) or "").strip()
        start   = m.end()
        end     = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            turns.append({"speaker": name, "role": role, "text": content})
    return turns


def _parse_turns(raw: str) -> list[dict]:
    body = _extract_after_header(raw)
    turns = _parse_motley_fool(body)
    if len(turns) >= 3:
        return turns
    turns = _parse_inline(body)
    if len(turns) >= 3:
        return turns
    if body.strip():
        return [{"speaker": "Unknown", "role": "", "text": body.strip()}]
    return []


# ── chunking ──────────────────────────────────────────────────────────────────

def _chunk(text: str) -> list[str]:
    tokens = ENC.encode(text)
    if len(tokens) <= CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunks.append(ENC.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── chunk schema (matches TranscriptChunk in transcript_scraper.py) ───────────

@dataclass
class TranscriptChunk:
    text:           str
    ticker:         str
    doc_type:       str   # always "earnings_transcript"
    fiscal_quarter: str   # e.g. "Q4 2025"
    filing_date:    str   # empty — not present in local .txt files
    speaker:        str
    speaker_role:   str
    turn_index:     int
    chunk_index:    int
    source_url:     str   # relative path of the source file


# ── local file → chunks ───────────────────────────────────────────────────────

def _file_to_chunks(path: Path) -> list[TranscriptChunk]:
    parsed = _parse_filename(path)
    if not parsed:
        print(f"  SKIP {path.name} — does not match TICKER_Q{{N}}_{{YEAR}}.txt")
        return []

    ticker, fiscal_quarter = parsed
    raw = path.read_text(encoding="utf-8")
    turns = _parse_turns(raw)

    if not turns:
        print(f"  FAIL {path.name} — could not parse speaker turns")
        return []

    print(f"  OK   {path.name} — {ticker} {fiscal_quarter}, {len(turns)} turns")

    chunks: list[TranscriptChunk] = []
    chunk_index = 0
    for turn_idx, turn in enumerate(turns):
        for piece in _chunk(turn["text"]):
            chunks.append(TranscriptChunk(
                text           = piece,
                ticker         = ticker,
                doc_type       = "earnings_transcript",
                fiscal_quarter = fiscal_quarter,
                filing_date    = "",
                speaker        = turn["speaker"],
                speaker_role   = turn["role"],
                turn_index     = turn_idx,
                chunk_index    = chunk_index,
                source_url     = str(path),
            ))
            chunk_index += 1

    return chunks


# ── Modal GPU function ────────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    secrets=[modal.Secret.from_dotenv(_ROOT)],
    timeout=600,
)
def embed_and_upsert(
    chunk_dicts: list[dict],
    ticker: str,
    index_name: str = INDEX_NAME,
) -> dict:
    """Embed a list of serialised TranscriptChunk dicts and upsert to Pinecone."""
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone

    print(f"[{ticker}] loading model …")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    print(f"[{ticker}] connecting to Pinecone …")
    pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)

    texts = [c["text"] for c in chunk_dicts]
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        embs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embeddings.extend(embs)
        print(f"[{ticker}] embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")

    records = []
    for chunk, emb in zip(chunk_dicts, embeddings):
        q_label = chunk["fiscal_quarter"].replace(" ", "_")
        vec_id  = (
            f"{chunk['ticker']}_transcript"
            f"_{q_label}_{chunk['turn_index']}_{chunk['chunk_index']}"
        )
        records.append({
            "id":       vec_id,
            "values":   emb.tolist(),
            "metadata": chunk,
        })

    for i in range(0, len(records), UPSERT_BATCH):
        batch = records[i : i + UPSERT_BATCH]
        index.upsert(vectors=batch, namespace=ticker)
        print(f"[{ticker}] upserted {min(i + UPSERT_BATCH, len(records))}/{len(records)}")

    return {
        "ticker":           ticker,
        "namespace":        ticker,
        "chunks_upserted":  len(records),
        "embedding_dim":    len(records[0]["values"]) if records else 0,
    }


# ── local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    ticker: str = "",
    file:   str = "",
    index_name: str = INDEX_NAME,
):
    transcripts_dir = Path(_ROOT) / "transcripts"
    if not transcripts_dir.exists():
        print(f"ERROR: {transcripts_dir} does not exist")
        return

    if file:
        paths = [Path(file)]
    else:
        paths = sorted(transcripts_dir.glob("*.txt"))

    if ticker:
        ticker_up = ticker.upper()
        paths = [p for p in paths if p.name.upper().startswith(ticker_up + "_")]

    if not paths:
        print("No .txt files found matching criteria.")
        return

    print(f"Found {len(paths)} file(s) to ingest:")

    # Parse all files locally, group by ticker
    by_ticker: dict[str, list[dict]] = {}
    for path in paths:
        chunks = _file_to_chunks(path)
        if not chunks:
            continue
        t = chunks[0].ticker
        by_ticker.setdefault(t, []).extend(asdict(c) for c in chunks)

    if not by_ticker:
        print("No chunks produced — check file format.")
        return

    # Embed + upsert per ticker via Modal
    for t, chunk_dicts in by_ticker.items():
        print(f"\n[{t}] sending {len(chunk_dicts)} chunks to Modal …")
        result = embed_and_upsert.remote(chunk_dicts, ticker=t, index_name=index_name)
        print(f"[{t}] done → {result}")
