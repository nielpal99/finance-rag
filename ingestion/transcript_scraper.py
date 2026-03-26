"""
Search for earnings call transcripts via Tavily API, chunk by speaker turn,
embed with bge-large-en-v1.5, and upsert to Pinecone.

Searches the last N calendar quarters per ticker. Tavily consistently returns
Motley Fool transcripts whose raw_content uses the markdown speaker format:
    **Name** -- *Role*
    <speech text>

Also handles the inline Seeking Alpha / plain-text format:
    Name (Role): <speech text>

Usage:
    python3 ingestion/transcript_scraper.py                    # all tickers, 8 quarters
    python3 ingestion/transcript_scraper.py --ticker NVDA
    python3 ingestion/transcript_scraper.py --ticker NVDA --quarters 2 --dry-run
"""

import os
import re
import time
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests
import tiktoken
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv(Path(__file__).parent.parent / ".env")

# ── constants ─────────────────────────────────────────────────────────────────

TAVILY_URL   = "https://api.tavily.com/search"
INDEX_NAME   = "finance-rag"
CHUNK_SIZE   = 400
CHUNK_OVERLAP = 50
EMBED_BATCH  = 64
UPSERT_BATCH = 100

COMPANY_NAMES = {
    "NVDA": "NVIDIA",
    "AMD":  "Advanced Micro Devices",
    "AVGO": "Broadcom",
    "TSM":  "Taiwan Semiconductor",
    "ANET": "Arista Networks",
    "MU":   "Micron Technology",
}

ALL_TICKERS = list(COMPANY_NAMES)

ENC = tiktoken.get_encoding("cl100k_base")

# ── speaker-turn regexes ──────────────────────────────────────────────────────

# Motley Fool markdown: **Name** -- *Role*  (role is optional)
_MF_SPEAKER = re.compile(
    r"\*\*([^*]{2,60})\*\*(?:\s*--\s*\*([^*]{2,80})\*)?"
)

# Inline format: Name (Role): or Name:  at start of line
_INLINE_SPEAKER = re.compile(
    r"^([A-Z][A-Za-z\.\s\-\']{1,50}?)\s*(?:\(([^)]{2,60})\))?\s*:\s",
    re.MULTILINE,
)

# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class TranscriptChunk:
    text:           str
    ticker:         str
    doc_type:       str   # always "earnings_transcript"
    fiscal_quarter: str   # e.g. "Q3 2025"
    filing_date:    str   # best-available date from the search result
    speaker:        str
    speaker_role:   str
    turn_index:     int
    chunk_index:    int
    source_url:     str


# ── quarter helpers ───────────────────────────────────────────────────────────

def last_n_calendar_quarters(n: int) -> list[tuple[int, int]]:
    """Return the last *n* completed calendar quarters as (year, quarter) tuples,
    newest first.  Today is 2026-03-24 → most recent complete quarter = Q4 2025."""
    # Q4 2025 is complete; Q1 2026 is in progress → start from (2025, 4)
    year, q = 2025, 4
    quarters = []
    for _ in range(n):
        quarters.append((year, q))
        q -= 1
        if q == 0:
            q = 4
            year -= 1
    return quarters


# ── Tavily search ─────────────────────────────────────────────────────────────

def _search_transcript(
    ticker: str,
    year: int,
    quarter: int,
) -> Optional[dict]:
    """Return the best Tavily result for one earnings call, or None."""
    company = COMPANY_NAMES[ticker]
    query = f"{company} {ticker} Q{quarter} {year} earnings call transcript"

    resp = requests.post(
        TAVILY_URL,
        json={
            "api_key":             os.environ["TAVILY_API_KEY"],
            "query":               query,
            "search_depth":        "advanced",
            "include_raw_content": True,
            "max_results":         5,
        },
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])

    # Prefer results whose URL contains "earnings-call-transcript" or "transcript"
    # and whose raw_content is long enough to be a real transcript (>5k chars).
    def _score(r: dict) -> tuple:
        url = r.get("url", "")
        rc  = r.get("raw_content") or ""
        url_bonus = 2 if "earnings-call-transcript" in url else (1 if "transcript" in url else 0)
        return (url_bonus, len(rc))

    candidates = [r for r in results if len(r.get("raw_content") or "") > 5000]
    if not candidates:
        return None
    return max(candidates, key=_score)


# ── transcript parsing ────────────────────────────────────────────────────────

def _extract_after_header(text: str) -> str:
    """Strip navigation / boilerplate before the actual transcript body."""
    for marker in ("## Prepared Remarks", "Prepared Remarks:", "Operator\n", "**Operator**"):
        idx = text.find(marker)
        if idx != -1:
            return text[idx:]
    return text


def _parse_motley_fool(text: str) -> list[dict]:
    """Parse Motley Fool markdown speaker turns.
    Format: **Name** -- *Role*  followed by content until next bold name.
    """
    turns = []
    # Split on every occurrence of **...**
    parts = _MF_SPEAKER.split(text)
    # split() with capturing groups returns: [pre, name, role, content, name, role, content, ...]
    # parts[0] is text before first match; then triplets of (name, role, content)
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
    """Parse inline 'Name (Role): text' format."""
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
    """Try Motley Fool format first, fall back to inline format."""
    body = _extract_after_header(raw)
    turns = _parse_motley_fool(body)
    if len(turns) >= 3:
        return turns
    turns = _parse_inline(body)
    if len(turns) >= 3:
        return turns
    # Last resort: treat whole body as a single unnamed chunk
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


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(
    tickers: list[str],
    quarters: int = 8,
    dry_run: bool = False,
) -> None:
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    q_list = last_n_calendar_quarters(quarters)

    for ticker in tickers:
        ticker = ticker.upper()
        print(f"\n{'═' * 60}")
        print(f"[{ticker}] processing {quarters} quarters …")
        all_chunks: list[TranscriptChunk] = []
        chunk_index = 0

        quarters_ok:   list[str] = []
        quarters_fail: list[str] = []

        for year, q in q_list:
            fiscal_quarter = f"Q{q} {year}"
            print(f"  [{ticker}] searching {fiscal_quarter} …")
            time.sleep(0.5)  # polite rate-limit

            result = _search_transcript(ticker, year, q)
            if not result:
                print(f"  [{ticker}] {fiscal_quarter} — FAIL: no transcript found by Tavily")
                quarters_fail.append(fiscal_quarter)
                continue

            url         = result["url"]
            raw         = result.get("raw_content") or ""
            filing_date = result.get("published_date", "") or ""

            turns = _parse_turns(raw)
            if not turns:
                print(f"  [{ticker}] {fiscal_quarter} — FAIL: could not parse speaker turns from {url}")
                quarters_fail.append(fiscal_quarter)
                continue

            print(f"  [{ticker}] {fiscal_quarter} — OK: {len(turns)} turns from {url}")
            quarters_ok.append(fiscal_quarter)

            for turn_idx, turn in enumerate(turns):
                for piece in _chunk(turn["text"]):
                    all_chunks.append(TranscriptChunk(
                        text           = piece,
                        ticker         = ticker,
                        doc_type       = "earnings_transcript",
                        fiscal_quarter = fiscal_quarter,
                        filing_date    = filing_date,
                        speaker        = turn["speaker"],
                        speaker_role   = turn["role"],
                        turn_index     = turn_idx,
                        chunk_index    = chunk_index,
                        source_url     = url,
                    ))
                    chunk_index += 1

        print(f"  [{ticker}] quarters OK   ({len(quarters_ok)}): {', '.join(quarters_ok) or 'none'}")
        print(f"  [{ticker}] quarters FAIL ({len(quarters_fail)}): {', '.join(quarters_fail) or 'none'}")
        print(f"  [{ticker}] total chunks: {len(all_chunks)}")
        if dry_run or not all_chunks:
            if dry_run:
                print(f"  [{ticker}] dry-run — skipping embed/upsert")
                _print_sample(all_chunks)
            continue

        # ── embed ──────────────────────────────────────────────────────────────
        texts = [c.text for c in all_chunks]
        embeddings = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i : i + EMBED_BATCH]
            embs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            embeddings.extend(embs)
            print(f"  [{ticker}] embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")

        # ── upsert ─────────────────────────────────────────────────────────────
        q_norm  = all_chunks[0].fiscal_quarter.replace(" ", "_")  # first quarter label
        records = []
        for chunk, emb in zip(all_chunks, embeddings):
            q_label = chunk.fiscal_quarter.replace(" ", "_")
            vec_id  = f"{ticker}_transcript_{q_label}_{chunk.turn_index}_{chunk.chunk_index}"
            records.append({
                "id":       vec_id,
                "values":   emb.tolist(),
                "metadata": asdict(chunk),
            })

        for i in range(0, len(records), UPSERT_BATCH):
            batch = records[i : i + UPSERT_BATCH]
            index.upsert(vectors=batch, namespace=ticker)
            print(f"  [{ticker}] upserted {min(i + UPSERT_BATCH, len(records))}/{len(records)}")

        print(f"  [{ticker}] done — {len(records)} vectors in namespace '{ticker}'")


def _print_sample(chunks: list[TranscriptChunk], n: int = 3) -> None:
    for c in chunks[:n]:
        print(f"\n  {'─'*50}")
        print(f"  chunk_index    : {c.chunk_index}")
        print(f"  fiscal_quarter : {c.fiscal_quarter}")
        print(f"  speaker        : {c.speaker}")
        print(f"  speaker_role   : {c.speaker_role}")
        print(f"  filing_date    : {c.filing_date}")
        print(f"  source_url     : {c.source_url}")
        print(f"  text preview   : {c.text[:300]}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",   default=None, help="Single ticker (default: all 6)")
    parser.add_argument("--quarters", type=int, default=8)
    parser.add_argument("--dry-run",  action="store_true", help="Parse only, skip embed/upsert")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else ALL_TICKERS
    run(tickers, quarters=args.quarters, dry_run=args.dry_run)
