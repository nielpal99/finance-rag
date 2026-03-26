"""
Fetch transcript URLs via Firecrawl API and save markdown to transcripts/.

Reads transcripts_todo.csv (columns: ticker, quarter, year, url).
Skips rows with empty url or where the output file already exists.
Saves to transcripts/TICKER_Q{N}_{YEAR}.txt.

Usage:
    python3 ingestion/firecrawl_ingestor.py
    python3 ingestion/firecrawl_ingestor.py --csv transcripts_todo.csv
    python3 ingestion/firecrawl_ingestor.py --force   # overwrite existing files
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── paths & env ───────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

TRANSCRIPTS_DIR = _ROOT / "transcripts"
LOGS_DIR        = _ROOT / "logs"
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "firecrawl_ingest.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v1/scrape"
MIN_TRANSCRIPT_LEN   = 3_000


# ── fetch ─────────────────────────────────────────────────────────────────────

def fetch_via_firecrawl(url: str) -> str:
    """POST to Firecrawl /v1/scrape and return the markdown content."""
    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not api_key:
        raise EnvironmentError("FIRECRAWL_API_KEY not set in .env")

    resp = requests.post(
        FIRECRAWL_SCRAPE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        json={"url": url, "formats": ["markdown"]},
        timeout=60,
    )
    resp.raise_for_status()

    data = resp.json()
    markdown = data.get("data", {}).get("markdown", "")
    return markdown.strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default=str(_ROOT / "transcripts_todo.csv"))
    parser.add_argument("--force", action="store_true",
                        help="Overwrite output files that already exist")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        log.error(f"CSV not found: {csv_path}")
        return

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    ok_list, skip_list, fail_list = [], [], []

    for row in rows:
        ticker  = row["ticker"].strip().upper()
        quarter = row["quarter"].strip().upper()
        year    = row["year"].strip()
        url     = row["url"].strip()

        label    = f"{ticker} {quarter} {year}"
        out_path = TRANSCRIPTS_DIR / f"{ticker}_{quarter}_{year}.txt"

        if not url:
            log.info(f"[{label}] SKIP: no URL")
            skip_list.append(label)
            continue

        if out_path.exists() and not args.force:
            log.info(f"[{label}] SKIP: {out_path.name} already exists")
            skip_list.append(label)
            continue

        log.info(f"[{label}] fetching via Firecrawl …")
        try:
            text = fetch_via_firecrawl(url)
        except Exception as e:
            log.error(f"[{label}] FAIL: {e}")
            fail_list.append(label)
            continue

        if len(text) < MIN_TRANSCRIPT_LEN:
            log.warning(f"[{label}] FAIL: {len(text)} chars returned (paywall or parse error)")
            fail_list.append(label)
            continue

        out_path.write_text(text, encoding="utf-8")
        log.info(f"[{label}] OK: {len(text):,} chars → {out_path.name}")
        ok_list.append(label)

    log.info(
        f"\n{'═'*60}\n"
        f"DONE  OK={len(ok_list)}  SKIP={len(skip_list)}  FAIL={len(fail_list)}\n"
        f"  OK:   {', '.join(ok_list) or 'none'}\n"
        f"  SKIP: {', '.join(skip_list) or 'none'}\n"
        f"  FAIL: {', '.join(fail_list) or 'none'}\n"
        f"{'═'*60}"
    )


if __name__ == "__main__":
    main()
