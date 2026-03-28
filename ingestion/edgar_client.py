"""
Fetches the most recent 10-K for a given ticker from EDGAR,
parses it by section, and yields chunks with metadata.
"""

import re
import warnings
import requests
import tiktoken
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dataclasses import dataclass
from typing import Iterator

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ── constants ────────────────────────────────────────────────────────────────

EDGAR_BASE = "https://data.sec.gov"
EDGAR_FILING_BASE = "https://www.sec.gov/Archives/edgar"
USER_AGENT = "nielpal niel@example.com"  # required by EDGAR fair-use policy

CHUNK_SIZE = 400      # tokens
CHUNK_OVERLAP = 50    # tokens

TICKER_CIK = {
    "NVDA": "0001045810",
    "AMD":  "0000002488",
    "AVGO": "0001730168",
    "TSM":  "0001046179",
    "ANET": "0001596532",
    "MU":   "0000723125",
    "CRWV": "0001769628",
}

# 10-K section header pattern (matches "Item 1.", "ITEM 1A.", etc.)
SECTION_RE = re.compile(
    r"(?:^|\n)\s*(ITEM\s+\d+[A-Z]?\.?\s+[A-Z][^\n]{3,80})",
    re.IGNORECASE,
)

ENC = tiktoken.get_encoding("cl100k_base")


# ── data model ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    ticker: str
    filing_type: str
    period_of_report: str
    filed_at: str
    accession_number: str
    section: str
    chunk_index: int
    source_url: str


# ── EDGAR helpers ─────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def get_latest_10k_meta(ticker: str) -> dict:
    """Return accession number, filed date, period, and document URL for the
    most recent annual report for *ticker*.

    Accepts 10-K (domestic) and 20-F (foreign private issuers, e.g. TSM).
    """
    ANNUAL_FORMS = {"10-K", "20-F", "S-1", "S-1/A"}

    cik = TICKER_CIK[ticker.upper()]
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    data = requests.get(url, headers=_headers(), timeout=30).json()

    filings = data["filings"]["recent"]
    forms = filings["form"]
    accessions = filings["accessionNumber"]
    filed_dates = filings["filingDate"]
    periods = filings["reportDate"]
    primary_docs = filings["primaryDocument"]

    for i, form in enumerate(forms):
        if form in ANNUAL_FORMS:
            acc = accessions[i]
            acc_path = acc.replace("-", "")
            doc = primary_docs[i]
            return {
                "accession_number": acc,
                "filed_at": filed_dates[i],
                "period_of_report": periods[i],
                "filing_type": form,
                "doc_url": f"{EDGAR_FILING_BASE}/data/{cik.lstrip('0')}/{acc_path}/{doc}",
                "index_url": f"{EDGAR_FILING_BASE}/data/{cik.lstrip('0')}/{acc_path}/",
            }
    raise ValueError(f"No annual filing (10-K or 20-F) found for {ticker}")


def fetch_filing_text(doc_url: str) -> str:
    """Download the primary 10-K document and return clean plain text."""
    resp = requests.get(doc_url, headers=_headers(), timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml")
    # Remove script/style noise
    for tag in soup(["script", "style", "ix:header", "ix:hidden"]):
        tag.decompose()
    return soup.get_text(separator="\n")


# ── chunking ──────────────────────────────────────────────────────────────────

def _token_len(text: str) -> int:
    return len(ENC.encode(text))


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split *text* into token-bounded chunks with overlap."""
    tokens = ENC.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(ENC.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += size - overlap
    return chunks


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (section_title, section_text) pairs.
    Falls back to a single 'Document' section if no headers found."""
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [("Document", text)]

    MIN_SECTION_TOKENS = 100  # skip TOC stubs (page numbers, short labels)

    # Normalise titles and keep only the LAST occurrence of each item number.
    # iXBRL docs repeat headers: first in the TOC (short body) then in the
    # actual content (long body). We want the content copy.
    item_re = re.compile(r"ITEM\s+(\d+[A-Z]?)", re.IGNORECASE)
    last_match_for_item: dict[str, int] = {}
    for idx, m in enumerate(matches):
        key = item_re.search(m.group(1))
        label = key.group(1).upper() if key else m.group(1).strip()
        last_match_for_item[label] = idx
    deduped = sorted(last_match_for_item.values())

    sections = []
    for rank, idx in enumerate(deduped):
        m = matches[idx]
        title = re.sub(r"\s+", " ", m.group(1).strip())
        start = m.end()
        # End at the next kept match (not just next raw match)
        if rank + 1 < len(deduped):
            end = matches[deduped[rank + 1]].start()
        else:
            end = len(text)
        body = text[start:end].strip()
        if body and _token_len(body) >= MIN_SECTION_TOKENS:
            sections.append((title, body))
    return sections


# ── public API ────────────────────────────────────────────────────────────────

def iter_chunks(ticker: str) -> Iterator[Chunk]:
    """Yield all chunks for the most recent 10-K of *ticker*."""
    meta = get_latest_10k_meta(ticker)
    text = fetch_filing_text(meta["doc_url"])
    sections = split_into_sections(text)

    chunk_index = 0
    for section_title, section_text in sections:
        for piece in chunk_text(section_text):
            yield Chunk(
                text=piece,
                ticker=ticker.upper(),
                filing_type=meta["filing_type"],
                period_of_report=meta["period_of_report"],
                filed_at=meta["filed_at"],
                accession_number=meta["accession_number"],
                section=section_title,
                chunk_index=chunk_index,
                source_url=meta["doc_url"],
            )
            chunk_index += 1


# ── main ──────────────────────────────────────────────────────────────────────

def _print_chunk(chunk: Chunk) -> None:
    print(f"{'─' * 60}")
    print(f"Chunk {chunk.chunk_index}")
    print(f"  ticker          : {chunk.ticker}")
    print(f"  filing_type     : {chunk.filing_type}")
    print(f"  period_of_report: {chunk.period_of_report}")
    print(f"  filed_at        : {chunk.filed_at}")
    print(f"  accession_number: {chunk.accession_number}")
    print(f"  section         : {chunk.section}")
    print(f"  source_url      : {chunk.source_url}")
    print(f"  text preview    :\n")
    print(chunk.text[:400])
    print()


if __name__ == "__main__":
    TARGETS = {"Item 1. Business", "Item 1A. Risk Factors", "Item 7. Management"}
    seen: dict[str, Chunk] = {}

    print("Fetching NVDA 10-K from EDGAR …\n")
    for chunk in iter_chunks("NVDA"):
        for target in TARGETS:
            if target not in seen and chunk.section.startswith(target[:10]):
                seen[target] = chunk
        if len(seen) == len(TARGETS):
            break

    for target in ["Item 1. Business", "Item 1A. Risk Factors", "Item 7. Management"]:
        chunk = seen.get(target)
        if chunk:
            _print_chunk(chunk)
        else:
            print(f"[NOT FOUND] {target}\n")
