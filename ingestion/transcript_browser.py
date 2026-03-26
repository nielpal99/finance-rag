"""
Browser-based earnings call transcript scraper for Motley Fool and Seeking Alpha.

Strategy (two-stage):
  1. URL discovery  — Tavily search finds the best transcript URL
                      for each ticker/quarter on the specified site.
  2. Full extraction — Playwright navigates to that URL, dismisses overlays,
                      and extracts the complete article body text.

Usage:
    python3 ingestion/transcript_browser.py --ticker AMD --quarters 2
    python3 ingestion/transcript_browser.py --ticker NVDA --quarters 8
    python3 ingestion/transcript_browser.py --ticker TSM --quarters-list Q1_2025,Q2_2025,Q3_2025,Q4_2025 --site seekingalpha.com
    python3 ingestion/transcript_browser.py --ticker MU  --quarters-list Q1_2024,Q2_2024,Q3_2024

Prerequisites:
    pip install playwright requests
    python3 -m playwright install chromium
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page, TimeoutError as PWTimeout

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
        logging.FileHandler(LOGS_DIR / "browser_scrape.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

TAVILY_URL = "https://api.tavily.com/search"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Motley Fool selectors first, then Seeking Alpha, then generic fallbacks
ARTICLE_SELECTORS = [
    "[data-id='article-body']",
    ".article-body",
    ".tailwind-article-body",
    "div[class*='ArticleBody']",
    # Seeking Alpha
    "[data-test-id='article-content']",
    ".article-content",
    "#article-content",
    "div[class*='article_content']",
    "article",
]

MIN_TRANSCRIPT_LEN = 3_000

COMPANY_NAMES = {
    "NVDA": "NVIDIA",
    "AMD":  "Advanced Micro Devices",
    "AVGO": "Broadcom",
    "TSM":  "Taiwan Semiconductor TSMC",
    "ANET": "Arista Networks",
    "MU":   "Micron Technology",
}

# Approximate call months per Q label (used to sharpen Tavily queries).
# Values are month names Motley Fool / SA articles typically include in titles.
# ANET / AVGO / AMD / NVDA follow calendar fiscal year.
# MU fiscal year ends Aug/Sep so Q labels shift.
_CALL_MONTH_HINTS: dict[str, dict[int, str]] = {
    "MU":   {1: "December", 2: "March", 3: "June", 4: "September"},
    # default (calendar-aligned companies): Q1→May, Q2→Aug, Q3→Nov, Q4→Feb
}
_DEFAULT_CALL_MONTHS = {1: "May", 2: "August", 3: "November", 4: "February"}

# ── quarter helpers ───────────────────────────────────────────────────────────

def last_n_calendar_quarters(n: int) -> list:
    year, q = 2025, 4
    quarters = []
    for _ in range(n):
        quarters.append((year, q))
        q -= 1
        if q == 0:
            q, year = 4, year - 1
    return quarters


def parse_quarters_list(spec: str) -> list:
    """Parse 'Q1_2025,Q4_2024' into [(2025,1),(2024,4)]."""
    result = []
    for part in spec.split(","):
        part = part.strip()
        m = re.match(r"Q(\d)_(\d{4})", part, re.IGNORECASE)
        if not m:
            raise ValueError(f"Bad quarter spec '{part}' — expected format Q1_2025")
        result.append((int(m.group(2)), int(m.group(1))))
    return result


# ── delays ────────────────────────────────────────────────────────────────────

def _pause(lo: float = 1.5, hi: float = 3.5) -> None:
    time.sleep(random.uniform(lo, hi))


# ── Stage 1: URL discovery via Tavily ─────────────────────────────────────────

def find_transcript_url(
    ticker: str, year: int, q: int, site: str = "fool.com"
) -> Optional[str]:
    """Use Tavily to find the best transcript URL on *site* for this quarter."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        log.error("TAVILY_API_KEY not set")
        return None

    company     = COMPANY_NAMES.get(ticker.upper(), ticker)
    month_hints = _CALL_MONTH_HINTS.get(ticker.upper(), _DEFAULT_CALL_MONTHS)
    month       = month_hints.get(q, "")
    # Include year twice and the approximate call month to anchor Tavily to the
    # right quarter rather than returning the most-recent article.
    query = (
        f"{company} {ticker} Q{q} {year} {month} {year} "
        f"earnings call transcript site:{site}"
    )

    try:
        resp = requests.post(
            TAVILY_URL,
            json={
                "api_key":      api_key,
                "query":        query,
                "search_depth": "basic",
                "max_results":  5,
            },
            timeout=20,
        )
        resp.raise_for_status()
        for r in resp.json().get("results", []):
            url = r.get("url", "")
            if site in url and "transcript" in url.lower():
                return url
        # Second pass: relax the "transcript" requirement for Seeking Alpha
        # which may use "earnings-call" in the URL instead
        if site != "fool.com":
            for r in resp.json().get("results", []):
                url = r.get("url", "")
                if site in url and ("earnings" in url.lower() or "transcript" in url.lower()):
                    return url
    except Exception as e:
        log.warning(f"Tavily error: {e}")

    return None


# ── Stage 2: full article extraction via Playwright ───────────────────────────

def _dismiss_overlays(page: Page) -> None:
    """Remove popup/paywall overlays via JS so they don't block text extraction."""
    try:
        page.evaluate("""
            ['#popup-container','[class*="pitch-container"]','[id*="popup"]',
             '[class*="paywall"]','[class*="modal"]','[class*="gate"]',
             '[class*="subscribe"]','[class*="regwall"]'].forEach(sel => {
                document.querySelectorAll(sel).forEach(el => el.remove());
            });
            document.body.style.overflow = 'auto';
        """)
    except Exception:
        pass


def _extract_text(page: Page) -> str:
    """Try article selectors in order; fall back to full body."""
    best = ""
    for sel in ARTICLE_SELECTORS:
        try:
            el = page.query_selector(sel)
            if el:
                t = el.inner_text()
                if len(t) > len(best):
                    best = t
        except Exception:
            continue
    if len(best) < MIN_TRANSCRIPT_LEN:
        try:
            body = page.query_selector("body")
            if body:
                t = body.inner_text()
                if len(t) > len(best):
                    best = t
        except Exception:
            pass
    return best.strip()


def _clean(text: str) -> str:
    lines, cleaned, blank_run = text.splitlines(), [], 0
    for line in lines:
        s = line.strip()
        if not s:
            blank_run += 1
            if blank_run <= 2:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(s)
    return "\n".join(cleaned).strip()


def fetch_article(page: Page, url: str) -> str:
    """Navigate to *url*, dismiss overlays, return cleaned article text."""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=40_000)
    except PWTimeout:
        log.warning(f"Timeout on {url}, attempting extraction anyway")
    except Exception as e:
        log.warning(f"Load error for {url}: {e}")
    _pause(2.5, 4.5)
    _dismiss_overlays(page)
    return _clean(_extract_text(page))


# ── quarter validation ────────────────────────────────────────────────────────

def _validate_quarter(text: str, ticker: str, year: int, q: int) -> bool:
    """Warn if the extracted text doesn't mention the expected year.

    Checks the first 600 chars (the header / call date line) for the target
    year.  Allows one-year tolerance for Q1 reports whose call falls in the
    following calendar year (e.g. ANET Q4 2025 call in Feb 2026).
    Returns True if the content looks right, False if it looks stale/wrong.
    """
    sample = text[:600].lower()
    if str(year) in sample:
        return True
    # Q4 companies call in early next year
    if q == 4 and str(year + 1) in sample:
        return True
    # Micron Q4 calls in September — headline year matches
    if ticker.upper() == "MU" and q == 4 and str(year) in sample:
        return True
    return False


# ── per-quarter orchestration ─────────────────────────────────────────────────

def scrape_quarter(
    page: Page,
    ticker: str,
    year: int,
    q: int,
    site: str = "fool.com",
) -> bool:
    label    = f"Q{q} {year}"
    out_path = TRANSCRIPTS_DIR / f"{ticker}_Q{q}_{year}.txt"

    log.info(f"[{ticker}] {label} — searching for URL on {site} …")
    url = find_transcript_url(ticker, year, q, site=site)
    if not url:
        log.warning(f"[{ticker}] {label} — FAIL: no URL found on {site}")
        return False

    log.info(f"[{ticker}] {label} — URL: {url}")
    _pause(1.0, 2.0)

    text = fetch_article(page, url)

    if len(text) < MIN_TRANSCRIPT_LEN:
        log.warning(
            f"[{ticker}] {label} — FAIL: {len(text)} chars extracted "
            f"(possible paywall or parse error)"
        )
        return False

    # Quarter validation — warn but still save so the operator can inspect
    if not _validate_quarter(text, ticker, year, q):
        log.warning(
            f"[{ticker}] {label} — WARN: extracted text does not appear to "
            f"reference {year} in header — possible wrong-quarter URL. "
            f"Saving anyway for manual review."
        )

    out_path.write_text(text, encoding="utf-8")
    log.info(f"[{ticker}] {label} — OK: {len(text):,} chars → {out_path.name}")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",        required=True)
    parser.add_argument("--quarters",      type=int, default=8,
                        help="Number of most-recent calendar quarters (ignored if --quarters-list is set)")
    parser.add_argument("--quarters-list", default=None,
                        help="Comma-separated explicit quarters, e.g. Q1_2025,Q4_2024")
    parser.add_argument("--site",          default="fool.com",
                        help="Site to search: fool.com (default) or seekingalpha.com")
    parser.add_argument("--no-headless",   dest="headless", action="store_false", default=True)
    args   = parser.parse_args()
    ticker = args.ticker.upper()

    if args.quarters_list:
        q_list = parse_quarters_list(args.quarters_list)
        log.info(f"Starting: {ticker}, explicit quarters: {args.quarters_list}, site: {args.site}")
    else:
        q_list = last_n_calendar_quarters(args.quarters)
        log.info(f"Starting: {ticker}, {args.quarters} quarters, site: {args.site}")

    ok_list, fail_list = [], []

    with sync_playwright() as pw:
        launch_args = ["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"]
        try:
            browser = pw.chromium.launch(headless=args.headless, args=launch_args)
        except Exception:
            log.info("Chromium failed, retrying with system Chrome …")
            browser = pw.chromium.launch(headless=args.headless, channel="chrome", args=launch_args)

        ctx = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
        )
        ctx.route("**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf}",
                  lambda r: r.abort())
        page = ctx.new_page()

        for year, q in q_list:
            try:
                if scrape_quarter(page, ticker, year, q, site=args.site):
                    ok_list.append(f"Q{q} {year}")
                else:
                    fail_list.append(f"Q{q} {year}")
            except Exception as e:
                log.error(f"[{ticker}] Q{q} {year} — error: {e}")
                fail_list.append(f"Q{q} {year}")
            _pause(2.0, 3.5)

        browser.close()

    log.info(
        f"\n{'═'*60}\n"
        f"[{ticker}] DONE  OK={len(ok_list)} FAIL={len(fail_list)}\n"
        f"  OK:   {', '.join(ok_list) or 'none'}\n"
        f"  FAIL: {', '.join(fail_list) or 'none'}\n"
        f"{'═'*60}"
    )


if __name__ == "__main__":
    main()
