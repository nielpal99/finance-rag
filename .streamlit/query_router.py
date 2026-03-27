from __future__ import annotations
import re
import snowflake.connector
import toml

# Keywords that signal a structured financial metric question
METRIC_KEYWORDS = [
    "revenue", "revenues", "earnings", "eps", "net income",
    "cash", "operating income", "assets", "profit", "quarterly",
    "annual", "q1", "q2", "q3", "q4", "fiscal", "2021", "2022",
    "2023", "2024", "how much", "what was", "what were"
]

COMPANY_ALIASES = {
    "nvidia": "NVDA",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    "arista": "ANET",
    "arista networks": "ANET",
    "broadcom": "AVGO",
    "micron": "MU",
}
TICKERS = ["NVDA", "AMD", "TSM", "ANET", "AVGO", "MU"]

def detect_ticker(query: str) -> str | None:
    q_lower = query.lower()
    q_upper = query.upper()
    
    # Check company name aliases first
    for alias, ticker in COMPANY_ALIASES.items():
        if alias in q_lower:
            return ticker
    
    # Fall back to ticker symbol match
    for ticker in TICKERS:
        if ticker in q_upper:
            return ticker
    
    return None

def is_metric_question(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in METRIC_KEYWORDS)

def route_query(query: str) -> str:
    if is_metric_question(query) and detect_ticker(query):
        return "snowflake"
    return "pinecone"

def query_snowflake(query: str, ticker: str) -> str:
    secrets = toml.load(".streamlit/secrets.toml")
    sf = secrets["snowflake"]
    conn = snowflake.connector.connect(**sf)
    cursor = conn.cursor()

    year_match = re.search(r"\b(20\d{2})\b", query)

    if year_match:
        year = year_match.group(1)
        cursor.execute("""
            SELECT DISTINCT concept, value, period_end, form
            FROM xbrl_facts
            WHERE ticker = %s
            AND YEAR(period_end) = %s
            ORDER BY period_end DESC
            LIMIT 20
        """, (ticker, int(year)))
    else:
        cursor.execute("""
            SELECT DISTINCT concept, value, period_end, form
            FROM xbrl_facts
            WHERE ticker = %s
            ORDER BY period_end DESC
            LIMIT 20
        """, (ticker,))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return f"No structured data found for {ticker} matching your query."

    result = f"**Structured financial data for {ticker}:**\n\n"
    for concept, value, period_end, form in rows:
        result += f"- {concept}: ${value:,.0f} (as of {period_end}, {form})\n\n"

    return result