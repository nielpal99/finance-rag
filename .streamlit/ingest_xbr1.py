import httpx
import snowflake.connector
import toml
from datetime import datetime

# Your AI infrastructure tickers + CIK numbers
TICKERS = {
    "NVDA": "0001045810",
    "AMD":  "0000002488",
    "TSM":  "0001046179",
    "ANET": "0001596532",
    "AVGO": "0001730168",
    "MU":   "0000723125",
}


IFRS_CONCEPTS = [
    "Revenue",
    "ProfitLoss",
    "BasicEarningsLossPerShare",
    "ProfitLossFromOperatingActivities",
    "Assets",
    "CashAndCashEquivalents",
]
# The financial concepts we care about
CONCEPTS = [
    "Revenues",
    "NetIncomeLoss",
    "EarningsPerShareBasic",
    "OperatingIncomeLoss",
    "Assets",
    "CashAndCashEquivalentsAtCarryingValue",
]

def fetch_xbrl(cik: str):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = httpx.get(url, headers={"User-Agent": "nielpal99@gmail.com"})
    r.raise_for_status()
    return r.json()

def extract_facts(ticker, cik, data):
    rows = []
    gaap = data.get("facts", {}).get("us-gaap", {})
    
    for concept in CONCEPTS:
        if concept not in gaap:
            continue
        units = gaap[concept].get("units", {})
        entries = units.get("USD", units.get("shares", []))
        
        for entry in entries:
            if entry.get("form") not in ("10-K", "10-Q", "20-F"):
                continue
            rows.append((
                ticker,
                cik,
                concept,
                entry.get("val"),
                entry.get("end"),
                entry.get("form"),
                entry.get("filed"),
            ))

    ifrs = data.get("facts", {}).get("ifrs-full", {})
    for concept in IFRS_CONCEPTS:
        if concept not in ifrs:
            continue
        entries = ifrs[concept].get("units", {}).get("USD", [])
        for entry in entries:
            if entry.get("form") != "20-F":
                continue
            rows.append((
                ticker,
                cik,
                concept,
                entry.get("val"),
                entry.get("end"),
                entry.get("form"),
                entry.get("filed"),
            ))  
    return rows

def load_to_snowflake(rows):
    secrets = toml.load(".streamlit/secrets.toml")
    sf = secrets["snowflake"]
    conn = snowflake.connector.connect(**sf)
    cursor = conn.cursor()
    
    cursor.executemany("""
        INSERT INTO xbrl_facts 
        (ticker, cik, concept, value, period_end, form, filed)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, rows)
    
    conn.commit()
    conn.close()
    print(f"✅ Loaded {len(rows)} rows into Snowflake")

if __name__ == "__main__":
    all_rows = []
    for ticker, cik in TICKERS.items():
        print(f"Fetching {ticker}...")
        data = fetch_xbrl(cik)
        rows = extract_facts(ticker, cik, data)
        all_rows.extend(rows)
        print(f"  → {len(rows)} facts extracted")
    
    load_to_snowflake(all_rows)