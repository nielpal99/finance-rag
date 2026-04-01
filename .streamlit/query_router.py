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
    # GPU & AI chip manufacturers
    "nvidia": "NVDA",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "intel": "INTC",
    "qualcomm": "QCOM",
    "marvell": "MRVL",
    "marvell technology": "MRVL",
    "arm holdings": "ARM",
    # GPU cloud
    "coreweave": "CRWV",
    "crwv": "CRWV",
    # Server & systems
    "super micro": "SMCI",
    "supermicro": "SMCI",
    # Foundry
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    # Semiconductor equipment
    "applied materials": "AMAT",
    "lam research": "LRCX",
    "kla corporation": "KLAC",
    "teradyne": "TER",
    "entegris": "ENTG",
    "onto innovation": "ONTO",
    "asml": "ASML",
    # Memory & storage
    "micron": "MU",
    "western digital": "WDC",
    "seagate": "STX",
    "seagate technology": "STX",
    # Networking & infrastructure
    "arista": "ANET",
    "arista networks": "ANET",
    "cisco": "CSCO",
    "ciena": "CIEN",
    "infinera": "INFN",
    # Hyperscalers
    "microsoft": "MSFT",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "meta platforms": "META",
    "facebook": "META",
    # Enterprise software & cloud data
    "oracle": "ORCL",
    "datadog": "DDOG",
    "mongodb": "MDB",
    "cloudflare": "NET",
    "confluent": "CFLT",
    "gitlab": "GTLB",
    # Networking & mixed-signal semiconductors
    "broadcom": "AVGO",
    "microchip technology": "MCHP",
    "microchip": "MCHP",
    "skyworks": "SWKS",
    "skyworks solutions": "SWKS",
    "qorvo": "QRVO",
    "macom": "MTSI",
    "macom technology": "MTSI",
    # Power & data center infrastructure
    "vistra": "VST",
    "eaton": "ETN",
    # Packaging
    "amkor": "AMKR",
    "amkor technology": "AMKR",
    # Consumer & enterprise tech
    "apple": "AAPL",
    "netflix": "NFLX",
    "salesforce": "CRM",
    "servicenow": "NOW",
    "palantir": "PLTR",
}
TICKERS = [
    # GPU & AI chip manufacturers
    "NVDA", "AMD", "INTC", "QCOM", "MRVL", "ARM",
    # GPU cloud
    "CRWV",
    # Server & systems
    "SMCI",
    # Foundry
    "TSM",
    # Semiconductor equipment
    "AMAT", "LRCX", "KLAC", "TER", "ENTG", "ONTO", "ASML",
    # Memory & storage
    "MU", "WDC", "STX",
    # Networking & infrastructure
    "ANET", "CSCO", "CIEN", "INFN",
    # Hyperscalers
    "MSFT", "GOOGL", "AMZN", "META",
    # Enterprise software & cloud data
    "ORCL", "SNOW", "DDOG", "MDB", "NET", "CFLT", "GTLB",
    # Networking & mixed-signal semiconductors
    "AVGO", "MCHP", "SWKS", "QRVO", "MTSI",
    # Power & data center infrastructure
    "VST", "ETN",
    # Packaging
    "AMKR",
    # Consumer & enterprise tech
    "AAPL", "NFLX", "CRM", "NOW", "PLTR",
]

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

def _format_concept(name: str) -> str:
    """Convert CamelCase XBRL concept to readable label: OperatingIncomeLoss → Operating Income Loss"""
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)

def query_snowflake(query: str, ticker: str) -> str:
    secrets = toml.load(".streamlit/secrets.toml")
    year_match = re.search(r"\b(20\d{2})\b", query)

    # Try financial_intelligence.marts.fct_company_metrics first (all 47 companies)
    try:
        conn = snowflake.connector.connect(**secrets["snowflake_monitor"])
        cursor = conn.cursor()
        if year_match:
            cursor.execute("""
                SELECT period_end, form,
                       revenue, gross_profit, operating_income, net_income,
                       r_and_d, capex, assets, cash,
                       gross_margin_pct, operating_margin_pct, net_margin_pct
                FROM financial_intelligence.marts.fct_company_metrics
                WHERE ticker = %s
                  AND YEAR(period_end) = %s
                ORDER BY period_end DESC
                LIMIT 10
            """, (ticker, int(year_match.group(1))))
        else:
            cursor.execute("""
                SELECT period_end, form,
                       revenue, gross_profit, operating_income, net_income,
                       r_and_d, capex, assets, cash,
                       gross_margin_pct, operating_margin_pct, net_margin_pct
                FROM financial_intelligence.marts.fct_company_metrics
                WHERE ticker = %s
                ORDER BY period_end DESC
                LIMIT 10
            """, (ticker,))
        rows = cursor.fetchall()
        conn.close()

        if rows:
            result = f"**Structured financial data for {ticker} (AI Infrastructure Monitor):**\n\n"
            for (period_end, form, revenue, gross_profit, operating_income,
                 net_income, r_and_d, capex, assets, cash,
                 gross_margin_pct, operating_margin_pct, net_margin_pct) in rows:
                result += f"**{period_end} ({form})**\n"
                if revenue is not None:
                    result += f"- Revenue: ${revenue:,.0f}\n"
                if gross_profit is not None:
                    gm = f" ({gross_margin_pct:.1f}% margin)" if gross_margin_pct is not None else ""
                    result += f"- Gross Profit: ${gross_profit:,.0f}{gm}\n"
                if operating_income is not None:
                    om = f" ({operating_margin_pct:.1f}% margin)" if operating_margin_pct is not None else ""
                    result += f"- Operating Income: ${operating_income:,.0f}{om}\n"
                if net_income is not None:
                    nm = f" ({net_margin_pct:.1f}% margin)" if net_margin_pct is not None else ""
                    result += f"- Net Income: ${net_income:,.0f}{nm}\n"
                if r_and_d is not None:
                    result += f"- R&D: ${r_and_d:,.0f}\n"
                if capex is not None:
                    result += f"- CapEx: ${capex:,.0f}\n"
                if assets is not None:
                    result += f"- Total Assets: ${assets:,.0f}\n"
                if cash is not None:
                    result += f"- Cash: ${cash:,.0f}\n"
                result += "\n"
            return result
    except Exception:
        pass

    # Fall back to finance_rag.xbrl.xbrl_facts for original 6 tickers
    conn = snowflake.connector.connect(**secrets["snowflake"])
    cursor = conn.cursor()
    if year_match:
        cursor.execute("""
            SELECT DISTINCT concept, value, period_end, form
            FROM xbrl_facts
            WHERE ticker = %s
              AND YEAR(period_end) = %s
            ORDER BY period_end DESC
            LIMIT 20
        """, (ticker, int(year_match.group(1))))
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
        result += f"- {_format_concept(concept)}: ${value:,.0f} (as of {period_end}, {form})\n\n"

    return result