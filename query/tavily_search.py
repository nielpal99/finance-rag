import os
from tavily import TavilyClient

try:
    import streamlit as st
    def get_secret(key):
        return st.secrets.get(key) or os.environ.get(key, "")
except ImportError:
    def get_secret(key):
        return os.environ.get(key, "")

TICKERS = ["NVDA", "AMD", "TSM", "ANET", "AVGO", "MU"]
COMPANY_NAMES = {
    "NVDA": "NVIDIA",
    "AMD": "AMD Advanced Micro Devices",
    "TSM": "TSMC Taiwan Semiconductor",
    "ANET": "Arista Networks",
    "AVGO": "Broadcom",
    "MU": "Micron Technology",
}

def get_client():
    return TavilyClient(api_key=get_secret("TAVILY_API_KEY"))

def tavily_search(query: str, tickers: list[str]) -> str:
    client = get_client()

    # Enrich query with company names for better results
    company_context = " ".join([COMPANY_NAMES.get(t, t) for t in tickers])
    enriched_query = f"{query} {company_context} earnings financials 2025 2026"

    response = client.search(
        query=enriched_query,
        search_depth="advanced",
        max_results=5,
        include_answer=True,
    )

    answer = response.get("answer", "")
    results = response.get("results", [])

    output = ""
    if answer:
        output += f"{answer}\n\n"

    if results:
        output += "**Sources:**\n"
        for r in results[:3]:
            output += f"- [{r['title']}]({r['url']})\n"

    return output.strip()

def should_use_tavily(answer: str, sources: list[dict]) -> bool:
    # Trigger Tavily if RAG couldn't find anything
    if "couldn't find" in answer.lower():
        return True
    # Trigger if all scores are low
    if sources and all(s.get("score", 0) < 0.3 for s in sources):
        return True
    return False
