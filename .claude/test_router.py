from query_router import route_query, detect_ticker, query_snowflake

test_queries = [
    "What was NVDA revenue in 2023?",
    "Explain TSMC's competitive moat",
    "What were AMD earnings last quarter?",
    "How does Arista Networks make money?",
]

for q in test_queries:
    route = route_query(q)
    ticker = detect_ticker(q)
    print(f"Q: {q}")
    print(f"→ Route: {route} | Ticker: {ticker}\n")