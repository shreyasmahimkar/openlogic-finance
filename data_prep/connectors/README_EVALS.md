# Box 1 connector evals

Each connector agent has an `eval/` folder with a schema-valid ADK evalset and a
`test_config.json` (scoring thresholds). They verify the agent takes the right
**tool trajectory** and holds its response contract — the non-deterministic part
that unit tests can't cover.

| Agent | Evalset | Expected trajectory |
|---|---|---|
| `market_data` | `market_data/eval/market_data.evalset.json` | `fetch_and_explain` |
| `financial_news` | `financial_news/eval/financial_news.evalset.json` | `check_news_cache → search_articles → save_news_to_csv` (cache-miss path) |
| `global_events` | `global_events/eval/global_events.evalset.json` | `get_global_events` |

## Validate (no keys) — runs in CI
`model_library/tests/test_evalsets.py` schema-checks every evalset repo-wide.

## Scored run (needs `GEMINI_API_KEY`)
```bash
adk eval data_prep/connectors/market_data \
    data_prep/connectors/market_data/eval/market_data.evalset.json \
    --config_file_path data_prep/connectors/market_data/eval/test_config.json
```
