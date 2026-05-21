# Data Prep (BOX 1)

Transformed from the former `data_ingestion/` directory, this box is responsible for market connections, processing pipelines, and quantitative feature stores.

## Target Structure & Subdirectories

- **`connectors/`**: Financial news, GDELT global events, and market data engines (e.g. yfinance, NYTimes API).
- **`pipelines/`**: Cleaning, parsing, storage, and transformation scripts.
- **`features/`**: Alternative data parsers, embedding generators, and feature stores.

## Purpose & Architectural Rule

This box specializes in raw financial data acquisition, validation, and feature engineering. All clean tables, vector embeddings, and alternative signals are generated here before being passed to `model_library` or `strategy_testing`.
