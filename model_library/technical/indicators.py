import logging
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)


def enrich_ohlcv_data(csv_path: str) -> str:
    """
    Reads an OHLCV CSV file, calculates MoE-F required technical indicators using
    pandas and numpy exclusively, and saves it to a new enriched CSV.

    Required Indicators:
    - MACD: EMA(12) - EMA(26)
    - Bollinger Bands: SMA(20) ± [2 × σ(20)]
    - 30-Day RSI
    - 30-Day CCI
    - 30-Day DX
    - 30-Day & 60-Day SMAs
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot enrich data. File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Close" not in df.columns:
        raise ValueError("CSV must contain a 'Close' column.")

    logger.info(f"Calculating technical indicators for {csv_path}...")

    # 1. SMAs
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()
    df["SMA_60"] = df["Close"].rolling(window=60).mean()

    # 2. MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26

    # 3. Bollinger Bands
    std_20 = df["Close"].rolling(window=20).std()
    df["Bollinger_Upper"] = df["SMA_20"] + (2 * std_20)
    df["Bollinger_Lower"] = df["SMA_20"] - (2 * std_20)

    # 4. 30-Day RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    # Standard RSI uses smoothed moving average (EMA)
    avg_gain = gain.ewm(alpha=1 / 30, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 30, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["RSI_30"] = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))

    # Ensure High and Low exist for CCI and DX
    if "High" in df.columns and "Low" in df.columns:
        # 5. 30-Day CCI
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        sma_tp = tp.rolling(window=30).mean()

        # Calculate Mean Deviation carefully to avoid deprecated df.mad()
        # pandas rolling.apply is slow, we use a rolling window directly
        def mean_deviation(x):
            return np.abs(x - np.mean(x)).mean()

        md = tp.rolling(window=30).apply(mean_deviation, raw=True)
        # Add epsilon to prevent division by zero
        df["CCI_30"] = (tp - sma_tp) / (0.015 * (md + 1e-8))

        # 6. 30-Day DX
        high_diff = df["High"].diff()
        low_diff = -df["Low"].diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

        tr1 = df["High"] - df["Low"]
        tr2 = np.abs(df["High"] - df["Close"].shift(1))
        tr3 = np.abs(df["Low"] - df["Close"].shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smooth with Wilder's method equivalent (alpha = 1/n)
        smoothed_plus_dm = pd.Series(plus_dm).ewm(alpha=1 / 30, adjust=False).mean()
        smoothed_minus_dm = pd.Series(minus_dm).ewm(alpha=1 / 30, adjust=False).mean()
        smoothed_tr = tr.ewm(alpha=1 / 30, adjust=False).mean()

        plus_di = 100 * (smoothed_plus_dm / (smoothed_tr + 1e-8))
        minus_di = 100 * (smoothed_minus_dm / (smoothed_tr + 1e-8))

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df["DX_30"] = dx
    else:
        logger.warning("High/Low columns missing. Skipping CCI and DX indicators.")

    # Save to _enriched.csv
    base, ext = os.path.splitext(csv_path)
    enriched_path = f"{base}_enriched{ext}"
    df.to_csv(enriched_path, index=False)

    logger.info(f"Enriched CSV securely saved to {enriched_path}")
    return enriched_path


def read_market_indicators(csv_path: str) -> str:
    """Reads the last 10 trading days of OHLCV and technical indicators from the enriched CSV.

    Args:
        csv_path: Absolute or relative path to the enriched market data CSV.

    Returns:
        A formatted string of the latest 10 days of indicators.
    """
    import os
    import pandas as pd

    if not os.path.exists(csv_path):
        return f"Error: File not found at {csv_path}"
    try:
        df = pd.read_csv(csv_path)
        last_10 = df.tail(10)
        cols = [
            "Date",
            "Close",
            "SMA_20",
            "SMA_30",
            "SMA_60",
            "MACD",
            "Bollinger_Upper",
            "Bollinger_Lower",
            "RSI_30",
            "CCI_30",
            "DX_30",
        ]
        cols = [c for c in cols if c in last_10.columns]
        sub_df = last_10[cols]
        header = " | ".join(cols)
        rows = []
        for _, row in sub_df.iterrows():
            rows.append(" | ".join([str(row[c]) for c in cols]))
        return header + "\n" + "-" * len(header) + "\n" + "\n".join(rows)
    except Exception as e:
        return f"Error reading indicators: {e}"


def apply_semantic_news_filter(
    csv_path: str, news_column_name: str = "news_text", similarity_threshold: float = 0.2
) -> str:
    """Applies semantic similarity filtering to cached news data.

    Args:
        csv_path: Path to the enriched CSV (used to locate the asset folder).
        news_column_name: Unused here, kept for schema compatibility.
        similarity_threshold: Threshold below which news articles are filtered out.

    Returns:
        High-signal news chunks.
    """
    import os
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    asset_dir = "assets"
    news_files = [
        f for f in os.listdir(asset_dir) if f.startswith("financial_news_") and f.endswith(".csv")
    ]
    if not news_files:
        return "No news data files found in assets directory."

    news_files.sort()
    latest_news_file = os.path.join(asset_dir, news_files[-1])
    try:
        df = pd.read_csv(latest_news_file)
    except Exception as e:
        return f"Error reading news file: {e}"

    parsed_articles = []
    for idx, row in df.iterrows():
        try:
            val = row.iloc[0]
            if isinstance(val, str) and (val.startswith("{") or val.startswith("[")):
                art_dict = eval(val)
            else:
                art_dict = {"headline": row.get("headline", ""), "snippet": row.get("snippet", "")}

            headline = art_dict.get("headline", "")
            snippet = art_dict.get("snippet", "")
            parsed_articles.append(
                {"headline": headline, "snippet": snippet, "text": f"{headline}. {snippet}"}
            )
        except Exception:
            continue

    if not parsed_articles:
        return "No articles could be parsed."

    baseline_query = "stock market economy finance interest rates inflation Federal Reserve SPY index trading earnings"
    texts = [baseline_query] + [art["text"] for art in parsed_articles]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)

    similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    filtered_articles = []
    for i, sim in enumerate(similarities):
        if sim >= similarity_threshold:
            filtered_articles.append(
                f"- **{parsed_articles[i]['headline']}** (Similarity: {sim:.2f})\n  {parsed_articles[i]['snippet']}"
            )

    if not filtered_articles:
        top_indices = similarities.argsort()[-3:][::-1]
        for idx in top_indices:
            filtered_articles.append(
                f"- **{parsed_articles[idx]['headline']}** (Similarity: {similarities[idx]:.2f} - Fallback)\n  {parsed_articles[idx]['snippet']}"
            )

    return "High-signal news chunks:\n\n" + "\n".join(filtered_articles)
