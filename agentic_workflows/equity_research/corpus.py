"""Sample earnings-call transcript snippets for the equity-research RAG slice.

Stand-in for real transcripts + 10-K MD&A that, in production, would be ingested
from GCS / S3 / Snowflake (see docs/DATA_PLATFORMS_ANALYSIS — future). Fictional
company "Nimbus Corp (NMBS)", plus mock real-world assets.
"""

SAMPLE_TRANSCRIPTS = [
    # ── NIMBUS CORP (NMBS) ───────────────────────────────────────────────────
    {
        "id": "nmbs-q4-guidance",
        "ticker": "NMBS",
        "source": "Nimbus Corp (NMBS) Q4 2025 Earnings Call — CFO prepared remarks",
        "text": (
            "For fiscal 2026 we guide revenue growth of 8 to 10 percent and operating "
            "margin expansion of roughly 120 basis points as our cloud mix increases. "
            "We expect free cash flow conversion above 90 percent and plan to resume "
            "share repurchases in the second half of the year."
        ),
    },
    {
        "id": "nmbs-q4-demand",
        "ticker": "NMBS",
        "source": "Nimbus Corp (NMBS) Q4 2025 Earnings Call — CEO Q&A",
        "text": (
            "Demand signals strengthened through the quarter; net revenue retention rose "
            "to 118 percent and enterprise bookings grew 24 percent year over year. We are "
            "seeing longer-duration contracts, which improves revenue visibility into 2026."
        ),
    },
    {
        "id": "nmbs-q4-risk",
        "ticker": "NMBS",
        "source": "Nimbus Corp (NMBS) Q4 2025 Earnings Call — Risk commentary",
        "text": (
            "We remain cautious on the macro environment. A stronger dollar is a 2 to 3 "
            "point headwind to reported revenue, and elevated interest rates may extend "
            "sales cycles for rate-sensitive customers in financial services."
        ),
    },
    {
        "id": "nmbs-q4-margins",
        "ticker": "NMBS",
        "source": "Nimbus Corp (NMBS) Q4 2025 Earnings Call — Margins & costs",
        "text": (
            "Gross margin reached 79 percent, up 200 basis points, driven by data-center "
            "efficiency. We expect continued leverage in R&D as a percent of revenue while "
            "absorbing higher AI-infrastructure depreciation over the next two years."
        ),
    },
    # ── APPLE (AAPL) ──────────────────────────────────────────────────────────
    {
        "id": "aapl-q4-guidance",
        "ticker": "AAPL",
        "source": "Apple Inc. (AAPL) Q4 2025 Earnings Call — CFO prepared remarks",
        "text": (
            "For fiscal 2026, we expect Apple revenue growth to accelerate, driven by iPhone "
            "demand and services momentum. Gross margin is guided to a record 46.5 to 47.5 percent, "
            "and we plan to return $110 billion to shareholders through repurchases and dividends."
        ),
    },
    {
        "id": "aapl-q4-demand",
        "ticker": "AAPL",
        "source": "Apple Inc. (AAPL) Q4 2025 Earnings Call — CEO Q&A",
        "text": (
            "The early adoption of Apple Intelligence has driven strong device upgrade cycles "
            "in North America and Europe. Services subscriber base reached 1.1 billion, growing "
            "14 percent year over year, representing strong recurring demand."
        ),
    },
    {
        "id": "aapl-q4-risk",
        "ticker": "AAPL",
        "source": "Apple Inc. (AAPL) Q4 2025 Earnings Call — Risk commentary",
        "text": (
            "Supply chain tightness in key high-performance components and foreign exchange fluctuations "
            "in Asia-Pacific present minor risks. Consumer spending volatility in mature markets "
            "could also affect hardware refresh rates."
        ),
    },
    {
        "id": "aapl-q4-margins",
        "ticker": "AAPL",
        "source": "Apple Inc. (AAPL) Q4 2025 Earnings Call — Margins & costs",
        "text": (
            "Services gross margin hit 74 percent, expanding 150 basis points due to scale. "
            "Hardware gross margin remained stable at 36.5 percent, supported by favorable "
            "component pricing and product mix shift towards Pro models."
        ),
    },
    # ── GOOGLE (GOOG) ─────────────────────────────────────────────────────────
    {
        "id": "goog-q4-guidance",
        "ticker": "GOOG",
        "source": "Alphabet Inc. (GOOG) Q4 2025 Earnings Call — CFO prepared remarks",
        "text": (
            "For fiscal 2026, Alphabet guides search and services revenue growth of 10 to 12 percent, "
            "with Google Cloud operating margins expected to exceed 25 percent as we optimize infrastructure. "
            "We expect higher capital expenditures to support AI development, resulting in elevated depreciation."
        ),
    },
    {
        "id": "goog-q4-demand",
        "ticker": "GOOG",
        "source": "Alphabet Inc. (GOOG) Q4 2025 Earnings Call — CEO Q&A",
        "text": (
            "Our AI assistant integration has seen strong consumer adoption, leading to a 15 percent "
            "increase in search engagement. Google Cloud backlog grew 30 percent, reflecting "
            "large enterprise multi-year commitments to our AI infrastructure."
        ),
    },
    {
        "id": "goog-q4-risk",
        "ticker": "GOOG",
        "source": "Alphabet Inc. (GOOG) Q4 2025 Earnings Call — Risk commentary",
        "text": (
            "Regulatory headwinds and antitrust scrutiny remain our primary risks, potentially "
            "impacting ad tech revenue. Additionally, higher hardware sales mix may slightly "
            "compress overall gross margins."
        ),
    },
    {
        "id": "goog-q4-margins",
        "ticker": "GOOG",
        "source": "Alphabet Inc. (GOOG) Q4 2025 Earnings Call — Margins & costs",
        "text": (
            "Google Cloud operating margin reached 22 percent in Q4, up from 18 percent last year. "
            "AI data-center efficiency offsets rising energy costs, keeping search margin stable."
        ),
    },
    # ── BITCOIN / CRYPTO MACRO (BTC) ──────────────────────────────────────────
    {
        "id": "btc-q4-guidance",
        "ticker": "BTC",
        "source": "Digital Asset Roundtable (BTC) Q4 2025 — Treasury Outlook",
        "text": (
            "For fiscal 2026, we expect institutional inflows into digital assets to accelerate, "
            "driven by spot ETF adoption. Corporate treasury strategies remain focused on buying "
            "and holding Bitcoin, targeting a 15 percent increase in BTC holdings."
        ),
    },
    {
        "id": "btc-q4-demand",
        "ticker": "BTC",
        "source": "Digital Asset Roundtable (BTC) Q4 2025 — Q&A Session",
        "text": (
            "Halving-induced supply constraints combined with growing global liquidity have "
            "historically driven bullish regimes. Layer-2 development is improving transaction "
            "velocity, showing strong utility and adoption."
        ),
    },
    {
        "id": "btc-q4-risk",
        "ticker": "BTC",
        "source": "Digital Asset Roundtable (BTC) Q4 2025 — Risk commentary",
        "text": (
            "Regulatory shifts, global energy restrictions on mining, and extreme volatility "
            "remain key risk factors that could affect short-term liquidations and leverage."
        ),
    },
    {
        "id": "btc-q4-margins",
        "ticker": "BTC",
        "source": "Digital Asset Roundtable (BTC) Q4 2025 — Protocol Activity",
        "text": (
            "Transaction fees stabilized at lower levels due to protocol efficiency, while "
            "network hashrate reached new all-time highs, reflecting robust security and miner conviction."
        ),
    },
    # ── S&P 500 / MACRO INDEX (SPY) ───────────────────────────────────────────
    {
        "id": "spy-q4-guidance",
        "ticker": "SPY",
        "source": "Macroeconomic Outlook (SPY) Q4 2025 — CFO prepared remarks",
        "text": (
            "For fiscal 2026, aggregate S&P 500 earnings growth is guided at 9 to 11 percent, "
            "supported by resilient consumer spending and moderate inflation. We anticipate the "
            "Federal Reserve will cut rates by 75 basis points, easing borrowing costs."
        ),
    },
    {
        "id": "spy-q4-demand",
        "ticker": "SPY",
        "source": "Macroeconomic Outlook (SPY) Q4 2025 — CEO Q&A",
        "text": (
            "Corporate profit margins remain near historic highs of 12.2 percent. Strength in "
            "technology and financials is offsetting minor weakness in real estate and energy sectors."
        ),
    },
    {
        "id": "spy-q4-risk",
        "ticker": "SPY",
        "source": "Macroeconomic Outlook (SPY) Q4 2025 — Risk commentary",
        "text": (
            "Geopolitical tensions, trade tariffs, and sticky service-sector inflation present "
            "key risks to corporate margin projections and consumer credit quality."
        ),
    },
    {
        "id": "spy-q4-margins",
        "ticker": "SPY",
        "source": "Macroeconomic Outlook (SPY) Q4 2025 — Margins & costs",
        "text": (
            "Operating leverage has improved across major sectors. Energy costs have moderated, "
            "while productivity gains from automation are helping control labor costs."
        ),
    },
]
