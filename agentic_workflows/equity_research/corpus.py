"""Sample earnings-call transcript snippets for the equity-research RAG slice.

Stand-in for real transcripts + 10-K MD&A that, in production, would be ingested
from GCS / S3 / Snowflake (see docs/DATA_PLATFORMS_ANALYSIS — future). Fictional
company "Nimbus Corp (NMBS)".
"""

SAMPLE_TRANSCRIPTS = [
    {
        "id": "nmbs-q4-guidance",
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
        "source": "Nimbus Corp (NMBS) Q4 2025 Earnings Call — CEO Q&A",
        "text": (
            "Demand signals strengthened through the quarter; net revenue retention rose "
            "to 118 percent and enterprise bookings grew 24 percent year over year. We are "
            "seeing longer-duration contracts, which improves revenue visibility into 2026."
        ),
    },
    {
        "id": "nmbs-q4-risk",
        "source": "Nimbus Corp (NMBS) Q4 2025 Earnings Call — Risk commentary",
        "text": (
            "We remain cautious on the macro environment. A stronger dollar is a 2 to 3 "
            "point headwind to reported revenue, and elevated interest rates may extend "
            "sales cycles for rate-sensitive customers in financial services."
        ),
    },
    {
        "id": "nmbs-q4-margins",
        "source": "Nimbus Corp (NMBS) Q4 2025 Earnings Call — Margins & costs",
        "text": (
            "Gross margin reached 79 percent, up 200 basis points, driven by data-center "
            "efficiency. We expect continued leverage in R&D as a percent of revenue while "
            "absorbing higher AI-infrastructure depreciation over the next two years."
        ),
    },
]
