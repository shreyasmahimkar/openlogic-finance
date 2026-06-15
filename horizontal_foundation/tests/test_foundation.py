from pathlib import Path
from horizontal_foundation.config.system_config import SystemConfig
from horizontal_foundation.interpretability.explain_engine import ExplanationEngine
from horizontal_foundation.core.base_connector import BaseConnector
from data_prep.connectors.market_data.tools import MarketDataConnector

def test_system_config():
    """Verifies that SystemConfig returns the correct defaults and resolves paths."""
    assert SystemConfig.DEFAULT_TICKER == "SPY"
    assert SystemConfig.DEFAULT_PERIOD == "10y"
    
    # Test path resolution
    test_path = SystemConfig.get_asset_path("test_file.csv")
    assert isinstance(test_path, Path)
    assert test_path.name == "test_file.csv"

def test_explanation_engine():
    """Verifies that the ExplanationEngine produces custom responses for beginner vs academic tiers."""
    mock_metadata = {
        "ticker": "AAPL",
        "rows_fetched": 100,
        "start_date": "2026-01-01",
        "end_date": "2026-05-01",
        "latest_close_price": 175.50
    }
    
    beginner_text = ExplanationEngine.explain_data_prep(mock_metadata, level="beginner")
    academic_text = ExplanationEngine.explain_data_prep(mock_metadata, level="academic")
    
    # Check Beginner content
    assert "🧸 Beginner Explanation" in beginner_text
    assert "AAPL" in beginner_text
    assert "175.50" in beginner_text
    assert "500 biggest businesses" in beginner_text
    
    # Check Academic content
    assert "🔬 Academic Quantitative Explanation" in academic_text
    assert "AAPL" in academic_text
    assert "175.50" in academic_text
    assert "stationarity drift" in academic_text

def test_connector_inheritance():
    """Verifies that MarketDataConnector correctly inherits from the BaseConnector interface."""
    connector = MarketDataConnector()
    assert isinstance(connector, BaseConnector)
