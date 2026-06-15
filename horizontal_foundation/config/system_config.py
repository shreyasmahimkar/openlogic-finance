from pathlib import Path


class SystemConfig:
    """Provides passive baseline system configurations and paths."""

    # repo root: system_config.py is at horizontal_foundation/config/, so parents[2].
    WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
    DEFAULT_ASSET_DIR = WORKSPACE_ROOT / "assets"
    DEFAULT_TICKER = "SPY"
    DEFAULT_PERIOD = "10y"

    YFINANCE_CACHE_TTL = 3600  # standard TTL in seconds for caches

    @classmethod
    def get_asset_path(cls, filename: str) -> Path:
        """Returns the absolute path for an asset, ensuring the assets directory exists."""
        cls.DEFAULT_ASSET_DIR.mkdir(parents=True, exist_ok=True)
        return cls.DEFAULT_ASSET_DIR / filename
