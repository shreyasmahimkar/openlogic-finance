"""
strategy_testing/lean_engine/lean_bridge.py

Fully-implemented bridge between OpenLogic Finance and the QuantConnect LEAN CLI.

Usage (standalone):
    python -m strategy_testing.lean_engine.lean_bridge

Usage (as library):
    from strategy_testing.lean_engine.lean_bridge import LeanEngineBridge

    bridge = LeanEngineBridge()
    result = bridge.run_backtest()
    print(result)
"""

import json
import logging
import os
import subprocess
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Default project root relative to this file ───────────────────────────────
_DEFAULT_PROJECT = Path(__file__).parent / "lean_project"


@dataclass
class BacktestResult:
    """Structured result returned after a LEAN backtest run."""

    strategy_name:   str
    ticker:          str
    fast_period:     int
    slow_period:     int
    success:         bool
    return_code:     int
    stdout:          str
    stderr:          str
    output_dir:      Optional[str]    = None
    started_at:      str              = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at:    Optional[str]    = None
    total_return_pct: Optional[float] = None   # parsed from LEAN logs when available

    def to_dict(self) -> dict:
        return {
            "strategy_name":    self.strategy_name,
            "ticker":           self.ticker,
            "fast_period":      self.fast_period,
            "slow_period":      self.slow_period,
            "success":          self.success,
            "return_code":      self.return_code,
            "total_return_pct": self.total_return_pct,
            "output_dir":       self.output_dir,
            "started_at":       self.started_at,
            "completed_at":     self.completed_at,
        }


class LeanEngineBridge:
    """
    Bridge to handle the QuantConnect Lean CLI local environment.

    This integrates Box 3 (Strategy Testing) with LEAN for robust backtesting
    and is designed to be callable as an ADK tool from the strategy_testing agent.

    Pre-requisites:
        pip install lean
        lean login          # one-time, uses your QC credentials
        lean init <project> # already done — lean_project/ is pre-initialised

    Env variables (optional overrides):
        LEAN_PROJECT_PATH   Path to the LEAN project directory
        LEAN_CLI_PATH       Path to the `lean` executable (default: resolved via PATH)
    """

    def __init__(
        self,
        project_path: Optional[str] = None,
        lean_cli:     Optional[str] = None,
    ):
        self.project_path = Path(
            project_path
            or os.getenv("LEAN_PROJECT_PATH", str(_DEFAULT_PROJECT))
        ).resolve()

        self.lean_cli = lean_cli or os.getenv("LEAN_CLI_PATH") or shutil.which("lean") or "lean"

        logger.info(
            f"[LeanEngineBridge] project_path={self.project_path} | lean_cli={self.lean_cli}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run_backtest(
        self,
        ticker:        str = "SPY",
        fast_period:   int = 50,
        slow_period:   int = 200,
        position_size: float = 1.0,
        output_dir:    Optional[str] = None,
    ) -> BacktestResult:
        """
        Trigger a local LEAN backtest for the SMA Golden Cross strategy.

        Args:
            ticker:        Asset ticker (e.g. "SPY", "QQQ", "AAPL").
            fast_period:   Fast SMA period (default 50).
            slow_period:   Slow SMA period (default 200).
            position_size: Fractional portfolio allocation on signal (default 1.0).
            output_dir:    Where LEAN writes results. Defaults to lean_project/backtests/.

        Returns:
            BacktestResult dataclass with success flag and parsed metrics.
        """
        strategy_name = f"SMA{fast_period}_SMA{slow_period}_{ticker}"
        started_at    = datetime.utcnow().isoformat()

        logger.info(f"[BACKTEST START] {strategy_name}")

        # ── Patch config.json with caller-supplied parameters ─────────────────
        config_path = self.project_path / "config.json"
        self._patch_config(config_path, ticker, fast_period, slow_period, position_size)

        # ── Build LEAN CLI command ────────────────────────────────────────────
        cmd = [
            self.lean_cli,
            "backtest",
            str(self.project_path),
            "--output", output_dir or str(self.project_path / "backtests" / strategy_name),
            "--log-file", str(self.project_path / "backtests" / f"{strategy_name}.log"),
        ]

        logger.info(f"[LEAN CMD] {' '.join(cmd)}")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,         # 10-minute safety timeout
            )

            completed_at = datetime.utcnow().isoformat()
            success      = proc.returncode == 0

            if success:
                logger.info(f"[BACKTEST OK] {strategy_name} completed in {proc.returncode}")
            else:
                logger.error(f"[BACKTEST FAIL] rc={proc.returncode}\n{proc.stderr[:500]}")

            # ── Try to extract total return from LEAN stdout ──────────────────
            total_return = self._parse_return(proc.stdout)

            return BacktestResult(
                strategy_name    = strategy_name,
                ticker           = ticker,
                fast_period      = fast_period,
                slow_period      = slow_period,
                success          = success,
                return_code      = proc.returncode,
                stdout           = proc.stdout,
                stderr           = proc.stderr,
                output_dir       = output_dir,
                started_at       = started_at,
                completed_at     = completed_at,
                total_return_pct = total_return,
            )

        except FileNotFoundError:
            msg = (
                f"LEAN CLI not found at '{self.lean_cli}'. "
                "Run `pip install lean` and `lean login` first."
            )
            logger.error(msg)
            return BacktestResult(
                strategy_name=strategy_name,
                ticker=ticker,
                fast_period=fast_period,
                slow_period=slow_period,
                success=False,
                return_code=-1,
                stdout="",
                stderr=msg,
            )

        except subprocess.TimeoutExpired:
            msg = f"LEAN backtest timed out after 600s for {strategy_name}"
            logger.error(msg)
            return BacktestResult(
                strategy_name=strategy_name,
                ticker=ticker,
                fast_period=fast_period,
                slow_period=slow_period,
                success=False,
                return_code=-2,
                stdout="",
                stderr=msg,
            )

    def check_lean_installed(self) -> dict:
        """
        Verify the LEAN CLI is installed and reachable.

        Returns:
            {"installed": bool, "version": str}
        """
        try:
            result = subprocess.run(
                [self.lean_cli, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return {
                "installed": result.returncode == 0,
                "version":   result.stdout.strip() or result.stderr.strip(),
            }
        except FileNotFoundError:
            return {"installed": False, "version": "lean not found"}

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _patch_config(
        self,
        config_path:   Path,
        ticker:        str,
        fast_period:   int,
        slow_period:   int,
        position_size: float,
    ) -> None:
        """Overwrite config.json parameters without touching other settings."""
        if not config_path.exists():
            logger.warning(f"config.json not found at {config_path}, skipping patch")
            return

        with open(config_path) as f:
            config = json.load(f)

        config.setdefault("parameters", {}).update(
            {
                "fast-period":    str(fast_period),
                "slow-period":    str(slow_period),
                "ticker":         ticker,
                "position-size":  str(position_size),
            }
        )

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        logger.debug(f"[CONFIG PATCH] {config_path} updated with {ticker} SMA{fast_period}/{slow_period}")

    @staticmethod
    def _parse_return(stdout: str) -> Optional[float]:
        """
        Attempt to extract 'Total Return' % from LEAN's stdout log lines.
        Returns None if not parseable.
        """
        for line in stdout.splitlines():
            if "Total Return" in line:
                parts = line.split()
                for part in parts:
                    clean = part.replace("%", "").replace(",", "")
                    try:
                        return float(clean)
                    except ValueError:
                        continue
        return None


# ── Standalone entry-point ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    bridge = LeanEngineBridge()

    # Quick install check
    check = bridge.check_lean_installed()
    print(f"\nLEAN CLI installed: {check['installed']} | version: {check['version']}\n")

    if check["installed"]:
        result = bridge.run_backtest(
            ticker      = "SPY",
            fast_period = 50,
            slow_period = 200,
        )
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("Install LEAN first: pip install lean && lean login")
