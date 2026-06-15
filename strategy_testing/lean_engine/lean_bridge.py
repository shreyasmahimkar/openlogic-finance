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
_DEFAULT_PROJECT = Path(__file__).parent / "logistic_regression_project"
_DEFAULT_WORKSPACE = Path(__file__).parent / "lean_workspace"
_VENV_LEAN = Path(__file__).parents[2] / ".openlogic-env" / "bin" / "lean"

# ── model_library signal files to sync into lean_project before every push ───
# key   = source path relative to repo root (model_library)
# value = destination filename inside lean_project/
_REPO_ROOT = Path(__file__).parents[2]
_SIGNAL_SYNC_MAP: dict[str, str] = {
    "model_library/technical/signals/sma_crossover_signal.py": "sma_crossover_signal.py",
    "model_library/ml_zoo/logistic_regression.py": "logistic_regression.py",
}


@dataclass
class BacktestResult:
    """Structured result returned after a LEAN backtest run."""

    strategy_name: str
    ticker: str
    fast_period: int
    slow_period: int
    success: bool
    return_code: int
    stdout: str
    stderr: str
    output_dir: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    total_return_pct: Optional[float] = None  # parsed from LEAN logs when available
    cagr_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    total_orders: Optional[int] = None
    full_summary: Optional[str] = None  # full stats table from lean cloud backtest

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "ticker": self.ticker,
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "success": self.success,
            "return_code": self.return_code,
            "total_return_pct": self.total_return_pct,
            "cagr_pct": self.cagr_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_orders": self.total_orders,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "full_summary": self.full_summary,
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
        lean_cli: Optional[str] = None,
    ):
        if project_path:
            p = Path(project_path)
            if not p.is_absolute():
                self.project_path = (_REPO_ROOT / p).resolve()
            else:
                self.project_path = p.resolve()
        else:
            self.project_path = Path(
                os.getenv("LEAN_PROJECT_PATH", str(_DEFAULT_PROJECT))
            ).resolve()

        # Prefer venv lean binary → env override → shutil.which (pyenv-safe fallback)
        self.lean_cli = (
            lean_cli
            or os.getenv("LEAN_CLI_PATH")
            or (str(_VENV_LEAN) if _VENV_LEAN.exists() else None)
            or shutil.which("lean")
            or "lean"
        )

        self.workspace_path = Path(
            os.getenv("LEAN_WORKSPACE_PATH", str(_DEFAULT_WORKSPACE))
        ).resolve()

        self.project_name = self.project_path.name

        logger.info(
            f"[LeanEngineBridge] project_path={self.project_path} "
            f"| workspace={self.workspace_path} | lean_cli={self.lean_cli} "
            f"| project_name={self.project_name}"
        )

        # Dynamically log in if credentials are provided via environment variables (e.g. Streamlit Secrets)
        qc_user_id = os.getenv("QC_USER_ID") or os.getenv("QUANTCONNECT_API_ID")
        qc_api_token = os.getenv("QC_API_TOKEN") or os.getenv("QUANTCONNECT_API_TOKEN")
        if qc_user_id and qc_api_token:
            try:
                credentials_path = Path.home() / ".lean" / "credentials"
                if not credentials_path.exists():
                    logger.info(
                        "[LeanEngineBridge] LEAN credentials file not found. Attempting automated login..."
                    )
                    env = self._get_subprocess_env()
                    login_cmd = [
                        self.lean_cli,
                        "login",
                        "--user-id",
                        qc_user_id,
                        "--api-token",
                        qc_api_token,
                    ]
                    login_proc = subprocess.run(login_cmd, capture_output=True, text=True, env=env)
                    if login_proc.returncode != 0:
                        logger.error(
                            f"[LeanEngineBridge] Automated login failed: {login_proc.stderr.strip()}"
                        )
                    else:
                        logger.info("[LeanEngineBridge] Automated login successful.")
            except Exception as e:
                logger.error(f"[LeanEngineBridge] Error during automated login: {str(e)}")

    # ── Public API ────────────────────────────────────────────────────────────

    def run_backtest(
        self,
        ticker: str = "SPY",
        fast_period: int = 50,
        slow_period: int = 200,
        position_size: float = 1.0,
        max_drawdown_pct: float = 0.15,
        probability_threshold: float = 0.5,
        rsi_period: int = 14,
        output_dir: Optional[str] = None,
    ) -> BacktestResult:
        """
        Trigger a local LEAN backtest for the SMA Golden Cross strategy.

        Args:
            ticker:        Asset ticker (e.g. "SPY", "QQQ", "AAPL").
            fast_period:   Fast SMA period (default 50).
            slow_period:   Slow SMA period (default 200).
            position_size: Fractional portfolio allocation on signal (default 1.0).
            max_drawdown_pct: Maximum allowed drawdown stop limit (default 0.15).
            output_dir:    Where LEAN writes results. Defaults to lean_project/backtests/.

        Returns:
            BacktestResult dataclass with success flag and parsed metrics.
        """
        strategy_name = f"SMA{fast_period}_SMA{slow_period}_{ticker}"
        started_at = datetime.utcnow().isoformat()

        logger.info(f"[BACKTEST START] {strategy_name}")

        # Ensure the project directory exists under lean_workspace
        dest_project_dir = self.workspace_path / self.project_name
        dest_project_dir.mkdir(parents=True, exist_ok=True)

        # Copy config.json to workspace project folder if missing there
        src_config = self.project_path / "config.json"
        dest_config = dest_project_dir / "config.json"
        if src_config.exists() and not dest_config.exists():
            shutil.copy2(src_config, dest_config)
            logger.info(f"[SYNC] Copied config.json → {dest_config}")

        # ── Patch config.json with caller-supplied parameters ─────────────────
        # Patch both the source project and the workspace copy
        for cfg_dir in [self.project_path, dest_project_dir]:
            config_path = cfg_dir / "config.json"
            self._patch_config(
                config_path,
                ticker,
                fast_period,
                slow_period,
                position_size,
                max_drawdown_pct,
                probability_threshold,
                rsi_period,
            )

        # ── Sync strategy files to workspace ────────────────────────────
        # 1. Always sync main.py
        src_main = self.project_path / "main.py"
        dest_main = dest_project_dir / "main.py"
        if src_main.exists():
            shutil.copy2(src_main, dest_main)
            logger.info(f"[SYNC] Copied {src_main} → {dest_main}")

        # 2. Sync model_library signal files (source of truth → project_dir → workspace)
        #    This ensures LEAN cloud always runs the latest signal logic from Box 2.
        for rel_src, dest_name in _SIGNAL_SYNC_MAP.items():
            src_signal = _REPO_ROOT / rel_src
            dest_local = self.project_path / dest_name  # project_dir/ (tracked)
            dest_ws = dest_project_dir / dest_name  # lean_workspace/project_name/ (cloud push)

            if not src_signal.exists():
                logger.warning(f"[SYNC SKIP] Signal source not found: {src_signal}")
                continue

            shutil.copy2(src_signal, dest_local)
            logger.info(f"[SYNC] model_library → project_dir: {dest_name}")

            shutil.copy2(src_signal, dest_ws)
            logger.info(f"[SYNC] model_library → lean_workspace: {dest_name}")

        # ── Cloud push then cloud backtest (no Docker required) ───────────────
        push_cmd = [self.lean_cli, "cloud", "push", "--project", self.project_name]
        bt_cmd = [self.lean_cli, "cloud", "backtest", self.project_name]

        logger.info(f"[LEAN PUSH] {' '.join(push_cmd)}")

        try:
            env = self._get_subprocess_env()
            push_proc = subprocess.run(
                push_cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.workspace_path),  # lean.json lives here
                env=env,
            )
            if push_proc.returncode != 0:
                logger.error(f"[PUSH FAIL] {push_proc.stderr[:500]}")

            logger.info(f"[LEAN BACKTEST] {' '.join(bt_cmd)}")
            proc = subprocess.run(
                bt_cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(self.workspace_path),  # lean.json lives here
                env=env,
            )

            completed_at = datetime.utcnow().isoformat()
            success = proc.returncode == 0
            combined_out = proc.stdout + "\n" + push_proc.stdout

            if success:
                logger.info(f"[BACKTEST OK] {strategy_name}")
            else:
                logger.error(f"[BACKTEST FAIL] rc={proc.returncode}\n{proc.stderr[:500]}")

            # ── Parse stats from cloud backtest table output ──────────────────
            total_return = self._parse_return(combined_out)
            cagr = self._parse_cagr(combined_out)
            drawdown = self._parse_drawdown(combined_out)
            orders = self._parse_orders(combined_out)
            full_summary = self._extract_summary_table(combined_out)

            return BacktestResult(
                strategy_name=strategy_name,
                ticker=ticker,
                fast_period=fast_period,
                slow_period=slow_period,
                success=success,
                return_code=proc.returncode,
                stdout=combined_out,
                stderr=proc.stderr,
                output_dir=output_dir,
                started_at=started_at,
                completed_at=completed_at,
                total_return_pct=total_return,
                cagr_pct=cagr,
                max_drawdown_pct=drawdown,
                total_orders=orders,
                full_summary=full_summary,
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
            env = self._get_subprocess_env()
            result = subprocess.run(
                [self.lean_cli, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
            )
            return {
                "installed": result.returncode == 0,
                "version": result.stdout.strip() or result.stderr.strip(),
            }
        except FileNotFoundError:
            return {"installed": False, "version": "lean not found"}

    def _get_subprocess_env(self) -> dict:
        """
        Prepare environment variables for the subprocess.
        To support read-only virtualenvs (like Streamlit Cloud), we copy the lean package
        to a writable directory in home and prepend it to PYTHONPATH so lean can dynamically
        write its modules-*.json file.
        """
        env = os.environ.copy()

        import importlib.util

        lean_src_dir = None
        try:
            spec = importlib.util.find_spec("lean")
            if spec and spec.submodule_search_locations:
                lean_src_dir = spec.submodule_search_locations[0]
        except Exception:
            pass

        if not lean_src_dir:
            import sys

            for path in sys.path:
                if not path:
                    continue
                candidate = os.path.join(path, "lean")
                if os.path.isdir(candidate) and os.path.exists(
                    os.path.join(candidate, "__init__.py")
                ):
                    lean_src_dir = candidate
                    break

        if lean_src_dir:
            try:
                writable_parent = Path.home() / ".lean_writable_pkg"
                writable_lean_dir = writable_parent / "lean"
                if not writable_lean_dir.exists():
                    writable_parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(lean_src_dir, writable_lean_dir, dirs_exist_ok=True)
                    logger.info(
                        f"[LeanEngineBridge] Copied lean package to writable path: {writable_lean_dir}"
                    )

                # Prepend the writable parent directory to PYTHONPATH
                existing_pythonpath = env.get("PYTHONPATH", "")
                if existing_pythonpath:
                    env["PYTHONPATH"] = os.path.pathsep.join(
                        [str(writable_parent), existing_pythonpath]
                    )
                else:
                    env["PYTHONPATH"] = str(writable_parent)
            except Exception as e:
                logger.error(f"[LeanEngineBridge] Failed to setup writable lean package copy: {e}")

        return env

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _patch_config(
        self,
        config_path: Path,
        ticker: str,
        fast_period: int,
        slow_period: int,
        position_size: float,
        max_drawdown_pct: float = 0.15,
        probability_threshold: float = 0.5,
        rsi_period: int = 14,
    ) -> None:
        """Overwrite config.json parameters without touching other settings."""
        if not config_path.exists():
            logger.warning(f"config.json not found at {config_path}, skipping patch")
            return

        with open(config_path) as f:
            config = json.load(f)

        config.setdefault("parameters", {}).update(
            {
                "fast-period": str(fast_period),
                "slow-period": str(slow_period),
                "ticker": ticker,
                "position-size": str(position_size),
                "max-drawdown-pct": str(max_drawdown_pct),
                "probability-threshold": str(probability_threshold),
                "rsi-period": str(rsi_period),
            }
        )

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        logger.debug(
            f"[CONFIG PATCH] {config_path} updated with {ticker} SMA{fast_period}/{slow_period}"
        )

    @staticmethod
    def _parse_return(stdout: str) -> Optional[float]:
        """
        Extract Net Profit / Total Return from both local and cloud LEAN output.
        Handles table rows like: │ Net Profit │ -3.738% │
        and plain log lines like: Total Return  -3.74%
        """
        import re

        # Cloud table format: │ Net Profit │ -3.738% │
        for line in stdout.splitlines():
            if "Net Profit" in line:
                match = re.search(r"([\-\+]?\d+\.?\d*)%", line)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass
            # Plain log format fallback: Total Return  -3.74%
            if "Total Return" in line:
                match = re.search(r"([\-\+]?\d+\.?\d*)%", line)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass
        return None

    @staticmethod
    def _parse_cagr(stdout: str) -> Optional[float]:
        import re

        for line in stdout.splitlines():
            # │ Compounding Annual  │ 8.072%         │
            if "Compounding Annual" in line:
                match = re.search(r"│\s*Compounding Annual\s*│\s*([\-\+]?\d+\.?\d*)%", line)
                if match:
                    return float(match.group(1))
        return None

    @staticmethod
    def _parse_drawdown(stdout: str) -> Optional[float]:
        import re

        for line in stdout.splitlines():
            # │ Drawdown           │ 19.900%          │
            if "Drawdown" in line and "Recovery" not in line:
                match = re.search(r"│\s*Drawdown\s*│\s*([\-\+]?\d+\.?\d*)%", line)
                if match:
                    return float(match.group(1))
        return None

    @staticmethod
    def _parse_orders(stdout: str) -> Optional[int]:
        import re

        for line in stdout.splitlines():
            # │ Total Orders       │ 7                │
            if "Total Orders" in line:
                match = re.search(r"│\s*Total Orders\s*│\s*(\d+)", line)
                if match:
                    return int(match.group(1))
        return None

    @staticmethod
    def _extract_summary_table(stdout: str) -> Optional[str]:
        """
        Extract the full statistics table block from lean cloud backtest stdout.
        Returns everything between the first and last box-drawing border lines.
        """
        lines = stdout.splitlines()
        table_lines = []
        in_table = False
        for line in lines:
            if "\u250c" in line or "\u2514" in line or "\u251c" in line:  # ┌ └ ├
                in_table = True
            if in_table:
                table_lines.append(line)
            if "\u2514" in line and in_table:  # └ = closing border
                break
        return (
            "\n".join(table_lines) if table_lines else stdout[-3000:]
        )  # fallback: last 3000 chars


# ── Standalone entry-point ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    bridge = LeanEngineBridge()

    # Quick install check
    check = bridge.check_lean_installed()
    print(f"\nLEAN CLI installed: {check['installed']} | version: {check['version']}\n")

    if check["installed"]:
        result = bridge.run_backtest(
            ticker="SPY",
            fast_period=50,
            slow_period=200,
        )
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("Install LEAN first: pip install lean && lean login")
