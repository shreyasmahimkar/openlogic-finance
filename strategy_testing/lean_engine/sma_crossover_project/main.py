# region imports
from AlgorithmImports import *
from sma_crossover_signal import StrategyConfig, SignalType, detect_crossover, drawdown_breached
# endregion

# ──────────────────────────────────────────────────────────────────────────────
# NOTE: sma_crossover_signal.py is the source of truth for all signal logic.
# It lives in model_library/technical/signals/ and is auto-synced here by
# lean_bridge.py before every cloud push. Do NOT edit the copy in lean_project/.
# ──────────────────────────────────────────────────────────────────────────────


class GoldenCrossSMAStrategy(QCAlgorithm):
    """
    LEAN Execution Adapter for the SMA Golden Cross strategy (Box 3).

    This class is responsible ONLY for LEAN-specific concerns:
      - Indicator registration and warm-up (self.sma, Resolution, etc.)
      - Order execution (set_holdings, liquidate)
      - Charting and structured logging

    All signal decisions (crossover detection, drawdown check) are delegated to:
      model_library/technical/signals/sma_crossover_signal.py

    Configurable via config.json parameters:
      - fast-period       (default: 50)
      - slow-period       (default: 200)
      - ticker            (default: SPY)
      - position-size     (default: 1.0 → 100% of portfolio)
      - max-drawdown-pct  (default: 0.15 → 15%)
    """

    def initialize(self):
        # ── Build StrategyConfig from LEAN parameters ─────────────────────────
        # StrategyConfig is defined in model_library — this is the only place
        # LEAN parameters are mapped to the shared config object.
        self._cfg = StrategyConfig(
            fast_period      = int(self.get_parameter("fast-period",      50)),
            slow_period      = int(self.get_parameter("slow-period",     200)),
            position_size    = float(self.get_parameter("position-size",  1.0)),
            max_drawdown_pct = float(self.get_parameter("max-drawdown-pct", 0.15)),
            ticker           = self.get_parameter("ticker", "SPY") or "SPY",
        )

        # ── Date Range ────────────────────────────────────────────────────────
        self.set_start_date(2016, 5, 27)
        self.set_end_date(2026, 5, 12)
        self.set_cash(100_000)

        # ── Universe ──────────────────────────────────────────────────────────
        equity = self.add_equity(self._cfg.ticker, Resolution.DAILY)
        equity.set_data_normalization_mode(DataNormalizationMode.ADJUSTED)
        self.symbol = equity.symbol

        # ── LEAN Indicators (LEAN manages streaming + warm-up) ────────────────
        # Indicator values are extracted each bar and passed to model_library
        # signal functions — LEAN does not own the decision logic.
        self.fast_sma = self.sma(self.symbol, self._cfg.fast_period, Resolution.DAILY)
        self.slow_sma = self.sma(self.symbol, self._cfg.slow_period, Resolution.DAILY)

        # Previous bar state — passed to detect_crossover() each tick
        self._prev_fast: Optional[float] = None
        self._prev_slow: Optional[float] = None

        # Peak portfolio value — passed to drawdown_breached() each tick
        self._peak_value: float = self.portfolio.total_portfolio_value

        # ── Warm-up ───────────────────────────────────────────────────────────
        self.set_warm_up(self._cfg.slow_period + 10, Resolution.DAILY)

        # ── Charts ────────────────────────────────────────────────────────────
        sma_chart = Chart("SMA Crossover")
        sma_chart.add_series(Series("Price",                         SeriesType.LINE,    0, "$"))
        sma_chart.add_series(Series(f"SMA{self._cfg.fast_period}",   SeriesType.LINE,    0, "$"))
        sma_chart.add_series(Series(f"SMA{self._cfg.slow_period}",   SeriesType.LINE,    0, "$"))
        self.add_chart(sma_chart)

        signal_chart = Chart("Trade Signals")
        signal_chart.add_series(Series("Golden Cross", SeriesType.SCATTER, 0, "$"))
        signal_chart.add_series(Series("Death Cross",  SeriesType.SCATTER, 0, "$"))
        self.add_chart(signal_chart)

        portfolio_chart = Chart("Portfolio Value")
        portfolio_chart.add_series(Series("Equity", SeriesType.LINE, 0, "$"))
        self.add_chart(portfolio_chart)

        # ── Counters ──────────────────────────────────────────────────────────
        self._golden_crosses: int = 0
        self._death_crosses:  int = 0

        self.log(
            f"[INIT] {self._cfg.ticker} | "
            f"SMA{self._cfg.fast_period}/{self._cfg.slow_period} | "
            f"Position={self._cfg.position_size * 100:.0f}% | "
            f"MaxDD={self._cfg.max_drawdown_pct * 100:.0f}%"
        )

    # ──────────────────────────────────────────────────────────────────────────
    def on_data(self, data: Slice):
        if self.is_warming_up:
            return

        if not self.fast_sma.is_ready or not self.slow_sma.is_ready:
            return

        if not data.bars.contains_key(self.symbol):
            return

        # ── Extract raw indicator values (LEAN's responsibility ends here) ────
        fast:  float = self.fast_sma.current.value
        slow:  float = self.slow_sma.current.value
        price: float = data[self.symbol].close

        # ── Charting ──────────────────────────────────────────────────────────
        self.plot("SMA Crossover", "Price",                       price)
        self.plot("SMA Crossover", f"SMA{self._cfg.fast_period}", fast)
        self.plot("SMA Crossover", f"SMA{self._cfg.slow_period}", slow)
        self.plot("Portfolio Value", "Equity", self.portfolio.total_portfolio_value)

        # ── Update peak value ─────────────────────────────────────────────────
        current_value: float = self.portfolio.total_portfolio_value
        if current_value > self._peak_value:
            self._peak_value = current_value

        # ── Drawdown Guard — decision delegated to model_library ──────────────
        if self.portfolio[self.symbol].is_long and drawdown_breached(current_value, self._peak_value, self._cfg.max_drawdown_pct):
            self.liquidate(self.symbol, tag="MAX_DRAWDOWN_STOP")
            self.log(
                f"[DRAWDOWN STOP] Liquidated @ ${price:.2f} | "
                f"Drawdown exceeded {self._cfg.max_drawdown_pct * 100:.0f}% limit"
            )
            self._peak_value = self.portfolio.total_portfolio_value  # Reset peak to cash balance

        # ── Crossover Signal — decision delegated to model_library ────────────
        signal: SignalType = detect_crossover(fast, slow, self._prev_fast, self._prev_slow)

        if signal == SignalType.GOLDEN_CROSS:
            if not self.portfolio[self.symbol].is_long:
                self.set_holdings(self.symbol, self._cfg.position_size)
                self._peak_value = self.portfolio.total_portfolio_value  # Reset peak value on trade entry
                self._golden_crosses += 1
                self.plot("Trade Signals", "Golden Cross", price)
                self.log(
                    f"[GOLDEN CROSS #{self._golden_crosses}] BUY {self._cfg.ticker} @ ${price:.2f} | "
                    f"SMA{self._cfg.fast_period}={fast:.2f} > SMA{self._cfg.slow_period}={slow:.2f}"
                )

        elif signal == SignalType.DEATH_CROSS:
            if self.portfolio[self.symbol].is_long:
                self.liquidate(self.symbol, tag="DEATH_CROSS")
                self._death_crosses += 1
                self.plot("Trade Signals", "Death Cross", price)
                self.log(
                    f"[DEATH CROSS #{self._death_crosses}] SELL {self._cfg.ticker} @ ${price:.2f} | "
                    f"SMA{self._cfg.fast_period}={fast:.2f} < SMA{self._cfg.slow_period}={slow:.2f}"
                )

        self._prev_fast = fast
        self._prev_slow = slow

    # ──────────────────────────────────────────────────────────────────────────
    def on_order_event(self, order_event: OrderEvent):
        if order_event.status == OrderStatus.FILLED:
            self.log(
                f"[ORDER FILLED] {order_event.direction} {order_event.fill_quantity} "
                f"{self._cfg.ticker} @ ${order_event.fill_price:.2f} | "
                f"Tag: {self.transactions.get_order_by_id(order_event.order_id).tag}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    def on_end_of_algorithm(self):
        final_value  = self.portfolio.total_portfolio_value
        total_return = ((final_value - 100_000) / 100_000) * 100

        self.log("=" * 60)
        self.log(f"  STRATEGY: SMA{self._cfg.fast_period}/{self._cfg.slow_period} on {self._cfg.ticker}")
        self.log(f"  Starting Cash  : $100,000.00")
        self.log(f"  Final Value    : ${final_value:,.2f}")
        self.log(f"  Total Return   : {total_return:.2f}%")
        self.log(f"  Golden Crosses : {self._golden_crosses}")
        self.log(f"  Death  Crosses : {self._death_crosses}")
        self.log(f"  Total Orders   : {self.transactions.orders_count}")
        self.log("=" * 60)
