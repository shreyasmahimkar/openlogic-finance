# region imports
from AlgorithmImports import *
# endregion


class GoldenCrossSMAStrategy(QCAlgorithm):
    """
    SMA 50/200 Golden Cross Strategy for QuantConnect LEAN Engine.

    Signal Logic:
      - Golden Cross: SMA50 crosses ABOVE SMA200 → Long (Buy & Hold)
      - Death Cross:  SMA50 crosses BELOW SMA200 → Flat (Liquidate)

    Configurable via config.json parameters:
      - fast-period  (default: 50)
      - slow-period  (default: 200)
      - ticker       (default: SPY)
      - position-size (default: 1.0 → 100% of portfolio)
    """

    def initialize(self):
        # ── Date Range ────────────────────────────────────────────────────────
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2026, 5, 12)
        self.set_cash(100_000)

        # ── Parameters (overridable from config.json) ─────────────────────────
        self.fast_period  = int(self.get_parameter("fast-period",   50))
        self.slow_period  = int(self.get_parameter("slow-period",  200))
        self.ticker       = self.get_parameter("ticker",          "SPY") or "SPY"
        self.position_size = float(self.get_parameter("position-size", 1.0))

        # ── Universe ──────────────────────────────────────────────────────────
        equity = self.add_equity(self.ticker, Resolution.DAILY)
        equity.set_data_normalization_mode(DataNormalizationMode.ADJUSTED)
        self.symbol = equity.symbol

        # ── Indicators ────────────────────────────────────────────────────────
        self.fast_sma = self.sma(self.symbol, self.fast_period, Resolution.DAILY)
        self.slow_sma = self.sma(self.symbol, self.slow_period, Resolution.DAILY)

        # Consolidate a crossover detector on both SMAs
        self._prev_fast = None
        self._prev_slow = None

        # ── Warm-up ───────────────────────────────────────────────────────────
        # Allow indicators to stabilise before trading
        self.set_warm_up(self.slow_period + 10, Resolution.DAILY)

        # ── Risk Management ───────────────────────────────────────────────────
        # Max drawdown stop: liquidate if portfolio drops >15% from peak
        self._peak_value = self.portfolio.total_portfolio_value
        self._max_drawdown_pct = 0.15

        # ── Charting ──────────────────────────────────────────────────────────
        sma_chart = Chart("SMA Crossover")
        sma_chart.add_series(Series("Price",  SeriesType.LINE, 0, "$"))
        sma_chart.add_series(Series(f"SMA{self.fast_period}", SeriesType.LINE, 0, "$"))
        sma_chart.add_series(Series(f"SMA{self.slow_period}", SeriesType.LINE, 0, "$"))
        self.add_chart(sma_chart)

        signal_chart = Chart("Trade Signals")
        signal_chart.add_series(Series("Golden Cross", SeriesType.SCATTER, 0, "$"))
        signal_chart.add_series(Series("Death Cross",  SeriesType.SCATTER, 0, "$"))
        self.add_chart(signal_chart)

        portfolio_chart = Chart("Portfolio Value")
        portfolio_chart.add_series(Series("Equity", SeriesType.LINE, 0, "$"))
        self.add_chart(portfolio_chart)

        # ── Trade Counter ─────────────────────────────────────────────────────
        self._golden_crosses = 0
        self._death_crosses   = 0

        self.log(
            f"[INIT] GoldenCross | Ticker={self.ticker} | "
            f"Fast SMA={self.fast_period} | Slow SMA={self.slow_period} | "
            f"Position Size={self.position_size*100:.0f}%"
        )

    # ──────────────────────────────────────────────────────────────────────────
    def on_data(self, data: Slice):
        if self.is_warming_up:
            return

        if not self.fast_sma.is_ready or not self.slow_sma.is_ready:
            return

        if not data.bars.contains_key(self.symbol):
            return

        fast  = self.fast_sma.current.value
        slow  = self.slow_sma.current.value
        price = data[self.symbol].close

        # ── Charting ──────────────────────────────────────────────────────────
        self.plot("SMA Crossover", "Price",                  price)
        self.plot("SMA Crossover", f"SMA{self.fast_period}", fast)
        self.plot("SMA Crossover", f"SMA{self.slow_period}", slow)
        self.plot("Portfolio Value", "Equity", self.portfolio.total_portfolio_value)

        # ── Drawdown Guard ────────────────────────────────────────────────────
        current_value = self.portfolio.total_portfolio_value
        if current_value > self._peak_value:
            self._peak_value = current_value

        drawdown = (self._peak_value - current_value) / self._peak_value
        if drawdown >= self._max_drawdown_pct and self.portfolio[self.symbol].is_long:
            self.liquidate(self.symbol, tag="MAX_DRAWDOWN_STOP")
            self.log(
                f"[DRAWDOWN STOP] Liquidated at ${price:.2f} | "
                f"Drawdown={drawdown*100:.1f}% exceeded {self._max_drawdown_pct*100:.0f}% limit"
            )
            self._prev_fast = fast
            self._prev_slow = slow
            return

        # ── First bar: seed previous values ──────────────────────────────────
        if self._prev_fast is None:
            self._prev_fast = fast
            self._prev_slow = slow
            return

        # ── Golden Cross: SMA50 crosses ABOVE SMA200 ──────────────────────────
        if self._prev_fast <= self._prev_slow and fast > slow:
            if not self.portfolio[self.symbol].is_long:
                self.set_holdings(self.symbol, self.position_size)
                self._golden_crosses += 1
                self.plot("Trade Signals", "Golden Cross", price)
                self.log(
                    f"[GOLDEN CROSS #{self._golden_crosses}] BUY {self.ticker} @ ${price:.2f} | "
                    f"SMA{self.fast_period}={fast:.2f} > SMA{self.slow_period}={slow:.2f}"
                )

        # ── Death Cross: SMA50 crosses BELOW SMA200 ───────────────────────────
        elif self._prev_fast >= self._prev_slow and fast < slow:
            if self.portfolio[self.symbol].is_long:
                self.liquidate(self.symbol, tag="DEATH_CROSS")
                self._death_crosses += 1
                self.plot("Trade Signals", "Death Cross", price)
                self.log(
                    f"[DEATH CROSS #{self._death_crosses}] SELL {self.ticker} @ ${price:.2f} | "
                    f"SMA{self.fast_period}={fast:.2f} < SMA{self.slow_period}={slow:.2f}"
                )

        self._prev_fast = fast
        self._prev_slow = slow

    # ──────────────────────────────────────────────────────────────────────────
    def on_order_event(self, order_event: OrderEvent):
        if order_event.status == OrderStatus.FILLED:
            self.log(
                f"[ORDER FILLED] {order_event.direction} {order_event.fill_quantity} "
                f"{self.ticker} @ ${order_event.fill_price:.2f} | "
                f"Tag: {self.transactions.get_order_by_id(order_event.order_id).tag}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    def on_end_of_algorithm(self):
        final_value  = self.portfolio.total_portfolio_value
        total_return = ((final_value - 100_000) / 100_000) * 100

        self.log("=" * 60)
        self.log(f"  STRATEGY SUMMARY: SMA{self.fast_period}/{self.slow_period} Golden Cross on {self.ticker}")
        self.log(f"  Starting Cash  : $100,000.00")
        self.log(f"  Final Value    : ${final_value:,.2f}")
        self.log(f"  Total Return   : {total_return:.2f}%")
        self.log(f"  Golden Crosses : {self._golden_crosses}")
        self.log(f"  Death  Crosses : {self._death_crosses}")
        self.log(f"  Total Orders   : {self.transactions.order_count}")
        self.log("=" * 60)
