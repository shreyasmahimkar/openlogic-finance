# region imports
from AlgorithmImports import *
from logistic_regression import (
    LogisticStrategyConfig,
    LogisticModelPayload,
    LRSignalType,
    predict_probability,
    evaluate_signal,
    engineer_features,
)
from sma_crossover_signal import drawdown_breached
# endregion

# ──────────────────────────────────────────────────────────────────────────────
# NOTE: logistic_regression.py and sma_crossover_signal.py are source of truth.
# They live in model_library/ and are auto-synced here by lean_bridge.py
# before every cloud push. Do NOT edit the copies in lean_project/.
# ──────────────────────────────────────────────────────────────────────────────


class LogisticRegressionStrategy(QCAlgorithm):
    """
    LEAN Execution Adapter for the Logistic Regression strategy (Box 3).

    This class is responsible ONLY for LEAN-specific concerns:
      - Indicator registration and warm-up (SMA, RSI, Resolution, etc.)
      - Order execution (set_holdings, liquidate)
      - Charting and structured logging

    All signal decisions (feature engineering, prediction probability, signal evaluation,
    drawdown check) are delegated to:
      model_library/ml_zoo/logistic_regression.py
      model_library/technical/signals/sma_crossover_signal.py

    Configurable via config.json parameters:
      - fast-period             (default: 50)
      - slow-period             (default: 200)
      - rsi-period              (default: 14)
      - probability-threshold   (default: 0.5)
      - ticker                  (default: SPY)
      - position-size           (default: 1.0)
      - max-drawdown-pct        (default: 0.15)
    """

    def initialize(self):
        # ── Build LogisticStrategyConfig from LEAN parameters ─────────────────
        self._cfg = LogisticStrategyConfig(
            ticker           = self.get_parameter("ticker", "SPY") or "SPY",
            fast_period      = int(self.get_parameter("fast-period", 50)),
            slow_period      = int(self.get_parameter("slow-period", 200)),
            rsi_period       = int(self.get_parameter("rsi-period", 14)),
            probability_threshold = float(self.get_parameter("probability-threshold", 0.5)),
            position_size    = float(self.get_parameter("position-size", 1.0)),
            max_drawdown_pct = float(self.get_parameter("max-drawdown-pct", 0.15)),
        )

        # ── Date Range ────────────────────────────────────────────────────────
        self.set_start_date(2016, 5, 27)
        self.set_end_date(2026, 5, 12)
        self.set_cash(100_000)

        # ── Universe ──────────────────────────────────────────────────────────
        equity = self.add_equity(self._cfg.ticker, Resolution.DAILY)
        equity.set_data_normalization_mode(DataNormalizationMode.ADJUSTED)
        self.symbol = equity.symbol

        # ── LEAN Indicators ───────────────────────────────────────────────────
        self.fast_sma = self.sma(self.symbol, self._cfg.fast_period, Resolution.DAILY)
        self.slow_sma = self.sma(self.symbol, self._cfg.slow_period, Resolution.DAILY)
        self.rsi = self.rsi(self.symbol, self._cfg.rsi_period, MovingAverageType.WILDERS, Resolution.DAILY)

        # Previous bar state variables
        self._prev_close: Optional[float] = None
        self._prev_prob: Optional[float] = None

        # Peak portfolio value for risk guard
        self._peak_value: float = self.portfolio.total_portfolio_value

        # ── Warm-up ───────────────────────────────────────────────────────────
        warmup_period = max(self._cfg.slow_period, self._cfg.rsi_period) + 10
        self.set_warm_up(warmup_period, Resolution.DAILY)

        # ── Pre-Trained Model Coefficients (Default) ──────────────────────────
        self._model = LogisticModelPayload(
            weights={
                "sma_ratio": 2.5,
                "rsi_norm": 0.5,
                "momentum": 1.0
            },
            intercept=0.1,
            feature_means={
                "sma_ratio": 0.005,
                "rsi_norm": 0.02,
                "momentum": 0.0003
            },
            feature_stds={
                "sma_ratio": 0.03,
                "rsi_norm": 0.35,
                "momentum": 0.015
            }
        )

        # ── Charts ────────────────────────────────────────────────────────────
        prob_chart = Chart("LR Model Probability")
        prob_chart.add_series(Series("Probability", SeriesType.LINE, 0, "%"))
        prob_chart.add_series(Series("Threshold", SeriesType.LINE, 0, "%"))
        self.add_chart(prob_chart)

        portfolio_chart = Chart("Portfolio Value")
        portfolio_chart.add_series(Series("Equity", SeriesType.LINE, 0, "$"))
        self.add_chart(portfolio_chart)

        # Counters
        self._buys: int = 0
        self._sells: int = 0

        self.log(
            f"[INIT] {self._cfg.ticker} | "
            f"SMA{self._cfg.fast_period}/{self._cfg.slow_period} | "
            f"RSI{self._cfg.rsi_period} | "
            f"Threshold={self._cfg.probability_threshold:.2f} | "
            f"Position={self._cfg.position_size * 100:.0f}% | "
            f"MaxDD={self._cfg.max_drawdown_pct * 100:.0f}%"
        )

    # ──────────────────────────────────────────────────────────────────────────
    def on_data(self, data: Slice):
        if self.is_warming_up:
            return

        if not self.fast_sma.is_ready or not self.slow_sma.is_ready or not self.rsi.is_ready:
            return

        if not data.bars.contains_key(self.symbol):
            return

        # ── Extract raw values ────────────────────────────────────────────────
        price: float = data[self.symbol].close
        fast:  float = self.fast_sma.current.value
        slow:  float = self.slow_sma.current.value
        rsi_val: float = self.rsi.current.value

        # ── Feature Engineering ───────────────────────────────────────────────
        raw_data = {
            "close": price,
            "fast_sma": fast,
            "slow_sma": slow,
            "rsi": rsi_val,
            "prev_close": self._prev_close if self._prev_close is not None else price
        }
        features = engineer_features(raw_data)

        # ── Predict Probability ────────────────────────────────────────────────
        prob = predict_probability(features, self._model)

        # ── Charting ──────────────────────────────────────────────────────────
        self.plot("LR Model Probability", "Probability", prob * 100.0)
        self.plot("LR Model Probability", "Threshold", self._cfg.probability_threshold * 100.0)
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

        # ── Evaluate Signal ───────────────────────────────────────────────────
        signal: LRSignalType = evaluate_signal(prob, self._prev_prob, self._cfg.probability_threshold)

        if signal == LRSignalType.BUY:
            if not self.portfolio[self.symbol].is_long:
                self.set_holdings(self.symbol, self._cfg.position_size)
                self._peak_value = self.portfolio.total_portfolio_value  # Reset peak value on trade entry
                self._buys += 1
                self.log(
                    f"[BUY #{self._buys}] BUY {self._cfg.ticker} @ ${price:.2f} | "
                    f"Prob={prob:.4f} > Threshold={self._cfg.probability_threshold:.2f}"
                )

        elif signal == LRSignalType.SELL:
            if self.portfolio[self.symbol].is_long:
                self.liquidate(self.symbol, tag="LR_PROB_EXIT")
                self._sells += 1
                self.log(
                    f"[SELL #{self._sells}] SELL {self._cfg.ticker} @ ${price:.2f} | "
                    f"Prob={prob:.4f} <= Threshold={self._cfg.probability_threshold:.2f}"
                )

        self._prev_close = price
        self._prev_prob = prob

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
        self.log(f"  STRATEGY: Logistic Regression Model on {self._cfg.ticker}")
        self.log(f"  Starting Cash  : $100,000.00")
        self.log(f"  Final Value    : ${final_value:,.2f}")
        self.log(f"  Total Return   : {total_return:.2f}%")
        self.log(f"  Buys           : {self._buys}")
        self.log(f"  Sells          : {self._sells}")
        self.log(f"  Total Orders   : {self.transactions.orders_count}")
        self.log("=" * 60)
