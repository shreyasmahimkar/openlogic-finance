"""
risk_management/portfolio/auditor.py

Active drawdown auditor and risk veto controls simulator (Box 4).
"""

import pandas as pd
from model_library.technical.signals.sma_crossover_signal import drawdown_breached


def run_audited_simulation(
    df: pd.DataFrame, max_drawdown_pct: float, initial_capital: float = 100000.0
) -> tuple[pd.DataFrame, bool, any, float, float]:
    """
    Simulates crossover strategy with active drawdown auditing (Box 4).
    If a breach occurs, liquidates to cash and resumes trading at the next golden crossover.

    Returns:
        tuple: (sim_df with Strat_Value_RM, risk_halted, halt_date, peak_value, final_value)
    """
    sim_df = df.copy()
    portfolio_values_rm = []
    cash_rm = initial_capital
    shares_rm = 0.0
    position_rm = 0  # 0: Cash, 1: Long
    peak_val_rm = initial_capital
    risk_halted = False
    halt_date = None

    for date, row in sim_df.iterrows():
        close = row["Close"]
        sig = row["Signal"]

        # 1. Execute signals
        if sig == "GOLDEN_CROSS" and position_rm == 0:
            shares_rm = cash_rm / close
            cash_rm = 0.0
            position_rm = 1
            # Reset peak value to start tracking drawdown for the new trade
            peak_val_rm = shares_rm * close
        elif sig == "DEATH_CROSS" and position_rm == 1:
            cash_rm = shares_rm * close
            shares_rm = 0.0
            position_rm = 0

        # 2. Calculate portfolio value
        current_val = cash_rm + (shares_rm * close)
        if current_val > peak_val_rm:
            peak_val_rm = current_val

        # 3. Check drawdown (only check when in a long position)
        if position_rm == 1 and drawdown_breached(
            current_val, peak_val_rm, threshold=max_drawdown_pct
        ):
            cash_rm = current_val
            shares_rm = 0.0
            position_rm = 0
            risk_halted = True
            if halt_date is None:
                # Record the first breach date for reporting
                halt_date = date
            # Reset peak to start tracking drawdown from our new cash balance
            peak_val_rm = cash_rm

        portfolio_values_rm.append(current_val)

    sim_df["Strat_Value_RM"] = portfolio_values_rm
    return sim_df, risk_halted, halt_date, peak_val_rm, cash_rm
