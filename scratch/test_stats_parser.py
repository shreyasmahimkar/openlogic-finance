import pandas as pd

raw_table = """
┌────────────────────────────┬──────────────────┬─────────────────────────────┬────────────────┐
│ Statistic                  │ Value            │ Statistic                   │ Value          │
├────────────────────────────┼──────────────────┼─────────────────────────────┼────────────────┤
│ Equity                     │ $118,633.78      │ Fees                        │ -$12.26        │
│ Holdings                   │ $0.00            │ Net Profit                  │ $18,633.78     │
│ Probabilistic Sharpe Ratio │ 0.217%           │ Return                      │ 18.63 %        │
│ Unrealized                 │ $0.00            │ Volume                      │ $866,316.53    │
├────────────────────────────┼──────────────────┼─────────────────────────────┼────────────────┤
│ Total Orders               │ 8                │ Average Win                 │ 4.36%          │
│ Average Loss               │ 0%               │ Compounding Annual Return   │ 1.729%         │
│ Drawdown                   │ 15.100%          │ Expectancy                  │ 0              │
│ Start Equity               │ 100000           │ End Equity                  │ 118633.78      │
│ Net Profit                 │ 18.634%          │ Sharpe Ratio                │ -0.263         │
│ Sortino Ratio              │ -0.139           │ Probabilistic Sharpe Ratio  │ 0.217%         │
│ Loss Rate                  │ 0%               │ Win Rate                    │ 100%           │
│ Profit-Loss Ratio          │ 0                │ Alpha                       │ -0.027         │
│ Beta                       │ 0.142            │ Annual Standard Deviation   │ 0.057          │
│ Annual Variance            │ 0.003            │ Information Ratio           │ -0.739         │
│ Tracking Error             │ 0.138            │ Treynor Ratio               │ -0.105         │
│ Total Fees                 │ $12.26           │ Estimated Strategy Capacity │ $2400000000.00 │
│ Lowest Capacity Asset      │ SPY R735QTJ8XC9X │ Portfolio Turnover          │ 0.22%          │
│ Drawdown Recovery          │ 1239             │                             │                │
└────────────────────────────┴──────────────────┴─────────────────────────────┴────────────────┘
"""

def parse_lean_summary(summary_str: str) -> pd.DataFrame:
    if not summary_str:
        return pd.DataFrame()
        
    records = []
    for line in summary_str.strip().split("\n"):
        if "│" in line and "Statistic" not in line:
            parts = [p.strip() for p in line.split("│")]
            if len(parts) >= 5:
                stat1 = parts[1]
                val1 = parts[2]
                stat2 = parts[3]
                val2 = parts[4]
                
                if stat1 and val1:
                    records.append({"Metric": stat1, "Value": val1})
                if stat2 and val2:
                    records.append({"Metric": stat2, "Value": val2})
                    
    return pd.DataFrame(records)

df_stats = parse_lean_summary(raw_table)
print(df_stats)
