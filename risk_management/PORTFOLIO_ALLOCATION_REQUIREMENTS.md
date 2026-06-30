# Requirements: Model Portfolio Capital Allocation Engine (Box 4)

**Status:** Phases 1–4 implemented (math core + allocator + tests + dashboard panel); see
"Implementation status" below. Phases 1.5/5 (VaR ceiling, agentic narrative, regime tilt) open.
**Owner:** TBD
**Box(es):** `risk_management/` (Box 4) — primary; consumes `model_library` (Box 2) /
`strategy_testing` (Box 3) metrics; surfaced in `interface/streamlit/` (Box 6).
**Spec home:** mirror into `docs/specs/0004-portfolio-allocation-engine.md` per
`AGENTS.md` §Workflow before implementation (Box 4 agent specs are an open backlog item).

---

## 1. One-line ask

> Given the strategies a user has selected and backtested, tell them **what % of
> portfolio capital to allocate to each one — and *why*** — from a *model-portfolio*
> (whole-book) perspective, subject to hard risk limits. The "Jim Simons / Medallion"
> framing: size by **statistical edge, volatility, and correlation**, not by gut.

Today Box 4 is purely *defensive* (a drawdown veto that liquidates after the fact).
This module adds the *constructive* half of risk management — **how much to bet on
each edge in the first place** — which is the discipline that actually compounds a
quant book. It is advisory (it recommends weights + an explanation); the existing
veto remains the hard backstop.

---

## Implementation status (this branch)

| Deliverable | File | State |
|---|---|---|
| Sizing primitives (Kelly / max-Sharpe / min-var / risk-parity / equal-weight) | [`portfolio/sizing.py`](portfolio/sizing.py) | ✅ pure, unit-tested |
| Allocator + constraints + grounded "why" | [`enterprise/allocator.py`](enterprise/allocator.py) | ✅ caps, leverage, halt-aware, correlation flags |
| Tests | [`tests/test_sizing.py`](tests/test_sizing.py), [`tests/test_allocator.py`](tests/test_allocator.py) | ✅ 18 tests green |
| Dashboard panel (Box 4 tab) | `interface/streamlit/app.py` → `render_allocation_panel` | ✅ method selector, donut, efficient frontier, 5-method comparison, metrics, "why", API expander |
| POC notebook | [`interface/notebooks/portfolio_allocation_meanvar_poc.ipynb`](../interface/notebooks/portfolio_allocation_meanvar_poc.ipynb) | ✅ executed |

Public API (what the dashboard calls):

```python
from risk_management.enterprise.allocator import StrategyStat, AllocationConfig, allocate
res = allocate([StrategyStat("Model A", ret_a), StrategyStat("Model B", ret_b)],
               AllocationConfig(method="max_sharpe", max_strategy_weight=0.60))
res.weights        # {strategy: % of capital};  res.cash_weight
res.explanation    # grounded "why" (Tier-1);   res.warnings  # caps / halts / ρ flags
```

Still open: §4 VaR/CVaR ceiling (Phase 1.5), the Tier-2 agentic narrative in `agents/`
(Phase 3 stretch), and the regime tilt (Phase 5).

## 2. What exists today (code-dig findings)

### 2.1 OpenLogic Finance — Box 4 as built

| Piece | File | What it does | Gap for this feature |
|---|---|---|---|
| Drawdown veto callback | `risk_management/portfolio/guardrail.py` | `make_risk_veto_callback()` → ADK `before_tool_callback`; `evaluate_trade(state, RiskLimits)`; latches `risk_halted`. Reuses `drawdown_breached`. | Per-trade *kill switch* only. No sizing, no allocation, no cross-strategy view. |
| Backtest auditor | `risk_management/portfolio/auditor.py` | `run_audited_simulation()` — all-in/all-out (`cash`→`shares`) with drawdown liquidation. | Binary 0%/100% exposure; no notion of "allocate 35%." |
| Box 4 dashboard tab | `interface/streamlit/app.py` (`tabs[3]`, ~L2021) | Drawdown-limit slider + risk-audit log terminal. | No allocation panel, no per-strategy weights, no "why." |
| Per-strategy metrics | `interface/streamlit/app.py` `compute_stats()` (~L630) | Already computes **Total Return, CAGR, Sharpe, Sortino, Info Ratio, Alpha, Beta, Max Drawdown** + raw daily return series, vs an SPY benchmark. | These are exactly the allocator inputs — but they are computed inline for display and **thrown away**, not exposed as a typed object. |
| Grounding pattern | `risk_management/governance/grounding.py` | "cite or abstain" grounding instruction + `is_grounded()`. | Reuse this idiom for the allocation *explanation* (no fabricated rationale). |
| Empty scaffolding | `risk_management/enterprise/`, `risk_management/agents/` | READMEs only. `enterprise/` is explicitly slated for "cross-strategy VaR, exposure limits, aggregate drawdown across the book"; `agents/` for a "narrative risk reviewer that explains *why* an exposure is unsafe." | **These are the intended homes for this module.** |

**Key reuse rule (`AGENTS.md` hard rule #3 — import, don't copy):** `drawdown_breached`
lives in `model_library/technical/signals/sma_crossover_signal.py` and is shared by both
guardrail and auditor. The allocator's risk checks must import the same canonical math, not
re-derive drawdown/vol.

### 2.2 Acadia Analytics — what to borrow

- **Vocabulary already exists:** `acadia_analytics/backend-development/backtest/backtest_routes.py`
  defines `position_size_type ∈ {'fixed', 'percentage', 'kelly', 'volatility_adjusted'}` on a
  strategy — but it is a DB field, and Kelly / portfolio-level metrics are listed as *future work*
  in `backtest/IMPLEMENTATION_SUMMARY.md` ("Position sizing algorithms (Kelly criterion,
  volatility-based)", "Portfolio-level metrics"). **OpenLogic can leapfrog this by actually building it.**
- **Kelly inputs are already computed:** `acadia .../backtest_modules/metrics_calculator.py`
  produces `win_rate (p)`, `avg_win`, `avg_loss`, `win_loss_ratio (b)`, `profit_factor`,
  `sharpe_ratio`, `sortino_ratio`, `max_drawdown_pct`. Kelly only needs `p` and `b`.
- **Regime-conditioned sizing:** `acadia .../aa_research/python/HMM/aadit-main.py` scales exposure
  by HMM regime (`POSITION_SIZES = {Bull: 1.0, Neutral: 0.5, Caution: 0.3, Bear: -0.5}`).
  A v2 enhancement: tilt the allocation by the current regime from Box 2.

**Net:** OpenLogic already produces every statistic the allocator needs; the work is to (a)
expose those stats as a typed contract, (b) add the allocation math, (c) add the explanation
layer, (d) add a Box 4 panel.

---

## 3. The math — allocation methods to support

All methods take the **N selected strategies** + their backtest stats and emit a weight vector
`w` (∑w ≤ 1; remainder = cash). Support a user-selectable **method** so the dashboard can show
how the recommendation changes with philosophy.

| # | Method | Formula (per strategy *i*) | Inputs needed | Notes / "Simons" rationale |
|---|---|---|---|---|
| 1 | **Fractional Kelly** *(default)* | `f_i = (p_i·b_i − (1−p_i)) / b_i`, then apply **half-Kelly** `w_i = λ·f_i`, `λ=0.5` | win rate `p`, win/loss ratio `b` (or μ,σ form `f=μ/σ²`) | Maximizes long-run log-growth. **Half/quarter-Kelly** is mandatory — full Kelly is too volatile (Simons/Thorp practice). Clip `f_i ≤ 0` → 0 (no edge ⇒ no bet). |
| 2 | **Mean-Variance / Max-Sharpe** (Markowitz) | `w ∝ Σ⁻¹ μ`, normalized; or solve max Sharpe s.t. constraints | mean daily returns `μ`, covariance matrix `Σ` (from the raw return series) | Rewards uncorrelated edges; penalizes redundant strategies. Needs the **return-series correlation matrix**, not just per-strategy stats. |
| 3 | **Risk Parity** | `w_i ∝ 1/σ_i`, then iterate to equal risk contribution `w_i·(Σw)_i` | per-strategy vol `σ`, covariance `Σ` | Robust when expected returns are unreliable (which they usually are). Good "humble" default. |
| 4 | **Volatility targeting** | scale gross exposure so portfolio σ = target (e.g. 10% annualized); `leverage = σ_target / σ_portfolio` | portfolio σ, target σ | Sits *on top* of any of the above. Where leverage > 1 surfaces (capped, see §4). |
| 5 | **Equal-weight (1/N)** | `w_i = 1/N` | none | Baseline to display alongside — the honest benchmark every fancier method must beat. |

**Shared requirements for the math layer**
- Pure, deterministic, unit-testable functions (no I/O, no LLM) in `risk_management/portfolio/`
  (sizing primitives: Kelly, covariance, risk-parity solver) and
  `risk_management/enterprise/` (the cross-strategy allocator that composes them + applies §4).
- Inputs sourced as **config**, never magic numbers (`AGENTS.md` rule — half-Kelly λ, target vol,
  caps, lookback all live in config/env via `horizontal_foundation`).
- Annualization consistent with `compute_stats` (`√252` daily).
- Degenerate-input handling: N=1, zero-variance series, perfectly-correlated strategies,
  negative-edge strategies, insufficient history → defined, safe behavior (documented, tested).

---

## 4. Risk constraints & guardrails (the binding ones)

The allocator is **advisory but bounded**. Recommended weights MUST pass these before display,
and the engine reports which constraint bound:

1. **Per-strategy cap** — `w_i ≤ max_strategy_weight` (e.g. 40%). No single edge dominates the book.
2. **Gross-leverage cap** — `∑|w_i| ≤ max_gross_leverage` (default 1.0 = no leverage; vol-targeting
   may request >1 but is hard-capped here).
3. **Correlation penalty** — down-weight strategies whose return series are highly correlated
   (ρ above a threshold) so the book isn't N copies of one bet. (This is the Medallion
   "diversify across uncorrelated signals" principle.)
4. **Drawdown / halt interaction** — reuse `guardrail.evaluate_trade` / `RiskLimits`: if a
   strategy (or the book) is in a `risk_halted` state, its recommended weight is **0** and the
   reason is surfaced. The allocator must never recommend sizing *into* a vetoed strategy.
5. **VaR / CVaR ceiling** *(enterprise/, v1.5)* — portfolio 95% VaR/CVaR (historical, from the
   joined return series) ≤ a configured ceiling; scale gross exposure down if breached.
6. **Min-history gate** — refuse to size a strategy with fewer than K observations; abstain with a reason.

These map onto the `enterprise/` charter (cross-strategy VaR, exposure limits, aggregate
drawdown) — implement them there.

**Hard rule preserved:** this module **recommends**, it does not place orders. `AGENTS.md` hard
rule #1 (no autonomous money movement) is unchanged; the drawdown veto in `guardrail.py` remains
the non-overridable backstop. Output is a recommendation object consumed by a human in Box 6.

---

## 5. The "why" — explanation layer (the differentiator)

The user asked specifically for **"how much … *and why.*"** The explanation is a first-class
output, not an afterthought.

- **Tier 1 — deterministic attribution (always on):** for each strategy, a structured breakdown
  of *what drove its weight*: edge (Sharpe/Kelly-f), standalone vol, correlation/diversification
  contribution, drawdown posture, and **which constraint (if any) bound it** (e.g. "capped at 40%",
  "halved by half-Kelly", "down-weighted: ρ=0.82 with Model A"). This is pure math — testable,
  reproducible, no LLM.
- **Tier 2 — narrative risk reviewer (agentic, optional):** an ADK agent in
  `risk_management/agents/` that turns the Tier-1 attribution into plain English
  ("Model B gets 28% because it has the highest risk-adjusted edge (Sharpe 1.4) and is only
  0.2 correlated with Model A, but it's capped below its raw Kelly size because its 22% max
  drawdown trips the strict limit."). **Must reuse the `governance/grounding.py` cite-or-abstain
  pattern**: every claim cites a Tier-1 number; no fabricated rationale. Grounded by `is_grounded()`.

---

## 6. Interface contract (typed I/O)

```
INPUT  StrategyStat (one per selected strategy) — promote compute_stats() to a real object:
  name, raw_total_return, cagr, sharpe, sortino, info_ratio, alpha, beta, max_drawdown,
  daily_return_series (pd.Series),            # required for Σ / correlation / VaR
  win_rate (p), win_loss_ratio (b),           # for Kelly (derive if absent)
  n_observations, is_halted (bool)

CONFIG AllocationConfig:
  method ∈ {kelly, mean_variance, risk_parity, vol_target, equal_weight}
  kelly_fraction (λ, default 0.5), target_vol, max_strategy_weight,
  max_gross_leverage, max_correlation, var_confidence, min_observations

OUTPUT AllocationResult:
  weights: {strategy_name: float}             # ∑ ≤ 1, remainder = cash_weight
  cash_weight: float
  per_strategy_attribution: {name: {kelly_f, vol, corr_penalty, bound_by, ...}}
  portfolio_metrics: {expected_sharpe, expected_vol, var_95, cvar_95, gross_leverage}
  explanation: str                            # Tier-2 narrative (grounded)
  warnings: [str]                             # abstentions, capped items, halts
```

Wire the result through ADK **session state** (consistent with `guardrail.py` reading
`portfolio_*` keys) so other boxes can consume it without coupling to the UI.

---

## 7. Dashboard requirements (Box 4 tab — `interface/streamlit/app.py`, `tabs[3]`)

Add a **"📊 Capital Allocation"** section alongside the existing drawdown-veto panel
(don't replace it — defensive + constructive risk live side by side):

1. **Method selector** (radio): Kelly / Mean-Variance / Risk Parity / Vol-Target / Equal-Weight.
2. **Allocation result:** a bar or donut of recommended **% per selected strategy + cash**, with
   the equal-weight (1/N) baseline shown for honesty.
3. **"Why" panel:** the Tier-1 attribution table (edge, vol, correlation, bound-by) + the Tier-2
   grounded narrative.
4. **Risk readout:** expected portfolio Sharpe, expected vol vs target, 95% VaR/CVaR, gross
   leverage, and any **capped/halted** flags (red) tied back to the drawdown veto.
5. **Config controls:** half-Kelly λ slider, target vol, per-strategy cap, max-correlation — all
   reactive, so the user sees weights move as risk appetite changes.
6. Respect the existing manual "box unlock" flow (Box 4 unlocks after Box 3).

Operates on the **strategies the user selected/compared in Box 2 and backtested in Box 3** — i.e.
"a *selected* strategy," per the ask.

---

## 8. Data dependencies (what upstream boxes must hand over)

- **Box 2/3 must expose the per-strategy daily return series**, not just the formatted stat
  strings. Today `compute_stats()` computes `returns = series.pct_change()` and discards it.
  Promote its output to the `StrategyStat` object in §6 (single source of truth — import, don't
  recompute). Correlation, covariance, and VaR all require the *joined* return matrix.
- Win-rate / win-loss-ratio: derive from the strategy's trade ledger if available (Acadia-style),
  else approximate from the daily series; document the approximation.
- Benchmark series (SPY) is already available for beta/alpha; keep it for the report.

---

## 9. Success criteria / acceptance

- **Correctness:** on a hand-checked 2-strategy fixture, Kelly, mean-variance, risk-parity, and
  equal-weight all produce the analytically-expected weights (unit tests, no API keys).
- **Constraints bind:** a strategy over the per-strategy cap is clipped; a highly-correlated pair
  is down-weighted; a `risk_halted` strategy gets weight 0 — each covered by a test, each
  surfaced in `warnings` with a reason (mirrors how `guardrail` reports its veto reason).
- **Explanation is grounded:** every number in the Tier-2 narrative traces to a Tier-1
  attribution field; `is_grounded()` passes; the agent abstains rather than fabricates when data
  is missing (eval-tested, per the repo's Phase-2 eval discipline).
- **No weakened veto:** the drawdown veto path is unchanged and still tested (`AGENTS.md`:
  any change that could weaken a veto must be called out + tested).
- **Dashboard:** Box 4 tab shows allocation %, the "why," and risk readout for the selected
  strategies without errors; weights react to the config sliders.

## 10. Testing requirements

- Unit tests for every math primitive (Kelly, covariance, risk-parity solver, vol-target,
  constraint application) under `risk_management/tests/` — deterministic, fixture-driven.
- A direct `run_audited_simulation` breach test is already an open backlog item; add the
  allocator tests next to it.
- Eval set for the narrative agent (grounded / abstains-when-missing), schema-valid ADK evalset
  so CI guards it without keys (scored runs need a model key) — same pattern as the Box 1 agents.

---

## 11. Architecture / where code lands (Box 4)

```
risk_management/
├── portfolio/
│   ├── sizing.py          # NEW: pure Kelly / covariance / risk-parity / vol-target primitives
│   ├── guardrail.py       # unchanged — hard drawdown veto (the backstop)
│   └── auditor.py         # unchanged
├── enterprise/
│   └── allocator.py       # NEW: cross-strategy allocator — composes sizing + §4 constraints + VaR
├── agents/
│   └── allocation_reviewer/   # NEW: ADK agent — Tier-2 grounded "why" narrative
├── governance/grounding.py    # reused by the reviewer (cite-or-abstain)
└── tests/                     # NEW: math + constraint + breach tests
```

Dependency direction stays one-way (Box 4 imports Box 2 math, never the reverse). The Streamlit
panel (Box 6) imports the `AllocationResult`, it does not embed the math.

---

## 12. Phased plan (smallest valuable slice first)

- **Phase 1 — Math core (offline, no UI):** `StrategyStat` contract + `sizing.py`
  (fractional-Kelly + equal-weight) + `allocator.py` with per-strategy & leverage caps. Full unit
  tests. *Deliverable: `allocate(stats, config) → AllocationResult` callable from a notebook.*
- **Phase 2 — Cross-strategy:** covariance/correlation, mean-variance + risk-parity, correlation
  penalty, vol-targeting, VaR/CVaR ceiling, halt-aware weights.
- **Phase 3 — Explanation:** Tier-1 attribution (deterministic) + Tier-2 grounded narrative agent + evalset.
- **Phase 4 — Dashboard:** the Box 4 "📊 Capital Allocation" panel; promote `compute_stats` to emit `StrategyStat`.
- **Phase 5 (stretch) — Regime tilt:** condition allocation on the Box 2 regime model
  (Acadia HMM-style exposure scaling).

---

## 13. Open questions (need a decision before Phase 1)

1. **Default method** — confirm fractional (half) Kelly as the headline recommendation, with the
   others shown for comparison? (Recommended.)
2. **Kelly inputs** — derive `p`/`b` from a trade ledger (preferred, Acadia-style) or from the
   daily-return μ/σ form `f = μ/σ²`? Affects what Box 3 must export.
3. **Leverage** — is gross > 1.0 ever allowed (vol-targeting up), or hard-cap at fully-invested
   for the open-source default?
4. **Capacity** — do we model per-strategy capital capacity / liquidity, or treat all strategies
   as infinitely scalable for v1? (Simons cared deeply about capacity; likely out of scope for v1.)
5. **Rebalance cadence** — is this a point-in-time recommendation, or does it need a rebalancing
   schedule + turnover cost model? (v1 = point-in-time.)

## 13a. POC findings — mean-variance, Model A vs Model B (SPY 10y)

Prototype: [`interface/notebooks/portfolio_allocation_meanvar_poc.ipynb`](../interface/notebooks/portfolio_allocation_meanvar_poc.ipynb)
(reproduces the engine's exact return series, then runs MV / Kelly / risk-parity / equal-weight).

**How mean-variance fits:** it is the only method that uses the **covariance between
strategies**, so it is the diversification referee — it decides whether a second strategy
*adds* return-per-unit-risk or is a redundant copy. Max-Sharpe solves `w ∝ Σ⁻¹μ` (long-only).

**Empirical result (10y SPY):**

| Strategy | CAGR | Vol | Sharpe | MaxDD |
|---|---|---|---|---|
| Model A (LR) | 5.4% | 12.7% | 0.48 | −43.6% |
| Model B (SMA) | 7.9% | 14.0% | 0.61 | −33.7% |

- **Correlation(A, B) ≈ 0.88** — both are long-only SPY timing overlays, in cash ~42% of the
  time, often the *same* days. Allocation by method: Max-Sharpe → **0% A / 100% B**;
  Min-Variance → 91% A / 9% B; Risk-Parity & Equal-Weight → ~50/50; Kelly → wants **~4×
  gross leverage** and a short.
- **Diversification gain ≈ 0:** the blended max-Sharpe (0.61) barely beats holding Model B
  alone. MV is being *honest* — at ρ=0.88 there is nothing to diversify, so a near-duplicate
  strategy earns ~0% of the book.

**What it proves for this spec:** (1) the **breadth gap** is the #1 priority — MV only creates
value when fed *uncorrelated* edges (add a mean-reversion / different-asset / regime strategy in
Box 2); (2) the §4 constraints are **necessary, not decorative** — raw Kelly's 4× leverage + short
is un-investable without long-only / leverage-cap / per-strategy-cap / correlation-penalty;
(3) method choice = risk philosophy (same data → five defensible 0–100% answers), so the Box 4
panel must let the user pick the method **and show the "why."**

## 14. Out of scope (v1)

- Placing or rebalancing real/paper orders (stays human-gated in Box 5; this module only advises).
- Intraday / continuous re-optimization, transaction-cost-aware turnover optimization.
- Tax-aware allocation, multi-account/household aggregation.
- Per-strategy capacity & market-impact modeling (revisit post-v1).
