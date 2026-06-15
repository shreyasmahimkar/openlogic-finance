# Backlog — per-box SDLC gaps (deferred)

Captured 2026-06-15 from an audit of the repo against *The New SDLC With Vibe
Coding*. Phases 0–5 built the **shared** harness (rulebook, CI, evals for the
flagship agent, Vertex deploy). This backlog is the **per-box** discipline that
is still uneven — deferred to a later iteration, not yet started.

Priority order is top-to-bottom. See `docs/AGENTIC_ENGINEERING_SDLC_PLAN.md` and
`docs/QUALITY_FLYWHEEL.md` for context.

## Maturity snapshot

| Box | AGENTS.md | Unit tests | Evals | Tools/MCP | Runtime guardrail | Spec |
|---|---|---|---|---|---|---|
| 1 Data Prep | ✅ | ❌ | ❌ | ⚠️ news=MCP; market/global=local | n/a | ❌ |
| 2 Model Library | ✅ | ✅ | ✅ | ✅ | n/a | ⚠️ template only |
| 3 Strategy Testing | ✅ | ❌ (`final_test.py` is a script) | ❌ | ✅ | n/a | ❌ |
| 4 Risk Mgmt | ✅ | ⚠️ guardrail tested | ❌ | logic exists + **ADK veto callback** ✅ | ✅ | ❌ |
| 5 Live/Paper Exec | ✅ | ❌ | ❌ | deploy ✅ | ❌ human-gate not coded | ❌ |
| 6 Interface | ✅ | ❌ | n/a | ✅ | n/a | ❌ |

## Items (prioritized)

1. ✅ **DONE — Box 4: Risk Auditor veto is real.** `risk_management/portfolio/guardrail.py`
   `make_risk_veto_callback()` is an ADK `before_tool_callback` that vetoes
   trade-shaped tool calls on drawdown breach / halt, reusing `drawdown_breached`.
   Covered by `risk_management/tests/test_guardrail.py` (9 tests). Still TODO:
   attach it to a real order-placing agent once Box 5 has one.

2. **Tests for Boxes 1, 3** (deterministic, no API keys, fast CI wins).
   - Box 1: `check_news_cache`, `save_news_to_csv`, `read_market_indicators`.
   - Box 3: convert `final_test.py` to pytest; test the LEAN summary parser + backtest math.
   - ~~Box 4: `run_audited_simulation`~~ — guardrail decision now covered (see #1);
     still worth a direct `run_audited_simulation` breach test.

3. **Evalsets for the Box 1 connector agents** (market_data / financial_news /
   global_events): right tool called, no hallucination, output contract held.

4. **Finish MCP in Box 1.** `financial_news` uses real MCP (`StdioServerParameters`);
   `market_data` and `global_events` use local FunctionTools and `mcp_client.py` is a
   stub. Standardize on MCP or delete the stub and document the decision.

5. **Local observability.** `enable_tracing` is deploy-only. Wire ADK tracing for
   local runs and connect the `horizontal_foundation/interpretability` engine to
   agent runs so drift/cost are visible without deploying.

6. **Housekeeping.**
   - Write real specs in `docs/specs/` for the Box 1/3/4/5 agents (each spec's
     success criteria seeds its eval rubric).
   - Fill or remove empty `__init__`-only dirs: `data_prep/features`,
     `data_prep/pipelines`, `risk_management/agents`, `risk_management/enterprise`,
     `live_paper_execution/paper_accounts`.
   - De-duplicate the LEAN projects' copies of `logistic_regression.py` /
     `sma_crossover_signal.py` (see `docs/memory/0003-import-dont-copy.md`).
   - Replace the ingestion stub (`coordinator.data_ingestion_stub`) with the live
     Yahoo Finance MCP.
