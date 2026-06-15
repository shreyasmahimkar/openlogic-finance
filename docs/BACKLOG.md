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
| 1 Data Prep | ✅ | ⚠️ tool tests | ✅ 3 agents | ✅ policy set (see 0005) | n/a | ✅ 0001–0003 |
| 2 Model Library | ✅ | ✅ | ✅ | ✅ | n/a | ⚠️ template only |
| 3 Strategy Testing | ✅ | ✅ parsers + MoE-F loop | ❌ | ✅ | n/a | ❌ |
| 4 Risk Mgmt | ✅ | ⚠️ guardrail tested | ❌ | logic exists + **ADK veto callback** ✅ | ✅ | ❌ |
| 5 Live/Paper Exec | ✅ | ❌ | ❌ | deploy ✅ | ❌ human-gate not coded | ❌ |
| 6 Interface | ✅ | ❌ | n/a | ✅ | n/a | ❌ |

## Items (prioritized)

1. ✅ **DONE — Box 4: Risk Auditor veto is real.** `risk_management/portfolio/guardrail.py`
   `make_risk_veto_callback()` is an ADK `before_tool_callback` that vetoes
   trade-shaped tool calls on drawdown breach / halt, reusing `drawdown_breached`.
   Covered by `risk_management/tests/test_guardrail.py` (9 tests). Still TODO:
   attach it to a real order-placing agent once Box 5 has one.

2. ✅ **DONE — Tests for Boxes 1 & 3** (deterministic, no API keys).
   - Box 1: `check_news_cache` / `save_news_to_csv` (`data_prep/tests/`),
     `read_market_indicators` + `enrich_ohlcv_data` (`model_library/tests/test_indicators.py`).
   - Box 3: LEAN stdout parsers + `extract_summary_table` and the MoE-F filter/Gibbs
     loop (`strategy_testing/tests/`); `final_test.py` refactored into a testable
     `run_simulation()`.
   - Remaining: a direct `run_audited_simulation` breach test (Box 4 decision is
     already covered by #1), and evals for the agents themselves (see #3).

3. ✅ **DONE — Evalsets for the Box 1 connector agents.** Schema-valid ADK evalsets
   + `test_config.json` for `market_data`, `financial_news`, `global_events`
   (expected tool trajectories). `test_evalsets.py` now scans **all** `eval/` dirs
   repo-wide, so they're guarded in CI without keys. See
   `data_prep/connectors/README_EVALS.md`. Scored runs need `GEMINI_API_KEY`.

4. ✅ **DONE — MCP policy set for Box 1** (`docs/memory/0005-mcp-policy.md`):
   MCP for external third-party services; FunctionTools for local deterministic
   computation. Deleted the dead `mcp_client.py` stub; fixed the hard-coded `uvx`
   path in `financial_news` (now `shutil.which`). Flagged
   `global_events.search_recent_events` (a simulated web search) as the next real
   MCP migration — needs a web-search MCP server + key, so left as documented TODO.

5. ✅ **DONE — Local observability.** `horizontal_foundation/observability.py`
   (`setup_tracing` / `setup_from_env`) wires OpenTelemetry for local runs;
   `OPENLOGIC_TRACING=1 adk run model_library/agentic_ai/moe_coordinator` now emits
   ADK trajectory spans. Interpretability engine documented as the human-readable
   layer on top. Tests in `horizontal_foundation/tests/test_observability.py`.

6. ✅ **DONE — Housekeeping.**
   - ✅ Specs for the Box 1 agents (`docs/specs/0001..0003`); Box 3/4/5 agent
     specs remain TODO.
   - ✅ Documented the skeleton dirs (`data_prep/features`, `data_prep/pipelines`,
     `risk_management/agents`, `risk_management/enterprise`,
     `live_paper_execution/paper_accounts`) with intent READMEs instead of leaving
     dead scaffolding.
   - ✅ LEAN copies are now generated from `model_library` via
     `scripts/sync_lean_strategies.py` (`make sync-lean`), guarded by
     `test_lean_sync.py`.
   - ✅ Replaced `data_ingestion_stub` with config-driven `resolve_ingestion_csv`
     (env override / optional live yfinance refresh / cached default).

## Newly found

- ✅ **FIXED — `SystemConfig.WORKSPACE_ROOT` off-by-one.** Was `parents[3]`
  (resolved *above* the repo → a stray `../assets/`); now `parents[2]`. Audited
  all consumers (`coordinator.py` already self-resolved; `market_data/tools.py`
  now uses the correct in-repo `assets/`). Regression test in `test_foundation.py`.
