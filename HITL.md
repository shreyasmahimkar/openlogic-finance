# HITL.md — Human-in-the-Loop Verification Guide

A manual checklist to verify this implementation and hunt for bugs the automated
tests can't catch. Work top to bottom; each step says **what to run**, **what you
should see**, and **what to look for** (likely failure points).

- **Branch under test:** `backlog/box4-risk-veto`
- **Scope:** Phases 0–5 (agentic-engineering SDLC) + Backlog items 1–6.
- **Baseline claims to confirm:** 108 tests pass, `ruff` clean, Streamlit boots,
  the MoE-F agent builds and runs on a Google account alone.

```bash
git fetch origin
git checkout backlog/box4-risk-veto
git log --oneline -6        # should show "Backlog #1..#6"
```

> Legend: ✅ = should pass automatically · 👀 = needs your eyes/judgment ·
> 🔑 = needs a credential (Gemini/NYT/GCP) you must supply.

---

## 0. Environment & reproducibility

```bash
make setup        # builds .venv from uv.lock, installs pre-commit hooks
make test         # full test suite
make lint         # ruff
```

- ✅ `make test` → **108 passed**.
- ✅ `make lint` → **All checks passed!**
- 👀 If `make setup` fails, note your Python (must be ≥3.11) and `uv` version.
- 👀 **Known wrinkle:** the repo also has a legacy hand-made env at
  `.openlogic-env/`. `make` targets use uv's `.venv`. Confirm you're not mixing
  them (`which python` after activating).

---

## 1. The MoE-F agent builds & runs

```bash
# Build only (no model calls) — should print the 4-stage pipeline:
python -c "from model_library.agentic_ai.moe_coordinator import agent as a; \
print(a.root_agent.name, [s.name for s in a.root_agent.sub_agents])"
```
- ✅ Prints `MoEF_Pipeline ['NIFTY_Ingestion_Pipeline', 'ParallelFilterPhase', 'SynthesizerAgent', 'PlottingAgent']`.

```bash
# 🔑 Full run needs a key. Put GEMINI_API_KEY in .env, then:
make run          # adk run model_library/agentic_ai/moe_coordinator
# or: make web    # then open http://localhost:8000 and pick "moe_coordinator"
```
- 👀 Ask it: *"Run the MoE-F forecast for SPY."* Watch the trajectory:
  `resolve_ingestion_csv → enrich_ohlcv_data → (expert swarm) → robust_gibbs_aggregation → render_moe_trajectories`.
- 👀 **Look for:** does the swarm actually complete? (Experts default to Gemini —
  if you see model-not-found errors, someone set `OPENLOGIC_HETEROGENEOUS_EXPERTS=1`
  without LiteLLM/keys.) Does a chart `moe_regimes.png` appear next to the package
  after ≥7 turns?
- 👀 **Bug-hunt:** confirm `resolve_ingestion_csv` returns a path **inside** the
  repo (`.../openlogic-finance/assets/SPY_10y.csv`), not `.../OpenSource/assets/`.

---

## 2. Streamlit dashboard

```bash
make web-dash     # streamlit run interface/streamlit/app.py
```
- ✅ Boots without error; 6 tabs (Box 1–6) render; title "OpenLogic Finance Dashboard".
- 👀 Change sidebar params (ticker, SMA periods) → simulations recompute, no traceback.
- 👀 **Look for:** the "Autonomous Agent Pipeline" run completes all 6 boxes; the
  Box 4 panel shows a drawdown **veto** firing on historical data.

---

## 3. Risk-veto guardrail (Box 4) — the highest-value change

This is the hard rule "the auditor can veto trades." Verify the decision logic:

```bash
python - <<'PY'
from risk_management.portfolio.guardrail import make_risk_veto_callback, RiskLimits
cb = make_risk_veto_callback(RiskLimits(max_drawdown_pct=0.15))

class Tool:  name = "place_order"
class Ctx:   state = {"portfolio_peak_value": 100000, "portfolio_current_value": 80000}
print("breach →", cb(Tool(), {"side":"buy","symbol":"SPY"}, Ctx()))   # expect VETOED
class Ctx2:  state = {"portfolio_peak_value": 100000, "portfolio_current_value": 95000}
print("ok    →", cb(Tool(), {"side":"buy"}, Ctx2()))                   # expect None
class Read:  name = "read_market_indicators"
print("read  →", cb(Read(), {"csv_path":"x"}, Ctx()))                  # expect None (not a trade)
PY
```
- ✅ 20% drawdown → `{'status': 'VETOED', ...}`; 5% drawdown → `None`; read tool → `None`.
- 👀 **Bug-hunt:** is the veto **latched**? After a breach, `state['risk_halted']`
  must be `True` and a later healthy trade must still be vetoed.
- 👀 **Not yet wired:** no live order-placing agent exists, so the callback isn't
  attached to anything in production. Confirm that's acceptable for now (it's
  documented as pending Box 5).

---

## 4. Model registry & "runs on a Google account alone"

```bash
python -c "from model_library.agentic_ai.model_registry import get_model; \
print(get_model('expert_technical'), get_model('orchestration'))"        # both gemini-2.5-flash
OPENLOGIC_HETEROGENEOUS_EXPERTS=1 python -c "from model_library.agentic_ai.model_registry import get_model; \
print(get_model('expert_fundamental'))"                                  # a LiteLlm wrapper / gpt-4o
```
- ✅ Default → Gemini everywhere. Flag set → non-Gemini expert.
- 👀 **Look for:** no hard-coded model strings left in agents (`grep -rn 'gpt-4o\|llama-3\|mixtral' model_library/ | grep -v registry`).

---

## 5. Evals (ADK)

```bash
make test     # includes test_evalsets.py (schema validity, no keys)
find . -name '*.evalset.json' -not -path './.openlogic-env/*'   # 5 files
```
- ✅ All evalsets schema-valid.
- 🔑 Scored run (optional, needs `GEMINI_API_KEY`):
  ```bash
  adk eval model_library/agentic_ai/moe_coordinator \
      model_library/agentic_ai/moe_coordinator/eval/trajectory.evalset.json \
      --config_file_path model_library/agentic_ai/moe_coordinator/eval/test_config.json
  ```
- 👀 **Look for:** the expected tool trajectory in the evalset matches what the
  agent actually does (tool **names** must line up, esp. `resolve_ingestion_csv`).

---

## 6. Local observability

```bash
OPENLOGIC_TRACING=1 python -c "
from model_library.agentic_ai.moe_coordinator import agent
from horizontal_foundation.observability import get_tracer
with get_tracer().start_as_current_span('manual_check'): pass
print('tracing configured')"
```
- ✅ Prints span JSON to the console (ConsoleSpanExporter) + "tracing configured".
- 👀 **Bug-hunt:** run a real `make run` with `OPENLOGIC_TRACING=1` and confirm ADK
  emits trajectory spans (not just the manual one).

---

## 7. LEAN strategy sync (no duplication drift)

```bash
make sync-lean                          # regenerates the LEAN copies
python scripts/sync_lean_strategies.py --check   # exit 0 = in sync
git status --porcelain                  # should be clean after sync
```
- ✅ `--check` exits 0; `make test` includes `test_lean_sync.py`.
- 👀 **Bug-hunt:** edit `model_library/ml_zoo/logistic_regression.py`, run
  `make sync-lean`, confirm both LEAN projects update and the test catches drift
  if you *don't* sync. Revert your edit afterward.

---

## 8. Vertex deploy guide (review-only unless you have GCP)

- 👀 Read [docs/DEPLOY_VERTEX.md](docs/DEPLOY_VERTEX.md). Confirm the steps are
  coherent and `deploy_vertex.py` is **env-driven** (no hard-coded project/bucket):
  ```bash
  python live_paper_execution/cloud_deploy/deploy_vertex.py   # should exit asking for GOOGLE_CLOUD_PROJECT/STAGING_BUCKET
  ```
- 🔑 Only attempt a real deploy if you have a billing-enabled GCP project (it costs money).

---

## 9. Docs ↔ reality consistency (catches drift bugs)

- 👀 [docs/BACKLOG.md](docs/BACKLOG.md) — every item 1–6 marked done; maturity table matches.
- 👀 [AGENTS.md](AGENTS.md) + per-box `AGENTS.md` — claims match the code you just ran.
- 👀 `grep -rn "research_papers_to_agents" .` (excluding `.git`) — only the historical note in the plan doc should remain.

---

## Known issues to confirm (already logged, not fixed)

1. 🐛 **`SystemConfig.WORKSPACE_ROOT` is off by one** (`horizontal_foundation/config/system_config.py`):
   `parents[3]` resolves *above* the repo, creating a stray `../assets/` with a
   duplicate CSV. Verify:
   ```bash
   python -c "from horizontal_foundation.config.system_config import SystemConfig as S; print(S.DEFAULT_ASSET_DIR)"
   ```
   👀 Should point **inside** the repo but currently doesn't. Needs a focused fix
   + audit of all `SystemConfig` consumers.
2. ⚠️ **`global_events.search_recent_events`** returns a *simulated* web-search
   result (stub) — flagged as the next MCP migration (`docs/memory/0005-mcp-policy.md`).
3. ⚠️ **`financial_news` MCP** needs `NYT_API_KEY` + network (`uvx` fetches the NYT
   MCP server) to actually return articles. Cache path works offline.
4. ⚠️ **`resolve_ingestion_csv` live path** (`OPENLOGIC_LIVE_INGEST=1`) calls
   yfinance (network); default is the cached CSV.

---

## Sign-off checklist

- [ ] `make test` → 108 passed
- [ ] `make lint` → clean
- [ ] MoE-F agent builds; full run completes on `GEMINI_API_KEY` (chart produced)
- [ ] Streamlit boots; 6 tabs; risk veto visible in Box 4
- [ ] Risk-veto guardrail vetoes on breach, latches halt, ignores non-trade tools
- [ ] Model registry defaults to Gemini; heterogeneous flag switches experts
- [ ] 5 evalsets schema-valid; trajectory tool-names match the agent
- [ ] `OPENLOGIC_TRACING=1` emits spans
- [ ] `make sync-lean --check` clean; drift is caught
- [ ] `deploy_vertex.py` refuses to run without env vars
- [ ] Confirmed the `SystemConfig` bug (#1 above) and agreed on follow-up

**Found a bug?** Note the step number + command + actual output, and whether it's a
blocker or a follow-up. File against the backlog or open an issue.
