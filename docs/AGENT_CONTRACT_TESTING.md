# Agent Contract Testing (TDD for multi-agent handoffs)

Multi-agent pipelines fail quietly. An agent passes state to the next via an
`output_key` (produced) and `{placeholder}` references in its instruction
(consumed). When those drift, the pipeline doesn't crash — it just carries the
wrong (or missing) data forward. This document is the **test-first discipline**
for catching that class of bug, plus a reusable analyzer that automates the
structural part.

## Why this exists (two real bugs)

An external handoff review of the MoE-F pipeline flagged two issues that were
**real** (verified in `model_library/agentic_ai/coordinator.py`):

1. **Completeness gap** — `MarketDataExtractor`'s instruction promised "OHLCV
   data *and news*" but its tool returns only one OHLCV path; news never flowed
   downstream. *The instruction over-promised vs. the actual output contract.*
2. **Referential-integrity gap** — the SBERT **news** filter referenced
   `{enriched_market_data}` (a price+indicator CSV), not a news artifact. *The
   handoff key was semantically wrong for the consuming step.*

Both are now fixed and locked in by `strategy_testing/tests/test_agent_contracts.py`.

## The contract (write it before the agents)

For each agent, state its contract explicitly:

| Agent | Consumes (`{...}`) | Produces (`output_key`) | Tools |
|---|---|---|---|
| MarketDataExtractor | — | `structured_market_data` (OHLCV path) | `resolve_ingestion_csv` |
| QuantitativeFeatureAgent | `{structured_market_data}` | `enriched_market_data` (indicator CSV) | `enrich_ohlcv_data` |
| SBERT_SemanticFilter | `{structured_market_data}` (to locate news) | `filtered_news_context` | `apply_semantic_news_filter` |
| Experts (×3) | `{enriched_market_data}`, `{filtered_news_context}` | `pred_*` | filter tool |
| SynthesizerAgent | (reads `pred_*` via tool/state) | `synthesized_history_context` | gibbs tool |
| PlottingAgent | `{synthesized_history_context}` | `final_status` | render tool |

**Rule of thumb:** every `{placeholder}` an agent consumes MUST be an upstream
agent's `output_key`; every claim in an instruction MUST be backed by a tool or a
consumed key.

## Three layers of checks

### 1. Structural — automated, runs in CI (no keys)
`strategy_testing/validation/agent_contracts.py::analyze_pipeline` walks the agent
tree in execution order and flags:
- **dangling_reference** — `{key}` consumed but never produced upstream (hard fail),
- **orphan_output** — `output_key` never referenced by a `{placeholder}`
  (informational — may be consumed by a tool via session state, or be terminal).

```python
from model_library.agentic_ai.coordinator import build_moef_level_3_system
from strategy_testing.validation.agent_contracts import analyze_pipeline

report = analyze_pipeline(build_moef_level_3_system())
print(report.summary())          # [OK] 8 agents · 0 dangling refs · 4 orphan outputs
assert report.ok()               # CI gate: no dangling references
```

This catches *dangling references and orphans* — but **not** semantic mismatches
(both example bugs above were structurally "valid" keys). That's layer 2.

### 2. Semantic completeness — review + LM-judge
Does each instruction's *claims* match what its tools/outputs actually deliver?
("Promises news but emits only a price path.") This needs judgment:
- a **review checklist** (below), and/or
- an **LM-as-judge** over (instruction, tools, output) — the same pattern as the
  RAG faithfulness judge (`strategy_testing/validation/llm_judge.py`). Specialized
  tools (e.g. Summit) automate scoring + prompt-rewriting for this layer.

### 3. Behavioral — evals
Trajectory + grounding evals confirm the agent actually *does* what the contract
says at runtime (`*/eval/*.evalset.json`, `rag_eval.py`).

## Review checklist for any new agent / handoff

- [ ] Every `{placeholder}` in the instruction is produced by an upstream `output_key`.
- [ ] The `output_key` is consumed somewhere downstream (or documented as terminal / tool-consumed).
- [ ] The instruction does not promise outputs the tools can't produce (no over-promising).
- [ ] The consumed key is *semantically right* for the step (a news step reads news, not prices).
- [ ] `analyze_pipeline` reports no dangling references.
- [ ] A trajectory eval covers the new tool call.

## Run it

```bash
pytest strategy_testing/tests/test_agent_contracts.py -q     # structural + regression
```

CI runs this with the full suite, so a future handoff regression fails the build.
