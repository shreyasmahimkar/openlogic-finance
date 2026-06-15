# MoE-F Coordinator (ADK app)

A **Level 3 multi-agent** implementation of the Mixture-of-Experts Filter (MoE-F)
described in `model_library/agentic_ai/docs/moef_agents_plan.md`, assembled with
Google ADK primitives (`SequentialAgent`, `ParallelAgent`, `LlmAgent`,
`FunctionTool`).

## Pipeline

```
moef_level_3_system (SequentialAgent)
├── NIFTY_Ingestion_Pipeline (SequentialAgent)
│   ├── MarketDataExtractor      → structured_market_data
│   ├── QuantitativeFeatureAgent → enriched_market_data   (technical indicators)
│   └── SBERT_SemanticFilter     → filtered_news_context
├── ParallelFilterPhase (ParallelAgent)   ← model_library.agentic_ai.experts
│   ├── Llama_Expert    → pred_llama
│   ├── GPT4o_Expert    → pred_gpt
│   └── Mixtral_Expert  → pred_mixtral     (each runs stochastic_filter_update)
├── SynthesizerAgent  → synthesized_history_context   (robust_gibbs_aggregation)
└── PlottingAgent     → final_status                  (7-day rolling chart)
```

## Run

Always launch from the **repo root** so the absolute `model_library...` imports
resolve.

```bash
# Interactive terminal (most reliable — points straight at this package)
adk run model_library/agentic_ai/moe_coordinator

# Web UI (lists agents under agentic_ai; pick "moe_coordinator")
adk web model_library/agentic_ai
```

Open the printed local URL (usually http://localhost:8000) and interact with
`moef_level_3_system`.

## Provenance & design notes

- **Reconstructed in Phase 0.** The original source was lost — only
  `__pycache__/*.pyc` survived and it was never committed to git. This package
  was rebuilt from the spec above and the committed twin `interface/cli/agent.py`.
- **Shared math is imported, not vendored.** Experts, the stochastic filter, the
  Gibbs aggregation, and the indicator enrichment all come from `model_library`.
  The lost original copied these in-tree; that duplication is intentionally gone.
- **Known follow-up (Phase 3):** `interface/cli/agent.py` still duplicates this
  assembly and should be refactored to import `root_agent` from here.
- **Not re-vendored:** the old `final_test.py` (a backtest harness — see
  `strategy_testing/backtesting/final_test.py`) and the `block_convey/`
  PRISMtrace client are intentionally not recreated here.

See `docs/AGENTIC_ENGINEERING_SDLC_PLAN.md` for the broader plan.
