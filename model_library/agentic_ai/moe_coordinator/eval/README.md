# MoE-F evals

ADK evalsets for the MoE-F coordinator. **Evals verify the non-deterministic
parts** (did the agent take the right trajectory and produce an acceptable
response) — the deterministic math is covered by `pytest` (`model_library/tests/`).

## Files

| File | Checks |
|---|---|
| `ingestion.evalset.json` | Phase-1 ingestion calls `data_ingestion_stub` then `enrich_ohlcv_data`. |
| `trajectory.evalset.json` | Full master trajectory: ingest → swarm → `robust_gibbs_aggregation` → `render_moe_trajectories`. |
| `test_config.json` | Scoring thresholds: `tool_trajectory_avg_score` (tool path) and `response_match_score` (final response). |

## Run

Requires a model key (`GEMINI_API_KEY`) since evals invoke the agent:

```bash
adk eval model_library/agentic_ai/moe_coordinator \
    model_library/agentic_ai/moe_coordinator/eval/trajectory.evalset.json \
    --config_file_path model_library/agentic_ai/moe_coordinator/eval/test_config.json
```

CI validates that these files are **well-formed** (schema-valid) without keys
(`model_library/tests/test_evalsets.py`); the scored run is a manual / keyed step.

## Extending

Add a case per behavior you want to lock in. Each spec's success criteria
(`docs/specs/`) should map to an eval case here. Rubric dimensions to grow into:
task success, tool-use quality, trajectory compliance, hallucination, response quality.
