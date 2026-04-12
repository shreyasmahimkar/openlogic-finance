# Integrating Data Ingestion with the MoE-F Coordinator

This implementation plan outlines the steps required to transition the `moe_coordinator` from a prototyped mock state into a fully functional multi-agent orchestrator. The goal is to naturally consume the daily close OHLCV historical dataset provided by the `data_ingestion_agent`, compute ground-truths, and dynamically pool the 3 expert models for stock forecasting.

## User Review Required

> [!IMPORTANT]
> The biggest challenge here is statefulness. The MoE-F paper describes a continuous time-series filter. This means the system needs to remember what each of the 3 experts predicted *yesterday* to properly penalize/reward them *today*.

## Proposed Changes

### `moe_coordinator` Logic Wiring

#### [MODIFY] [agent.py](file:///Users/shreyas/gitrepos/OpenSource/openlogic-finance/moe_coordinator/agent.py)
* **Real History Processing**: Alter `invoke_moe_filter` to dynamically load the CSV stored at `market_context_url` (from `/shared-assets` or locally) via `pandas`. Compute `most_recent_close_change` dynamically using the final two records in the dataset instead of expecting it explicitly from the user.
* **Persistent Cache Management**: Replace the `dummy_last_preds = [0.8, 0.4, 0.1]` line with a read to a local `last_preds.json` or SQLite dataset that stores previous expert responses.
* **Expert Sub-Agent Polling**: Replace `new_expert_forecasts = [0.55, 0.60, 0.90]` with parallel ADK `Agent.run()` invocations to `expert_technician`, `expert_fundamental`, and `expert_contrarian`. The recent slice of the OHLCV CSV must be injected into the prompt of each expert.
* **Cache Rewriting**: Upon aggregating the results, overwrite the persistent DB with the brand new forecasts so they can be evaluated during the next run.

#### [MODIFY] [experts.py](file:///Users/shreyas/gitrepos/OpenSource/openlogic-finance/moe_coordinator/experts.py)
* Refine the system prompts so each expert understands *how* to read the serialized Pandas CSV snippet passed to them. 

## Open Questions

> [!WARNING]
> Please review and provide feedback on the following implementation constraints:

1. **Wait, how do we query the internal sub-agents?** Do we want to invoke the `expert_*` instances defined in `experts.py` synchronously using `Agent.run(user_prompt)`, or async, triggering them concurrently?
2. **State DB format**: Is writing a simple `last_predictions.json` file inside the `moe_coordinator` directory acceptable for persisting historical expert guesses between runs, or is there an overarching preferred Database?
3. **CSV Parsing Window**: Should the coordinator pass the **entire** 10-year OHLCV file to the experts, or should we slice it down to the last 30-60 days to save on prompt tokens and improve float prediction accuracy? 

## Verification Plan

### Manual Verification
- We will boot the `moe_coordinator` locally and invoke it via Python tests to trigger a forecast.
- We will verify that a `last_predictions.json` is generated successfully.
- We will view logs to explicitly monitor that the three experts each return an actual float prediction rather than hallucinating texts, validating `[float(pred) for pred in expert_calls]` parses perfectly.
