# data_prep/pipelines (planned)

End-to-end ingestion pipelines that chain the Box 1 connectors
(`market_data`, `financial_news`, `global_events`) into a single dataset build.

**Status:** scaffolding. The MoE-F coordinator currently sequences ingestion
itself (`model_library/agentic_ai/coordinator.py`); reusable standalone pipelines
land here. See `docs/BACKLOG.md`.
