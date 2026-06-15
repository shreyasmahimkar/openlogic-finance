# data_prep/features (planned)

Feature-engineering transforms for Box 1 — turning raw OHLCV/news into model
inputs. Technical-indicator math currently lives in
`model_library/technical/indicators.py` (imported, not duplicated); reusable
feature pipelines that compose those indicators belong here as they grow.

**Status:** scaffolding. No modules yet — see `docs/BACKLOG.md`.
