# AGENTS.md — live_paper_execution (Box 5: Live & Paper Execution)

Root rules: [`/AGENTS.md`](../AGENTS.md).

Trading connectivity, Docker simulator environments, and GCP/Vertex deploy rigs.
**The highest-risk box** — it touches real money paths.

## Public surface

- `paper_accounts/` — paper-trading account management.
- `simulators/docker-compose.yml` — isolated Docker execution environments.
- `cloud_deploy/deploy_vertex.py` — package the ADK app to **Vertex AI Agent Engine**.
- `cloud_deploy/Dockerfile` — container build.
- `serving/predict.py` — load a promoted model + score (Vertex/SageMaker endpoint in prod) — the *deploy* half of the MDLC.
- `monitoring/drift.py` — the *monitor* half: data/prediction drift (PSI), performance decay, and a retrain trigger. See `docs/EQUITY_RESEARCH.md`.

## Rules

- **No autonomous money movement.** An agent never places a live order or transfer
  on its own — execution is human-gated and subject to the Risk Auditor veto
  (Box 4). This is the repo's first hard rule; it is absolute here.
- Start in **paper** mode; live trading requires explicit human sign-off per session.
- `deploy_vertex.py` must read `PROJECT_ID` / `STAGING_BUCKET` from config/env (no
  hard-coded constants — Phase 4 hardening) and ship only **eval-gated** builds.
- Wire OpenTelemetry on every deployed run so agent decisions are auditable.
