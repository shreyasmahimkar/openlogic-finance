# live_paper_execution/paper_accounts (planned)

Paper-trading account state (Box 5): positions, cash, and order tickets for
simulated execution. This is where an order-placing agent's tools live — and the
natural attach point for the Box 4 risk-veto callback
(`risk_management/portfolio/guardrail.make_risk_veto_callback`).

**Status:** scaffolding. No live broker connectivity yet. See `docs/BACKLOG.md`.
Hard rule: no autonomous money movement (root `AGENTS.md`).
