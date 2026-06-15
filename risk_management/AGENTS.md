# AGENTS.md — risk_management (Box 4: Risk Management)

Root rules: [`/AGENTS.md`](../AGENTS.md).

Enforces risk limits and operates **active Risk Auditor Agents that can veto
trades**. This box is a guardrail, not advisory.

## Public surface

- `portfolio/auditor.py` — the Risk Auditor (VaR limits, drawdown checks, trade veto).
- `agents/` — risk-focused ADK agents.
- `enterprise/` — enterprise-level / aggregate risk.

## Rules

- The auditor is a **hard guardrail**: in Phase 3 it becomes an ADK
  `before_model` / `before_tool` callback that can block any trade-shaped action.
  Its veto is not optional and not overridable by another agent.
- Risk thresholds are **config**, sourced from `horizontal_foundation`/env — never magic numbers buried in logic.
- Any change that could weaken a veto path must be called out explicitly in review and covered by a test.
