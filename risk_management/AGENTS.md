# AGENTS.md — risk_management (Box 4: Risk Management)

Root rules: [`/AGENTS.md`](../AGENTS.md).

Enforces risk limits and operates **active Risk Auditor Agents that can veto
trades**. This box is a guardrail, not advisory.

## Public surface

- `portfolio/guardrail.py` — **`make_risk_veto_callback()`**: the Risk Auditor as
  an ADK `before_tool_callback`. Detects trade-shaped tool calls and vetoes them
  (short-circuits the tool) when the drawdown limit is breached or risk is halted.
  Attach it to any agent that can place orders.
- `portfolio/auditor.py` — `run_audited_simulation`: the backtest-time drawdown
  auditor (shares the `drawdown_breached` math with the guardrail).
- `governance/grounding.py` — responsible-AI grounding controls for the Equity Research Assistant: the grounding instruction + `is_grounded()` citation check (cite or abstain — no fabricated guidance).
- `governance/approval.py` — `make_research_approval_callback()`: the **human-in-the-loop** gate (an ADK `before_tool_callback`) that blocks publishing a recommendation until a human approves. Same idiom as `portfolio/guardrail.py`. See `docs/EQUITY_RESEARCH.md`.
- `agents/` — risk-focused ADK agents.
- `enterprise/` — enterprise-level / aggregate risk.

## Rules

- The auditor is a **hard guardrail**, now live as an ADK `before_tool_callback`
  (`guardrail.make_risk_veto_callback`). Wire it onto every order-placing agent;
  returning a dict from the callback blocks the tool. Its veto is not optional and
  not overridable by another agent.
- Risk thresholds are **config**, sourced from `horizontal_foundation`/env — never magic numbers buried in logic.
- Any change that could weaken a veto path must be called out explicitly in review and covered by a test.
