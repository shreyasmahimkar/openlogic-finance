# The Quality Flywheel (Phase 5 — the compounding loop)

Phases 0–4 built the factory. Phase 5 is **keeping it sharp**: a repeating loop
that compounds quality instead of letting the harness drift. This is process, not
a one-time deliverable.

## The loop

```
        ┌──────────────────────────────────────────────────────┐
        ▼                                                        │
1. Benchmark ──► 2. Cluster failures ──► 3. Optimize ──► 4. Regression-gate ──► 5. Monitor
   (run evals)     (group root causes)     (prompts/        (lock the fix in       (watch prod
                                            tools/context)    CI evals + tests)      traces)
```

1. **Benchmark** — run the eval suite (`make` + `adk eval`) against a fixed dataset each iteration.
2. **Cluster failures** — group failing cases by root cause (bad tool choice, hallucinated output, wrong trajectory, missing context).
3. **Optimize** — fix the *cause*: tighten an `AGENTS.md` rule, a tool docstring, a model in the registry, or a prompt. Prefer harness changes over one-off patches.
4. **Regression-gate** — add the failing case to an evalset / test so CI catches it forever. Each cycle widens coverage.
5. **Monitor** — read production traces (OpenTelemetry from Agent Engine); new failure modes feed back into step 1.

## Operating rules

- **Harness files are infrastructure.** `AGENTS.md`, specs, evalsets, the model
  registry, and CI are reviewed in PRs and owned by named engineers — same bar as
  service code (root `AGENTS.md`, workflow).
- **Set the bar at the eval, not the demo.** A new agent ships into a shared
  workflow only with eval coverage + an explicit rubric, the way a service ships
  only with test coverage.
- **Every incident becomes a regression test.** If an agent does something it
  shouldn't, add a rule to `AGENTS.md` *and* a case to the evals.

## Known remaining work (tracked, post-Phase-5)

The per-box SDLC gaps (risk-veto callback, per-box tests/evals, MCP completion,
local observability, specs, skeleton cleanup, LEAN de-dup) are catalogued and
prioritized in **[`docs/BACKLOG.md`](BACKLOG.md)** — deferred to a later iteration.
