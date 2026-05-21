# Strategy Testing (BOX 3)

Integrates local simulators, backtesting scripts, and QuantConnect LEAN bridges to test strategy efficacy.

## Target Structure & Subdirectories

- **`lean_engine/`**: QuantConnect LEAN workspaces, local engine sync tools, and Python-to-C# bridges.
- **`backtesting/`**: Lightweight vector/event-driven simulators and local evaluation rigs.

## Purpose & Architectural Rule

This box provides the evaluation sandboxes. Before any strategy goes live, it must pass rigorous simulation evaluations using either local event-driven backtesters or standard QuantConnect LEAN backtests.
