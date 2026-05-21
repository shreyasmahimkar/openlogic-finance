# OpenLogic Finance - Developer Architecture Guide (6-Box Model)

Welcome to the **OpenLogic Finance** quantitative and agentic AI trading framework. This document details our transition to the **6-Box Model**, establishing a strict boundary between our horizontal infrastructure foundations and vertical agent orchestration layers.

```
                  ┌──────────────────────────────────────────┐
                  │          AGENTIC WORKFLOWS (Vertical)    │
                  │   Primitives  │  Orchestrators  │  Tools │
                  └────────────────────┬─────────────────────┘
                                       │
  ┌────────────────────────────────────▼────────────────────────────────────┐
  │                                6-BOX MODEL                              │
  ├───────────────────┬───────────────────┬───────────────────┬─────────────┤
  │    BOX 1          │    BOX 2          │    BOX 3          │    BOX 4    │
  │    Data Prep      │    Model Library  │  Strategy Testing │  Risk Mgmt  │
  ├───────────────────┼───────────────────┼───────────────────┼─────────────┤
  │    BOX 5          │    BOX 6          │                   │             │
  │  Live Execution   │    Interface      │                   │             │
  └───────────────────┴───────────────────┴───────────────────┴─────────────┘
                                       ▲
                  ┌────────────────────┴─────────────────────┐
                  │       HORIZONTAL FOUNDATION (Infrastructure)│
                  │       Config    │     Utils    │    Core    │
                  └──────────────────────────────────────────┘
```

---

## 1. Directory Structure

Below is the clean structural map of the repository after the migration:

```
openlogic-finance/
├── horizontal_foundation/         # The horizontal baseline all layers sit on
│   ├── config/                    # Global environment, credentials, system parameters
│   ├── utils/                     # Shared logging, math, data structures, helper functions
│   └── core/                      # Base primitives used universally
├── agentic_workflows/             # The vertical orchestration box cutting across layers
│   ├── primitives/                # Base ADK agent templates, agent memories, thought loops
│   ├── orchestrators/             # MoE-F coordinators and cross-box agent dispatchers
│   └── tools/                     # Global tool registries accessible by agents
├── data_prep/                     # BOX 1 (Transformed from data_ingestion/)
│   ├── connectors/                # Financial news, GDELT global events, market data engines
│   ├── pipelines/                 # Cleaning, parsing, storage, transformation scripts
│   └── features/                  # Alternative data parsers, embedding generators, feature stores
├── model_library/                 # BOX 2
│   ├── agentic_ai/                # Stochastic filtering, MoE expert configs, ADK agent blueprints
│   ├── ml_zoo/                    # Traditional ML models (XGBoost, HMM, filters)
│   └── technical/                 # Quant indicators and atomic signal math (sma_crossover, etc.)
├── strategy_testing/              # BOX 3
│   ├── lean_engine/               # QuantConnect LEAN workspaces, local engine syncs, Python bridges
│   └── backtesting/               # Lightweight vector/event-driven simulators, local evaluation rigs
├── risk_management/               # BOX 4 (NEW BOX: Separate from execution)
│   ├── portfolio/                 # Position sizing, joint distribution estimation, correlation analysis
│   ├── enterprise/                # Value-at-Risk (VaR), Conditional VaR, drawdown caps, margin rules
│   └── agents/                    # Risk-auditor agents capable of rejecting trades
├── live_paper_execution/          # BOX 5 (Transformed from execution_layer/)
│   ├── simulators/                # Containerized mock execution environments (Docker Compose)
│   ├── paper_accounts/            # Interactive broker/exchange paper APIs and state managers
│   └── cloud_deploy/              # GCP accelerators, Vertex AI endpoints, production infra configs
└── interface/                     # BOX 6 (Three UX targets)
    ├── notebooks/                 # Notebook-as-a-Service environments, research templates, tutorials
    ├── cli/                       # Command-line interface agents and dev terminal interactive tools
    └── streamlit/                 # Streamlit dashboard, multi-agent monitoring UI, strategy visualizers
```

---

## 2. Description of the 6 Boxes

### Horizontal Foundation
- **Role**: Provides baseline services to all layers of the ecosystem.
- **Rules**: Strictly passive. It must not import from higher levels (`agentic_workflows`, `data_prep`, `model_library`, etc.).

### Box 1: Data Prep (`data_prep/`)
- **Role**: Acquires, sanitizes, and prepares data.
- **Ingestion**: Manages both structured market feeds (Yahoo Finance) and unstructured alternatives (GDELT, NYTimes financial news).

### Box 2: Model Library (`model_library/`)
- **Role**: Mathematical, statistical, and cognitive logic.
- **Components**: Hosts stochastic filters (Wonham-Shiryaev filters), Gibbs PAC-Bayes aggregation solvers, traditional models (filters, XGBoost), and signal indicators.

### Box 3: Strategy Testing (`strategy_testing/`)
- **Role**: Performance estimation.
- **Bridge**: Interfaces with the local QuantConnect LEAN CLI to backtest strategies against high-fidelity historical data.

### Box 4: Risk Management (`risk_management/`)
- **Role**: Operational guardrails.
- **Boundary**: Completely isolated from execution layer. Enforces portfolio constraints, Value-at-Risk parameters, and embeds active **Risk Auditor Agents** that can dynamically veto trading instructions.

### Box 5: Live & Paper Execution (`live_paper_execution/`)
- **Role**: Interface to active market execution.
- **Deploy**: Supports local docker-compose simulator swarms and GCP Vertex AI production deployments.

### Box 6: Interface (`interface/`)
- **Role**: Interaction layer.
- **Targets**: Serves quantitative researchers via Jupyter Notebook templates (`notebooks/`), developers via shell agents (`cli/`), and end-users/monitors via beautiful Streamlit dashboards (`streamlit/`).

---

## 3. Strict Dependency Boundaries

To maintain high-quality system engineering, developers must respect the following architectural rules:
1. **Vertical Isolation**: Vertical orchestrators (`agentic_workflows/`) coordinate calls to Box 1-4 but should remain lightweight and decoupled from the actual mathematical/ingestion implementation details.
2. **Execution De-coupling**: `live_paper_execution/` must never size positions or evaluate drawdowns internally. It purely executes sized, approved parameters passed downstream through the `risk_management/` barrier.
3. **Passive Foundation**: Any general-purpose utility or helper MUST be moved to `horizontal_foundation/utils/` rather than creating redundant modules in feature packages.
