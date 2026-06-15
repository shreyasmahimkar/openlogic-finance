# OpenLogic Finance | Institutional Multi-Agent Streamlit Dashboard

Welcome to the institutional dashboard for OpenLogic Finance. This web-based interface implements a visual, side-by-side comparison of **Model A (Logistic Regression Strategy)** vs. **Model B (SMA Crossover Strategy)** using the standard **OpenLogic Finance 6-Box Architecture**.

---

## ⚡ Global Layout & Core Execution Modes

The dashboard supports two core modes of execution:
1. **Autonomous Agent Mode (`⚡ Run Autonomous Agent Pipeline`)**: Triggers an automated multi-agent synchronization sequence executing sequentially from Box 1 through Box 6. It displays a real-time hacker terminal log showing active dialogue from specialized agents (Market Data, Feature Engineer, Backtester, Risk Auditor, Execution Broker) and a central progress tracking indicator.
2. **Manual Intervention Mode (`🛠️ Enter Manual Box-by-Box Mode`)**: Unlocks a tab-based step-by-step layout. It allows quantitative developers and risk auditors to manually configure parameters (Fast/Slow moving average periods, RSI lookbacks, Decision thresholds, and Drawdown veto limits) and inspect each modular box individually, recalculating and updating interactive Plotly charts instantly.

---

## 📦 6-Box Architectural Layout

### 📦 Box 1: Data Prep
- **Ingestion & Data Health Audit**: Tracks historical boundary validation, data cleanliness scores (100%), and observations ingested (2,516 rows).
- **Timezone Safety**: Localizes Est/UTC daily boundaries to avoid slicing alignment errors.
- **Visual Pricing Trajectory**: An interactive Plotly chart superimposing Close Prices with technical indicator overlays (Fast SMA, Slow SMA) and an independent Relative Strength Index (RSI) subplot.
- **Explanation Narratives**: Integrates the `ExplanationEngine` to generate multi-fidelity descriptions, from *Beginner Friendly (Teddy Bear)* to *Academic Quantitative (Jim Simons Level)*.

### 🔬 Box 2: Model Library & Signal Logic
- **Strict Comparative View of Mathematical Frameworks**:
  - **Model A (Logistic Regression)**: Outlines standard features space formulas ($\text{sma\_ratio}$, $\text{rsi\_norm}$, $\text{momentum}$). Renders a side-by-side comparative bar chart projecting scaled feature weights to raw feature space weights ($\text{raw\_weight\_i} = \frac{w_i}{\sigma_i}$).
  - **Model B (SMA Crossover)**: Details trend-following crossover mechanics.
- **Real-Time Signal Ledger**: A dynamic DataFrame preview highlighting standard `GOLDEN_CROSS` / `DEATH_CROSS` triggers in green/red highlights.

### 🧪 Box 3: Strategy Testing
- **High-Fidelity Quantitative Sandbox**: Simulates historical trading outcomes using a zero-dependency, event-driven portfolio simulator.
- **Key Statistics Matrix**: Displays institutional metrics side-by-side: Total Return, CAGR, Sharpe Ratio, Sortino Ratio, Information Ratio (Info Ratio), Alpha, and Beta.
- **Interactive Equity Curves**: Compares the equity growth of Model A, Model B, and the Benchmark Asset (SPY Buy & Hold) starting with a $100,000 baseline.

### 🛡️ Box 4: Risk Management & Audit
- **Active Risk Auditor Drawdown Stop**: Configures standard (15% Stop) or strict (8% Stop) drawdown limits, or customize dynamically with a risk slider.
- **Risk Veto Interventions**: Simulates active veto halts during historical crashes (COVID crash of Feb/Mar 2020), liquidating positions to cash and logs active incidents to a dedicated terminal.

### ⚡ Box 5: Live & Paper Execution Architecture
- **API Connectivity Profiles**: Contrasts spot crypto (Binance REST API) vs. traditional retail brokers (Interactive Brokers Gateway REST API), highlighting order routing logic, transaction costs, and slippage.
- **Simulated Paper Trading Terminal**: An interactive terminal executing paper order tickets on-demand with randomized slippage modeling.

### 📈 Box 6: System Orchestration & Health
- **Multi-Agent Telemetry**: Traces agent synchronization actions in a retro-terminal environment.
- **Model Confidence Distribution**: Plots frequency distribution histograms for Model A probability predictions to evaluate drift metrics.

---

## 🛠️ Verification & Startup Guide

To launch the dashboard locally inside the unified environment, follow these steps:

### 1. Activate the Virtual Environment
```bash
# Navigate to the workspace root
cd /Users/shreyas/gitrepos/OpenSource/openlogic-finance

# Activate the local virtual environment
source .openlogic-env/bin/activate
```

### 2. Install Dashboard Dependencies
Make sure you have all required packages installed. Since the repository recommends `uv`, you can use it, or fall back to standard `pip`:
```bash
# Using uv (Recommended)
uv pip install -r interface/streamlit/requirements.txt

# Or using standard pip
pip install -r interface/streamlit/requirements.txt
```

### 3. Verify Dependency Alignment
Ensure that your packages are compiled and correct:
```bash
python -m py_compile interface/streamlit/app.py
```

### 4. Run the Dashboard
You can run the Streamlit application either from the repository root or from within the `interface/streamlit` directory.

#### Option A: Running from the Repository Root (Recommended)
```bash
streamlit run interface/streamlit/app.py
```

#### Option B: Running from the Streamlit Directory
```bash
cd interface/streamlit
streamlit run app.py
```

Streamlit will launch a local development server (typically at `http://localhost:8501`) and automatically open the institutional dashboard in your web browser.
