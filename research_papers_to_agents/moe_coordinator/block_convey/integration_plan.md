# PRISMtrace Integration Plan for MoE-F Coordinator

Based on the `PRISMtrace_Getting_Started_Guide.pdf`, here is the step-by-step plan and code to integrate Block Convey telemetry into your existing `moe_coordinator`.

## Step 1: Install Dependencies & Config
Add the SDK to your virtual environment:
```bash
uv pip install blockconvey-monitor>=0.1.3
```

Ensure your `.env` file contains:
```env
PRISMTRACE_API_KEY="pt-sk-..."
PRISMTRACE_PROJECT_ID="uuid-..."
```

## Step 2: The Telemetry Helper (`block_convey/prismtrace_client.py`)
PRISMtrace advises using a non-blocking daemon thread so we don't slow down the agent. Create this file:

```python
import os
import requests
import threading
from dotenv import load_dotenv

load_dotenv()

def send_trace(user_input, output, model, latency_ms, step, step_order, session_id):
    """Synchronous POST to PRISMtrace"""
    try:
        requests.post(
            'https://prismtrace.blockconvey.com/api/traces',
            json={
                'project_id': os.getenv('PRISMTRACE_PROJECT_ID'),
                'api_key': os.getenv('PRISMTRACE_API_KEY'),
                'input_messages': [{'role':'user', 'content': str(user_input)}],
                'output_message': str(output),
                'model': model,
                'latency_ms': latency_ms,
                'metadata': {
                    'agent_name': 'MoE-F Coordinator',
                    'agent_id': 'moe-f-001',
                    'session_id': session_id,
                    'step': step,
                    'step_order': step_order
                }
            }, 
            timeout=10
        )
    except Exception as e:
        print(f"Telemetry failed: {e}")

def send_trace_async(user_input, output, model, latency_ms, step, step_order, session_id):
    """Fire-and-forget asynchronous wrapper."""
    t = threading.Thread(
        target=send_trace,
        args=(user_input, output, model, latency_ms, step, step_order, session_id),
        daemon=True
    )
    t.start()
```

## Step 3: Modifying `final_test.py` for Trajectory State
To group multiple steps into a "Trajectory", PRISMtrace needs the `session_id` and `step_order`. Modify the `final_test.py` simulation loop:

```python
import uuid
import time
from block_convey.prismtrace_client import send_trace_async

# Inside run_test() simulation loop:
for idx, row in df.iterrows():
    # 1. Generate unique session ID for this step
    turn_id = str(uuid.uuid4())
    state.set("session_id", turn_id)
    state.set("step_order", 1)  # Initialize counter
    
    # ... your existing logic ...
    
    # Example tracking LLM Expert:
    t0 = time.time()
    pred_llama = float(np.clip(news + np.random.normal(0, 0.1), 0.0, 1.0))
    ms0 = int((time.time()-t0)*1000)
    
    send_trace_async(
        user_input=f"Analyze {row['Date']} indicators", 
        output=str(pred_llama), 
        model="llama-3-8b", 
        latency_ms=ms0,
        step="llm_call",
        step_order=state.get("step_order"),
        session_id=turn_id
    )
    state.set("step_order", state.get("step_order") + 1)
```

## Step 4: Tracking Mathematical Filters (`filters.py`)
You can trace your filter updates as well by passing the state.

```python
from block_convey.prismtrace_client import send_trace_async
import time

def stochastic_filter_update(expert_name: str, prediction: float, ground_truth: float, state):
    t0 = time.time()
    
    # ... existing Bayesian logic ...
    
    ms = int((time.time()-t0)*1000)
    
    # Log the math update to PRISMtrace
    send_trace_async(
        user_input=f"{expert_name} pred: {prediction}", 
        output="Filter updated successfully", 
        model="math_guardrail", # Distinguishes local code from LLM
        latency_ms=ms,
        step="stochastic_update",
        step_order=state.get("step_order", 0),
        session_id=state.get("session_id", "unknown")
    )
    if state.get("step_order"):
        state.set("step_order", state.get("step_order") + 1)
```

By passing the same `turn_id` throughout the loop for that specific day, PRISMtrace's Trajectory tab will visually link:
1. Data Fetch
2. Expert 1 / 2 / 3 Inferences
3. Stochastic Filter Updates
4. Aggregation

Let me know if you want me to actively modify the `.py` files in the repository to insert this!
