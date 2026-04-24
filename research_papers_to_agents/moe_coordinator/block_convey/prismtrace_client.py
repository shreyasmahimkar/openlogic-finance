import os
import requests
from concurrent.futures import ThreadPoolExecutor
import atexit
from dotenv import load_dotenv

load_dotenv()

# Global thread pool for non-blocking telemetry
_executor = ThreadPoolExecutor(max_workers=20)

def _shutdown_executor():
    # wait=False prevents the script from hanging at the end if the mock server is unreachable
    _executor.shutdown(wait=False, cancel_futures=True)

atexit.register(_shutdown_executor)

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
            timeout=1
        )
    except Exception as e:
        print(f"Telemetry failed: {e}")

def send_trace_async(user_input, output, model, latency_ms, step, step_order, session_id):
    """Fire-and-forget asynchronous wrapper using ThreadPoolExecutor."""
    try:
        _executor.submit(
            send_trace,
            user_input, output, model, latency_ms, step, step_order, session_id
        )
    except RuntimeError:
        pass  # Executor might be closed during shutdown
