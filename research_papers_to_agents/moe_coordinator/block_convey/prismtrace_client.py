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

from opentelemetry import trace

def send_trace(user_input, output, model, latency_ms, step, step_order, session_id, trace_id=None, span_id=None):
    """Synchronous POST to PRISMtrace"""
    try:
        metadata = {
            'agent_name': 'MoE-F Coordinator',
            'agent_id': 'moe-f-001',
            'session_id': session_id,
            'step': step,
            'step_order': step_order
        }
        if trace_id:
            metadata['trace_id'] = trace_id
        if span_id:
            metadata['parent_span_id'] = span_id
            
        requests.post(
            'https://prismtrace.blockconvey.com/api/traces',
            json={
                'project_id': os.getenv('PRISMTRACE_PROJECT_ID'),
                'api_key': os.getenv('PRISMTRACE_API_KEY'),
                'input_messages': [{'role':'user', 'content': str(user_input)}],
                'output_message': str(output),
                'model': model,
                'latency_ms': latency_ms,
                'metadata': metadata
            }, 
            timeout=1
        )
    except Exception as e:
        print(f"Telemetry failed: {e}")

def send_trace_async(user_input, output, model, latency_ms, step, step_order, session_id):
    """Fire-and-forget asynchronous wrapper using ThreadPoolExecutor."""
    try:
        # Extract ADK's native OpenTelemetry context before offloading to background thread
        current_span = trace.get_current_span()
        ctx = current_span.get_span_context()
        trace_id = None
        span_id = None
        if ctx and ctx.is_valid:
            trace_id = format(ctx.trace_id, "032x")
            span_id = format(ctx.span_id, "016x")
            
        _executor.submit(
            send_trace,
            user_input, output, model, latency_ms, step, step_order, session_id, trace_id, span_id
        )
    except RuntimeError:
        pass  # Executor might be closed during shutdown
