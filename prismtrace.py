import os
import time
import requests
import threading
from typing import Any, Optional

class PRISMtraceADKAdapter:
    """
    Google ADK Adapter for PRISMtrace telemetry.
    Wired into LlmAgent callback hooks.
    """
    def __init__(self, api_key: str, project_id: str, agent_name: str = "my-adk-agent"):
        self.api_key = api_key
        self.project_id = project_id
        self.agent_name = agent_name
        
        # Track session/trajectory counters
        self._step_order = {}
        self._session_ids = {}

    def _send_trace_sync(
        self,
        input_content: str,
        output_content: str,
        model: str,
        latency_ms: int,
        step: str,
        step_order: int,
        session_id: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        is_error: bool = False,
        error_msg: Optional[str] = None,
        agent_name: Optional[str] = None
    ):
        try:
            metadata = {
                'agent_name': agent_name or self.agent_name,
                'agent_id': 'adk-agent-001',
                'session_id': session_id,
                'step': step,
                'step_order': step_order
            }
            if trace_id:
                metadata['trace_id'] = trace_id
            if span_id:
                metadata['parent_span_id'] = span_id
            if is_error:
                metadata['is_error'] = True
                metadata['error_message'] = error_msg

            requests.post(
                'https://prismtrace.blockconvey.com/api/traces',
                json={
                    'project_id': self.project_id,
                    'api_key': self.api_key,
                    'input_messages': [{'role': 'user', 'content': str(input_content)}],
                    'output_message': str(output_content),
                    'model': model,
                    'latency_ms': latency_ms,
                    'metadata': metadata
                },
                timeout=5
            )
        except Exception as e:
            # Silent fallback to prevent slowing down execution
            pass

    def _send_trace_async(self, **kwargs):
        t = threading.Thread(
            target=self._send_trace_sync,
            kwargs=kwargs,
            daemon=True
        )
        t.start()

    def before_agent(self, callback_context: Any = None, *args, **kwargs):
        """Hook called before the agent begins execution."""
        ctx = callback_context or kwargs.get("callback_context")
        if not ctx:
            return
        session_id = getattr(ctx, "session_id", "live_adk_run")
        self._session_ids[id(ctx)] = session_id
        self._step_order[id(ctx)] = 1
        
        # Capture start time
        if hasattr(ctx, "state") and ctx.state is not None:
            ctx.state["agent_start_time"] = time.time()

    def after_agent(self, callback_context: Any = None, *args, **kwargs):
        """Hook called after the agent completes execution."""
        ctx = callback_context or kwargs.get("callback_context")
        if not ctx:
            return
        session_id = self._session_ids.get(id(ctx), getattr(ctx, "session_id", "live_adk_run"))
        
        start_time = time.time()
        if hasattr(ctx, "state") and ctx.state is not None:
            start_time = ctx.state.get("agent_start_time", time.time())
        latency_ms = int((time.time() - start_time) * 1000)
        
        step_order = self._step_order.get(id(ctx), 1)
        
        output_content = "Agent completed"
        if hasattr(ctx, "state") and ctx.state is not None:
            output_content = str(ctx.state.get("enriched_market_data", "Agent completed"))
            
        agent_name = self.agent_name
        if hasattr(ctx, "agent") and hasattr(ctx.agent, "name"):
            agent_name = ctx.agent.name
        elif hasattr(ctx, "agent_name") and ctx.agent_name:
            agent_name = ctx.agent_name

        self._send_trace_async(
            input_content="Agent execution started",
            output_content=output_content,
            model="adk-agent-pipeline",
            latency_ms=latency_ms,
            step="agent_run",
            step_order=step_order,
            session_id=session_id,
            agent_name=agent_name
        )

    def before_model(self, callback_context: Any = None, *args, **kwargs):
        """Hook called before the LLM model is invoked."""
        ctx = callback_context or kwargs.get("callback_context")
        if not ctx:
            return
        if hasattr(ctx, "state") and ctx.state is not None:
            ctx.state["model_start_time"] = time.time()

    def after_model(self, callback_context: Any = None, *args, **kwargs):
        """Hook called after the LLM model is invoked."""
        ctx = callback_context or kwargs.get("callback_context")
        if not ctx:
            return
        session_id = self._session_ids.get(id(ctx), getattr(ctx, "session_id", "live_adk_run"))
        
        start_time = time.time()
        if hasattr(ctx, "state") and ctx.state is not None:
            start_time = ctx.state.get("model_start_time", time.time())
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Get input prompt and output completion
        input_prompt = ""
        if hasattr(ctx, "messages") and ctx.messages:
            input_prompt = str(ctx.messages[-1])
        else:
            input_prompt = "LLM invocation"
            
        output_response = ""
        if hasattr(ctx, "response"):
            output_response = str(ctx.response)
            
        model_name = "gemini-2.5-flash"
        if hasattr(ctx, "model"):
            model_name = str(ctx.model)

        agent_name = self.agent_name
        if hasattr(ctx, "agent") and hasattr(ctx.agent, "name"):
            agent_name = ctx.agent.name
        elif hasattr(ctx, "agent_name") and ctx.agent_name:
            agent_name = ctx.agent_name

        step_order = self._step_order.get(id(ctx), 1)
        self._send_trace_async(
            input_content=input_prompt,
            output_content=output_response,
            model=model_name,
            latency_ms=latency_ms,
            step="llm_call",
            step_order=step_order,
            session_id=session_id,
            agent_name=agent_name
        )
        self._step_order[id(ctx)] = step_order + 1

    def before_tool(self, tool_context: Any = None, *args, **kwargs):
        """Hook called before a tool is executed."""
        ctx = tool_context or kwargs.get("tool_context")
        if not ctx:
            return
        if hasattr(ctx, "state") and ctx.state is not None:
            ctx.state["tool_start_time"] = time.time()

    def after_tool(self, tool_context: Any = None, *args, **kwargs):
        """Hook called after a tool is executed."""
        ctx = tool_context or kwargs.get("tool_context")
        if not ctx:
            return
        session_id = self._session_ids.get(id(ctx), getattr(ctx, "session_id", "live_adk_run"))
        
        start_time = time.time()
        if hasattr(ctx, "state") and ctx.state is not None:
            start_time = ctx.state.get("tool_start_time", time.time())
        latency_ms = int((time.time() - start_time) * 1000)
        
        tool_name = "unknown_tool"
        tool_args = ""
        if hasattr(ctx, "tool_call"):
            tool_name = getattr(ctx.tool_call, "name", "tool")
            tool_args = str(getattr(ctx.tool_call, "args", ""))
        elif "tool" in kwargs:
            tool_obj = kwargs["tool"]
            tool_name = getattr(tool_obj, "name", "tool")
            
        if "args" in kwargs:
            tool_args = str(kwargs["args"])
            
        tool_output = ""
        if hasattr(ctx, "tool_output"):
            tool_output = str(ctx.tool_output)
        elif "tool_response" in kwargs:
            tool_output = str(kwargs["tool_response"])

        agent_name = self.agent_name
        if hasattr(ctx, "agent") and hasattr(ctx.agent, "name"):
            agent_name = ctx.agent.name
        elif hasattr(ctx, "agent_name") and ctx.agent_name:
            agent_name = ctx.agent_name

        step_order = self._step_order.get(id(ctx), 1)
        self._send_trace_async(
            input_content=f"Call tool {tool_name} with args: {tool_args}",
            output_content=tool_output,
            model=f"tool:{tool_name}",
            latency_ms=latency_ms,
            step="tool_call",
            step_order=step_order,
            session_id=session_id,
            agent_name=agent_name
        )
        self._step_order[id(ctx)] = step_order + 1

    def record_model_error(self, exc: Exception, callback_context: Any = None, *args, **kwargs):
        """Explicitly record a model or execution error and send to telemetry."""
        ctx = callback_context or kwargs.get("callback_context")
        if not ctx:
            return
        session_id = self._session_ids.get(id(ctx), getattr(ctx, "session_id", "live_adk_run"))
        step_order = self._step_order.get(id(ctx), 1)
        
        agent_name = self.agent_name
        if hasattr(ctx, "agent") and hasattr(ctx.agent, "name"):
            agent_name = ctx.agent.name
        elif hasattr(ctx, "agent_name") and ctx.agent_name:
            agent_name = ctx.agent_name

        self._send_trace_async(
            input_content="Error triggered during model run",
            output_content=f"Exception: {str(exc)}",
            model="error_logger",
            latency_ms=0,
            step="error_handling",
            step_order=step_order,
            session_id=session_id,
            is_error=True,
            error_msg=str(exc),
            agent_name=agent_name
        )
