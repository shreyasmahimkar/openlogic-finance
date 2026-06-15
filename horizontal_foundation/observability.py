"""Local observability for agent runs (horizontal foundation).

The Vertex deploy enables tracing in the cloud (`deploy_vertex.py`); this module
brings the same OpenTelemetry tracing to **local** runs so drift, latency, and
tool calls are visible without deploying. ADK emits spans automatically once a
global tracer provider is configured — so calling `setup_tracing()` before a
`Runner`/`adk run` is enough to capture the agent's trajectory.

The `horizontal_foundation/interpretability` engine is the human-readable layer
on top of these machine traces.

Usage::

    from horizontal_foundation.observability import setup_tracing, get_tracer
    setup_tracing()                       # console exporter (or set OPENLOGIC_TRACE_EXPORTER)
    with get_tracer().start_as_current_span("my_step"):
        ...

Enable for any local entrypoint without code changes by exporting
`OPENLOGIC_TRACING=1` and calling `setup_from_env()`.
"""

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

SERVICE_NAME = "openlogic-finance"


def setup_tracing(
    service_name: str = SERVICE_NAME,
    exporter: SpanExporter | None = None,
) -> TracerProvider:
    """Configure the global OTel tracer provider (idempotent).

    Keys off the actual global provider — OTel allows the provider to be set only
    once per process, so a second call reuses it rather than overriding it.

    Args:
        service_name: resource service.name attached to every span.
        exporter: a span exporter; defaults to ConsoleSpanExporter. Pass an
            InMemorySpanExporter in tests to assert on captured spans.
    """
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        # Already configured in this process — attach the extra exporter if any.
        if exporter is not None:
            current.add_span_processor(SimpleSpanProcessor(exporter))
        return current

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(SimpleSpanProcessor(exporter or ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider


def setup_from_env() -> TracerProvider | None:
    """Enable tracing iff `OPENLOGIC_TRACING=1`. Safe to call from any entrypoint."""
    if os.environ.get("OPENLOGIC_TRACING") != "1":
        return None
    return setup_tracing()


def get_tracer(name: str = SERVICE_NAME):
    """Return an OTel tracer (configures a default provider if none is set)."""
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        setup_tracing()
    return trace.get_tracer(name)
