"""Unit tests for local OTel tracing setup (no network)."""

import importlib

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def _fresh_module():
    # Reset the module-level _CONFIGURED guard between tests for isolation.
    import horizontal_foundation.observability as obs

    importlib.reload(obs)
    return obs


def test_setup_emits_spans_to_exporter():
    obs = _fresh_module()
    exporter = InMemorySpanExporter()
    obs.setup_tracing(exporter=exporter)

    with obs.get_tracer("test").start_as_current_span("ingest"):
        pass

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["ingest"]


def test_setup_is_idempotent():
    obs = _fresh_module()
    p1 = obs.setup_tracing()
    p2 = obs.setup_tracing()
    assert p1 is p2  # second call returns the same provider, no duplicate setup


def test_setup_from_env_off_by_default(monkeypatch):
    obs = _fresh_module()
    monkeypatch.delenv("OPENLOGIC_TRACING", raising=False)
    assert obs.setup_from_env() is None


def test_setup_from_env_on(monkeypatch):
    obs = _fresh_module()
    monkeypatch.setenv("OPENLOGIC_TRACING", "1")
    assert obs.setup_from_env() is not None
