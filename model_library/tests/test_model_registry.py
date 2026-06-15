"""Unit tests for the central model registry (model_library/agentic_ai/model_registry.py)."""

import pytest

from model_library.agentic_ai import model_registry as mr


def test_defaults_are_gemini(monkeypatch):
    # Ensure no env overrides leak in from the runner.
    monkeypatch.delenv("OPENLOGIC_HETEROGENEOUS_EXPERTS", raising=False)
    for role in mr.ROLES:
        monkeypatch.delenv(f"OPENLOGIC_MODEL_{role.upper()}", raising=False)
        assert mr.get_model(role) == "gemini-2.5-flash"


def test_unknown_role_raises():
    with pytest.raises(ValueError):
        mr.get_model("not_a_role")


def test_per_role_env_override(monkeypatch):
    # A Gemini override stays a bare string (no LiteLlm wrapping needed).
    monkeypatch.setenv("OPENLOGIC_MODEL_ORCHESTRATION", "gemini-2.5-pro")
    assert mr.get_model("orchestration") == "gemini-2.5-pro"


def test_heterogeneous_flag_changes_experts(monkeypatch):
    monkeypatch.delenv("OPENLOGIC_MODEL_EXPERT_FUNDAMENTAL", raising=False)
    monkeypatch.setenv("OPENLOGIC_HETEROGENEOUS_EXPERTS", "1")
    # Non-Gemini ids resolve to something other than the bare Gemini default.
    resolved = mr.get_model("expert_fundamental")
    assert resolved != "gemini-2.5-flash"
    # Orchestration is unaffected by the experts flag.
    assert mr.get_model("orchestration") == "gemini-2.5-flash"
