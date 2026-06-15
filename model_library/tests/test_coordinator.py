"""Unit tests for the canonical MoE-F builder (model_library/agentic_ai/coordinator.py).

These build the ADK agent tree (no model calls) and exercise the plotter, so they
run without API keys.
"""

import os

from model_library.agentic_ai.coordinator import (
    build_moef_level_3_system,
    render_moe_trajectories,
)


def test_pipeline_shape():
    system = build_moef_level_3_system()
    assert system.name == "MoEF_Pipeline"
    stages = [s.name for s in system.sub_agents]
    assert stages == [
        "NIFTY_Ingestion_Pipeline",
        "ParallelFilterPhase",
        "SynthesizerAgent",
        "PlottingAgent",
    ]
    swarm = system.sub_agents[1]
    assert [e.name for e in swarm.sub_agents] == ["Llama_Expert", "GPT4o_Expert", "Mixtral_Expert"]


def test_two_builds_are_independent():
    """Regression: ADK agents may have only one parent. Building twice in one
    process must produce fully independent trees (factory, not shared singleton)."""
    a = build_moef_level_3_system()
    b = build_moef_level_3_system()
    assert a is not b
    assert a.sub_agents[1] is not b.sub_agents[1]  # the swarm is fresh each build


class _FakeState:
    def __init__(self, data):
        self._d = data

    def get(self, key, default=None):
        return self._d.get(key, default)


def test_render_accumulates_history(tmp_path):
    state = _FakeState({"final_prediction": 0.7, "current_ground_truth": 0.6})
    # Below 7 points → accumulating, no chart yet.
    msg = render_moe_trajectories(state, artifact_dir=str(tmp_path))
    assert "Accumulating" in msg
    assert os.path.exists(tmp_path / "moe_history.csv")

    # Reach 7 points → chart is rendered.
    for _ in range(6):
        msg = render_moe_trajectories(state, artifact_dir=str(tmp_path))
    assert "Chart rendered" in msg
    assert os.path.exists(tmp_path / "moe_regimes.png")
