"""CLI entrypoint for the MoE-F pipeline.

Thin wrapper over the shared canonical builder
(`model_library/agentic_ai/coordinator.py`). Kept for backward-compatible imports
(`live_paper_execution/cloud_deploy/deploy_vertex.py`,
`strategy_testing/backtesting/final_test.py`). New code should import from the
coordinator module or use the ADK app at `model_library/agentic_ai/moe_coordinator/`.
"""

import os

from model_library.agentic_ai.coordinator import build_moef_level_3_system
from model_library.agentic_ai.coordinator import render_moe_trajectories as _render

_ARTIFACT_DIR = os.path.dirname(__file__)

moef_level_3_system = build_moef_level_3_system(artifact_dir=_ARTIFACT_DIR)
root_agent = moef_level_3_system


def render_moe_trajectories(state) -> str:
    """Backward-compatible plotter (imported by final_test); writes under interface/cli/."""
    return _render(state, _ARTIFACT_DIR)
