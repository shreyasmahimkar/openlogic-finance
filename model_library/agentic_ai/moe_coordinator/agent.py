"""MoE-F Coordinator — ADK app entrypoint (`adk run model_library/agentic_ai/moe_coordinator`).

Thin wrapper. The pipeline is assembled by the shared canonical builder in
`model_library/agentic_ai/coordinator.py`; this module just binds artifacts to
this package directory and exposes `root_agent` for ADK discovery.

See README.md for provenance (the source was reconstructed in Phase 0 after the
original was lost to orphaned bytecode) and the full pipeline description.
"""

import os

from horizontal_foundation.observability import setup_from_env
from model_library.agentic_ai.coordinator import build_moef_level_3_system

# Enable local OTel tracing when OPENLOGIC_TRACING=1 (no-op otherwise). ADK then
# emits spans for the agent's trajectory; see horizontal_foundation/observability.py.
setup_from_env()

# Artifacts (moe_history.csv / moe_regimes.png) land next to this package.
moef_level_3_system = build_moef_level_3_system(artifact_dir=os.path.dirname(__file__))

# ADK entrypoint: `adk run/web` discovers `root_agent`.
root_agent = moef_level_3_system
