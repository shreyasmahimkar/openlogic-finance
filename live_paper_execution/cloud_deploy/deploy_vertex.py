"""Deploy the MoE-F coordinator to Vertex AI Agent Engine.

Configuration is read from the environment (no hard-coded project/bucket):

    GOOGLE_CLOUD_PROJECT     GCP project id            (required)
    STAGING_BUCKET           gs://... staging bucket    (required)
    GOOGLE_CLOUD_LOCATION    region (default us-central1)

Usage:
    export GOOGLE_CLOUD_PROJECT=my-project
    export STAGING_BUCKET=gs://my-bucket
    uv run python live_paper_execution/cloud_deploy/deploy_vertex.py

See docs/DEPLOY_VERTEX.md for the full walkthrough.
"""

import os
import sys

import vertexai
from google.adk.agent_engines import AdkApp

# Import the canonical agent (built once by the shared coordinator builder).
from model_library.agentic_ai.moe_coordinator.agent import moef_level_3_system


def deploy_to_gcp():
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    staging_bucket = os.environ.get("STAGING_BUCKET")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    missing = [
        name
        for name, val in (("GOOGLE_CLOUD_PROJECT", project), ("STAGING_BUCKET", staging_bucket))
        if not val
    ]
    if missing:
        sys.exit(
            f"Missing required environment variable(s): {', '.join(missing)}.\n"
            "See docs/DEPLOY_VERTEX.md."
        )
    if not staging_bucket.startswith("gs://"):
        sys.exit("STAGING_BUCKET must be a gs:// URI, e.g. gs://my-bucket.")

    print(f"Authenticating with Vertex AI (project={project}, location={location})...")
    client = vertexai.Client(project=project, location=location)

    print("Packaging ADK application (MoE-F Level-3 pipeline)...")
    app = AdkApp(agent=moef_level_3_system, enable_tracing=True)  # OpenTelemetry traces

    print("Deploying to Vertex AI Agent Engine. This may take several minutes...")
    remote_agent = client.agent_engines.create(
        agent=app,
        config={
            "requirements": ["google-cloud-aiplatform[agent_engines,adk]"],
            "staging_bucket": staging_bucket,
        },
    )
    print("Deployed. Agent Engine resource:")
    print(f"  {getattr(remote_agent, 'resource_name', remote_agent)}")
    return remote_agent


if __name__ == "__main__":
    deploy_to_gcp()
