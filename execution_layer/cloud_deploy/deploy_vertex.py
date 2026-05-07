import vertexai
from google.adk.agent_engines import AdkApp
from .agent import moef_level_3_system

def deploy_to_gcp():
    print("Authenticating with Google Cloud IAM...")
    # Initialize the Vertex AI SDK
    client = vertexai.Client(project="YOUR_PROJECT_ID", location="us-central1")
    
    print("Packaging ADK Application...")
    # Package the MoE-F App
    app = AdkApp(agent=moef_level_3_system)
    
    print("Deploying to Vertex AI Agent Engine. This may take several minutes.")
    # Deploy to Vertex AI Agent Engine
    remote_agent = client.agent_engines.create(
        agent=app,
        config={
            "requirements": ["google-cloud-aiplatform[agent_engines,adk]"],
            "staging_bucket": "gs://YOUR_STAGING_BUCKET",
        }
    )
    print("Successfully mapped Level 3 MAS to GCP Native Agent Engine!")
    return remote_agent

if __name__ == "__main__":
    deploy_to_gcp()
