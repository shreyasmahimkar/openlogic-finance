# Deploying the MoE-F Coordinator to Vertex AI Agent Engine

A step-by-step guide to deploying the MoE-F coordinator to **Vertex AI Agent
Engine** using your own Google account. Thanks to the Phase 3 model registry, the
whole pipeline runs on **Gemini via Vertex** — no OpenAI/Groq keys required.

> **What you get:** a managed, autoscaling endpoint running the
> `moef_level_3_system` ADK pipeline, with OpenTelemetry tracing, that you can
> query over the Vertex AI SDK.

---

## 0. What you need

- A Google account with access to **Google Cloud Console** (https://console.cloud.google.com).
- A GCP **project** with **billing enabled** (Agent Engine + Gemini calls cost money — see §7).
- The **gcloud CLI** installed locally (https://cloud.google.com/sdk/docs/install).
- This repo set up locally: `make setup` (or `uv sync`).

---

## 1. Create / pick a project and enable billing

```bash
gcloud projects create openlogic-finance-demo      # or reuse an existing project
gcloud config set project openlogic-finance-demo
```

Then in the Console, link a billing account: **Billing → Link a billing account**.
(Agent Engine and Gemini require billing even on free-tier credits.)

## 2. Enable the required APIs

```bash
gcloud services enable aiplatform.googleapis.com storage.googleapis.com
```

`aiplatform` covers both Vertex AI Agent Engine and Vertex-hosted Gemini.

## 3. Create a staging bucket

Agent Engine stages your packaged agent in Cloud Storage. Create a bucket **in
the same region** you will deploy to:

```bash
gcloud storage buckets create gs://openlogic-finance-staging \
    --location=us-central1
```

## 4. Authenticate (Application Default Credentials)

```bash
gcloud auth application-default login
```

This logs in your personal Google account and writes ADC that the Vertex SDK
picks up automatically. Your account needs these IAM roles on the project
(grant via **IAM & Admin → IAM**, or you already have them as project Owner):

- **Vertex AI User** (`roles/aiplatform.user`)
- **Storage Admin** (`roles/storage.admin`) on the staging bucket
- **Service Account User** (`roles/iam.serviceAccountUser`)

## 5. Configure the deploy environment

The deploy script (`live_paper_execution/cloud_deploy/deploy_vertex.py`) is fully
env-driven — no editing code:

```bash
export GOOGLE_CLOUD_PROJECT=openlogic-finance-demo
export STAGING_BUCKET=gs://openlogic-finance-staging
export GOOGLE_CLOUD_LOCATION=us-central1        # optional, this is the default

# Tell ADK/GenAI to use Vertex-hosted Gemini (not the public API key path):
export GOOGLE_GENAI_USE_VERTEXAI=TRUE
```

> **Why `GOOGLE_GENAI_USE_VERTEXAI=TRUE`?** The agents resolve to
> `gemini-2.5-flash` via the model registry. This flag routes those calls through
> **your project's Vertex AI** (billed to your project, no separate API key)
> instead of the public Gemini API. On Agent Engine itself this is the default;
> setting it locally makes a pre-deploy smoke test behave the same way.

## 6. Deploy

```bash
make deploy
# equivalently:
uv run python live_paper_execution/cloud_deploy/deploy_vertex.py
```

The script:
1. initializes the Vertex client for your project/region,
2. wraps the canonical `moef_level_3_system` in an `AdkApp` (tracing enabled),
3. uploads to the staging bucket and creates the Agent Engine instance.

This takes a few minutes. On success it prints the **Agent Engine resource name**
(`projects/.../locations/.../reasoningEngines/NNNN`) — save it.

## 7. Test the deployed agent

```python
import vertexai

client = vertexai.Client(project="openlogic-finance-demo", location="us-central1")
agent = client.agent_engines.get("projects/.../reasoningEngines/NNNN")  # your resource name

for event in agent.stream_query(message="Run the MoE-F forecast for SPY."):
    print(event)
```

You can also see it under **Vertex AI → Agent Engine** in the Console, with
traces under **Vertex AI → Traces** (OpenTelemetry is enabled in the deploy).

## 8. Costs & cleanup

Agent Engine bills for the managed runtime **for as long as it exists**, plus
per-call Gemini tokens. To avoid ongoing charges, delete it when done:

```python
agent.delete(force=True)
```

or **Agent Engine → (your engine) → Delete** in the Console. Also empty the
staging bucket if you no longer need it.

---

## Local sanity check before deploying (optional, recommended)

Run the pipeline locally first so you don't pay for a broken deploy:

```bash
export GEMINI_API_KEY=...          # public API key path, for quick local runs
make run                           # adk run model_library/agentic_ai/moe_coordinator
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Missing required environment variable(s)` | Export `GOOGLE_CLOUD_PROJECT` and `STAGING_BUCKET` (§5). |
| `PermissionDenied` / 403 | Grant the IAM roles in §4; confirm `gcloud config get-value project`. |
| `404 ... API not enabled` | Re-run §2; enabling can take a minute to propagate. |
| Bucket region mismatch | Staging bucket and `GOOGLE_CLOUD_LOCATION` must match (§3). |
| Experts error on non-Gemini models | You set `OPENLOGIC_HETEROGENEOUS_EXPERTS=1` without LiteLLM/provider keys — unset it to use the Gemini defaults. |

See also `model_library/agentic_ai/docs/ADK_USAGE_GUIDE.md` and
`docs/AGENTIC_ENGINEERING_SDLC_PLAN.md` (§6, §9 Phase 4).
