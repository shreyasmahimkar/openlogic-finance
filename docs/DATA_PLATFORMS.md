# Data platforms — AWS, Snowflake, Databricks & SQL (P4)

How the Equity Research Assistant uses enterprise data platforms — built
**local-first** so the SQL/interfaces are real and run offline, with the
**identical code** targeting the cloud in production (just swap the connection).
GCP/ADK is the agentic brain; these platforms feed and govern it.

```
   Databricks (Spark)          Snowflake (SQL)              AWS / GCS
   feature engineering   ──▶   feature store +       ──▶   object storage
   (Delta, MLflow)             audit log (SQL)             (transcripts, models)
        │                          │                           │
        └──────────────┬───────────┴────────────┬──────────────┘
                       ▼                        ▼
                ADK agent (Gemini/Vertex) + return model
```

## What's implemented (local stand-ins, identical code path)

| Platform | Local stand-in (runs offline) | Module | Production swap |
|---|---|---|---|
| **Snowflake** (SQL) | SQLite | `data_prep/feature_store.py` | same SQL on a Snowflake connection |
| **Snowflake** (governance) | SQLite | `risk_management/governance/audit.py` | Snowflake table + row-level access / masking |
| **AWS S3 / GCS** | local filesystem | `horizontal_foundation/storage.py` | `boto3` / `google-cloud-storage`, same put/get/list |
| **Databricks** (Spark) | pandas | `data_prep/pipelines/feature_pipeline.py` | same feature logic as a Spark/Delta job |

## Snowflake — the governed SQL system of record

- **Feature store** (`feature_store.py`): write engineered features; **point-in-time
  reads** (`WHERE date <= ? ORDER BY date DESC LIMIT 1` — no lookahead leakage);
  **monitoring marts** (`COUNT/AVG/MIN/MAX`) that power drift dashboards.
- **Audit log** (`audit.py`): every retrieval, score, recommendation, and **human
  approval** is a queryable SQL row — "why BUY, and who signed off?". In prod:
  masking policies on PII, row-level access.
- **SQL on display:** point-in-time joins, aggregation marts, and audit queries.

## Databricks — scale-out feature engineering

- `feature_pipeline.run_feature_pipeline()` computes the model features and lands
  them in the store. Locally it's pandas; the Databricks job runs the **same
  transform with Spark over Delta Lake**, plus MLflow-tracked training (P2 model)
  and nightly batch scoring written back to the feature store.

## AWS — genuine multi-cloud, kept small

- **S3** holds the transcript corpus + serialized models (read by the RAG ingester
  and the serving layer) — `LocalObjectStore` is the offline stand-in.
- **SageMaker** can host the return model endpoint that the GCP-hosted agent calls
  cross-cloud (the `serving/predict.py` contract is endpoint-agnostic).
- **Bedrock** is an optional alternative LLM/embeddings provider (provider-agnostic
  by design — the model registry already abstracts this).

## SQL — woven throughout

Feature engineering (point-in-time), monitoring aggregations (drift marts), and
the governance **audit trail** are all real SQL — see `feature_store.py` and
`audit.py`.

## Cost-aware adoption order

1. **Local-first** (this repo): SQLite + local FS + pandas — the SQL/Spark/feature
   code is identical to cloud, so the skill is demonstrated with zero spend.
2. Add a **Snowflake trial** → run the same feature/audit SQL on real warehouses.
3. Add **Databricks** (Community/trial) → run the Spark feature + training job.
4. Add the **AWS slice** (S3 + one SageMaker endpoint) → cross-cloud call from the
   agent; tear down to avoid charges.
