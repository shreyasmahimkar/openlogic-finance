# Live & Paper Execution (BOX 5)

Transformed from the former `execution_layer/` directory, this box coordinates mock execution environments, paper accounts, and cloud deployment pipelines.

## Target Structure & Subdirectories

- **`simulators/`**: Containerized mock execution environments (e.g. Docker Compose rigs).
- **`paper_accounts/`**: Interactive Broker / exchange paper APIs and state managers.
- **`cloud_deploy/`**: GCP accelerators, Vertex AI endpoints, and production infra build files.

## Purpose & Architectural Rule

This box provides the interface to live or sandbox markets. It is isolated from the predictive signals and risk management loops to handle pure transactional execution and API connection concerns.
