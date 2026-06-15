# OpenLogic Finance — common entrypoints. Run `make help` for the list.
# Standardizes on uv (`.venv` from uv.lock). See AGENTS.md.

export PYTHONPATH := .

.DEFAULT_GOAL := help
.PHONY: help setup lock test lint fmt run web web-dash hooks deploy sync-lean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Build the env from the lockfile + install pre-commit hooks
	uv sync --extra dev --extra interface
	uv run pre-commit install

lock: ## Re-resolve dependencies into uv.lock
	uv lock

test: ## Run the test suite
	uv run pytest -q

lint: ## Lint with ruff
	uv run ruff check .

fmt: ## Auto-format with ruff
	uv run ruff format .

run: ## Run the MoE-F coordinator (interactive). Launch from repo root.
	uv run adk run model_library/agentic_ai/moe_coordinator

web: ## Launch the ADK web UI, then pick "moe_coordinator"
	uv run adk web model_library/agentic_ai

web-dash: ## Launch the Streamlit monitoring dashboard
	uv run streamlit run interface/streamlit/app.py

hooks: ## Run all pre-commit hooks against the whole tree
	uv run pre-commit run --all-files

deploy: ## Deploy the MoE-F agent to Vertex AI Agent Engine (see docs/DEPLOY_VERTEX.md)
	uv run python live_paper_execution/cloud_deploy/deploy_vertex.py

sync-lean: ## Regenerate the LEAN project strategy copies from model_library
	uv run python scripts/sync_lean_strategies.py
