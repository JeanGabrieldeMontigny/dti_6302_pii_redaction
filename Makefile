CONFIG ?= configs/default.yaml
export MLFLOW_DISABLE_AGENT_HINT := 1
RUN := uv run python -m pipelines.cli --config $(CONFIG)

# Local Airflow: state lives in .airflow/ (git-ignored); DAGs are read from pipelines/dags.
AIRFLOW_ENV := AIRFLOW_HOME=$(CURDIR)/.airflow \
	AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/pipelines/dags \
	AIRFLOW__CORE__LOAD_EXAMPLES=False \
	NO_PROXY='*' OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
DAG_ID := pii_redaction_pipeline

.PHONY: help setup lint format test data validate baseline pipeline airflow airflow-run mlflow train evaluate serve monitor clean

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Create .venv and install locked dependencies (dev + airflow groups)
	uv sync

lint:  ## Run code-quality checks
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-format code
	uv run ruff format .
	uv run ruff check --fix .

test:  ## Run unit, integration and DAG tests
	uv run pytest

data:  ## Download raw data, validate it, build leakage-free splits, validate them
	$(RUN) ingest validate_raw preprocess validate_processed

validate:  ## Re-run data validation on existing raw + processed data
	$(RUN) validate_raw validate_processed

baseline:  ## Evaluate the regex baseline on validation and log it to MLflow
	$(RUN) evaluate_baseline

pipeline:  ## Run the whole pipeline without Airflow
	$(RUN) all

airflow:  ## Start a local Airflow (UI on http://localhost:8080; login printed on start)
	$(AIRFLOW_ENV) uv run airflow standalone

airflow-run:  ## Run the DAG once in-process via `airflow dags test` (no scheduler needed)
	$(AIRFLOW_ENV) uv run airflow db migrate >/dev/null
	$(AIRFLOW_ENV) uv run airflow dags reserialize >/dev/null
	$(AIRFLOW_ENV) uv run airflow dags test $(DAG_ID) -c '{"config": "$(CONFIG)"}'

mlflow:  ## Open the MLflow UI on http://localhost:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

train:  ## Train the Keras model (TODO: checkpoint 3)
	@echo "TODO: implement Keras training step" && exit 1

evaluate:  ## Evaluate a trained model (TODO: checkpoint 3)
	@echo "TODO: implement evaluation" && exit 1

serve:  ## Run the in-browser redaction demo (TODO: checkpoint 4)
	@echo "TODO: implement serving demo" && exit 1

monitor:  ## Generate the monitoring report (TODO: checkpoint 4)
	@echo "TODO: implement monitoring report" && exit 1

clean:  ## Remove caches (keeps data, MLflow runs and Airflow state)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path './.venv/*' -prune -exec rm -rf {} +
