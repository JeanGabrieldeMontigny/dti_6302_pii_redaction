CONFIG ?= configs/default.yaml
export MLFLOW_DISABLE_AGENT_HINT := 1
RUN := uv run python -m pipelines.cli --config $(CONFIG)

# Local Airflow: state lives in .airflow/ (git-ignored); DAGs are read from pipelines/dags.
AIRFLOW_ENV := AIRFLOW_HOME=$(CURDIR)/.airflow \
	AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/pipelines/dags \
	AIRFLOW__CORE__LOAD_EXAMPLES=False \
	NO_PROXY='*' OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
DAG_ID := pii_redaction_pipeline

.PHONY: help setup lint format test data validate baseline pipeline airflow airflow-run mlflow \
	train experiments evaluate serve monitor clean docker-build docker-train status decode-sweep \
	compare redact hooks train-gpu \
	airflow-docker-up airflow-docker-run airflow-docker-logs airflow-docker-down

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

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

train:  ## Train + evaluate one Keras tagger (CONFIG=configs/experiments/<name>.yaml)
	$(RUN) train

status:  ## Progress of running/interrupted trainings (epoch, ETA)
	uv run python -m pipelines.status

decode-sweep:  ## Recall/precision curve over decoding settings for the latest model of CONFIG
	$(RUN) decode_sweep

compare:  ## Write reports/experiments/comparison.md (all evaluated models side by side)
	$(RUN) compare

redact:  ## Redact TEXT="..." with a trained model (MODEL=e2 optional, WORD_SNAP=1 optional)
	uv run python -m src.serving.redactor $(if $(MODEL),--model $(MODEL)) $(if $(WORD_SNAP),--word-snap) "$(TEXT)"

hooks:  ## Install the git pre-commit hooks (once per clone)
	uv run pre-commit install

train-gpu:  ## Train CONFIG on an NVIDIA GPU in Docker (Linux/Windows hosts with NVIDIA drivers)
	docker compose --profile gpu run --rm --build trainer-gpu train --config $(CONFIG)

EXPERIMENTS ?= $(sort $(wildcard configs/experiments/*.yaml))
experiments:  ## Train every experiment config in configs/experiments/ (one MLflow run each)
	@for c in $(EXPERIMENTS); do echo "=== $$c ==="; uv run python -m pipelines.cli --config $$c train || exit 1; done

evaluate:  ## Evaluate a trained model (TODO: checkpoint 3)
	@echo "TODO: implement evaluation" && exit 1

serve:  ## Run the in-browser redaction demo (TODO: checkpoint 4)
	@echo "TODO: implement serving demo" && exit 1

monitor:  ## Generate the monitoring report (TODO: checkpoint 4)
	@echo "TODO: implement monitoring report" && exit 1

# --- Docker (all platforms; the only way to run Airflow on Windows) -------------------------------
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
GIT_DIRTY := $(shell test -z "$$(git status --porcelain --untracked-files=no 2>/dev/null)" && echo false || echo true)
DOCKER_MOUNTS := -v "$(CURDIR)/data:/app/data" -v "$(CURDIR)/reports:/app/reports" \
	-v "$(CURDIR)/artifacts:/app/artifacts"

docker-build:  ## Build the pipeline image (records the git commit for MLflow lineage)
	docker build -t pii-redaction --build-arg GIT_COMMIT=$(GIT_COMMIT) --build-arg GIT_DIRTY=$(GIT_DIRTY) .

docker-train:  ## Train CONFIG inside the pipeline image (MLflow run stays in the container)
	docker run --rm $(DOCKER_MOUNTS) pii-redaction python -m pipelines.cli train --config $(CONFIG)

airflow-docker-up:  ## Start Airflow (http://localhost:8080) + its MLflow UI (:5001) in Docker
	docker compose up -d --build airflow
	@echo "Airflow: http://localhost:8080   MLflow (Docker runs): http://localhost:5001"

airflow-docker-run:  ## Trigger one DAG run in the Docker Airflow and wait for it (CONFIG=...)
	docker compose exec airflow airflow-run $(CONFIG)

airflow-docker-logs:  ## Follow the Docker Airflow logs
	docker compose logs -f airflow

airflow-docker-down:  ## Stop Docker Airflow (keeps its state; add -v to docker compose down to wipe)
	docker compose down

clean:  ## Remove caches (keeps data, MLflow runs and Airflow state)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path './.venv/*' -prune -exec rm -rf {} +
