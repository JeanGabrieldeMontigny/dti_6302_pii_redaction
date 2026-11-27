# Local Airflow (+ MLflow UI) for the training pipeline. Works on every OS, including Windows where
# Airflow cannot run natively. Dependencies are baked into the image; the repository itself is
# mounted at /app at runtime (see docker-compose.yml), so code and DAG changes apply without a
# rebuild. Rebuild only when uv.lock changes.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# git: lineage tags (commit / dirty) for MLflow runs, read from the mounted repository.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app \
    # Windows checkouts have CRLF line endings; compare them to the LF index without flagging
    # every file as modified.
    && git config --system core.autocrlf true

ENV UV_PROJECT_ENVIRONMENT=/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    PATH="/venv/bin:$PATH" PYTHONPATH=/app PYTHONUNBUFFERED=1

WORKDIR /build
COPY pyproject.toml uv.lock .python-version ./
# All default groups (dev + airflow); the project itself is imported from /app via PYTHONPATH.
RUN uv sync --locked --no-install-project

COPY docker/airflow-entrypoint.sh /usr/local/bin/airflow-entrypoint
COPY docker/airflow-run.sh /usr/local/bin/airflow-run
RUN chmod +x /usr/local/bin/airflow-entrypoint /usr/local/bin/airflow-run

WORKDIR /app
EXPOSE 8080 5001
ENTRYPOINT ["airflow-entrypoint"]
