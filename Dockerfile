# Reproducible environment for the training pipeline (no dev/Airflow groups).
# Mount data/ and reports/ to keep outputs (MLflow runs stay inside the container), e.g.
#   docker run -v $PWD/data:/app/data -v $PWD/reports:/app/reports pii-redaction
# The deployment target is the browser (on-device inference); a serving image
# for the demo web app will be added once the export format is decided.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# Install dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-default-groups --no-install-project

COPY README.md ./
COPY src ./src
COPY pipelines ./pipelines
COPY configs ./configs
RUN uv sync --locked --no-default-groups

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "pipelines.cli", "all"]
