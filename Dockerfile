# Reproducible environment for the training pipeline (no dev/Airflow groups).
# Build with `make docker-build` (or `scripts\run.cmd docker-build`), which records the git commit.
# Mount data/, reports/ and artifacts/ to keep outputs (MLflow runs stay inside the container):
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" \
#     -v "$PWD/artifacts:/app/artifacts" pii-redaction \
#     python -m pipelines.cli train --config configs/experiments/e1_small_cnn_subset.yaml
# The default command runs every step, including training with the (slow) default config.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 GIT_PYTHON_REFRESH=quiet

# Install dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-default-groups --no-install-project

COPY README.md ./
COPY src ./src
COPY pipelines ./pipelines
COPY configs ./configs
RUN uv sync --locked --no-default-groups

# The image has no .git, so the commit is passed in at build time and recorded on MLflow runs
# (src/tracking.py falls back to these). Kept last so a new commit doesn't invalidate the
# dependency layers above.
ARG GIT_COMMIT=unknown
ARG GIT_DIRTY=unknown
ENV GIT_COMMIT=$GIT_COMMIT GIT_DIRTY=$GIT_DIRTY

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "pipelines.cli", "all"]
