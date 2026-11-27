# GPU training image (NVIDIA). TensorFlow has no native Windows GPU support, so on Windows GPU
# training runs here, through Docker Desktop's GPU passthrough. The repository is mounted at /app
# at runtime (docker-compose.yml, service `trainer-gpu`), so code changes need no rebuild.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app \
    && git config --system core.autocrlf true

ENV UV_PROJECT_ENVIRONMENT=/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    PATH="/venv/bin:$PATH" PYTHONPATH=/app PYTHONUNBUFFERED=1

WORKDIR /build
COPY pyproject.toml uv.lock .python-version ./
# Runtime dependencies + the locked CUDA/cuDNN wheels (dependency group `gpu`).
RUN uv sync --locked --no-install-project --no-default-groups --group gpu

# TensorFlow does not find every pip-installed CUDA library on its own (libcusolver is missed),
# so register all of them with the dynamic linker.
RUN find /venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d \
        > /etc/ld.so.conf.d/nvidia-pip.conf \
    && ldconfig

# - Allocate GPU memory as needed instead of reserving all of it up front.
# - TensorFlow 2.21 ships no kernels for the newest GPUs (e.g. RTX 50xx, compute capability
#   12.0), so CUDA compiles them from PTX on first use (~1-2 min). Keep that cache in a volume
#   so it only happens once.
ENV TF_FORCE_GPU_ALLOW_GROWTH=true \
    CUDA_CACHE_PATH=/cache/cuda \
    CUDA_CACHE_MAXSIZE=4294967296 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app
ENTRYPOINT ["python", "-m", "pipelines.cli"]
