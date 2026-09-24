# ITK-SNAP DLS server (nnInteractive, SAM2, ADPKD client)
# Build from the repo root: docker compose -f docker/compose.yml build
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# The PyPI torch wheels for Linux bundle CUDA, so no CUDA base image is needed;
# the GPU is provided by the compose device reservation.
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/models/hf-cache

WORKDIR /app

# Dependencies first, so code changes do not reinstall torch
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY itksnap_dls ./itksnap_dls
RUN uv sync --frozen --no-dev

EXPOSE 8911

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8911/status', timeout=3)"]

CMD ["python", "-m", "itksnap_dls", "--host", "0.0.0.0", "--port", "8911", "--models-path", "/models"]
