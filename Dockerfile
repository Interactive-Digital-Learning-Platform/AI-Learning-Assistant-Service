# syntax=docker/dockerfile:1
###############################################################################
# AI Learning Assistant Service
#
# Multi-stage image intended for:
#   - build/push:  GitHub Actions (docker buildx) -> Azure Container Registry
#   - deploy:      Azure Container Apps
#
# One image, two workloads (override the command per Container App):
#   API     : uvicorn app.main:app --host 0.0.0.0 --port 8002   (default CMD)
#   Worker  : arq app.worker.WorkerSettings
#
# The API's models (embedder, reranker, BM25) are downloaded at build time and
# baked into the image, so replicas start without fetching ~600 MB from the
# Hugging Face Hub on every cold start.
###############################################################################

ARG PYTHON_VERSION=3.13

###############################################################################
# Stage 1 — builder: resolve and install dependencies into a self-contained venv
###############################################################################
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

# - bytecode-compile installed packages for faster cold starts
# - copy (don't hardlink) from the uv cache so the venv is relocatable
# - never let uv fetch its own Python; use the interpreter in this image
# - raise the HTTP timeout so the large torch CPU wheel doesn't hit the 30s default
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_HTTP_TIMEOUT=180

WORKDIR /app

# Third-party dependencies only (project itself is not installed; the app runs
# from source). This layer is rebuilt only when the manifests change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project

###############################################################################
# Stage 2 — models: pre-download the models the API loads at startup
###############################################################################
FROM builder AS models

# Defaults mirror app/core/config.py. If you change EMBEDDING_MODEL /
# RERANK_MODEL / SPARSE_MODEL on the Container App, rebuild with matching
# --build-arg values (or set HF_HUB_OFFLINE=0 to allow a runtime download).
ARG EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ARG SPARSE_MODEL=Qdrant/bm25

ENV HF_HOME=/models/huggingface \
    FASTEMBED_CACHE_DIR=/models/fastembed

# Load each model exactly the way the app does, so everything it needs at
# runtime (weights, tokenizers, any remote code) lands in the cache.
RUN /app/.venv/bin/python - <<EOF
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder, SentenceTransformer

SentenceTransformer("${EMBEDDING_MODEL}", device="cpu")
CrossEncoder("${RERANK_MODEL}")
SparseTextEmbedding(model_name="${SPARSE_MODEL}", cache_dir="/models/fastembed")
EOF

###############################################################################
# Stage 3 — runtime: slim image with only what's needed to run the service
###############################################################################
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# System packages:
#   libmagic1 : file-type sniffing of uploads (python-magic) in
#               app/controllers/attachment_controllers.py
#   libgomp1  : OpenMP runtime for torch / onnxruntime (sentence-transformers,
#               fastembed) wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user
RUN groupadd --system app && useradd --system --gid app --home-dir /app app

WORKDIR /app

# - PYTHONPATH: lets the `arq` worker import `app.worker` from the source tree
# - HF_HUB_OFFLINE: models are baked in, so never call out to the Hub at startup
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_DIR=/app/.cache/fastembed \
    HF_HUB_OFFLINE=1

# Dependencies, then baked models, then application code — most-volatile last.
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=models --chown=app:app /models/huggingface /app/.cache/huggingface
COPY --from=models --chown=app:app /models/fastembed /app/.cache/fastembed
COPY --chown=app:app app ./app
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app alembic.ini ./alembic.ini

USER app

EXPOSE 8002

# Hits the "/" liveness ping in app/main.py.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8002/', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
