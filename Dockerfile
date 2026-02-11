# Stage 1: Build — install dependencies with uv
FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable

# Stage 2: Runtime — lean image with only what's needed
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY vidqc/ vidqc/
COPY config.yaml .
COPY models/ models/

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["python", "-m", "vidqc"]
