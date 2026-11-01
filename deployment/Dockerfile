# syntax=docker/dockerfile:1
FROM python:3.15-slim@sha256:94d3426f1def7db4a106a232ac121bfd975393e5dfd40a619b955fae45fe7913
WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy package metadata first for layer caching
COPY pyproject.toml ./
COPY README.md ./
COPY src/ agent/ aurelius_cli/ gateway/ acp_adapter/ cron/ plugins/ tools/ skills/ scripts/ ./

# Install dependencies
RUN uv sync --no-dev

# Auth must be configured for production. Set AURELIUS_API_KEYS or AURELIUS_API_KEY
# before binding to 0.0.0.0. The AURELIUS_AUTH_REQUIRED flag can be set to false
# for local-only deployments (not recommended for production).
ENV AURELIUS_AUTH_REQUIRED=true

EXPOSE 8080

# Health check against the actual /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Bind loopback by default. Use --host 0.0.0.0 only with AURELIUS_API_KEYS set.
ENTRYPOINT ["python", "-m", "gateway.aurelius_api", "--host", "127.0.0.1", "--port", "8080"]
