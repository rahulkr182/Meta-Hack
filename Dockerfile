# Build stage
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files and install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml /build/
COPY README.md /build/
COPY sql_env/ /build/sql_env/
COPY server/ /build/server/

RUN pip install --no-cache-dir -e .

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Create a non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appgroup pyproject.toml README.md openenv.yaml inference.py /app/
COPY --chown=appuser:appgroup sql_env/ /app/sql_env/
COPY --chown=appuser:appgroup server/ /app/server/

# Switch to non-root user
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "sql_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
