# Hugging Face Spaces Dockerfile
# HF Spaces expects port 7860

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY openenv.yaml /app/openenv.yaml
COPY inference.py /app/inference.py
COPY sql_env/ /app/sql_env/
COPY server/ /app/server/

# Install all dependencies (openenv-core comes from pyproject.toml)
RUN pip install --no-cache-dir -e .

# HF Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run server
CMD ["uvicorn", "sql_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
