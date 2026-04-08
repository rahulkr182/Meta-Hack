# Hugging Face Spaces Dockerfile
# HF Spaces expects port 7860

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY sql_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy application code
COPY sql_env/ /app/sql_env/

# HF Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()" || exit 1

# Run server on port 7860 (HF Spaces requirement)
CMD ["uvicorn", "sql_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
