"""
FastAPI application for the SQL Query Environment.

This module creates an HTTP server that exposes the SqlEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. "
        "Install with: pip install openenv-core[core]"
    ) from e

# Import from local models.py (PYTHONPATH includes /app in Docker)
try:
    from models import SqlAction, SqlObservation
except ImportError:
    from ..models import SqlAction, SqlObservation

from .sql_environment import SqlEnvironment


# Create the app with web interface
app = create_app(
    SqlEnvironment,
    SqlAction,
    SqlObservation,
    env_name="sql_query_env",
    max_concurrent_envs=8,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution.

    This function enables running the server without Docker:
        python -m sql_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 7860)
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
