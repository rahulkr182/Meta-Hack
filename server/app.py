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
    python server/app.py
"""

import os
import sys

# Ensure the repo root is on sys.path so sql_env can be imported
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Lazy app creation — validator may not have openenv installed
app = None

try:
    from openenv.core.env_server.http_server import create_app

    try:
        from models import SqlAction, SqlObservation
    except ImportError:
        from sql_env.models import SqlAction, SqlObservation

    try:
        from server.sql_environment import SqlEnvironment
    except ImportError:
        from sql_env.server.sql_environment import SqlEnvironment

    app = create_app(
        SqlEnvironment,
        SqlAction,
        SqlObservation,
        env_name="sql_query_env",
        max_concurrent_envs=8,
    )
except Exception:
    # App creation deferred to main() if top-level import fails
    pass


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution.

    This function enables running the server:
        python server/app.py
        python -m server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 7860)
    """
    global app

    if app is None:
        from openenv.core.env_server.http_server import create_app
        from sql_env.models import SqlAction, SqlObservation
        from sql_env.server.sql_environment import SqlEnvironment

        app = create_app(
            SqlEnvironment,
            SqlAction,
            SqlObservation,
            env_name="sql_query_env",
            max_concurrent_envs=8,
        )

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
