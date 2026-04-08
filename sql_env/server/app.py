"""FastAPI application for the SQL Query Environment.

Creates the HTTP server with REST and WebSocket endpoints following
the OpenEnv specification.
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from sql_env.models import SqlAction, SqlObservation, SqlState
from sql_env.server.sql_environment import SqlEnvironment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    logger.info("SQL Query Environment starting up...")
    yield
    logger.info("SQL Query Environment shutting down...")
    env.close()


# ---------------------------------------------------------------------------
# Create app and environment
# ---------------------------------------------------------------------------

env = SqlEnvironment(max_attempts=3)

app = FastAPI(
    title="SQL Query Environment",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to write "
        "SQL queries against database schemas."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/schema")
async def schema():
    """Return the JSON schemas for Action, Observation, and State."""
    return {
        "action": SqlAction.model_json_schema(),
        "observation": SqlObservation.model_json_schema(),
        "state": SqlState.model_json_schema(),
    }


@app.post("/reset")
async def reset(data: Optional[Dict[str, Any]] = None):
    """Reset the environment. Accepts optional parameters in the body."""
    params = data or {}
    obs = env.reset(**params)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "info": obs.metadata,
    }


@app.post("/step")
async def step(data: Dict[str, Any]):
    """Execute a step with the given action."""
    action = SqlAction(**data.get("action", data))
    obs = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "info": obs.metadata,
    }


@app.get("/state")
async def state():
    """Get the current environment state."""
    return env.state.model_dump()


# ---------------------------------------------------------------------------
# WebSocket Endpoint (OpenEnv standard)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for real-time environment interaction."""
    await ws.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({
                    "type": "error",
                    "data": {"message": "Invalid JSON", "code": "INVALID_JSON"},
                })
                continue

            msg_type = msg.get("type", "")

            if msg_type == "reset":
                params = msg.get("data", {})
                obs = env.reset(**params)
                await ws.send_json({
                    "type": "observation",
                    "data": obs.model_dump(),
                })

            elif msg_type == "step":
                try:
                    action_data = msg.get("data", {})
                    action = SqlAction(**action_data)
                    obs = env.step(action)
                    await ws.send_json({
                        "type": "observation",
                        "data": obs.model_dump(),
                    })
                except Exception as e:
                    await ws.send_json({
                        "type": "error",
                        "data": {
                            "message": str(e),
                            "code": "EXECUTION_ERROR",
                        },
                    })

            elif msg_type == "state":
                await ws.send_json({
                    "type": "state",
                    "data": env.state.model_dump(),
                })

            elif msg_type == "close":
                await ws.close()
                break

            else:
                await ws.send_json({
                    "type": "error",
                    "data": {
                        "message": f"Unknown message type: {msg_type}",
                        "code": "UNKNOWN_TYPE",
                    },
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_json({
                "type": "error",
                "data": {"message": str(e), "code": "EXECUTION_ERROR"},
            })
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Run with: uvicorn sql_env.server.app:app --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
