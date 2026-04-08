"""Client for the SQL Query Environment.

Provides both sync and async interfaces for interacting with the
SQL environment server over HTTP and WebSocket.
"""

import json
from typing import Any, Dict, Optional

import httpx


class SqlEnvClient:
    """HTTP client for the SQL Query Environment.

    Provides synchronous access to the environment's REST API.
    For WebSocket-based interaction, use SqlEnvAsyncClient.

    Example:
        >>> client = SqlEnvClient(base_url="http://localhost:8000")
        >>> result = client.reset(task_id="easy_01")
        >>> print(result["observation"]["question"])
        >>> result = client.step("SELECT * FROM employees")
        >>> print(result["observation"]["feedback"])
        >>> print(result["reward"])
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        """Initialize the client.

        Args:
            base_url: URL of the environment server.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def schema(self) -> Dict[str, Any]:
        """Get action/observation/state JSON schemas."""
        resp = self._client.get("/schema")
        resp.raise_for_status()
        return resp.json()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reset the environment.

        Args:
            seed: Random seed for task selection.
            episode_id: Custom episode ID.
            task_id: Specific task to load.
            difficulty: Filter by difficulty.

        Returns:
            Reset response with observation, reward, done.
        """
        params = {}
        if seed is not None:
            params["seed"] = seed
        if episode_id:
            params["episode_id"] = episode_id
        if task_id:
            params["task_id"] = task_id
        if difficulty:
            params["difficulty"] = difficulty

        resp = self._client.post("/reset", json=params)
        resp.raise_for_status()
        return resp.json()

    def step(self, sql_query: str) -> Dict[str, Any]:
        """Submit a SQL query to the environment.

        Args:
            sql_query: The SQL query to execute.

        Returns:
            Step response with observation, reward, done.
        """
        resp = self._client.post("/step", json={
            "action": {"sql_query": sql_query}
        })
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        """Get current environment state."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SqlEnvAsyncClient:
    """Async WebSocket client for the SQL Query Environment.

    Example:
        >>> import asyncio
        >>> async def main():
        ...     async with SqlEnvAsyncClient("ws://localhost:8000/ws") as client:
        ...         obs = await client.reset(task_id="easy_01")
        ...         obs = await client.step("SELECT * FROM employees")
        ...         state = await client.state()
        >>> asyncio.run(main())
    """

    def __init__(self, ws_url: str = "ws://localhost:8000/ws"):
        self.ws_url = ws_url
        self._ws = None

    async def connect(self):
        """Connect to the WebSocket server."""
        import websockets
        self._ws = await websockets.connect(self.ws_url)

    async def close(self):
        """Close the WebSocket connection."""
        if self._ws:
            await self._ws.send(json.dumps({"type": "close"}))
            await self._ws.close()
            self._ws = None

    async def _send_and_recv(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message and receive the response."""
        if self._ws is None:
            raise RuntimeError("Not connected. Call connect() first.")
        await self._ws.send(json.dumps(message))
        response = await self._ws.recv()
        return json.loads(response)

    async def reset(self, **kwargs) -> Dict[str, Any]:
        """Reset the environment."""
        return await self._send_and_recv({"type": "reset", "data": kwargs})

    async def step(self, sql_query: str) -> Dict[str, Any]:
        """Submit a SQL query."""
        return await self._send_and_recv({
            "type": "step",
            "data": {"sql_query": sql_query},
        })

    async def state(self) -> Dict[str, Any]:
        """Get current state."""
        return await self._send_and_recv({"type": "state"})

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()
