"""Server app — re-exports from sql_env.server for OpenEnv compatibility."""
from sql_env.server.app import app  # noqa: F401


def main():
    """Entry point for the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
