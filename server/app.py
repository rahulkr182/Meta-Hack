"""Server app — re-exports from sql_env.server for compatibility."""
from sql_env.server.app import app  # noqa: F401
from sql_env.server.app import main  # noqa: F401

if __name__ == "__main__":
    main()
