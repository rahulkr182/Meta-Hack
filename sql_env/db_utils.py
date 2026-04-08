"""SQLite database utilities for the SQL Query Environment.

Provides helpers to create isolated in-memory SQLite databases,
seed them with schema + data, and execute queries safely.
"""

import sqlite3
from typing import Any, List, Optional, Tuple


def create_database(schema_sql: str, seed_sql: str) -> sqlite3.Connection:
    """Create an in-memory SQLite database with the given schema and seed data.

    Args:
        schema_sql: SQL statements to create tables (CREATE TABLE ...).
        seed_sql: SQL statements to insert seed data (INSERT INTO ...).

    Returns:
        A sqlite3.Connection to the in-memory database.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(schema_sql)
    conn.executescript(seed_sql)
    conn.commit()
    return conn


def execute_query(
    conn: sqlite3.Connection,
    sql: str,
    timeout_seconds: float = 5.0,
) -> Tuple[Optional[List[Tuple[Any, ...]]], Optional[List[str]], Optional[str]]:
    """Execute a SQL query safely and return results.

    Args:
        conn: SQLite connection.
        sql: SQL query to execute.
        timeout_seconds: Maximum time for query execution.

    Returns:
        Tuple of (rows, column_names, error_message).
        - rows: List of result tuples, or None if error.
        - column_names: List of column names, or None if error.
        - error_message: Error string, or None if success.
    """
    try:
        # Basic SQL injection / danger guard: disallow writes
        sql_upper = sql.strip().upper()
        for forbidden in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"):
            if sql_upper.startswith(forbidden):
                return None, None, f"Write operations are not allowed. Your query starts with '{forbidden}'."

        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        column_names = (
            [desc[0] for desc in cursor.description] if cursor.description else []
        )
        return rows, column_names, None
    except sqlite3.Error as e:
        return None, None, f"SQL Error: {str(e)}"
    except Exception as e:
        return None, None, f"Execution Error: {str(e)}"


def format_results(
    rows: Optional[List[Tuple[Any, ...]]],
    column_names: Optional[List[str]],
    max_rows: int = 50,
) -> str:
    """Format query results as a human-readable table string.

    Args:
        rows: List of result tuples.
        column_names: List of column names.
        max_rows: Maximum number of rows to include.

    Returns:
        Formatted string representation of the results.
    """
    if rows is None or column_names is None:
        return "(no results)"

    if not rows:
        return f"Columns: {', '.join(column_names)}\n(0 rows)"

    # Compute column widths
    col_widths = [len(name) for name in column_names]
    display_rows = rows[:max_rows]
    for row in display_rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    # Build header
    header = " | ".join(
        name.ljust(col_widths[i]) for i, name in enumerate(column_names)
    )
    separator = "-+-".join("-" * w for w in col_widths)

    # Build rows
    lines = [header, separator]
    for row in display_rows:
        line = " | ".join(
            str(val).ljust(col_widths[i]) for i, val in enumerate(row)
        )
        lines.append(line)

    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")

    lines.append(f"\n({len(rows)} row{'s' if len(rows) != 1 else ''} total)")
    return "\n".join(lines)
