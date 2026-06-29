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
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(schema_sql)
    conn.executescript(seed_sql)
    conn.commit()
    return conn


import re
import time

class TimeoutException(Exception):
    pass

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
        # Deep SQL injection guard: strip comments and strings to find actual command
        sql_clean = re.sub(r'--.*', '', sql)
        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
        sql_clean = re.sub(r"'[^']*'", "''", sql_clean)
        sql_upper = sql_clean.strip().upper()
        
        # Check first token
        first_token = sql_upper.split()[0] if sql_upper.split() else ""
        
        forbidden_commands = {
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", 
            "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "VACUUM"
        }
        
        if first_token in forbidden_commands:
            return None, None, f"Write operations are not allowed. Your query starts with '{first_token}'."
            
        # Specific check for PRAGMA (allow only read-only PRAGMAs)
        if first_token == "PRAGMA":
            # Very basic check: block setting pragmas (which typically have an '=' or are 'PRAGMA schema.name(value)')
            # EXCEPT table_info which is PRAGMA table_info(name)
            if "=" in sql_upper or ( "(" in sql_upper and not "TABLE_INFO" in sql_upper):
                 return None, None, "Setting PRAGMA values is not allowed."

        # Setup timeout handler
        start_time = time.time()
        def progress_handler():
            if time.time() - start_time > timeout_seconds:
                raise TimeoutException(f"Query exceeded {timeout_seconds}s timeout")
            return 0
            
        conn.set_progress_handler(progress_handler, 1000)

        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        column_names = (
            [desc[0] for desc in cursor.description] if cursor.description else []
        )
        
        # Clear progress handler
        conn.set_progress_handler(None, 0)
        
        return rows, column_names, None
    except TimeoutException as e:
        conn.set_progress_handler(None, 0)
        return None, None, f"Timeout Error: {str(e)}"
    except sqlite3.Error as e:
        conn.set_progress_handler(None, 0)
        return None, None, f"SQL Error: {str(e)}"
    except Exception as e:
        conn.set_progress_handler(None, 0)
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
        formatted_row = []
        for i, val in enumerate(row):
            val_str = str(val)
            # Truncate long strings
            if len(val_str) > 50:
                 val_str = val_str[:47] + "..."
            
            # Right-align numeric types (int, float)
            if isinstance(val, (int, float)):
                 formatted_row.append(val_str.rjust(col_widths[i]))
            else:
                 formatted_row.append(val_str.ljust(col_widths[i]))
                 
        lines.append(" | ".join(formatted_row))

    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")

    lines.append(f"\n({len(rows)} row{'s' if len(rows) != 1 else ''} total)")
    return "\n".join(lines)
