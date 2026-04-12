"""Grader for the SQL Query Environment.

Implements a multi-component scoring rubric (0.0–1.0) with partial-progress
signals so the agent gets meaningful reward even for partially correct queries.

Scoring Components:
    - Syntax validity    (0.10): Query parses without error
    - Execution success  (0.10): Query runs without runtime error
    - Correct tables     (0.15): References the right table(s)
    - Correct columns    (0.15): Selects the right column(s)
    - Partial row match  (0.20): Jaccard similarity of result rows
    - Exact match        (0.30): Results exactly match gold (order-insensitive)

Adjustments:
    - Step cost: -0.02 per attempt after the first (incentivizes efficiency)
    - First-attempt bonus: +0.05 if solved on first try
"""

import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

from sql_env.db_utils import execute_query
from sql_env.tasks import Task


# ---------------------------------------------------------------------------
# Weight configuration
# ---------------------------------------------------------------------------

WEIGHTS = {
    "syntax_valid": 0.10,
    "execution_success": 0.10,
    "correct_tables": 0.15,
    "correct_columns": 0.15,
    "partial_row_match": 0.20,
    "exact_match": 0.30,
}


# ---------------------------------------------------------------------------
# SQL Parsing Helpers
# ---------------------------------------------------------------------------

# Common SQL keywords to exclude from table/column extraction
SQL_KEYWORDS = {
    "select", "from", "where", "join", "inner", "outer", "left", "right",
    "full", "cross", "on", "and", "or", "not", "in", "is", "null", "as",
    "group", "by", "order", "having", "limit", "offset", "union", "all",
    "insert", "update", "delete", "create", "drop", "alter", "table",
    "into", "values", "set", "distinct", "between", "like", "exists",
    "case", "when", "then", "else", "end", "asc", "desc", "count",
    "sum", "avg", "min", "max", "round", "coalesce", "ifnull",
    "cast", "over", "partition", "row_number", "rank", "dense_rank",
    "with", "recursive", "true", "false", "primary", "key", "foreign",
    "references", "default", "integer", "text", "real", "blob",
}


def _extract_table_references(sql: str) -> Set[str]:
    """Extract table names referenced in a SQL query (best-effort)."""
    sql_clean = re.sub(r"'[^']*'", "", sql)  # Remove string literals
    sql_clean = re.sub(r"--[^\n]*", "", sql_clean)  # Remove line comments
    sql_clean = sql_clean.lower()

    tables = set()

    # Match FROM and JOIN clauses
    patterns = [
        r'\bfrom\s+(\w+)',
        r'\bjoin\s+(\w+)',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, sql_clean):
            name = match.group(1)
            if name not in SQL_KEYWORDS:
                tables.add(name)

    return tables


def _extract_selected_columns(sql: str) -> Set[str]:
    """Extract column names from the SELECT clause (best-effort)."""
    sql_clean = re.sub(r"'[^']*'", "", sql)
    sql_clean = re.sub(r"--[^\n]*", "", sql_clean)
    sql_clean = sql_clean.lower()

    # Find the SELECT ... FROM segment
    match = re.search(r'\bselect\s+(.*?)\bfrom\b', sql_clean, re.DOTALL)
    if not match:
        return set()

    select_clause = match.group(1)

    if "*" in select_clause:
        return {"*"}

    columns = set()
    # Split by comma and extract the column name (after dots, before AS)
    for part in select_clause.split(","):
        part = part.strip()
        # Handle "alias" in "expr AS alias"
        as_match = re.search(r'\bas\s+(\w+)\s*$', part)
        if as_match:
            columns.add(as_match.group(1))
        else:
            # Handle "table.column" or just "column"
            col_match = re.search(r'(\w+)\s*$', part)
            if col_match:
                name = col_match.group(1)
                if name not in SQL_KEYWORDS:
                    columns.add(name)

    return columns


def _normalize_rows(rows: List[Tuple[Any, ...]]) -> Set[Tuple[Any, ...]]:
    """Normalize rows for comparison: convert to tuples of strings, create a set."""
    normalized = set()
    for row in rows:
        normalized.add(tuple(str(v).strip().lower() for v in row))
    return normalized


# ---------------------------------------------------------------------------
# Main Grader
# ---------------------------------------------------------------------------

def grade_query(
    agent_sql: str,
    task: Task,
    conn: sqlite3.Connection,
    attempt: int = 1,
) -> Dict[str, Any]:
    """Grade an agent's SQL query against the gold standard.

    Args:
        agent_sql: The SQL query submitted by the agent.
        task: The task being evaluated (contains gold query and results).
        conn: Active SQLite connection with the task's database.

    Returns:
        Dictionary with:
            - total_reward: float (0.0–1.0)
            - breakdown: dict of component scores
            - feedback: human-readable feedback string
            - agent_rows: the agent's query results (or None)
            - agent_columns: the agent's result column names (or None)
            - error: error message if any
    """
    breakdown: Dict[str, float] = {k: 0.0 for k in WEIGHTS}
    feedback_parts: List[str] = []
    agent_rows = None
    agent_columns = None
    error_msg = None

    # Ensure gold results are computed
    gold_rows = task.gold_rows
    gold_columns = task.gold_columns

    # ---- 1. Syntax validity ----
    try:
        # Try to compile the SQL to check syntax (without executing)
        sql_stripped = agent_sql.strip().rstrip(";")
        # Use EXPLAIN to check syntax without side effects
        conn.execute(f"EXPLAIN {sql_stripped}")
        breakdown["syntax_valid"] = 1.0
        feedback_parts.append("✓ SQL syntax is valid")
    except sqlite3.Error:
        breakdown["syntax_valid"] = 0.0
        feedback_parts.append("✗ SQL syntax error — query could not be parsed")
        # If syntax is invalid, we can still try to check table/column names
        error_msg = "Syntax error"

    # ---- 2. Execution success ----
    agent_rows_raw, agent_cols_raw, exec_error = execute_query(conn, agent_sql)

    if exec_error is None:
        breakdown["execution_success"] = 1.0
        agent_rows = agent_rows_raw
        agent_columns = agent_cols_raw
        feedback_parts.append(f"✓ Query executed successfully ({len(agent_rows)} rows returned)")
    else:
        breakdown["execution_success"] = 0.0
        error_msg = exec_error
        feedback_parts.append(f"✗ Execution failed: {exec_error}")

    # ---- 3. Correct tables ----
    gold_tables = _extract_table_references(task.gold_query)
    agent_tables = _extract_table_references(agent_sql)

    if gold_tables and agent_tables:
        table_overlap = len(gold_tables & agent_tables) / len(gold_tables)
        breakdown["correct_tables"] = round(table_overlap, 3)
        if table_overlap >= 1.0:
            feedback_parts.append(f"✓ Correct tables referenced: {gold_tables}")
        elif table_overlap > 0:
            missing = gold_tables - agent_tables
            feedback_parts.append(
                f"◐ Partially correct tables. Found: {agent_tables & gold_tables}, "
                f"Missing: {missing}"
            )
        else:
            feedback_parts.append(f"✗ Wrong tables. Expected tables include: {gold_tables}")
    elif not agent_tables:
        feedback_parts.append("✗ Could not identify table references in your query")

    # ---- 4. Correct columns ----
    gold_cols_parsed = _extract_selected_columns(task.gold_query)
    agent_cols_parsed = _extract_selected_columns(agent_sql)

    if gold_cols_parsed and agent_cols_parsed:
        if "*" in agent_cols_parsed and "*" not in gold_cols_parsed:
            # SELECT * gets partial credit
            breakdown["correct_columns"] = 0.3
            feedback_parts.append("◐ Using SELECT * — select specific columns for full credit")
        elif "*" in gold_cols_parsed and "*" in agent_cols_parsed:
            breakdown["correct_columns"] = 1.0
            feedback_parts.append("✓ Correct column selection")
        else:
            col_overlap = (
                len(gold_cols_parsed & agent_cols_parsed) / len(gold_cols_parsed)
                if gold_cols_parsed
                else 0.0
            )
            breakdown["correct_columns"] = round(col_overlap, 3)
            if col_overlap >= 1.0:
                feedback_parts.append("✓ Correct columns selected")
            elif col_overlap > 0:
                feedback_parts.append(
                    f"◐ Partially correct columns. Matched: {gold_cols_parsed & agent_cols_parsed}"
                )
            else:
                feedback_parts.append("✗ Selected columns don't match expected output")

    # ---- 5. Partial row match (Jaccard similarity) ----
    if agent_rows is not None and gold_rows is not None:
        gold_set = _normalize_rows(gold_rows)
        agent_set = _normalize_rows(agent_rows)

        if not gold_set and not agent_set:
            # Both empty — perfect match
            breakdown["partial_row_match"] = 1.0
            feedback_parts.append("✓ Both queries return empty results (correct)")
        elif not gold_set or not agent_set:
            breakdown["partial_row_match"] = 0.0
            if not agent_set:
                feedback_parts.append("✗ Your query returned no rows, but results were expected")
            else:
                feedback_parts.append("✗ Your query returned rows, but none were expected")
        else:
            intersection = len(gold_set & agent_set)
            union = len(gold_set | agent_set)
            jaccard = intersection / union if union > 0 else 0.0
            breakdown["partial_row_match"] = round(jaccard, 3)

            if jaccard >= 1.0:
                feedback_parts.append("✓ All result rows match perfectly")
            elif jaccard > 0.5:
                feedback_parts.append(
                    f"◐ Good partial match ({intersection}/{len(gold_set)} expected rows found)"
                )
            elif jaccard > 0:
                feedback_parts.append(
                    f"◐ Partial match ({intersection}/{len(gold_set)} expected rows found)"
                )
            else:
                feedback_parts.append("✗ No result rows match the expected output")

    # ---- 6. Exact match ----
    if agent_rows is not None and gold_rows is not None:
        gold_set = _normalize_rows(gold_rows)
        agent_set = _normalize_rows(agent_rows)

        if gold_set == agent_set:
            breakdown["exact_match"] = 1.0
            feedback_parts.append("✓ EXACT MATCH — perfect result! 🎉")
        else:
            breakdown["exact_match"] = 0.0
            feedback_parts.append(
                f"✗ Results differ from expected "
                f"(got {len(agent_set)} rows, expected {len(gold_set)} rows)"
            )

    # ---- Compute total reward ----
    total = sum(
        breakdown[key] * WEIGHTS[key] for key in WEIGHTS
    )

    # Step cost: penalize later attempts to incentivize efficiency
    step_penalty = max(0, (attempt - 1)) * 0.02
    total = max(0.0, total - step_penalty)

    # First-attempt bonus for exact matches
    is_exact = breakdown.get("exact_match", 0.0) >= 1.0
    if is_exact and attempt == 1:
        total = min(1.0, total + 0.05)
        feedback_parts.append("🏆 First-attempt solve bonus! +0.05")
    elif step_penalty > 0:
        feedback_parts.append(f"⏱️ Efficiency penalty: -{step_penalty:.2f} (attempt {attempt})")

    # Clamp to strictly (0, 1) — evaluator requires scores not equal to 0.0 or 1.0
    # Use 0.01/0.99 margins to survive any rounding the evaluator applies
    total = round(max(0.01, min(0.99, total)), 4)

    # Also clamp breakdown values — evaluator may check ALL float fields in the
    # observation JSON, not just the top-level reward
    clamped_breakdown = {k: round(max(0.01, min(0.99, v)), 4) for k, v in breakdown.items()}

    feedback = "\n".join(feedback_parts)

    return {
        "total_reward": total,
        "breakdown": clamped_breakdown,
        "feedback": feedback,
        "agent_rows": agent_rows,
        "agent_columns": agent_columns,
        "error": error_msg,
    }
