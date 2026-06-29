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
    "correct_tables": 0.10,
    "correct_columns": 0.10,
    "correct_column_order": 0.05,
    "partial_row_match": 0.15,
    "correct_row_order": 0.10,
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
    "lead", "lag", "ntile", "cume_dist", "percent_rank", "first_value",
    "last_value", "nth_value", "with", "recursive", "true", "false", 
    "primary", "key", "foreign", "references", "default", "integer",
    "text", "real", "blob", "numeric", "boolean", "date", "datetime",
}


def _extract_table_references(sql: str) -> Set[str]:
    """Extract table names referenced in a SQL query (best-effort)."""
    sql_clean = re.sub(r"'[^']*'", "", sql)  # Remove string literals
    sql_clean = re.sub(r"--[^\n]*", "", sql_clean)  # Remove line comments
    sql_clean = sql_clean.lower()

    tables = set()

    # Identify CTE names to exclude them from actual table references
    cte_names = set()
    cte_matches = re.finditer(r'\b(?:with|with\s+recursive)\s+(\w+)\s+as\s*\(', sql_clean)
    for m in cte_matches:
        cte_names.add(m.group(1))
    
    # Also catch subsequent CTEs in a comma-separated list
    # e.g., WITH cte1 AS (...), cte2 AS (...)
    cte_list_matches = re.finditer(r'\bwith\s+.*?\)\s*,\s*(\w+)\s+as\s*\(', sql_clean, re.DOTALL)
    for m in cte_list_matches:
         cte_names.add(m.group(1))

    # Match FROM, JOIN, and UPDATE clauses
    patterns = [
        r'\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bupdate\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\binto\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, sql_clean):
            name = match.group(1)
            if name not in SQL_KEYWORDS and name not in cte_names:
                tables.add(name)

    return tables


def _extract_selected_columns(sql: str) -> List[str]:
    """Extract column names from the SELECT clause (best-effort), maintaining order."""
    sql_clean = re.sub(r"'[^']*'", "", sql)
    sql_clean = re.sub(r"--[^\n]*", "", sql_clean)
    sql_clean = sql_clean.lower()

    # Try to extract the outer SELECT statement, robust to subqueries and CTEs
    # A simple but effective heuristic is to find the LAST SELECT ... FROM block
    # or handle unions by extracting from all SELECT ... FROM segments that are not in parens.
    
    # Remove nested parens contents to simplify outer select extraction
    prev_len = -1
    while len(sql_clean) != prev_len:
        prev_len = len(sql_clean)
        sql_clean = re.sub(r'\([^()]*\)', '()', sql_clean)

    columns = []
    
    # Find all SELECT ... FROM that remain in the un-nested structure
    matches = list(re.finditer(r'\bselect\s+(.*?)\bfrom\b', sql_clean, re.DOTALL))
    
    if not matches:
        # Fallback if FROM is missing
        match = re.search(r'\bselect\s+(.*)', sql_clean, re.DOTALL)
        if match:
             matches = [match]
        else:
            return []

    # Process all selected blocks (helps with UNIONs)
    for match in matches:
        select_clause = match.group(1)

        if "*" in select_clause:
            columns.append("*")
            continue

        # Split by comma and extract the column name (after dots, before AS)
        for part in select_clause.split(","):
            part = part.strip()
            if not part or part == '()': 
                continue
                
            # Handle "alias" in "expr AS alias"
            as_match = re.search(r'\bas\s+(\w+)\s*$', part)
            if as_match:
                columns.append(as_match.group(1))
            else:
                # Handle "table.column" or just "column"
                # e.g. "e.first_name" -> "first_name"
                col_match = re.search(r'(?:[a-zA-Z0-9_]+\.)?([a-zA-Z0-9_]+)\s*$', part)
                if col_match:
                    name = col_match.group(1)
                    if name not in SQL_KEYWORDS:
                        columns.append(name)
                        
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
        try:
            conn.execute(f"EXPLAIN {sql_stripped}")
        except sqlite3.Error:
             # Fallback to complete_statement or prepare if EXPLAIN fails on valid complex queries
             if not sqlite3.complete_statement(sql_stripped + ";"):
                  raise
             # Also try dry-run prepare if possible, but EXPLAIN is usually best in SQLite
             
        breakdown["syntax_valid"] = 1.0
        feedback_parts.append("✓ SQL syntax is valid")
    except sqlite3.Error as e:
        breakdown["syntax_valid"] = 0.0
        feedback_parts.append(f"✗ SQL syntax error: {str(e)}")
        # If syntax is invalid, we can still try to check table/column names
        error_msg = f"Syntax error: {str(e)}"

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

    # ---- 4. Correct columns & Column Order ----
    gold_cols_list = _extract_selected_columns(task.gold_query)
    agent_cols_list = _extract_selected_columns(agent_sql)
    
    gold_cols_parsed = set(gold_cols_list)
    agent_cols_parsed = set(agent_cols_list)

    if gold_cols_parsed and agent_cols_parsed:
        if "*" in agent_cols_parsed and "*" not in gold_cols_parsed:
            # SELECT * gets partial credit
            breakdown["correct_columns"] = 0.3
            feedback_parts.append("◐ Using SELECT * — select specific columns for full credit")
        elif "*" in gold_cols_parsed and "*" in agent_cols_parsed:
            breakdown["correct_columns"] = 1.0
            breakdown["correct_column_order"] = 1.0
            feedback_parts.append("✓ Correct column selection")
        else:
            # Semantic matching for aliases
            normalized_gold = {c.replace("emp_count", "employee_count").replace("avg_sal", "avg_salary") for c in gold_cols_parsed}
            normalized_agent = {c.replace("emp_count", "employee_count").replace("avg_sal", "avg_salary") for c in agent_cols_parsed}
            
            col_overlap = (
                len(normalized_gold & normalized_agent) / len(normalized_gold)
                if normalized_gold
                else 0.0
            )
            breakdown["correct_columns"] = round(col_overlap, 3)
            
            # Check column order if exact set matches
            if col_overlap >= 1.0:
                 feedback_parts.append("✓ Correct columns selected")
                 # Order check
                 if len(gold_cols_list) == len(agent_cols_list) and all(
                      g.replace("emp_count", "employee_count").replace("avg_sal", "avg_salary") == 
                      a.replace("emp_count", "employee_count").replace("avg_sal", "avg_salary")
                      for g, a in zip(gold_cols_list, agent_cols_list)
                 ):
                      breakdown["correct_column_order"] = 1.0
                 else:
                      breakdown["correct_column_order"] = 0.5
                      feedback_parts.append("◐ Columns are correct but out of order")
            elif col_overlap > 0:
                feedback_parts.append(
                    f"◐ Partially correct columns. Matched: {gold_cols_parsed & agent_cols_parsed}"
                )
            else:
                feedback_parts.append("✗ Selected columns don't match expected output")

    # ---- 5. Partial row match (Jaccard similarity) & Row Order ----
    if agent_rows is not None and gold_rows is not None:
        gold_set = _normalize_rows(gold_rows)
        agent_set = _normalize_rows(agent_rows)
        
        # Row Order Match
        gold_norm_list = [tuple(str(v).strip().lower() for v in r) for r in gold_rows]
        agent_norm_list = [tuple(str(v).strip().lower() for v in r) for r in agent_rows]

        if not gold_set and not agent_set:
            # Both empty — perfect match
            breakdown["partial_row_match"] = 1.0
            breakdown["correct_row_order"] = 1.0
            feedback_parts.append("✓ Both queries return empty results (correct)")
        elif not gold_set or not agent_set:
            breakdown["partial_row_match"] = 0.0
            breakdown["correct_row_order"] = 0.0
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
                feedback_parts.append("✓ All result rows match perfectly (unordered)")
                
                # Check Row Order
                if gold_norm_list == agent_norm_list:
                     breakdown["correct_row_order"] = 1.0
                     feedback_parts.append("✓ Result row order is strictly correct")
                else:
                     breakdown["correct_row_order"] = 0.0
                     feedback_parts.append("✗ Result row order differs from expected (check ORDER BY)")
                     
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
