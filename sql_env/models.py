"""
Data models for the SQL Query Environment.

The sql_query_env environment provides SQL query writing tasks against
in-memory SQLite databases. The agent submits SQL queries and receives
graded feedback with partial-progress reward signals.
"""

from typing import Any, Dict, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class SqlAction(Action):
    """Action submitted by the agent: a SQL query string."""

    sql_query: str = Field(
        ...,
        description="The SQL query the agent wants to execute against the database.",
        min_length=1,
    )


class SqlObservation(Observation):
    """Observation returned to the agent after reset() or step()."""

    # Task context (always present)
    schema_description: str = Field(
        default="",
        description="Human-readable database schema (CREATE TABLE statements).",
    )
    question: str = Field(
        default="",
        description="Natural-language question the agent must answer with SQL.",
    )
    task_id: str = Field(default="", description="ID of the current task.")
    task_difficulty: str = Field(
        default="easy",
        description="Difficulty tier: 'easy', 'medium', 'hard', or 'expert'.",
    )

    # Step feedback (populated after step(), empty after reset())
    execution_result: Optional[str] = Field(
        default=None,
        description="Stringified result of executing the agent's SQL.",
    )
    execution_error: Optional[str] = Field(
        default=None,
        description="Error message if the SQL failed to execute.",
    )
    feedback: str = Field(
        default="",
        description="Human-readable grader feedback.",
    )
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown from the grader.",
    )


class SqlState(State):
    """Internal state of the SQL environment episode."""

    current_task_id: str = Field(default="", description="Current task ID.")
    task_difficulty: str = Field(default="easy", description="Current task difficulty.")
    attempts: int = Field(default=0, ge=0, description="Attempts used on this task.")
    max_attempts: int = Field(
        default=3, ge=1, description="Max attempts allowed per task."
    )
    accumulated_reward: float = Field(
        default=0.0, description="Sum of rewards earned in this episode."
    )
    best_reward: float = Field(
        default=0.0, description="Best single-step reward in this episode."
    )
    is_solved: bool = Field(
        default=False, description="Whether the agent achieved a perfect score."
    )
