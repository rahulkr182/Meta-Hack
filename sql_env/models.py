"""Typed models for the SQL Query Environment.

Defines the Action, Observation, and State types used by the environment,
client, and grader. All models use Pydantic for validation and serialization.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SqlAction(BaseModel):
    """Action submitted by the agent: a SQL query string."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    sql_query: str = Field(
        ...,
        description="The SQL query the agent wants to execute against the database.",
        min_length=1,
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata for the action.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SqlObservation(BaseModel):
    """Observation returned to the agent after reset() or step()."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

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
        description="Difficulty tier: 'easy', 'medium', or 'hard'.",
    )

    # Step feedback (populated after step(), empty after reset())
    execution_result: Optional[str] = Field(
        default=None,
        description="Stringified result of executing the agent's SQL, or the error message.",
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

    # Standard OpenEnv fields
    done: bool = Field(default=False, description="Whether the episode has ended.")
    reward: Optional[float] = Field(
        default=None, description="Reward for this step (0.0–1.0)."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata.",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SqlState(BaseModel):
    """Internal state of the SQL environment episode."""

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
    )

    episode_id: Optional[str] = Field(
        default=None, description="Unique identifier for this episode."
    )
    step_count: int = Field(default=0, ge=0, description="Steps taken so far.")
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
