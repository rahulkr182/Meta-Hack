"""
SQL Query Environment Implementation.

Manages an in-memory SQLite database per episode. The agent submits SQL queries,
which are executed and graded against gold-standard answers.
"""

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from sql_env.db_utils import create_database, execute_query, format_results
from sql_env.grader import grade_query
from sql_env.tasks import Task, get_random_task, get_task, TASKS

try:
    from models import SqlAction, SqlObservation, SqlState
except ImportError:
    from ..models import SqlAction, SqlObservation, SqlState


class SqlEnvironment(Environment):
    """
    OpenEnv-compliant SQL query writing environment.

    Each episode presents a database schema and a natural-language question.
    The agent submits SQL queries which are executed against an in-memory
    SQLite database. Graded feedback with partial reward signals is returned.

    Design:
    - Multi-step episodes: reset() provides task, step() validates SQL
    - Episode ends on: exact match OR max attempts exhausted
    - Blind mode for expert tasks: schema is hidden

    Example:
        >>> env = SqlEnvironment()
        >>> obs = env.reset(task_id='easy_01')
        >>> print(obs.question)
        >>> obs = env.step(SqlAction(sql_query="SELECT ..."))
        >>> print(obs.reward, obs.done)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, max_attempts: int = 3):
        """Initialize the environment.

        Args:
            max_attempts: Maximum number of attempts per task.
        """
        self._max_attempts = max_attempts
        self._state = SqlState(episode_id=str(uuid4()), step_count=0)
        self._current_task: Optional[Task] = None
        self._db_conn = None
        self._last_observation: Optional[SqlObservation] = None
        self._blind_mode: bool = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        blind_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> SqlObservation:
        """Reset the environment to a new episode.

        Args:
            seed: Random seed for task selection.
            episode_id: Custom episode ID (auto-generated if not provided).
            task_id: Specific task ID to use (overrides difficulty).
            difficulty: Filter task selection by difficulty tier.
            blind_mode: If True, hide schema from agent.
            **kwargs: Additional reset parameters.

        Returns:
            Initial observation with schema and question.
        """
        # Clean up previous episode
        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None

        # Select task
        if task_id:
            self._current_task = get_task(task_id)
        else:
            self._current_task = get_random_task(difficulty=difficulty, seed=seed)

        # Create fresh in-memory database
        self._db_conn = create_database(
            self._current_task.schema_sql,
            self._current_task.seed_sql,
        )

        # Blind mode: default to blind for expert tasks
        if blind_mode is not None:
            self._blind_mode = blind_mode
        else:
            self._blind_mode = self._current_task.difficulty == "expert"

        # Initialize state
        self._state = SqlState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_task_id=self._current_task.task_id,
            task_difficulty=self._current_task.difficulty,
            attempts=0,
            max_attempts=self._max_attempts,
            accumulated_reward=0.0,
            best_reward=0.0,
            is_solved=False,
        )

        # Schema description: hidden in blind mode
        if self._blind_mode:
            schema_desc = (
                "[BLIND MODE] Schema is not provided. "
                "Use queries like:\n"
                "  SELECT name, sql FROM sqlite_master WHERE type='table';\n"
                "to discover the database structure before answering."
            )
        else:
            schema_desc = self._current_task.schema_sql.strip()

        # Build initial observation
        obs = SqlObservation(
            schema_description=schema_desc,
            question=self._current_task.question,
            task_id=self._current_task.task_id,
            task_difficulty=self._current_task.difficulty,
            execution_result=None,
            execution_error=None,
            feedback=f"Task: {self._current_task.question}\n"
                     f"Difficulty: {self._current_task.difficulty}\n"
                     f"{'[BLIND MODE] Discover the schema first!' if self._blind_mode else 'Hint: ' + self._current_task.hint}\n"
                     f"You have {self._max_attempts} attempts. Submit a SQL query.",
            reward_breakdown={},
            done=False,
            reward=0.0,
            metadata={
                "task_id": self._current_task.task_id,
                "difficulty": self._current_task.difficulty,
                "max_attempts": self._max_attempts,
                "blind_mode": self._blind_mode,
            },
        )
        self._last_observation = obs
        return obs

    def step(self, action: SqlAction) -> SqlObservation:  # type: ignore[override]
        """Execute the agent's SQL query and return graded feedback.

        Args:
            action: SqlAction containing the agent's SQL query.

        Returns:
            Observation with execution results, reward, and feedback.
        """
        if self._current_task is None or self._db_conn is None:
            return SqlObservation(
                feedback="ERROR: Environment not initialized. Call reset() first.",
                done=True,
                reward=0.0,
            )

        if self._state.is_solved:
            return SqlObservation(
                schema_description=self._current_task.schema_sql.strip(),
                question=self._current_task.question,
                task_id=self._current_task.task_id,
                task_difficulty=self._current_task.difficulty,
                feedback="Task already solved! Call reset() to try a new task.",
                done=True,
                reward=0.0,
            )

        # Increment counters
        self._state.step_count += 1
        self._state.attempts += 1

        # Grade the query
        grade_result = grade_query(
            agent_sql=action.sql_query,
            task=self._current_task,
            conn=self._db_conn,
            attempt=self._state.attempts,
        )

        reward = grade_result["total_reward"]
        self._state.accumulated_reward += reward
        self._state.best_reward = max(self._state.best_reward, reward)

        # Format the agent's results for display
        exec_result_str = None
        exec_error_str = grade_result.get("error")
        if grade_result["agent_rows"] is not None:
            exec_result_str = format_results(
                grade_result["agent_rows"],
                grade_result["agent_columns"],
            )

        # Check if done
        is_exact = grade_result["breakdown"].get("exact_match", 0.0) >= 1.0
        attempts_exhausted = self._state.attempts >= self._state.max_attempts

        if is_exact:
            self._state.is_solved = True

        done = is_exact or attempts_exhausted

        # Build feedback
        feedback_lines = [grade_result["feedback"]]
        if not done:
            remaining = self._state.max_attempts - self._state.attempts
            feedback_lines.append(
                f"\nAttempts remaining: {remaining}/{self._state.max_attempts}"
            )
        elif is_exact:
            feedback_lines.append(f"\n🎉 Solved in {self._state.attempts} attempt(s)!")
        else:
            feedback_lines.append(
                f"\n❌ Out of attempts. Best reward: {self._state.best_reward:.4f}"
            )
            feedback_lines.append(
                f"\nCorrect query was:\n{self._current_task.gold_query.strip()}"
            )

        obs = SqlObservation(
            schema_description=self._current_task.schema_sql.strip(),
            question=self._current_task.question,
            task_id=self._current_task.task_id,
            task_difficulty=self._current_task.difficulty,
            execution_result=exec_result_str,
            execution_error=exec_error_str,
            feedback="\n".join(feedback_lines),
            reward_breakdown=grade_result["breakdown"],
            done=done,
            reward=reward,
            metadata={
                "attempt": self._state.attempts,
                "max_attempts": self._state.max_attempts,
                "best_reward": self._state.best_reward,
                "accumulated_reward": self._state.accumulated_reward,
                "is_solved": self._state.is_solved,
            },
        )
        self._last_observation = obs
        return obs

    @property
    def state(self) -> SqlState:
        """Get the current environment state."""
        return self._state

    def close(self) -> None:
        """Clean up resources."""
        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None
