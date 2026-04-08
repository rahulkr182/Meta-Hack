"""SQL Query Environment — Server-side implementation.

Implements the OpenEnv Environment interface (reset/step/state) for the
SQL query writing task. Manages an in-memory SQLite database per episode
and uses the grader for scoring.
"""

from typing import Any, Dict, Optional
from uuid import uuid4

from sql_env.db_utils import create_database, execute_query, format_results
from sql_env.grader import grade_query
from sql_env.models import SqlAction, SqlObservation, SqlState
from sql_env.tasks import Task, get_random_task, get_task, TASKS


class SqlEnvironment:
    """OpenEnv-compliant SQL query writing environment.

    The agent receives a database schema and a natural-language question,
    then submits SQL queries. The environment executes them against an
    in-memory SQLite database and returns graded feedback with partial
    reward signals.

    Lifecycle:
        1. reset() — picks a task, creates a fresh DB, returns schema + question
        2. step(action) — executes the agent's SQL, grades it, returns feedback
        3. state — returns the current episode state

    The episode ends when:
        - The agent achieves a perfect score (exact match), or
        - The agent exhausts all attempts (default: 3)
    """

    def __init__(self, max_attempts: int = 3):
        """Initialize the environment.

        Args:
            max_attempts: Maximum number of attempts per task.
        """
        self._max_attempts = max_attempts
        self._state = SqlState()
        self._current_task: Optional[Task] = None
        self._db_conn = None
        self._last_observation: Optional[SqlObservation] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs: Any,
    ) -> SqlObservation:
        """Reset the environment to a new episode.

        Args:
            seed: Random seed for task selection.
            episode_id: Custom episode ID (auto-generated if not provided).
            task_id: Specific task ID to use (overrides difficulty).
            difficulty: Filter task selection by difficulty tier.
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

        # Build initial observation
        obs = SqlObservation(
            schema_description=self._current_task.schema_sql.strip(),
            question=self._current_task.question,
            task_id=self._current_task.task_id,
            task_difficulty=self._current_task.difficulty,
            execution_result=None,
            execution_error=None,
            feedback=f"Task: {self._current_task.question}\n"
                     f"Difficulty: {self._current_task.difficulty}\n"
                     f"Hint: {self._current_task.hint}\n"
                     f"You have {self._max_attempts} attempts. Submit a SQL query.",
            reward_breakdown={},
            done=False,
            reward=0.0,
            metadata={
                "task_id": self._current_task.task_id,
                "difficulty": self._current_task.difficulty,
                "max_attempts": self._max_attempts,
            },
        )
        self._last_observation = obs
        return obs

    def step(
        self,
        action: SqlAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SqlObservation:
        """Execute the agent's SQL query and return graded feedback.

        Args:
            action: SqlAction containing the agent's SQL query.
            timeout_s: Optional timeout (unused, SQLite is fast).
            **kwargs: Additional arguments.

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
            # Show the gold query on final failure
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
