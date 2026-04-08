"""Tests for the SQL Query Environment."""

import sys
sys.path.insert(0, ".")

import pytest
from sql_env.models import SqlAction, SqlObservation, SqlState
from sql_env.tasks import TASKS, get_task, get_tasks_by_difficulty, get_all_task_ids
from sql_env.db_utils import create_database, execute_query, format_results
from sql_env.grader import grade_query
from sql_env.server.sql_environment import SqlEnvironment


# ---------------------------------------------------------------------------
# Task Registry Tests
# ---------------------------------------------------------------------------

class TestTasks:
    def test_all_tasks_registered(self):
        """All 9 tasks should be registered."""
        assert len(TASKS) == 9

    def test_difficulty_distribution(self):
        """3 tasks per difficulty tier."""
        for diff in ("easy", "medium", "hard"):
            tasks = get_tasks_by_difficulty(diff)
            assert len(tasks) == 3, f"Expected 3 {diff} tasks, got {len(tasks)}"

    def test_get_task_by_id(self):
        task = get_task("easy_01")
        assert task.task_id == "easy_01"
        assert task.difficulty == "easy"

    def test_get_task_unknown_raises(self):
        with pytest.raises(KeyError):
            get_task("nonexistent")

    def test_gold_queries_execute(self):
        """All gold queries should execute without errors."""
        for task_id, task in TASKS.items():
            conn = create_database(task.schema_sql, task.seed_sql)
            rows, cols, err = execute_query(conn, task.gold_query)
            conn.close()
            assert err is None, f"Task {task_id} gold query failed: {err}"
            assert rows is not None
            assert len(rows) > 0 or task_id in [], f"Task {task_id} returned no rows"


# ---------------------------------------------------------------------------
# DB Utils Tests
# ---------------------------------------------------------------------------

class TestDbUtils:
    def test_create_database(self):
        conn = create_database(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT);",
            "INSERT INTO test (id, val) VALUES (1, 'hello');",
        )
        rows, cols, err = execute_query(conn, "SELECT * FROM test;")
        conn.close()
        assert err is None
        assert rows == [(1, "hello")]
        assert cols == ["id", "val"]

    def test_write_operations_blocked(self):
        conn = create_database("CREATE TABLE t (id INT);", "")
        _, _, err = execute_query(conn, "INSERT INTO t VALUES (1);")
        conn.close()
        assert err is not None
        assert "Write operations" in err

    def test_format_results(self):
        result = format_results([(1, "alice"), (2, "bob")], ["id", "name"])
        assert "alice" in result
        assert "bob" in result
        assert "2 rows total" in result

    def test_format_empty(self):
        result = format_results([], ["id"])
        assert "0 rows" in result


# ---------------------------------------------------------------------------
# Grader Tests
# ---------------------------------------------------------------------------

class TestGrader:
    def test_perfect_score(self):
        """Gold query should get 1.0 reward."""
        task = get_task("easy_01")
        conn = create_database(task.schema_sql, task.seed_sql)
        result = grade_query(task.gold_query, task, conn)
        conn.close()
        assert result["total_reward"] >= 0.99, (
            f"Gold query got {result['total_reward']}: {result['breakdown']}"
        )

    def test_syntax_error_gets_partial(self):
        """A query with syntax errors should still get 0 but not crash."""
        task = get_task("easy_01")
        conn = create_database(task.schema_sql, task.seed_sql)
        result = grade_query("SELCT * FORM employees;", task, conn)
        conn.close()
        assert result["total_reward"] < 0.5
        assert result["total_reward"] >= 0.0

    def test_wrong_query_partial_credit(self):
        """SELECT * FROM employees should get partial credit (right table)."""
        task = get_task("easy_01")
        conn = create_database(task.schema_sql, task.seed_sql)
        result = grade_query("SELECT * FROM employees;", task, conn)
        conn.close()
        # Should get syntax + execution + partial table credit
        assert result["total_reward"] > 0.2
        assert result["total_reward"] < 1.0
        assert result["breakdown"]["syntax_valid"] == 1.0
        assert result["breakdown"]["execution_success"] == 1.0

    def test_all_gold_queries_score_high(self):
        """Every gold query should score >= 0.99."""
        for task_id, task in TASKS.items():
            conn = create_database(task.schema_sql, task.seed_sql)
            result = grade_query(task.gold_query, task, conn)
            conn.close()
            assert result["total_reward"] >= 0.99, (
                f"Task {task_id}: gold got {result['total_reward']}\n"
                f"Breakdown: {result['breakdown']}"
            )


# ---------------------------------------------------------------------------
# Environment Tests
# ---------------------------------------------------------------------------

class TestEnvironment:
    def test_reset_returns_observation(self):
        env = SqlEnvironment()
        obs = env.reset(task_id="easy_01")
        assert isinstance(obs, SqlObservation)
        assert obs.question != ""
        assert obs.schema_description != ""
        assert obs.done is False
        env.close()

    def test_step_returns_reward(self):
        env = SqlEnvironment()
        env.reset(task_id="easy_01")
        obs = env.step(SqlAction(sql_query="SELECT 1;"))
        assert obs.reward is not None
        assert 0.0 <= obs.reward <= 1.0
        env.close()

    def test_episode_ends_on_perfect_score(self):
        env = SqlEnvironment()
        env.reset(task_id="easy_01")
        task = TASKS["easy_01"]
        obs = env.step(SqlAction(sql_query=task.gold_query))
        assert obs.done is True
        assert obs.reward >= 0.99
        env.close()

    def test_episode_ends_after_max_attempts(self):
        env = SqlEnvironment(max_attempts=2)
        env.reset(task_id="easy_01")
        env.step(SqlAction(sql_query="SELECT 1;"))
        obs = env.step(SqlAction(sql_query="SELECT 2;"))
        assert obs.done is True
        env.close()

    def test_state_tracking(self):
        env = SqlEnvironment()
        env.reset(task_id="easy_01")
        assert env.state.step_count == 0
        assert env.state.attempts == 0
        env.step(SqlAction(sql_query="SELECT 1;"))
        assert env.state.step_count == 1
        assert env.state.attempts == 1
        env.close()

    def test_multiple_episodes(self):
        env = SqlEnvironment()
        # Episode 1
        obs1 = env.reset(task_id="easy_01")
        ep1_id = env.state.episode_id
        env.step(SqlAction(sql_query="SELECT 1;"))

        # Episode 2 (should be fresh)
        obs2 = env.reset(task_id="easy_02")
        ep2_id = env.state.episode_id
        assert ep1_id != ep2_id
        assert env.state.step_count == 0
        env.close()


# ---------------------------------------------------------------------------
# Model Validation Tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_action_validates_query(self):
        action = SqlAction(sql_query="SELECT 1;")
        assert action.sql_query == "SELECT 1;"

    def test_action_rejects_empty(self):
        with pytest.raises(Exception):
            SqlAction(sql_query="")

    def test_observation_defaults(self):
        obs = SqlObservation()
        assert obs.done is False
        assert obs.reward is None

    def test_state_defaults(self):
        state = SqlState()
        assert state.step_count == 0
        assert state.attempts == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
