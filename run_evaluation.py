"""Comprehensive evaluation harness for the SQL Query Environment.

Runs the gold-standard queries for all tasks to verify the grader
gives them a perfect 1.0 score, then runs the baseline.

Usage:
    python run_evaluation.py
"""

import sys
sys.path.insert(0, ".")

from sql_env.server.sql_environment import SqlEnvironment
from sql_env.models import SqlAction
from sql_env.tasks import TASKS, get_all_task_ids


def run_gold_evaluation():
    """Verify that gold queries score 1.0 on every task."""
    print("\n=== Gold Query Verification ===\n")
    env = SqlEnvironment(max_attempts=1)
    all_pass = True

    for task_id in sorted(get_all_task_ids()):
        task = TASKS[task_id]
        obs = env.reset(task_id=task_id, seed=0)

        # Submit the gold query
        action = SqlAction(sql_query=task.gold_query)
        obs = env.step(action)

        status = "✓ PASS" if obs.reward >= 0.99 else "✗ FAIL"
        if obs.reward < 0.99:
            all_pass = False

        print(f"  {task_id:<15} {task.difficulty:<10} reward={obs.reward:.4f}  {status}")
        if obs.reward < 0.99:
            print(f"    Breakdown: {obs.reward_breakdown}")
            print(f"    Feedback: {obs.feedback[:200]}")

    env.close()
    print(f"\n  {'ALL GOLD QUERIES PASS ✓' if all_pass else 'SOME GOLD QUERIES FAILED ✗'}\n")
    return all_pass


def run_partial_credit_demo():
    """Demonstrate that partially correct queries get partial credit."""
    print("\n=== Partial Credit Demonstration ===\n")
    env = SqlEnvironment(max_attempts=3)

    # Task: "List all employees in the Sales department"
    obs = env.reset(task_id="easy_01")
    print(f"  Task: {obs.question}")
    print()

    # Attempt 1: Wrong query entirely
    action1 = SqlAction(sql_query="SELECT 1;")
    obs1 = env.step(action1)
    print(f"  Attempt 1: 'SELECT 1;'")
    print(f"    Reward: {obs1.reward:.4f}  (syntax + execution credit only)")
    print(f"    Breakdown: {obs1.reward_breakdown}")
    print()

    # Attempt 2: Right table but wrong columns
    action2 = SqlAction(sql_query="SELECT * FROM employees;")
    obs2 = env.step(action2)
    print(f"  Attempt 2: 'SELECT * FROM employees;'")
    print(f"    Reward: {obs2.reward:.4f}  (partial table + column credit)")
    print(f"    Breakdown: {obs2.reward_breakdown}")
    print()

    # Attempt 3: Correct query
    action3 = SqlAction(sql_query="""
        SELECT e.first_name, e.last_name, e.salary
        FROM employees e
        JOIN departments d ON e.department_id = d.id
        WHERE d.name = 'Sales'
        ORDER BY e.last_name;
    """)
    obs3 = env.step(action3)
    print(f"  Attempt 3: correct query")
    print(f"    Reward: {obs3.reward:.4f}  (perfect score!)")
    print(f"    Breakdown: {obs3.reward_breakdown}")
    print(f"    Done: {obs3.done}, Solved: {obs3.metadata.get('is_solved')}")

    env.close()


def main():
    gold_pass = run_gold_evaluation()
    run_partial_credit_demo()

    if not gold_pass:
        print("ERROR: Gold query verification failed!")
        sys.exit(1)

    print("\n✅ All evaluations passed.\n")


if __name__ == "__main__":
    main()
