#!/usr/bin/env python3
"""Baseline Inference Script for the SQL Query Environment.

Uses the OpenAI API client to run a model against the environment.
Reads API credentials from environment variables (OPENAI_API_KEY).
Produces a reproducible baseline score on all tasks.

Usage:
    export OPENAI_API_KEY="sk-..."
    python3 baseline_inference.py

    # Or with a different model:
    python3 baseline_inference.py --model gpt-4o-mini

    # Dry run (no API needed, uses gold queries as upper bound):
    python3 baseline_inference.py --dry-run
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, ".")

from sql_env.server.sql_environment import SqlEnvironment
from sql_env.models import SqlAction
from sql_env.tasks import TASKS, get_all_task_ids


# ---------------------------------------------------------------------------
# OpenAI API caller
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert SQL query writer. You are given a database schema and a natural-language question.
Write the correct SQL query that answers the question.

Rules:
- Output ONLY the SQL query, nothing else
- Do NOT include markdown code fences
- Do NOT include explanations
- Use standard SQLite-compatible SQL
- Use appropriate JOINs, GROUP BY, HAVING, window functions as needed
- Return results sorted logically when applicable"""


def call_openai(
    schema: str,
    question: str,
    hint: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    attempt: int = 1,
    previous_feedback: str = "",
) -> str:
    """Call the OpenAI Chat Completions API to generate a SQL query.

    Args:
        schema: Database schema (CREATE TABLE statements).
        question: Natural-language question to answer.
        hint: Hint about SQL concepts needed.
        api_key: OpenAI API key.
        model: Model name.
        attempt: Current attempt number.
        previous_feedback: Feedback from previous attempt (for self-correction).

    Returns:
        Generated SQL query string.
    """
    import httpx

    user_msg = f"""Database Schema:
{schema}

Question: {question}

Hint: {hint}"""

    if attempt > 1 and previous_feedback:
        user_msg += f"""

Previous attempt feedback:
{previous_feedback}

Please fix your query based on this feedback."""

    user_msg += "\n\nSQL query:"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if present
    if content.startswith("```sql"):
        content = content[6:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    return content.strip()


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------

def run_baseline(
    api_key: str,
    model: str,
    max_attempts: int = 3,
    dry_run: bool = False,
) -> Dict[str, Dict]:
    """Run the baseline agent (OpenAI model) on all tasks.

    Each task allows up to max_attempts. After a failed attempt, the grader
    feedback is fed back into the model prompt for self-correction.

    Args:
        api_key: OpenAI API key.
        model: Model name.
        max_attempts: Max attempts per task.
        dry_run: If True, use gold queries instead of calling the API.

    Returns:
        Dict mapping task_id -> result info.
    """
    env = SqlEnvironment(max_attempts=max_attempts)
    results = {}

    for task_id in sorted(get_all_task_ids()):
        task = TASKS[task_id]
        obs = env.reset(task_id=task_id, seed=42)

        best_reward = 0.0
        best_query = ""
        solved = False
        previous_feedback = ""
        attempts_used = 0

        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt

            if dry_run:
                sql_query = task.gold_query
            else:
                try:
                    sql_query = call_openai(
                        schema=obs.schema_description,
                        question=obs.question,
                        hint=task.hint,
                        api_key=api_key,
                        model=model,
                        attempt=attempt,
                        previous_feedback=previous_feedback,
                    )
                except Exception as e:
                    print(f"  ⚠ API error on {task_id} attempt {attempt}: {e}")
                    sql_query = "SELECT 1;"

            # Submit to environment
            obs = env.step(SqlAction(sql_query=sql_query))

            if obs.reward > best_reward:
                best_reward = obs.reward
                best_query = sql_query

            if obs.done:
                solved = obs.reward_breakdown.get("exact_match", 0.0) >= 1.0
                break

            previous_feedback = obs.feedback
            if not dry_run:
                time.sleep(0.5)

        results[task_id] = {
            "difficulty": task.difficulty,
            "reward": best_reward,
            "query": best_query,
            "solved": solved,
            "attempts": attempts_used,
        }

        status = "✓ SOLVED" if solved else "✗"
        print(f"  {task_id:<15} {task.difficulty:<10} reward={best_reward:.4f}  "
              f"attempts={attempts_used}  {status}")

    env.close()
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results: Dict[str, Dict], model: str) -> None:
    """Print formatted baseline results."""
    print(f"\n{'='*70}")
    print(f"  SQL Query Environment — Baseline Inference Results")
    print(f"  Model: {model}")
    print(f"{'='*70}")
    print(f"\n{'Task ID':<15} {'Difficulty':<12} {'Reward':>8} {'Solved':>8} {'Attempts':>10}")
    print("-" * 70)

    by_diff: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    solved_count = 0

    for task_id, info in sorted(results.items()):
        solved = "✓" if info["solved"] else "✗"
        print(f"{task_id:<15} {info['difficulty']:<12} {info['reward']:>8.4f} "
              f"{solved:>8} {info['attempts']:>10}")
        by_diff[info["difficulty"]].append(info["reward"])
        if info["solved"]:
            solved_count += 1

    print("-" * 70)
    print(f"\nSummary:")
    print(f"  {'Tier':<12} {'Mean Reward':>12} {'Solved':>8} {'Total':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*8}")

    all_rewards = []
    for diff in ["easy", "medium", "hard"]:
        rewards = by_diff[diff]
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        s = sum(1 for info in results.values()
                if info["difficulty"] == diff and info["solved"])
        all_rewards.extend(rewards)
        print(f"  {diff:<12} {mean_r:>12.4f} {s:>8} {len(rewards):>8}")

    overall = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    print(f"\n  {'OVERALL':<12} {overall:>12.4f} {solved_count:>8} {len(all_rewards):>8}")
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline inference using OpenAI API on the SQL Query Environment"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=3,
        help="Max attempts per task (default: 3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Dry run: use gold queries instead of calling OpenAI API",
    )

    args = parser.parse_args()

    # Read API key from environment
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        print("Or use --dry-run to test without API calls.")
        sys.exit(1)

    model = args.model
    print(f"Model: {model}")
    print(f"Max attempts: {args.max_attempts}")
    if args.dry_run:
        print("MODE: DRY RUN (using gold queries, no API calls)")
    print()

    results = run_baseline(
        api_key=api_key,
        model=model,
        max_attempts=args.max_attempts,
        dry_run=args.dry_run,
    )

    print_results(results, model)

    # Save results
    output_path = "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": model,
            "results": results,
        }, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
