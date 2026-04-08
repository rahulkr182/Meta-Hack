#!/usr/bin/env python3
"""Baseline Inference Script for the SQL Query Environment.

Uses the OpenAI client to run a model against the environment.
Reads API credentials from environment variables:
  - API_BASE_URL: Base URL for the OpenAI-compatible API
  - MODEL_NAME: Model to use for inference
  - HF_TOKEN: Hugging Face token for authentication (no default)

Produces a reproducible baseline score on all tasks.

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py
"""

import os
import sys
import json
import time

# ---------------------------------------------------------------------------
# Environment variables (required by hackathon checklist)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ---------------------------------------------------------------------------
# OpenAI client (required by hackathon checklist)
# ---------------------------------------------------------------------------

from openai import OpenAI

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "EMPTY",
)

# ---------------------------------------------------------------------------
# Environment imports
# ---------------------------------------------------------------------------

sys.path.insert(0, ".")

from sql_env.server.sql_environment import SqlEnvironment
from sql_env.models import SqlAction
from sql_env.tasks import TASKS, get_all_task_ids

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert SQL query writer. You are given a database schema and a natural-language question.
Write the correct SQL query that answers the question.

Rules:
- Output ONLY the SQL query, nothing else
- Do NOT include markdown code fences
- Do NOT include explanations
- Use standard SQLite-compatible SQL
- Use appropriate JOINs, GROUP BY, HAVING, window functions as needed"""


def build_prompt(schema, question, hint, attempt=1, prev_feedback=""):
    """Build the user prompt for the LLM."""
    prompt = f"Database Schema:\n{schema}\n\nQuestion: {question}\n\nHint: {hint}"
    if attempt > 1 and prev_feedback:
        prompt += f"\n\nPrevious attempt feedback:\n{prev_feedback}\n\nPlease fix your query."
    prompt += "\n\nSQL query:"
    return prompt


def call_llm(schema, question, hint, attempt=1, prev_feedback=""):
    """Call the LLM via OpenAI client to generate a SQL query."""
    user_msg = build_prompt(schema, question, hint, attempt, prev_feedback)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    content = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```sql"):
        content = content[6:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    return content.strip()


# ---------------------------------------------------------------------------
# Main inference loop with structured logging (START/STEP/END)
# ---------------------------------------------------------------------------

def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        print("Set it with: export HF_TOKEN='hf_...'")
        sys.exit(1)

    max_attempts = 3
    env = SqlEnvironment(max_attempts=max_attempts)
    all_task_ids = sorted(get_all_task_ids())

    # === START ===
    print(f"START | model={MODEL_NAME} | api={API_BASE_URL} | tasks={len(all_task_ids)}")

    results = {}
    total_reward = 0.0

    for task_id in all_task_ids:
        task = TASKS[task_id]
        obs = env.reset(task_id=task_id, seed=42)

        best_reward = 0.0
        best_query = ""
        solved = False
        prev_feedback = ""
        attempts_used = 0

        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt

            try:
                sql_query = call_llm(
                    schema=obs.schema_description,
                    question=obs.question,
                    hint=task.hint,
                    attempt=attempt,
                    prev_feedback=prev_feedback,
                )
            except Exception as e:
                print(f"STEP | task={task_id} | attempt={attempt} | error={e}")
                sql_query = "SELECT 1;"

            # Submit to environment
            obs = env.step(SqlAction(sql_query=sql_query))

            # === STEP ===
            print(f"STEP | task={task_id} | attempt={attempt} | reward={obs.reward:.4f} | done={obs.done}")

            if obs.reward > best_reward:
                best_reward = obs.reward
                best_query = sql_query

            if obs.done:
                solved = obs.reward_breakdown.get("exact_match", 0.0) >= 1.0
                break

            prev_feedback = obs.feedback
            time.sleep(0.5)

        total_reward += best_reward
        results[task_id] = {
            "difficulty": task.difficulty,
            "reward": best_reward,
            "solved": solved,
            "attempts": attempts_used,
            "query": best_query,
        }

    env.close()

    # === END ===
    mean_reward = total_reward / len(all_task_ids) if all_task_ids else 0.0
    solved_count = sum(1 for r in results.values() if r["solved"])

    print(f"END | mean_reward={mean_reward:.4f} | solved={solved_count}/{len(all_task_ids)}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  SQL Query Environment — Inference Results")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'='*70}")
    print(f"\n{'Task ID':<15} {'Difficulty':<12} {'Reward':>8} {'Solved':>8} {'Attempts':>10}")
    print("-" * 70)

    for task_id, info in sorted(results.items()):
        s = "✓" if info["solved"] else "✗"
        print(f"{task_id:<15} {info['difficulty']:<12} {info['reward']:>8.4f} {s:>8} {info['attempts']:>10}")

    print("-" * 70)
    print(f"\nOverall: mean_reward={mean_reward:.4f}, solved={solved_count}/{len(all_task_ids)}")

    # Save results
    with open("inference_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "results": results}, f, indent=2)
    print(f"\nResults saved to inference_results.json")


if __name__ == "__main__":
    main()
