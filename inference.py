"""
Inference Script for SQL Query Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import subprocess
import textwrap

# Auto-install dependencies if missing
try:
    from openai import OpenAI
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai>=1.0.0", "-q"])
    from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables (MANDATORY)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "sql-query-env"
MAX_STEPS = 3
TEMPERATURE = 0.0
MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# OpenAI Client (MANDATORY)
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sql_env.server.sql_environment import SqlEnvironment
from sql_env.models import SqlAction
from sql_env.tasks import TASKS, get_all_task_ids

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert SQL query writer specializing in SQLite. You are given a database schema
and a natural-language question. Write the correct SQL query that answers the question.

Rules:
- Output ONLY the SQL query, nothing else
- Do NOT include markdown code fences, explanations, or comments
- Use standard SQLite-compatible SQL
- Use explicit column names (never SELECT *)
- Use table aliases for clarity (e.g., e for employees, d for departments)
- Always include ORDER BY when the question mentions sorting or ranking
- For aggregations, use meaningful column aliases with AS

SQLite-specific notes:
- Use strftime('%Y-%m', date_col) for month extraction
- Use ROUND(value, 2) for decimal rounding
- Window functions (RANK, ROW_NUMBER, LAG, SUM OVER) are supported
- Use WITH RECURSIVE for recursive CTEs
- Use CASE WHEN ... THEN ... ELSE ... END for conditional logic
""").strip()

FEW_SHOT_EXAMPLES = textwrap.dedent("""
Example 1:
Question: How many employees are in each department?
SQL: SELECT d.name AS department, COUNT(e.id) AS emp_count FROM employees e JOIN departments d ON e.department_id = d.id GROUP BY d.id, d.name ORDER BY emp_count DESC;

Example 2:
Question: Find the running total of order amounts by date.
SQL: SELECT order_date, total_amount, SUM(total_amount) OVER (ORDER BY order_date, id) AS running_total FROM orders ORDER BY order_date, id;
""").strip()


def build_user_prompt(schema, question, hint, attempt=1, prev_feedback="", difficulty="easy"):
    prompt = f"Database Schema:\n{schema}\n\nQuestion: {question}"
    if hint and not hint.startswith("[BLIND"):
        prompt += f"\n\nHint: {hint}"
    if difficulty in ("hard", "expert"):
        prompt += f"\n\n{FEW_SHOT_EXAMPLES}"
    if attempt > 1 and prev_feedback:
        prompt += f"\n\nYour previous attempt got this feedback:\n{prev_feedback}\n\nFix the issues and try again."
    prompt += "\n\nSQL query:"
    return prompt


def call_llm(schema, question, hint, attempt=1, prev_feedback="", difficulty="easy"):
    """Call the LLM via OpenAI Client."""
    user_msg = build_user_prompt(schema, question, hint, attempt, prev_feedback, difficulty)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
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
# Main
# ---------------------------------------------------------------------------

def run_task(env, task_id):
    """Run a single task and emit structured [START]/[STEP]/[END] logs."""
    task = TASKS[task_id]

    # [START]
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards = []
    step_count = 0
    best_reward = 0.0

    try:
        obs = env.reset(task_id=task_id, seed=42)

        prev_feedback = ""
        last_error = None

        for attempt in range(1, MAX_STEPS + 1):
            step_count = attempt

            try:
                sql_query = call_llm(
                    schema=obs.schema_description,
                    question=obs.question,
                    hint=task.hint,
                    attempt=attempt,
                    prev_feedback=prev_feedback,
                    difficulty=task.difficulty,
                )
                last_error = None
            except Exception as e:
                sql_query = "SELECT 1;"
                last_error = str(e)

            # Step
            obs = env.step(SqlAction(sql_query=sql_query))
            # Clamp reward to strictly (0, 1) — evaluator rejects 0.0 and 1.0
            reward = max(0.01, min(0.99, obs.reward))
            rewards.append(reward)

            if reward > best_reward:
                best_reward = reward

            done = obs.done

            # Sanitize error and action for single-line output
            error_str = last_error if last_error else (
                obs.execution_error if obs.execution_error else "null"
            )
            error_str = str(error_str).replace('\n', ' ').replace('\r', '')
            action_str = sql_query.replace('\n', ' ').replace('\r', '').strip()

            # [STEP] — use :.4f to avoid rounding 0.001→0.00 or 0.999→1.00
            print(f"[STEP] step={step_count} action={action_str} reward={reward:.4f} done={'true' if done else 'false'} error={error_str}", flush=True)

            if done:
                break

            prev_feedback = obs.feedback

    except Exception as e:
        # Ensure [END] is always emitted even on exception
        if not rewards:
            rewards = [0.01]
            step_count = 1
        error_msg = str(e).replace('\n', ' ')
        print(f"[STEP] step={step_count} action=ERROR reward=0.0100 done=true error={error_msg}", flush=True)

    # [END] — always emitted; clamp each reward and use :.4f
    clamped = [max(0.01, min(0.99, r)) for r in rewards]
    rewards_str = ",".join(f"{r:.4f}" for r in clamped)
    success = best_reward >= 0.99
    print(f"[END] success={'true' if success else 'false'} steps={step_count} rewards={rewards_str}", flush=True)

    return best_reward


def main():
    env = SqlEnvironment(max_attempts=MAX_STEPS)
    all_task_ids = sorted(get_all_task_ids())

    total_score = 0.0
    results = {}

    for task_id in all_task_ids:
        score = run_task(env, task_id)
        total_score += score
        results[task_id] = score

    env.close()

    # Summary
    mean_score = total_score / len(all_task_ids) if all_task_ids else 0.0
    print(f"\n--- Summary: mean_score={mean_score:.4f} ({sum(1 for s in results.values() if s >= 0.99)}/{len(all_task_ids)} solved) ---")

    # Save results
    with open("inference_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "results": results, "mean_score": mean_score}, f, indent=2)


if __name__ == "__main__":
    main()
