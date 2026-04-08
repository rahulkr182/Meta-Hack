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
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
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

BENCHMARK = "sql-query-env"
MAX_STEPS = 3
TEMPERATURE = 0.0
MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# OpenAI Client (MANDATORY)
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "EMPTY",
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
You are an expert SQL query writer. You are given a database schema and a natural-language question.
Write the correct SQL query that answers the question.

Rules:
- Output ONLY the SQL query, nothing else
- Do NOT include markdown code fences
- Do NOT include explanations
- Use standard SQLite-compatible SQL
""").strip()


def build_user_prompt(schema, question, hint, attempt=1, prev_feedback=""):
    prompt = f"Database Schema:\n{schema}\n\nQuestion: {question}\n\nHint: {hint}"
    if attempt > 1 and prev_feedback:
        prompt += f"\n\nPrevious attempt feedback:\n{prev_feedback}\nPlease fix your query."
    prompt += "\n\nSQL query:"
    return prompt


def call_llm(schema, question, hint, attempt=1, prev_feedback=""):
    """Call the LLM via OpenAI Client."""
    user_msg = build_user_prompt(schema, question, hint, attempt, prev_feedback)

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
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    obs = env.reset(task_id=task_id, seed=42)

    rewards = []
    step_count = 0
    best_reward = 0.0
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
            )
            last_error = None
        except Exception as e:
            sql_query = "SELECT 1;"
            last_error = str(e)

        # Step
        obs = env.step(SqlAction(sql_query=sql_query))
        reward = obs.reward
        rewards.append(reward)

        if reward > best_reward:
            best_reward = reward

        done = obs.done
        error_str = last_error if last_error else (obs.last_action_error if hasattr(obs, 'last_action_error') and obs.last_action_error else "null")

        # [STEP]
        print(f"[STEP] step={step_count} action={sql_query!r} reward={reward:.2f} done={'true' if done else 'false'} error={error_str}")

        if done:
            break

        prev_feedback = obs.feedback

    # Score normalized to [0, 1]
    score = best_reward  # already in [0, 1]
    success = score >= 0.99
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # [END]
    print(f"[END] success={'true' if success else 'false'} steps={step_count} score={score:.2f} rewards={rewards_str}")

    return score


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        print("Set it with: export HF_TOKEN='hf_...'")
        sys.exit(1)

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
