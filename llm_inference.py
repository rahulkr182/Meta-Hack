#!/usr/bin/env python3
"""LLM-based Inference Script for the SQL Query Environment.

Uses an LLM (via OpenAI-compatible API) to generate SQL queries for all tasks.
Supports OpenAI, Together AI, Hugging Face Inference, and any OpenAI-compatible
endpoint.

Usage:
    # With OpenAI
    export OPENAI_API_KEY="sk-..."
    python3 llm_inference.py

    # With Together AI
    export TOGETHER_API_KEY="..."
    python3 llm_inference.py --provider together

    # With Hugging Face Inference
    export HF_TOKEN="hf_..."
    python3 llm_inference.py --provider huggingface

    # With a custom OpenAI-compatible endpoint
    python3 llm_inference.py --base-url http://localhost:11434/v1 --api-key ollama --model llama3

    # Dry run (print prompts without calling API)
    python3 llm_inference.py --dry-run
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")

from sql_env.server.sql_environment import SqlEnvironment
from sql_env.models import SqlAction
from sql_env.tasks import TASKS, get_all_task_ids


# ---------------------------------------------------------------------------
# Provider configurations
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "default_model": "meta-llama/Llama-3-70b-chat-hf",
    },
    "huggingface": {
        "base_url": "https://api-inference.huggingface.co/v1",
        "env_key": "HF_TOKEN",
        "default_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
}


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert SQL query writer. You will be given a database schema and a natural-language question. Your task is to write the correct SQL query that answers the question.

Rules:
- Write ONLY the SQL query, nothing else
- Do NOT include markdown code fences (```) 
- Do NOT include explanations
- Use standard SQL syntax compatible with SQLite
- Use appropriate JOINs, GROUP BY, HAVING, window functions as needed
- Return results sorted in a logical order when applicable"""


def build_user_prompt(schema: str, question: str, hint: str, attempt: int = 1, previous_feedback: str = "") -> str:
    """Build the user prompt for the LLM."""
    prompt = f"""Database Schema:
{schema}

Question: {question}

Hint: {hint}"""

    if attempt > 1 and previous_feedback:
        prompt += f"""

Previous attempt feedback:
{previous_feedback}

Please fix your query based on this feedback."""

    prompt += "\n\nWrite the SQL query:"
    return prompt


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

def call_llm(
    messages: List[Dict[str, str]],
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Call an OpenAI-compatible API and return the response text."""
    import httpx

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"].strip()

    # Clean up: remove markdown code fences if present
    if content.startswith("```sql"):
        content = content[6:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    return content.strip()


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_llm_inference(
    base_url: str,
    api_key: str,
    model: str,
    max_attempts: int = 3,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Dict]:
    """Run LLM inference on all tasks with multi-turn feedback.

    The agent gets up to max_attempts per task. After each failed attempt,
    the grader's feedback is included in the next prompt so the LLM can
    self-correct.
    """
    env = SqlEnvironment(max_attempts=max_attempts)
    results = {}

    all_ids = sorted(get_all_task_ids())

    for task_id in all_ids:
        task = TASKS[task_id]
        obs = env.reset(task_id=task_id, seed=42)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task_id} ({task.difficulty})")
            print(f"Question: {task.question}")

        best_reward = 0.0
        best_query = ""
        previous_feedback = ""
        solved = False

        for attempt in range(1, max_attempts + 1):
            # Build prompt
            user_prompt = build_user_prompt(
                schema=obs.schema_description,
                question=obs.question,
                hint=task.hint,
                attempt=attempt,
                previous_feedback=previous_feedback,
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            if dry_run:
                print(f"\n--- Task {task_id}, Attempt {attempt} ---")
                print(f"Prompt:\n{user_prompt[:500]}...")
                sql_query = task.gold_query  # Use gold for dry run
            else:
                try:
                    sql_query = call_llm(
                        messages=messages,
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                    )
                except Exception as e:
                    print(f"  ✗ API error on {task_id} attempt {attempt}: {e}")
                    sql_query = "SELECT 1;"  # Fallback

            if verbose:
                print(f"\n  Attempt {attempt}: {sql_query[:80]}...")

            # Submit to environment
            action = SqlAction(sql_query=sql_query)
            obs = env.step(action)

            if verbose:
                print(f"  Reward: {obs.reward:.4f}")
                if obs.reward < 1.0:
                    print(f"  Feedback: {obs.feedback[:200]}")

            if obs.reward > best_reward:
                best_reward = obs.reward
                best_query = sql_query

            if obs.done:
                solved = obs.reward_breakdown.get("exact_match", 0.0) >= 1.0
                break

            previous_feedback = obs.feedback

            # Small delay to respect rate limits
            if not dry_run:
                time.sleep(0.5)

        results[task_id] = {
            "difficulty": task.difficulty,
            "reward": best_reward,
            "query": best_query,
            "solved": solved,
            "attempts_used": attempt,
        }

        status = "✓ SOLVED" if solved else "✗"
        print(f"  {task_id:<15} {task.difficulty:<10} reward={best_reward:.4f}  "
              f"attempts={attempt}  {status}")

    env.close()
    return results


def print_summary(results: Dict[str, Dict], model: str) -> None:
    """Print a formatted summary of LLM inference results."""
    print(f"\n{'='*70}")
    print(f"  SQL Query Environment — LLM Inference Results")
    print(f"  Model: {model}")
    print(f"{'='*70}")
    print(f"\n{'Task ID':<15} {'Difficulty':<12} {'Reward':>8} {'Solved':>8} {'Attempts':>10}")
    print("-" * 70)

    by_diff: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    solved_count = 0

    for task_id, info in sorted(results.items()):
        solved = "✓" if info["solved"] else "✗"
        print(f"{task_id:<15} {info['difficulty']:<12} {info['reward']:>8.4f} "
              f"{solved:>8} {info['attempts_used']:>10}")
        by_diff[info["difficulty"]].append(info["reward"])
        if info["solved"]:
            solved_count += 1

    print("-" * 70)
    print(f"\nSummary:")
    print(f"  {'Tier':<12} {'Mean Reward':>12} {'Solved':>8} {'Total':>8}")

    all_rewards = []
    for diff in ["easy", "medium", "hard"]:
        rewards = by_diff[diff]
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        s = sum(1 for r in rewards if r >= 0.99)
        all_rewards.extend(rewards)
        print(f"  {diff:<12} {mean_r:>12.4f} {s:>8} {len(rewards):>8}")

    overall = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    print(f"\n  {'OVERALL':<12} {overall:>12.4f} {solved_count:>8} {len(all_rewards):>8}")
    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-based inference on the SQL Query Environment"
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()) + ["custom"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--api-key", help="API key (or set via env var)")
    parser.add_argument("--model", help="Model name")
    parser.add_argument(
        "--max-attempts", type=int, default=3,
        help="Max attempts per task (default: 3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without calling API",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Resolve provider config
    if args.provider == "custom":
        base_url = args.base_url or "http://localhost:11434/v1"
        api_key = args.api_key or "none"
        model = args.model or "llama3"
    else:
        config = PROVIDERS[args.provider]
        base_url = args.base_url or config["base_url"]
        api_key = args.api_key or os.environ.get(config["env_key"], "")
        model = args.model or config["default_model"]

    if not api_key and not args.dry_run:
        env_key = PROVIDERS.get(args.provider, {}).get("env_key", "API_KEY")
        print(f"ERROR: No API key provided. Set {env_key} env var or use --api-key")
        sys.exit(1)

    print(f"Provider: {args.provider}")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Max attempts: {args.max_attempts}")
    if args.dry_run:
        print("MODE: DRY RUN (using gold queries)")
    print()

    results = run_llm_inference(
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_attempts=args.max_attempts,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    print_summary(results, model)

    # Save results to JSON
    output_path = "llm_inference_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": model,
            "provider": args.provider,
            "results": results,
        }, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
