---
title: SQL Query Environment
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# 🧠 SQL Query Environment — OpenEnv

> An AI agent learns to write SQL queries from natural language — with schema discovery, multi-turn correction, and progressive difficulty. Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv).

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-v1.0-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What Is This?

A **complete, real-world OpenEnv environment** where an AI agent is given a database schema and a natural-language question, and must produce the correct SQL query. The environment executes the query against an in-memory SQLite database and returns graded feedback with **partial-progress reward signals**.

This simulates a genuinely useful real-world skill — data analysts and engineers write SQL queries every day.

### Why SQL Query Writing?

| ✅ Requirement | How We Meet It |
|---|---|
| Real-world task | SQL querying is one of the most common data tasks |
| Objectively gradable | Execute the query, compare results row-by-row |
| Natural difficulty tiers | Simple SELECT → JOINs → Window functions → CTEs + correlated subqueries |
| Rich partial signals | 6-component reward + step cost + first-attempt bonus |
| Novel mechanics | **Blind mode** — agent must discover schema before querying |
| No external dependencies | SQLite runs in-process — fully reproducible |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│  Agent / Training Loop                      │
│  ┌──────────────────────────────────────┐   │
│  │ SqlEnvClient (HTTP or WebSocket)     │   │
│  │  - reset(task_id, difficulty)        │   │
│  │  - step(sql_query)                   │   │
│  │  - state()                           │   │
│  └──────────┬───────────────────────────┘   │
└─────────────┼───────────────────────────────┘
              │ HTTP / WebSocket
┌─────────────▼───────────────────────────────┐
│  FastAPI Server (Docker Container)          │
│  ┌──────────────────────────────────────┐   │
│  │ SqlEnvironment                       │   │
│  │  - In-memory SQLite per episode      │   │
│  │  - Grader (6-component scoring)      │   │
│  │  - 9 tasks (easy/medium/hard)        │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

---

## 📦 Quick Start

### Installation

```bash
git clone https://github.com/rahulkr182/Meta-Hack.git
cd Meta-Hack
pip install -e ".[dev]"
```

### Run the Server

```bash
uvicorn sql_env.server.app:app --host 0.0.0.0 --port 8000
```

### Use the Client

```python
from sql_env.client import SqlEnvClient

with SqlEnvClient(base_url="http://localhost:8000") as client:
    # Reset to a specific task
    result = client.reset(task_id="easy_01")
    print(result["observation"]["question"])
    # → "List all employees in the Sales department..."

    # Submit a SQL query
    result = client.step("""
        SELECT e.first_name, e.last_name, e.salary
        FROM employees e
        JOIN departments d ON e.department_id = d.id
        WHERE d.name = 'Sales'
        ORDER BY e.last_name;
    """)
    print(result["reward"])         # → 1.0
    print(result["observation"]["feedback"])  # → "✓ EXACT MATCH — perfect result! 🎉"
```

### Direct Environment Usage (No Server)

```python
from sql_env.server.sql_environment import SqlEnvironment
from sql_env.models import SqlAction

env = SqlEnvironment(max_attempts=3)
obs = env.reset(task_id="easy_01")
print(obs.question)

obs = env.step(SqlAction(sql_query="SELECT * FROM employees;"))
print(f"Reward: {obs.reward:.4f}")  # Partial credit!
print(obs.feedback)  # Detailed grading breakdown
```

---

## 🔤 Action / Observation / State Spaces

### Action

| Field | Type | Description |
|-------|------|-------------|
| `sql_query` | `str` | The SQL query to execute (required, non-empty) |
| `metadata` | `dict` | Optional metadata |

### Observation

| Field | Type | Description |
|-------|------|-------------|
| `schema_description` | `str` | Database CREATE TABLE statements |
| `question` | `str` | Natural-language question to answer |
| `task_id` | `str` | Current task identifier |
| `task_difficulty` | `str` | "easy", "medium", or "hard" |
| `execution_result` | `str?` | Formatted query results (table) |
| `execution_error` | `str?` | Error message if query failed |
| `feedback` | `str` | Detailed grader feedback |
| `reward_breakdown` | `dict` | Per-component reward scores |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward for this step (0.0–1.0) |

### State

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | `str` | Unique episode identifier |
| `step_count` | `int` | Total steps taken |
| `current_task_id` | `str` | Active task ID |
| `task_difficulty` | `str` | Active task difficulty |
| `attempts` | `int` | Attempts used |
| `max_attempts` | `int` | Maximum attempts (default: 3) |
| `accumulated_reward` | `float` | Sum of all rewards |
| `best_reward` | `float` | Best single-step reward |
| `is_solved` | `bool` | Whether perfect score achieved |

---

## 📊 Tasks (4 Difficulty Tiers × 13 Tasks)

### Easy (Simple SELECT + WHERE)

| ID | Question | SQL Concepts |
|----|----------|-------------|
| `easy_01` | List all employees in the Sales department | JOIN + WHERE |
| `easy_02` | How many orders were placed in 2024? | COUNT + WHERE |
| `easy_03` | What are the distinct product categories? | SELECT from table |

### Medium (JOINs + Aggregation)

| ID | Question | SQL Concepts |
|----|----------|-------------|
| `medium_01` | Total revenue per product category | Multi-JOIN + GROUP BY + SUM |
| `medium_02` | Employees managing more than 2 people | Self-JOIN + HAVING |
| `medium_03` | Customers with at least 2 orders | JOIN + GROUP BY + HAVING |

### Hard (Window Functions + Subqueries)

| ID | Question | SQL Concepts |
|----|----------|-------------|
| `hard_01` | Rank departments by average salary | AVG + RANK() OVER |
| `hard_02` | Products never ordered | LEFT JOIN + IS NULL |
| `hard_03` | Cumulative total of daily revenue | SUM() OVER (running total) |

### Expert (CTEs + Correlated Subqueries + Multi-step Analysis) 🔥

| ID | Question | SQL Concepts |
|----|----------|-------------|
| `expert_01` | Full management chain per employee | Recursive CTE (WITH RECURSIVE) |
| `expert_02` | Department budget status classification | CASE + GROUP BY + conditional aggregation |
| `expert_03` | Employees above their department average | Correlated subquery / derived table |
| `expert_04` | Monthly revenue growth percentage | CTE + LAG() window function |

---

## 🏆 Reward Function

The grader uses a **6-component scoring rubric** with meaningful partial progress:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Syntax validity** | 0.10 | Query parses without error |
| **Execution success** | 0.10 | Query runs without runtime error |
| **Correct tables** | 0.15 | References the right table(s) |
| **Correct columns** | 0.15 | Selects the right columns |
| **Partial row match** | 0.20 | Jaccard similarity of result rows |
| **Exact match** | 0.30 | Results exactly match gold standard |

**Total = Σ (component_score × weight)** → 0.0 to 1.0

#### Adjustments

| Modifier | Value | Description |
|----------|-------|-------------|
| 🏆 First-attempt bonus | +0.05 | Rewarded for solving on first try |
| ⏱️ Step cost | -0.02/attempt | Penalty for each extra attempt after the first |
| SELECT * penalty | 0.30 | Using `SELECT *` only gets 30% column credit |

### Partial Progress Example

```
"SELECT 1;"                          → 0.20  (syntax + execution only)
"SELECT * FROM employees;"           → 0.35  (+ partial table/column credit)
"SELECT first_name FROM employees;"  → 0.45  (+ better column credit)
"SELECT ... correct query ..."       → 1.00  (perfect score!)
```

---

## 📈 Baseline Results

The deterministic baseline generates `SELECT * FROM <first_table> LIMIT 10` for every task:

```
Task ID         Difficulty   Reward   Solved
─────────────────────────────────────────────
easy_01         easy         0.3500   ✗
easy_02         easy         0.3500   ✗
easy_03         easy         0.2750   ✗
medium_01       medium       0.3500   ✗
medium_02       medium       0.2750   ✗
medium_03       medium       0.2750   ✗
hard_01         hard         0.3500   ✗
hard_02         hard         0.2750   ✗
hard_03         hard         0.2750   ✗
─────────────────────────────────────────────
OVERALL                      0.3083
```

Run it yourself: `python3 baseline_inference.py`

### LLM-based Agent

We also include an **LLM-based inference script** that uses any OpenAI-compatible API to generate SQL queries. The agent gets multi-turn feedback — after a failed attempt, the grader's feedback is included in the next prompt, enabling self-correction.

```bash
# With OpenAI
export OPENAI_API_KEY="sk-..."
python3 llm_inference.py

# With Together AI
export TOGETHER_API_KEY="..."
python3 llm_inference.py --provider together

# With Hugging Face
export HF_TOKEN="hf_..."
python3 llm_inference.py --provider huggingface

# With local Ollama
python3 llm_inference.py --provider custom --base-url http://localhost:11434/v1 --model llama3

# Dry run (no API calls, uses gold queries)
python3 llm_inference.py --dry-run
```

| Flag | Description |
|------|-------------|
| `--provider` | `openai`, `together`, `huggingface`, or `custom` |
| `--model` | Override the default model |
| `--max-attempts` | Attempts per task (default: 3, with feedback loop) |
| `--verbose` / `-v` | Show full query + feedback per attempt |
| `--dry-run` | Print prompts only, no API calls |

---

## 🐳 Docker

### Build

```bash
docker build -t sql-query-env:latest .
```

### Run

```bash
docker run -p 7860:7860 sql-query-env:latest
```

### Verify

```bash
curl http://localhost:7860/health
# → {"status": "healthy"}
```

---

## 🚀 Deployed on Hugging Face Spaces

**Live:** [https://huggingface.co/spaces/Rahulkr1/sql-query-env](https://huggingface.co/spaces/Rahulkr1/sql-query-env)

```bash
# Test the live deployment
curl https://Rahulkr1-sql-query-env.hf.space/health
# → {"status": "healthy"}

curl -X POST https://Rahulkr1-sql-query-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy_01"}'
```

To deploy your own:
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select "Docker" as the SDK
3. Push this repository to the Space

---

## 🧪 Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run evaluation harness (gold query verification + partial credit demo)
python3 run_evaluation.py

# Run baseline inference
python3 baseline_inference.py
```

---

## 📁 Project Structure

```
Meta-Hack/
├── sql_env/
│   ├── __init__.py              # Package exports
│   ├── models.py                # Action, Observation, State (Pydantic)
│   ├── tasks.py                 # 13 tasks with schema, seed data, gold queries
│   ├── grader.py                # 6-component scoring + step cost + first-attempt bonus
│   ├── db_utils.py              # SQLite helpers (create, query, format)
│   ├── client.py                # HTTP + WebSocket clients
│   ├── openenv.yaml             # OpenEnv manifest
│   └── server/
│       ├── sql_environment.py   # Environment (reset/step/state + blind mode)
│       ├── app.py               # FastAPI server (REST + WebSocket)
│       ├── requirements.txt     # Docker dependencies
│       └── Dockerfile           # Container image
├── inference.py                 # Hackathon submission inference script
├── baseline_inference.py        # Deterministic baseline script
├── llm_inference.py             # LLM-based inference (OpenAI/Together/HF)
├── run_evaluation.py            # Gold verification + partial credit demo
├── tests/
│   └── test_sql_env.py          # Unit tests
├── Dockerfile                   # Top-level Docker build
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

---

## 🔗 API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/schema` | JSON schemas for Action/Observation/State |
| GET | `/state` | Current environment state |
| POST | `/reset` | Reset environment (body: `{task_id, difficulty, seed}`) |
| POST | `/step` | Submit action (body: `{action: {sql_query: "..."}}`) |

### WebSocket

Connect to `ws://host:8000/ws` and send JSON messages:

```json
{"type": "reset", "data": {"task_id": "easy_01"}}
{"type": "step", "data": {"sql_query": "SELECT ..."}}
{"type": "state"}
{"type": "close"}
```

---

## 📄 License

MIT
